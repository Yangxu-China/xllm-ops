/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "x_attention_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef OP_LOGE
#include <cmath>
#include <iostream>
#define OP_LOGE(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")
#endif

namespace optiling {

enum InputIndex {
    QUERY = 0,
    SHARED_KEY_BLOCK,
    SHARED_VALUE_BLOCK,
    UNSHARED_KEY_BLOCK,
    UNSHARED_VALUE_BLOCK,
    UNSHARED_BLOCK_TABLE,
    SHARED_KV_LENS,
    DECODE_STEP,
    SHARED_BLOCK_TABLE,
};

constexpr int64_t TS_TND = 128;
constexpr int64_t ML_W = 8;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t UNSHARED_Q_TILE = 128;
constexpr int64_t UNSHARED_KV_TILE = 256;
constexpr int64_t FLOAT_BLOCK_SIZE = 8;
constexpr int64_t SCALE_VALUE_ATTR_INDEX = 0;

class TilingXAttentionV2Func {
public:
    explicit TilingXAttentionV2Func(gert::TilingContext *tiling_context)
        : tiling_context_(tiling_context) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ParseShapeAndAttrs();
    void FillCoreSplitAndRanges();
    void SetWorkspaces();

    gert::TilingContext *tiling_context_ = nullptr;
    XAttentionV2TilingData tiling_data_;
    int64_t numTokens_ = 0;
    int64_t qHeadNum_ = 0;
    int64_t headDim_ = 0;
    int64_t batch_ = 0;
    int64_t beamSize_ = 0;
    int64_t kvHeadNum_ = 0;
    int64_t maxDecodeStep_ = 0;
    int64_t sharedTotal_ = 0;
    bool isSharedPaged_ = false;
    bool isUnsharedPaged_ = false;
    int64_t sharedTableMaxBlocks_ = 0;
    float scaleValue_ = 0.0f;
    int64_t groupSize_ = 1;
    int64_t sharedCoreNum_ = 0;
    int64_t unsharedCoreNum_ = 0;
    uint32_t cubeCoreNum_ = 0;
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

inline int64_t UnsharedMergeFactor(int64_t beam, int64_t hkv, int64_t groupSize, int64_t maxDecodeStep)
{
    int64_t cap = std::min(TS_TND / maxDecodeStep, TS_TND / groupSize);
    cap = std::min(cap, beam * hkv);
    return std::max<int64_t>(1, cap);
}

ge::graphStatus TilingXAttentionV2Func::ParseShapeAndAttrs()
{
    auto queryShape = tiling_context_->GetInputShape(QUERY)->GetStorageShape();
    auto sharedKeyBlockShape = tiling_context_->GetInputShape(SHARED_KEY_BLOCK)->GetStorageShape();
    auto unsharedKeyBlockShape = tiling_context_->GetInputShape(UNSHARED_KEY_BLOCK)->GetStorageShape();
    auto sharedTableShapePtr = tiling_context_->GetOptionalInputShape(SHARED_BLOCK_TABLE);
    auto unsharedTableShapePtr = tiling_context_->GetOptionalInputShape(UNSHARED_BLOCK_TABLE);

    numTokens_ = queryShape.GetDim(0);
    qHeadNum_ = queryShape.GetDim(1);
    headDim_ = queryShape.GetDim(2);
    kvHeadNum_ = sharedKeyBlockShape.GetDim(1);
    sharedTotal_ = sharedKeyBlockShape.GetDim(0);
    batch_ = unsharedKeyBlockShape.GetDim(0);
    beamSize_ = unsharedKeyBlockShape.GetDim(1);
    maxDecodeStep_ = unsharedKeyBlockShape.GetDim(3);
    isSharedPaged_ = (sharedTableShapePtr != nullptr);
    isUnsharedPaged_ = (unsharedTableShapePtr != nullptr);
    if (isSharedPaged_) {
        sharedTableMaxBlocks_ = sharedTableShapePtr->GetStorageShape().GetDim(1);
    }
    groupSize_ = qHeadNum_ / kvHeadNum_;
    if (batch_ <= 0 || beamSize_ <= 0 || numTokens_ != batch_ * beamSize_) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "x_attention_v2: numTokens(%ld) must equal batch(%ld) * beam(%ld).",
                numTokens_, batch_, beamSize_);
        return ge::GRAPH_FAILED;
    }
    if (qHeadNum_ % kvHeadNum_ != 0) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "x_attention_v2: query heads(%ld) must be divisible by kv heads(%ld).",
                qHeadNum_, kvHeadNum_);
        return ge::GRAPH_FAILED;
    }
    if (qHeadNum_ / kvHeadNum_ > 128) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "x_attention_v2: GQA group size(%ld) > 128 unsupported (requires Hq <= 128 * Hkv).",
                qHeadNum_ / kvHeadNum_);
        return ge::GRAPH_FAILED;
    }
    if (maxDecodeStep_ > 128) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "x_attention_v2: maxDecodeStep(%ld) > 128 unsupported.", maxDecodeStep_);
        return ge::GRAPH_FAILED;
    }
    if (headDim_ != BLOCK_SIZE) {
        OP_LOGE(tiling_context_->GetNodeName(), "x_attention_v2: head_dim must be 128, got %ld.", headDim_);
        return ge::GRAPH_FAILED;
    }
    scaleValue_ = static_cast<float>(1.0 / std::sqrt(1.0 * headDim_));
    auto attrs = tiling_context_->GetAttrs();
    if (attrs != nullptr) {
        const auto *attr_scale = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
        if (attr_scale != nullptr && *attr_scale > 0.0f) {
            scaleValue_ = *attr_scale;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void TilingXAttentionV2Func::FillCoreSplitAndRanges()
{
    int64_t sharedTilesPerUnit = std::max<int64_t>(1, CeilDiv(beamSize_, TS_TND));
    int64_t gMerge = UnsharedMergeFactor(beamSize_, kvHeadNum_, groupSize_, maxDecodeStep_);
    int64_t itemsPerBatch = CeilDiv(beamSize_ * kvHeadNum_, gMerge);
    int64_t maxPrompt = batch_ > 0 ? sharedTotal_ / batch_ : sharedTotal_;
    int64_t kvLoop = CeilDiv(std::max<int64_t>(maxPrompt, 1), TS_TND);
    int64_t sharedWork = batch_ * qHeadNum_ * sharedTilesPerUnit;
    int64_t unsharedWork = batch_ * itemsPerBatch;
    double ratio = static_cast<double>(unsharedWork) / (static_cast<double>(sharedWork * kvLoop) + 0.001);
    int64_t unsharedCores = static_cast<int64_t>(std::round(
        static_cast<double>(cubeCoreNum_) * ratio / (1.0 + ratio)));
    unsharedCores = std::max<int64_t>(6, std::min<int64_t>(cubeCoreNum_ - 2, unsharedCores));
    int64_t sharedCores = cubeCoreNum_ - unsharedCores;
    sharedCoreNum_ = sharedCores;
    unsharedCoreNum_ = unsharedCores;
}

void TilingXAttentionV2Func::SetWorkspaces()
{
    auto platform_info =
        platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
    size_t systemWorkspaceSize = static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
    int64_t flatRows = numTokens_ * qHeadNum_;
    size_t userWorkspaceSize = static_cast<size_t>(
        2 * flatRows * headDim_ * 2 + 4 * flatRows * ML_W * 4);
    size_t *workspace = tiling_context_->GetWorkspaceSizes(1);
    workspace[0] = systemWorkspaceSize + userWorkspaceSize;
}

ge::graphStatus TilingXAttentionV2Func::RunTiling()
{
    auto platform_info =
        platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
    cubeCoreNum_ = platform_info.GetCoreNumAic();
    auto ret = ParseShapeAndAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    FillCoreSplitAndRanges();

    tiling_data_.set_batch(batch_);
    tiling_data_.set_beam(beamSize_);
    tiling_data_.set_hq(qHeadNum_);
    tiling_data_.set_hkv(kvHeadNum_);
    tiling_data_.set_shared_total(sharedTotal_);
    tiling_data_.set_u_maxds(maxDecodeStep_);
    tiling_data_.set_scale(scaleValue_);
    tiling_data_.set_sbt_stride(isSharedPaged_ ? sharedTableMaxBlocks_ : 0);
    tiling_data_.set_shared_core_num(sharedCoreNum_);
    tiling_data_.set_total_cores(sharedCoreNum_ + unsharedCoreNum_);

    SetWorkspaces();

    uint64_t tilingKey = GET_TPL_TILING_KEY(isSharedPaged_ ? 1u : 0u, isUnsharedPaged_ ? 1u : 0u);
    tiling_context_->SetTilingKey(tilingKey);

    tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                              tiling_context_->GetRawTilingData()->GetCapacity());
    tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
    tiling_context_->SetBlockDim(static_cast<uint32_t>(sharedCoreNum_ + unsharedCoreNum_));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingXAttentionV2Func tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "x_attention_v2 tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XAttentionV2)
    .Tiling(TilingFunc);
} // namespace optiling
