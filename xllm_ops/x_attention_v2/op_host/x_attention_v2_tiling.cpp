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
#include <vector>
#include <algorithm>
#include <numeric>
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
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t SCALE_VALUE_ATTR_INDEX = 0;

class TilingXAttentionV2Func {
public:
    explicit TilingXAttentionV2Func(gert::TilingContext *tiling_context)
        : tiling_context_(tiling_context) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ParseShapeAndAttrs();
    bool SelectTiling();
    void BuildTaskTable();
    void SetWorkspaces();
    void SetTilingKey();

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
    float scaleValue_ = 0.0f;
    int64_t groupSize_ = 1;
    uint32_t cubeCoreNum_ = 0;

    // Task-local tiling
    int64_t sharedM_ = 0;
    int64_t beamsPerTask_ = 0;
    int64_t unsharedN_ = 0;
    int64_t taskCount_ = 0;
    int64_t decodeStep_ = 0;  // runtime value from decode_step tensor
    std::vector<int32_t> taskTable_;
};

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

ge::graphStatus TilingXAttentionV2Func::ParseShapeAndAttrs()
{
    auto queryShape = tiling_context_->GetInputShape(QUERY)->GetStorageShape();
    auto sharedKeyBlockShape = tiling_context_->GetInputShape(SHARED_KEY_BLOCK)->GetStorageShape();
    auto unsharedKeyBlockShape = tiling_context_->GetInputShape(UNSHARED_KEY_BLOCK)->GetStorageShape();

    numTokens_ = queryShape.GetDim(0);
    qHeadNum_ = queryShape.GetDim(1);
    headDim_ = queryShape.GetDim(2);
    kvHeadNum_ = sharedKeyBlockShape.GetDim(1);
    sharedTotal_ = sharedKeyBlockShape.GetDim(0);
    batch_ = unsharedKeyBlockShape.GetDim(0);
    beamSize_ = unsharedKeyBlockShape.GetDim(1);
    maxDecodeStep_ = unsharedKeyBlockShape.GetDim(3);
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
    if (headDim_ != BLOCK_SIZE) {
        OP_LOGE(tiling_context_->GetNodeName(), "x_attention_v2: head_dim must be 128, got %ld.", headDim_);
        return ge::GRAPH_FAILED;
    }
    if (maxDecodeStep_ > 128) {
        OP_LOGE(tiling_context_->GetNodeName(),
                "x_attention_v2: maxDecodeStep(%ld) > 128 unsupported.", maxDecodeStep_);
        return ge::GRAPH_FAILED;
    }

    // Read decode_step runtime value from tensor (mirrors Python: int(decode_step.item()))
    auto decodeStepTensor = tiling_context_->GetInputTensor(DECODE_STEP);
    if (decodeStepTensor != nullptr && decodeStepTensor->GetData<int32_t>() != nullptr) {
        decodeStep_ = *decodeStepTensor->GetData<int32_t>();
    } else {
        decodeStep_ = maxDecodeStep_;  // fallback
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

bool TilingXAttentionV2Func::SelectTiling()
{
    // Mirrors Python _select_tiling: try shared_m=128 first, then 64
    if (groupSize_ != 2 && groupSize_ != 4) return false;
    if (decodeStep_ < 1 || decodeStep_ > 4) return false;

    for (int64_t shared_m : {128, 64}) {
        int64_t bpt = shared_m / groupSize_;
        if (beamSize_ % bpt != 0) continue;
        if (bpt * maxDecodeStep_ > TS_TND) continue;
        int64_t task_count = batch_ * kvHeadNum_ * (beamSize_ / bpt);
        if (shared_m == 128 && task_count < static_cast<int64_t>(cubeCoreNum_) * 2) continue;
        sharedM_ = shared_m;
        beamsPerTask_ = bpt;
        unsharedN_ = bpt * maxDecodeStep_;
        return true;
    }
    return false;
}

void TilingXAttentionV2Func::BuildTaskTable()
{
    // Mirrors Python _build_task_table
    int64_t beamGroups = beamSize_ / beamsPerTask_;
    std::vector<std::array<int32_t, 6>> tasks;
    int64_t sharedTokenStart = 0;

    // Read shared_kv_lens from input tensor (mirrors Python: shared_kv_lens.cpu().tolist())
    auto sharedKvLensTensor = tiling_context_->GetInputTensor(SHARED_KV_LENS);
    std::vector<int32_t> sharedLens(batch_);
    if (sharedKvLensTensor != nullptr && sharedKvLensTensor->GetData<int32_t>() != nullptr) {
        const int32_t *data = sharedKvLensTensor->GetData<int32_t>();
        for (int64_t i = 0; i < batch_; i++) {
            sharedLens[i] = data[i];
        }
    } else {
        int64_t avgLen = batch_ > 0 ? sharedTotal_ / batch_ : 0;
        for (int64_t i = 0; i < batch_; i++) {
            sharedLens[i] = static_cast<int32_t>(avgLen);
        }
    }

    for (int64_t ridx = 0; ridx < batch_; ridx++) {
        int32_t slen = sharedLens[ridx];
        int32_t stiles = static_cast<int32_t>(CeilDiv(slen, TS_TND));
        for (int64_t kh = 0; kh < kvHeadNum_; kh++) {
            for (int64_t bg = 0; bg < beamGroups; bg++) {
                int32_t bs = static_cast<int32_t>(bg * beamsPerTask_);
                tasks.push_back({static_cast<int32_t>(sharedTokenStart),
                                static_cast<int32_t>(ridx),
                                static_cast<int32_t>(kh),
                                bs, slen, stiles});
            }
        }
        sharedTokenStart += slen;
    }

    // Sort by (-shared_tiles, ridx, kh, beam_start)
    std::sort(tasks.begin(), tasks.end(), [](const auto &a, const auto &b) {
        if (a[5] != b[5]) return a[5] > b[5];
        if (a[1] != b[1]) return a[1] < b[1];
        if (a[2] != b[2]) return a[2] < b[2];
        return a[3] < b[3];
    });

    taskCount_ = static_cast<int64_t>(tasks.size());
    taskTable_.resize(taskCount_ * 6);
    for (int64_t i = 0; i < taskCount_; i++) {
        for (int j = 0; j < 6; j++) {
            taskTable_[i * 6 + j] = tasks[i][j];
        }
    }
}

void TilingXAttentionV2Func::SetWorkspaces()
{
    auto platform_info =
        platform_ascendc::PlatformAscendC(tiling_context_->GetPlatformInfo());
    size_t systemWorkspaceSize = static_cast<size_t>(platform_info.GetLibApiWorkSpaceSize());
    // Group-local: no user workspace for partials (results stay in UB)
    // But we need workspace for: task_table, permuted Q/K/V, permuted output
    // task_table: taskCount_ * 6 * sizeof(int32_t)
    // permuted Q: same size as query (numTokens_ * qHeadNum_ * headDim_ * dtype_size)
    // permuted K/V: same size as unshared_k/v
    // permuted O: same size as query
    size_t dtype_size = 2;  // bf16/fp16
    size_t taskTableSize = static_cast<size_t>(taskCount_ * 6 * sizeof(int32_t));
    size_t permQSize = static_cast<size_t>(numTokens_ * qHeadNum_ * headDim_ * dtype_size);
    size_t permKVSize = static_cast<size_t>(batch_ * beamSize_ * kvHeadNum_ * maxDecodeStep_ * headDim_ * dtype_size);
    size_t userWorkspaceSize = taskTableSize + permQSize + permKVSize * 2 + permQSize;  // task_table + q_perm + uk_perm + uv_perm + o_perm
    size_t *workspace = tiling_context_->GetWorkspaceSizes(1);
    workspace[0] = systemWorkspaceSize + userWorkspaceSize;
}

void TilingXAttentionV2Func::SetTilingKey()
{
    // TilingKey: SharedM (0=64, 1=128) — mirrors Python XAttnV2TilingKey
    uint64_t key = (sharedM_ == 128) ? 1 : 0;
    tiling_context_->SetTilingKey(key);
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
    if (!SelectTiling()) {
        OP_LOGE(tiling_context_->GetNodeName(), "x_attention_v2: no valid tiling for group=%ld, ds=%ld, beam=%ld",
                groupSize_, decodeStep_, beamSize_);
        return ge::GRAPH_FAILED;
    }
    BuildTaskTable();

    tiling_data_.set_hq(qHeadNum_);
    tiling_data_.set_hkv(kvHeadNum_);
    tiling_data_.set_batch(batch_);
    tiling_data_.set_beam_size(beamSize_);
    tiling_data_.set_shared_m(sharedM_);
    tiling_data_.set_group(groupSize_);
    tiling_data_.set_unshared(decodeStep_);
    tiling_data_.set_max_ds(maxDecodeStep_);
    tiling_data_.set_shared_total(sharedTotal_);
    tiling_data_.set_scale(scaleValue_);
    tiling_data_.set_num_tokens(numTokens_);
    tiling_data_.set_total_cores(static_cast<int64_t>(cubeCoreNum_));
    tiling_data_.set_task_count(taskCount_);
    tiling_data_.set_beams_per_task(beamsPerTask_);
    tiling_data_.set_unshared_n(unsharedN_);

    SetWorkspaces();
    SetTilingKey();

    tiling_data_.SaveToBuffer(tiling_context_->GetRawTilingData()->GetData(),
                              tiling_context_->GetRawTilingData()->GetCapacity());
    tiling_context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
    tiling_context_->SetBlockDim(static_cast<uint32_t>(cubeCoreNum_));
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
