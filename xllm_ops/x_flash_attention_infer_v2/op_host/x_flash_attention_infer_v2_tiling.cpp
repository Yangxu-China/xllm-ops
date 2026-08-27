/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version  2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "x_flash_attention_infer_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cmath>
#include <algorithm>
#include <vector>

#ifndef OP_LOGE
#include <iostream>
#define OP_LOGE(nodeName, fmt, ...) \
  printf(fmt, ##__VA_ARGS__);       \
  printf("\n")
#endif

namespace optiling {

enum InputIndex {
    QUERY = 0,
    KEY_CACHE,
    VALUE_CACHE,
    MASK,
    BLOCK_TABLE,
    ACTUAL_Q_LENS,
    ACTUAL_KV_LENS,
    EXTRA_TILING,
};

enum AttrIndex {
    LAYOUT_ATTR = 0,
    QHEAD_ATTR,
    KVHEAD_ATTR,
    SCALE_ATTR,
};

constexpr int32_t TILE_D = 128;
constexpr int32_t TILE_N = 256;
constexpr int32_t NZ_FRACTAL = 16;
constexpr int32_t MAX_CORES = 64;
constexpr int32_t MAX_SPLITS = 8;

constexpr int32_t HEADER_INTS = 8;
constexpr int32_t CORE_FIELDS = 4;
constexpr int32_t ITEM_FIELDS = 8;
constexpr int32_t MERGE_FIELDS = 6;

struct PlanItem {
    int32_t batch;
    int32_t kv_head;
    int32_t group_tile;
    int32_t kv_lo;
    int32_t kv_hi;
    int32_t out_slot;
    int32_t rows;
    int32_t direct;
};

struct PlanMerge {
    int32_t batch;
    int32_t kv_head;
    int32_t group_tile;
    int32_t slot0;
    int32_t nsplit;
    int32_t rows;
};

struct PlanCoreRange {
    int32_t item_begin;
    int32_t item_end;
    int32_t merge_begin;
    int32_t merge_end;
};

static int32_t CeilDiv(int32_t a, int32_t b) { return (a + b - 1) / b; }

static int32_t ChooseTileM(int32_t group_size) {
    return group_size <= 16 ? 16 : 32;
}

static int32_t GroupTiles(int32_t group_size) {
    return CeilDiv(group_size, 128);
}

static int32_t RowsOfGroup(int32_t group_size, int32_t g) {
    int32_t remaining = group_size - g * 128;
    return std::min(128, remaining);
}

static std::vector<std::pair<int32_t, int32_t>> SplitPoints(int32_t kv_len, int32_t splits) {
    int32_t tiles = CeilDiv(kv_len, TILE_N);
    int32_t n = std::max(1, std::min(splits, tiles));
    std::vector<std::pair<int32_t, int32_t>> out;
    for (int32_t i = 0; i < n; i++) {
        int32_t lo_tile = tiles * i / n;
        int32_t hi_tile = tiles * (i + 1) / n;
        if (hi_tile <= lo_tile) continue;
        out.push_back({lo_tile * TILE_N, std::min(hi_tile * TILE_N, kv_len)});
    }
    if (out.empty()) out.push_back({0, kv_len});
    return out;
}

static std::vector<int32_t> GroupCosts(int32_t batch, int32_t kv_head, int32_t group_size,
                                       const std::vector<int32_t>& kv_lens) {
    int32_t gt = GroupTiles(group_size);
    std::vector<int32_t> out;
    for (int32_t b = 0; b < batch; b++) {
        int32_t tiles = CeilDiv(kv_lens[b], TILE_N);
        for (int32_t h = 0; h < kv_head * gt; h++) {
            out.push_back(tiles);
        }
    }
    return out;
}

static int32_t ItemCost(int32_t kv_lo, int32_t kv_hi) {
    return CeilDiv(kv_hi - kv_lo, TILE_N);
}

static std::vector<int32_t> PenalisedCosts(const std::vector<int32_t>& costs,
                                           float item_penalty, int32_t block_size) {
    constexpr int32_t COST_UNIT = 16;
    float scaled = item_penalty * static_cast<float>(block_size) / 128.0f;
    int32_t extra = static_cast<int64_t>(std::lround(scaled * COST_UNIT));
    std::vector<int32_t> out;
    out.reserve(costs.size());
    for (auto c : costs) out.push_back(c * COST_UNIT + extra);
    return out;
}

static std::vector<int32_t> ChooseSplits(int32_t batch, int32_t kv_head, int32_t group_size,
                                         const std::vector<int32_t>& kv_lens, int32_t num_cores) {
    auto costs = GroupCosts(batch, kv_head, group_size, kv_lens);
    int32_t total = 0;
    for (auto c : costs) total += c;
    if (total == 0) {
        return std::vector<int32_t>(costs.size(), 1);
    }
    double ideal = static_cast<double>(total) / num_cores;
    auto penalised = PenalisedCosts(costs, 0.50f, 128);

    int32_t maxCost = *std::max_element(costs.begin(), costs.end());
    int64_t startTarget = std::max(static_cast<int64_t>(maxCost), std::max(static_cast<int64_t>(1), static_cast<int64_t>(ideal)));

    std::vector<int32_t> best;
    int32_t bestMakespan = INT32_MAX;
    int32_t bestSum = INT32_MAX;
    std::vector<int32_t> last;

    for (int32_t target = startTarget; target > 0; target--) {
        std::vector<int32_t> candidate;
        for (auto c : costs) {
            candidate.push_back(std::max(1, std::min(MAX_SPLITS, CeilDiv(c, target))));
        }
        bool same = (candidate == last);
        last = candidate;
        if (same) continue;

        int32_t sum = 0;
        for (auto s : candidate) sum += s;
        int32_t maxLoad = 0;
        for (size_t i = 0; i < candidate.size(); i++) {
            int32_t n = std::max(1, std::min(candidate[i], costs[i]));
            int32_t per = costs[i] / n;
            int32_t rem = costs[i] - per * n;
            int32_t load = 0;
            for (int32_t j = 0; j < n; j++) {
                load += per + (j < rem ? 1 : 0);
            }
            maxLoad = std::max(maxLoad, load);
        }
        if (maxLoad < bestMakespan || (maxLoad == bestMakespan && sum < bestSum)) {
            best = candidate;
            bestMakespan = maxLoad;
            bestSum = sum;
        }
    }
    if (best.empty()) {
        return std::vector<int32_t>(costs.size(), 1);
    }
    return best;
}

static std::vector<std::vector<int32_t>> PackFree(const std::vector<int32_t>& costs, int32_t num_cores) {
    std::vector<std::pair<int32_t, int32_t>> heap;
    for (int32_t c = 0; c < num_cores; c++) heap.push_back({0, c});
    auto cmp = [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second > b.second;
    };
    std::make_heap(heap.begin(), heap.end(), cmp);

    std::vector<std::vector<int32_t>> bins(num_cores);
    std::vector<int32_t> order(costs.size());
    for (size_t i = 0; i < costs.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return costs[a] > costs[b]; });

    for (auto idx : order) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        auto [load, core] = heap.back();
        heap.pop_back();
        bins[core].push_back(idx);
        heap.push_back({load + costs[idx], core});
        std::push_heap(heap.begin(), heap.end(), cmp);
    }
    for (auto& b : bins) std::sort(b.begin(), b.end());
    return bins;
}

static std::vector<std::pair<int32_t, int32_t>> BalanceContiguous(const std::vector<int32_t>& costs, int32_t num_cores) {
    int32_t n = static_cast<int64_t>(costs.size());
    if (n == 0) return std::vector<std::pair<int32_t, int32_t>>(num_cores, {0, 0});

    int32_t maxCost = *std::max_element(costs.begin(), costs.end());
    int32_t sum = 0;
    for (auto c : costs) sum += c;

    auto feasible = [&](int32_t limit) -> bool {
        int32_t used = 1, cur = 0;
        for (auto c : costs) {
            if (c > limit) return false;
            if (cur + c > limit) { used++; cur = c; if (used > num_cores) return false; }
            else cur += c;
        }
        return true;
    };

    int32_t lo = maxCost, hi = sum;
    while (lo < hi) {
        int32_t mid = (lo + hi) / 2;
        if (feasible(mid)) hi = mid; else lo = mid + 1;
    }

    std::vector<std::pair<int32_t, int32_t>> ranges;
    int32_t start = 0, cur = 0;
    for (int32_t i = 0; i < n; i++) {
        int32_t coresLeft = num_cores - static_cast<int64_t>(ranges.size()) - 1;
        bool mustClose = (cur > 0) && ((n - i) <= coresLeft);
        if (cur > 0 && (cur + costs[i] > lo || mustClose)) {
            ranges.push_back({start, i});
            start = i; cur = costs[i];
        } else {
            cur += costs[i];
        }
    }
    ranges.push_back({start, n});
    while (static_cast<int64_t>(ranges.size()) < num_cores) ranges.push_back({n, n});
    return ranges;
}

class TilingXFlashAttentionInferV2Func {
public:
    explicit TilingXFlashAttentionInferV2Func(gert::TilingContext *ctx) : ctx_(ctx) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ParseInputs();
    void BuildPlan();
    void SetWorkspaces();
    uint64_t ComputeTilingKey() const;

    gert::TilingContext *ctx_ = nullptr;
    XFlashAttentionInferV2TilingData tiling_data_;

    int64_t batch_ = 0;
    int64_t qHead_ = 0;
    int64_t kvHead_ = 0;
    int64_t headDim_ = 0;
    int64_t blockSize_ = 0;
    int64_t groupSize_ = 1;
    int64_t numTokens_ = 0;
    int64_t maxKvLen_ = 0;
    int64_t maxBlocksPerBatch_ = 0;
    int64_t cubeCoreNum_ = 0;
    int64_t tileM_ = 16;
    float scale_ = 0.0f;
    bool isNz_ = false;

    std::vector<int32_t> kvLens_;
    std::vector<PlanItem> items_;
    std::vector<PlanMerge> merges_;
    std::vector<PlanCoreRange> coreRanges_;
    int32_t numSlots_ = 0;
    bool mergeFree_ = true;
    bool narrowTail_ = false;

    std::vector<int32_t> planBuf_;
};

ge::graphStatus TilingXFlashAttentionInferV2Func::ParseInputs()
{
    auto queryShape = ctx_->GetInputShape(QUERY)->GetStorageShape();
    auto keyShape = ctx_->GetInputShape(KEY_CACHE)->GetStorageShape();
    auto blockTableShape = ctx_->GetInputShape(BLOCK_TABLE)->GetStorageShape();
    auto kvLensShape = ctx_->GetInputShape(ACTUAL_KV_LENS)->GetStorageShape();

    numTokens_ = static_cast<int64_t>(queryShape.GetDim(0));
    qHead_ = static_cast<int64_t>(queryShape.GetDim(1));
    headDim_ = static_cast<int64_t>(queryShape.GetDim(2));
    batch_ = static_cast<int64_t>(kvLensShape.GetDim(0));
    maxBlocksPerBatch_ = static_cast<int64_t>(blockTableShape.GetDim(1));

    if (keyShape.GetDimNum() == 4) {
        blockSize_ = static_cast<int64_t>(keyShape.GetDim(1));
        kvHead_ = static_cast<int64_t>(keyShape.GetDim(2));
        isNz_ = false;
    } else if (keyShape.GetDimNum() == 4 && keyShape.GetDim(3) == NZ_FRACTAL) {
        blockSize_ = static_cast<int64_t>(keyShape.GetDim(2));
        kvHead_ = static_cast<int64_t>(keyShape.GetDim(1));
        isNz_ = true;
    } else {
        blockSize_ = static_cast<int64_t>(keyShape.GetDim(1));
        kvHead_ = static_cast<int64_t>(keyShape.GetDim(2));
        isNz_ = false;
    }

    if (headDim_ != TILE_D) {
        OP_LOGE(ctx_->GetNodeName(), "head_dim must be 128, got %d.", headDim_);
        return ge::GRAPH_FAILED;
    }
    if (qHead_ <= 0 || kvHead_ <= 0 || qHead_ % kvHead_ != 0) {
        OP_LOGE(ctx_->GetNodeName(), "qHead(%d) must be divisible by kvHead(%d).", qHead_, kvHead_);
        return ge::GRAPH_FAILED;
    }
    groupSize_ = qHead_ / kvHead_;
    if (groupSize_ > 128) {
        OP_LOGE(ctx_->GetNodeName(), "GQA group size %d > 128 unsupported.", groupSize_);
        return ge::GRAPH_FAILED;
    }
    tileM_ = ChooseTileM(groupSize_);

    auto attrs = ctx_->GetAttrs();
    if (attrs != nullptr) {
        const auto *pScale = attrs->GetAttrPointer<float>(SCALE_ATTR);
        if (pScale != nullptr && *pScale > 0.0f) {
            scale_ = *pScale;
        } else {
            scale_ = static_cast<float>(1.0 / std::sqrt(1.0 * headDim_));
        }
    } else {
        scale_ = static_cast<float>(1.0 / std::sqrt(1.0 * headDim_));
    }

    auto kvLensTensor = ctx_->GetInputTensor(ACTUAL_KV_LENS);
    if (kvLensTensor != nullptr) {
        auto hostData = kvLensTensor->GetData<int32_t>();
        if (hostData != nullptr) {
            for (int32_t b = 0; b < batch_; b++) {
                kvLens_.push_back(hostData[b]);
                maxKvLen_ = std::max(maxKvLen_, static_cast<int64_t>(hostData[b]));
            }
        }
    }
    if (kvLens_.empty()) {
        maxKvLen_ = blockSize_ * maxBlocksPerBatch_;
        kvLens_.resize(batch_, maxKvLen_);
    }

    return ge::GRAPH_SUCCESS;
}

void TilingXFlashAttentionInferV2Func::BuildPlan()
{
    auto splits = ChooseSplits(batch_, kvHead_, groupSize_, kvLens_, cubeCoreNum_);
    int32_t gt = GroupTiles(groupSize_);
    int32_t gidx = 0;
    int32_t slot = 0;

    for (int32_t b = 0; b < batch_; b++) {
        int32_t kvLen = kvLens_[b];
        for (int32_t h = 0; h < kvHead_; h++) {
            for (int32_t g = 0; g < gt; g++) {
                int32_t rows = RowsOfGroup(groupSize_, g);
                auto intervals = SplitPoints(kvLen, splits[gidx]);
                gidx++;

                if (intervals.size() == 1) {
                    items_.push_back({b, h, g, intervals[0].first, intervals[0].second,
                                      0, rows, 1});
                } else {
                    int32_t slot0 = slot;
                    for (auto& [lo, hi] : intervals) {
                        items_.push_back({b, h, g, lo, hi, slot, rows, 0});
                        slot++;
                    }
                    merges_.push_back({b, h, g, slot0, static_cast<int64_t>(intervals.size()), rows});
                }
            }
        }
    }
    numSlots_ = slot;
    mergeFree_ = merges_.empty();

    std::vector<int32_t> costs;
    costs.reserve(items_.size());
    for (auto& it : items_) {
        costs.push_back(ItemCost(it.kv_lo, it.kv_hi));
    }
    auto pen = PenalisedCosts(costs, 0.50f, blockSize_);

    auto bins = PackFree(pen, cubeCoreNum_);
    std::vector<int32_t> reorderedItems;
    coreRanges_.clear();
    int32_t at = 0;
    for (int32_t c = 0; c < cubeCoreNum_; c++) {
        for (auto idx : bins[c]) reorderedItems.push_back(idx);
        coreRanges_.push_back({at, at + static_cast<int64_t>(bins[c].size()), 0, 0});
        at += static_cast<int64_t>(bins[c].size());
    }
    std::vector<PlanItem> newItems;
    newItems.reserve(reorderedItems.size());
    for (auto idx : reorderedItems) newItems.push_back(items_[idx]);
    items_ = std::move(newItems);

    auto mergeRanges = BalanceContiguous(std::vector<int32_t>(merges_.size(), 1), cubeCoreNum_);
    for (int32_t c = 0; c < cubeCoreNum_ && c < static_cast<int64_t>(coreRanges_.size()); c++) {
        coreRanges_[c].merge_begin = mergeRanges[c].first;
        coreRanges_[c].merge_end = mergeRanges[c].second;
    }
    while (static_cast<int64_t>(coreRanges_.size()) < MAX_CORES) {
        int32_t ni = static_cast<int64_t>(items_.size());
        int32_t nm = static_cast<int64_t>(merges_.size());
        coreRanges_.push_back({ni, ni, nm, nm});
    }

    int32_t itemBase = HEADER_INTS + MAX_CORES * CORE_FIELDS;
    int32_t mergeBase = itemBase + static_cast<int64_t>(items_.size()) * ITEM_FIELDS;
    int32_t total = mergeBase + static_cast<int64_t>(merges_.size()) * MERGE_FIELDS;

    planBuf_.resize(total, 0);
    planBuf_[0] = cubeCoreNum_;
    planBuf_[1] = static_cast<int64_t>(items_.size());
    planBuf_[2] = static_cast<int64_t>(merges_.size());
    planBuf_[3] = maxBlocksPerBatch_;
    planBuf_[4] = itemBase;
    planBuf_[5] = mergeBase;
    planBuf_[6] = numSlots_;

    for (int32_t c = 0; c < MAX_CORES; c++) {
        int32_t off = HEADER_INTS + c * CORE_FIELDS;
        planBuf_[off] = coreRanges_[c].item_begin;
        planBuf_[off + 1] = coreRanges_[c].item_end;
        planBuf_[off + 2] = coreRanges_[c].merge_begin;
        planBuf_[off + 3] = coreRanges_[c].merge_end;
    }
    for (size_t i = 0; i < items_.size(); i++) {
        int32_t off = itemBase + static_cast<int64_t>(i) * ITEM_FIELDS;
        planBuf_[off] = items_[i].batch;
        planBuf_[off + 1] = items_[i].kv_head;
        planBuf_[off + 2] = items_[i].group_tile;
        planBuf_[off + 3] = items_[i].kv_lo;
        planBuf_[off + 4] = items_[i].kv_hi;
        planBuf_[off + 5] = items_[i].out_slot;
        planBuf_[off + 6] = items_[i].rows;
        planBuf_[off + 7] = items_[i].direct;
    }
    for (size_t i = 0; i < merges_.size(); i++) {
        int32_t off = mergeBase + static_cast<int64_t>(i) * MERGE_FIELDS;
        planBuf_[off] = merges_[i].batch;
        planBuf_[off + 1] = merges_[i].kv_head;
        planBuf_[off + 2] = merges_[i].group_tile;
        planBuf_[off + 3] = merges_[i].slot0;
        planBuf_[off + 4] = merges_[i].nsplit;
        planBuf_[off + 5] = merges_[i].rows;
    }

    if (!mergeFree_ && maxKvLen_ > TILE_N) {
        int32_t tiles = CeilDiv(maxKvLen_, TILE_N);
        int32_t lastSpan = maxKvLen_ - (tiles - 1) * TILE_N;
        int32_t nowBytes = tiles * TILE_N;
        int32_t thenBytes = (tiles - 1) * TILE_N + CeilDiv(lastSpan, NZ_FRACTAL) * NZ_FRACTAL;
        if (nowBytes > 0 && (1.0 - static_cast<double>(thenBytes) / nowBytes) >= 0.05) {
            narrowTail_ = true;
        }
    }
}

void TilingXFlashAttentionInferV2Func::SetWorkspaces()
{
    auto platform = platform_ascendc::PlatformAscendC(ctx_->GetPlatformInfo());
    size_t systemWsSize = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());

    int64_t planBytes = static_cast<int64_t>(planBuf_.size()) * sizeof(int32_t);
    int64_t planAligned = (planBytes + 63) & ~63;

    int64_t wsAccumSize = 0;
    int64_t wsStateSize = 0;
    if (!mergeFree_) {
        constexpr int32_t STATE_LANES = 64;
        wsAccumSize = static_cast<int64_t>(numSlots_) * TILE_D * 2 * sizeof(float);
        wsStateSize = static_cast<int64_t>(numSlots_) * STATE_LANES * sizeof(float);
    }

    size_t userWsSize = static_cast<size_t>(planAligned + wsAccumSize + wsStateSize);
    size_t *ws = ctx_->GetWorkspaceSizes(1);
    ws[0] = systemWsSize + userWsSize;

    tiling_data_.set_plan_offset(systemWsSize);
    tiling_data_.set_ws_accum_offset(static_cast<int64_t>(systemWsSize) + planAligned);
    tiling_data_.set_ws_state_offset(static_cast<int64_t>(systemWsSize) + planAligned + wsAccumSize);
    tiling_data_.set_plan_size(planBytes);
}

uint64_t TilingXFlashAttentionInferV2Func::ComputeTilingKey() const
{
    uint64_t key = 0;
    int32_t bsCode = 0;
    switch (blockSize_) {
        case 16: bsCode = 0; break;
        case 32: bsCode = 1; break;
        case 64: bsCode = 2; break;
        case 128: bsCode = 3; break;
        default: bsCode = 3; break;
    }
    int32_t tmCode = (tileM_ == 16) ? 0 : 1;
    int32_t fmtCode = isNz_ ? 1 : 0;
    int32_t mfCode = mergeFree_ ? 1 : 0;
    int32_t ntCode = narrowTail_ ? 1 : 0;

    key = (static_cast<uint64_t>(bsCode) << 4) |
          (static_cast<uint64_t>(tmCode) << 3) |
          (static_cast<uint64_t>(fmtCode) << 2) |
          (static_cast<uint64_t>(mfCode) << 1) |
          (static_cast<uint64_t>(ntCode));
    return key;
}

ge::graphStatus TilingXFlashAttentionInferV2Func::RunTiling()
{
    auto platform = platform_ascendc::PlatformAscendC(ctx_->GetPlatformInfo());
    cubeCoreNum_ = static_cast<int64_t>(platform.GetCoreNumAic());

    auto ret = ParseInputs();
    if (ret != ge::GRAPH_SUCCESS) return ret;

    BuildPlan();
    SetWorkspaces();

    tiling_data_.set_q_head(qHead_);
    tiling_data_.set_kv_head(kvHead_);
    tiling_data_.set_batch(batch_);
    tiling_data_.set_num_tokens(numTokens_);
    tiling_data_.set_head_dim(headDim_);
    tiling_data_.set_block_size(blockSize_);
    tiling_data_.set_group_size(groupSize_);
    tiling_data_.set_max_kv_len(maxKvLen_);
    tiling_data_.set_max_blocks_per_batch(maxBlocksPerBatch_);
    tiling_data_.set_num_cores(cubeCoreNum_);
    tiling_data_.set_tile_m(tileM_);
    tiling_data_.set_scale(scale_);

    uint64_t tilingKey = ComputeTilingKey();
    ctx_->SetTilingKey(tilingKey);

    tiling_data_.SaveToBuffer(ctx_->GetRawTilingData()->GetData(),
                              ctx_->GetRawTilingData()->GetCapacity());
    ctx_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
    ctx_->SetBlockDim(static_cast<uint32_t>(cubeCoreNum_));

    auto rawTilingData = ctx_->GetRawTilingData();
    auto dataPtr = rawTilingData->GetData();
    auto capacity = rawTilingData->GetCapacity();
    auto dataSize = rawTilingData->GetDataSize();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingXFlashAttentionInferV2Func tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "x_flash_attention_infer_v2 tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XFlashAttentionInferV2)
    .Tiling(TilingFunc);
} // namespace optiling
