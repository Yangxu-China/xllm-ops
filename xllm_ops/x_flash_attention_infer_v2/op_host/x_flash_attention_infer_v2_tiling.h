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

#ifndef X_FLASH_ATTENTION_INFER_V2_TILING_H_
#define X_FLASH_ATTENTION_INFER_V2_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(XFlashAttentionInferV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, q_head)
TILING_DATA_FIELD_DEF(int64_t, kv_head)
TILING_DATA_FIELD_DEF(int64_t, batch)
TILING_DATA_FIELD_DEF(int64_t, num_tokens)
TILING_DATA_FIELD_DEF(int64_t, head_dim)
TILING_DATA_FIELD_DEF(int64_t, block_size)
TILING_DATA_FIELD_DEF(int64_t, group_size)
TILING_DATA_FIELD_DEF(int64_t, max_kv_len)
TILING_DATA_FIELD_DEF(int64_t, max_blocks_per_batch)
TILING_DATA_FIELD_DEF(int64_t, num_cores)
TILING_DATA_FIELD_DEF(int64_t, tile_m)
TILING_DATA_FIELD_DEF(float, scale)
TILING_DATA_FIELD_DEF(int64_t, plan_offset)
TILING_DATA_FIELD_DEF(int64_t, ws_accum_offset)
TILING_DATA_FIELD_DEF(int64_t, ws_state_offset)
TILING_DATA_FIELD_DEF(int64_t, plan_size)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(XFlashAttentionInferV2, XFlashAttentionInferV2TilingData)

} // namespace optiling

#endif // X_FLASH_ATTENTION_INFER_V2_TILING_H_
