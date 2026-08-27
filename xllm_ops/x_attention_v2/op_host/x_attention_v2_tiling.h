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

#ifndef X_ATTENTION_V2_TILING_H_
#define X_ATTENTION_V2_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(XAttentionV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, batch)
TILING_DATA_FIELD_DEF(int64_t, beam)
TILING_DATA_FIELD_DEF(int64_t, hq)
TILING_DATA_FIELD_DEF(int64_t, hkv)
TILING_DATA_FIELD_DEF(int64_t, shared_total)
TILING_DATA_FIELD_DEF(int64_t, u_maxds)
TILING_DATA_FIELD_DEF(float, scale)
TILING_DATA_FIELD_DEF(int64_t, sbt_stride)
TILING_DATA_FIELD_DEF(int64_t, shared_core_num)
TILING_DATA_FIELD_DEF(int64_t, total_cores)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(XAttentionV2, XAttentionV2TilingData)

} // namespace optiling

#endif // X_ATTENTION_V2_TILING_H_
