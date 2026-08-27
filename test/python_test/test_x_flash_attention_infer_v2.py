#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version  2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for x_flash_attention_infer_v2 (paged-KV flash decoding, PyPTO Pro).

Extends v1 test_x_flash_attention_infer.py with broader shapes and golden cache.
qSeqlen=1 decode; causal mask is empty so golden is plain paged full-attention.
"""

import os
import math
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_CACHE_DIR = os.path.join(WORKSPACE, "golden_cache")
torch.manual_seed(0)

_DTYPE_NAME = {torch.float16: "fp16", torch.bfloat16: "bf16"}


def _cache_key(dtype, q_head, kv_head, batch, kv_seqlen, block_size, kv_format):
    return f"{_DTYPE_NAME[dtype]}_{q_head}_{kv_head}_{batch}_{kv_seqlen}_{block_size}_{kv_format}"


def _save_golden(key, data):
    os.makedirs(GOLDEN_CACHE_DIR, exist_ok=True)
    torch.save(data, os.path.join(GOLDEN_CACHE_DIR, f"{key}.pt"))


def _load_golden(key):
    p = os.path.join(GOLDEN_CACHE_DIR, f"{key}.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, weights_only=False)


def _paged_gather_kv(cache, block_table, kv_seqlen, batch, block_size):
    kv_head = cache.shape[2]
    head_dim = cache.shape[3]
    out = torch.zeros((batch, kv_seqlen, kv_head, head_dim), dtype=cache.dtype)
    for b in range(batch):
        pos = 0
        blk = 0
        while pos < kv_seqlen:
            block_id = int(block_table[b, blk].item())
            take = min(block_size, kv_seqlen - pos)
            out[b, pos:pos + take] = cache[block_id, :take]
            pos += take
            blk += 1
    return out


def _golden_decode(query, key, value, batch, q_head, kv_head, scale):
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    group = q_head // kv_head
    head_dim = q.shape[-1]
    out = torch.zeros((batch, q_head, head_dim), dtype=torch.float32)
    for b in range(batch):
        for h in range(q_head):
            kvh = h // group
            qh = q[b, h]
            kh = k[b, :, kvh]
            vh = v[b, :, kvh]
            scores = (kh @ qh) * scale
            out[b, h] = torch.softmax(scores, dim=-1) @ vh
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_head, kv_head", [
    (8, 8), (16, 8), (32, 8), (16, 4), (32, 32), (32, 1), (64, 8),
])
@pytest.mark.parametrize("batch", [1, 2, 6])
@pytest.mark.parametrize("kv_seqlen", [128, 512, 2048, 8192])
@pytest.mark.parametrize("block_size", [128])
def test_x_flash_attention_infer_v2(dtype, q_head, kv_head, batch, kv_seqlen, block_size):
    device_id = int(os.environ.get("ASCEND_DEVICE_ID", 0))
    torch_npu.npu.set_device(device_id)

    head_dim = 128
    q_seqlen = 1
    num_tokens = batch * q_seqlen
    scale = 1.0 / math.sqrt(head_dim)
    kv_format = "ND"

    ckey = _cache_key(dtype, q_head, kv_head, batch, kv_seqlen, block_size, kv_format)
    cached = _load_golden(ckey)

    if cached is not None:
        query = cached["query"]
        key_cache = cached["key_cache"]
        value_cache = cached["value_cache"]
        block_table = cached["block_table"]
        golden = cached["golden"]
    else:
        blocks_per_batch = (kv_seqlen + block_size - 1) // block_size
        num_blocks = batch * blocks_per_batch

        query = torch.randn(num_tokens, q_head, head_dim, dtype=dtype)
        key_cache = torch.randn(num_blocks, block_size, kv_head, head_dim, dtype=dtype)
        value_cache = torch.randn(num_blocks, block_size, kv_head, head_dim, dtype=dtype)

        block_table = torch.zeros(batch, blocks_per_batch, dtype=torch.int32)
        for b in range(batch):
            for j in range(blocks_per_batch):
                block_table[b, j] = b * blocks_per_batch + j

        gathered_k = _paged_gather_kv(key_cache, block_table, kv_seqlen, batch, block_size)
        gathered_v = _paged_gather_kv(value_cache, block_table, kv_seqlen, batch, block_size)
        query_bhd = query.view(batch, q_head, head_dim)
        golden = _golden_decode(query_bhd, gathered_k, gathered_v, batch, q_head, kv_head, scale)

        _save_golden(ckey, {
            "query": query, "key_cache": key_cache, "value_cache": value_cache,
            "block_table": block_table, "golden": golden,
        })

    actual_q_lens = torch.arange(1, batch + 1, dtype=torch.int32)
    actual_kv_lens = torch.full((batch,), kv_seqlen, dtype=torch.int32)

    out = custom_ops.x_flash_attention_infer_v2_npu(
        query.npu(), key_cache.npu(), value_cache.npu(),
        block_table.npu(), actual_q_lens.npu(), actual_kv_lens.npu(),
        q_head, kv_head, scale, batch, kv_seqlen, layout="TND",
    )
    out = out.cpu().view(batch, q_head, head_dim).to(torch.float32)

    if dtype == torch.bfloat16:
        assert torch.allclose(out, golden, atol=0.01, rtol=0.01)
    else:
        assert torch.allclose(out, golden, atol=0.001, rtol=0.001)
