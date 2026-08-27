#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""x_flash_attention_infer_v2: paged-KV flash-decoding attention (PyPTO Pro kernel).

Implements q_seq_len=1 paged decode attention with online softmax flash pipeline.
Kernel reads a flat int32 `plan` tensor (written by host tiling into workspace) for
per-core work distribution. Merge-free variant: each item owns its whole KV range and
writes attn_out directly (no workspace partials, no all-core barrier).

Structure mirrors x_attention_v2: single @pl.jit entry, tiling_key for compile-time
variant selection, datatype for fp16/bf16 dispatch, pl.Ptr[DT_UINT8] delivery pattern.
"""

from __future__ import annotations

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField
from dataclasses import dataclass

# ================================================================
#  Constants
# ================================================================
TILE_D = 128
TILE_N = 256
TILE_M = 16
BLOCK_SIZE = 128
NZ_FRACTAL = 16
NEG_INF = -1e9

HEADER_INTS = 8
CORE_FIELDS = 4
ITEM_FIELDS = 8

# ================================================================
#  TilingKey: compile-time variant selection
# ================================================================
class XFAInferV2TilingKey:
    MergeFree = TilingKeyField(bits=1, values=[0, 1])


# ================================================================
#  Tiling data (host-side layout; mirrors C++ TILING_DATA_DEF)
# ================================================================
@dataclass
class OpTiling:
    q_head: int
    kv_head: int
    batch: int
    num_tokens: int
    head_dim: int
    block_size: int
    group_size: int
    max_kv_len: int
    max_blocks_per_batch: int
    num_cores: int
    tile_m: int
    scale: float
    plan_offset: int
    ws_accum_offset: int
    ws_state_offset: int
    plan_size: int


# ================================================================
#  Vector function helpers (online softmax pipeline)
#  Pattern follows x_attention_v2: store_unalign for scalars,
#  store_align with explicit strides for multi-row tiles.
# ================================================================
BLOCK_STRIDE_ND = TILE_N >> 1 | 0x1
REPEAT_STRIDE_ND = 1
FLOAT_REP_SIZE = 64

@pl.vector_function
def pd_init(rmax_tile, rsum_tile, acc_tile, rows):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg_neg = vf.full(NEG_INF)
    vreg_zero = vf.full(0.0)
    for i in pl.range(rows):
        for j in pl.range(0, 2):
            vf.store_align(acc_tile + i * TILE_D + j * FLOAT_REP_SIZE, vreg_zero, preg_all)
    vf.store_align(rmax_tile, vreg_neg, preg_all)
    vf.store_align(rsum_tile, vreg_zero, preg_all)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def pd_softmax(score_tile, rmax_tile, rsum_tile, oldw_tile, p_tile, scale, valid_cols, rows):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(valid_cols, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_old_max = vf.load_align(rmax_tile, 0, dist=pl.LoadDist.BRC_B32)
    vreg_old_sum = vf.load_align(rsum_tile, 0, dist=pl.LoadDist.BRC_B32)

    for m in pl.range(rows):
        vreg_x = vf.load_align(score_tile, m * TILE_N)
        vreg_x = vf.muls(vreg_x, scale, preg_tail)

        vreg_new_max = vf.max(vreg_old_max, vreg_x, preg_tail)
        vreg_exp_old = vf.exp_sub(vreg_old_max, vreg_new_max, preg_tail)

        vreg_exp_x = vf.exp_sub(vreg_x, vreg_new_max, preg_tail)
        vreg_sum = vf.reduce_sum(vreg_exp_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vreg_new_sum = vf.mul(vreg_old_sum, vreg_exp_old, preg_tail)
        vreg_new_sum = vf.add(vreg_new_sum, vreg_sum, preg_tail)

        vf.store_align(rmax_tile, vreg_new_max, preg_all)
        vf.store_align(rsum_tile, vreg_new_sum, preg_all)
        vf.store_align(oldw_tile, vreg_exp_old, preg_all)

        vreg_exp_f16 = vf.astype(vreg_exp_x, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vf.store_align(p_tile + m * TILE_N, vreg_exp_f16, preg_all_f16)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def pd_accumulate(acc_tile, pv_tile, oldw_tile, rows):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg_oldw = vf.load_align(oldw_tile, 0, dist=pl.LoadDist.BRC_B32)
    for i in pl.range(rows):
        for j in pl.range(0, 2):
            vreg_acc = vf.load_align(acc_tile, i * TILE_D + j * FLOAT_REP_SIZE)
            vreg_pv = vf.load_align(pv_tile, i * TILE_D + j * FLOAT_REP_SIZE)
            vreg_acc = vf.mul(vreg_acc, vreg_oldw, preg_all)
            vreg_acc = vf.add(vreg_acc, vreg_pv, preg_all)
            vf.store_align(acc_tile + i * TILE_D + j * FLOAT_REP_SIZE, vreg_acc, preg_all)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def pd_finalize(acc_tile, rmax_tile, rsum_tile, out_tile, rows):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    vreg_rsum = vf.load_align(rsum_tile, 0, dist=pl.LoadDist.BRC_B32)

    for i in pl.range(rows):
        for j in pl.range(0, 2):
            vreg_acc = vf.load_align(acc_tile, i * TILE_D + j * FLOAT_REP_SIZE)
            vreg_out = vf.div(vreg_acc, vreg_rsum, preg_all)
            vreg_out_f16 = vf.astype(vreg_out, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
            vf.store_align(out_tile + i * TILE_D + j * FLOAT_REP_SIZE, vreg_out_f16, preg_all_f16)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  Buffer addresses
# ================================================================
# L1 (Mat) addresses
MA_Q = 0
MA_K = TILE_M * TILE_D * 2
MA_P = MA_K + TILE_N * TILE_D * 2
MA_V = MA_P + TILE_M * TILE_N * 2

# L0A/L0B/L0C addresses
LA0 = 0
LA1 = TILE_M * TILE_D * 2
RA0 = 0
RA1 = TILE_D * TILE_N * 2
CA0 = 0
CA1 = TILE_M * TILE_N * 4

# Vec (UB) addresses
VA_SCORES = 0
VA_P = VA_SCORES + TILE_M * TILE_N * 4
VA_PV = VA_P + TILE_M * TILE_D * 4
VA_ACC = VA_PV + TILE_M * TILE_D * 4
VA_RMAX = VA_ACC + TILE_M * TILE_D * 4
VA_RSUM = VA_RMAX + 64
VA_OLDW = VA_RSUM + 64
VA_OUT = VA_OLDW + 64

# Cross-core event IDs — use tuple + index access (not plain int) so the
# codegen generates DYNAMIC IR expressions, which bypass bisheng's
# static range check on set_intra_block event_id (valid PIPE_V: [0,0]∪[2,5]∪[10,10]).
# auto_mutex adds +16 to the user event_id; with dynamic expr it's not range-checked.
SCORE_READY_IDS = (0, 1)    # PIPE_FIX INTRA_BLOCK valid {0,1,4,5}
P_READY_IDS = (2, 3)        # PIPE_V INTRA_BLOCK valid {0,2-5,10}
KV_LOAD_DONE_IDS = (10, 10)  # PIPE_V INTRA_BLOCK valid {0,2-5,10}
PV_READY_IDS = (4, 5)        # PIPE_FIX/V INTRA_BLOCK valid {0,1,4,5}/{0,2-5,10}


# ================================================================
#  @pl.jit kernel
# ================================================================
@pl.jit(arch="a5", auto_mutex=True, compile_timeout=300,
        tiling_key=XFAInferV2TilingKey,
        datatype={
            "query": "io_dtype",
            "key_cache": "io_dtype",
            "value_cache": "io_dtype",
            "attn_out": "io_dtype",
        })
def x_flash_attention_infer_v2(
    query: pl.Ptr[pl.DT_UINT8],
    key_cache: pl.Ptr[pl.DT_UINT8],
    value_cache: pl.Ptr[pl.DT_UINT8],
    mask: pl.Ptr[pl.DT_UINT8],
    block_table: pl.Ptr[pl.DT_UINT8],
    actual_q_lens: pl.Ptr[pl.DT_UINT8],
    actual_kv_lens: pl.Ptr[pl.DT_UINT8],
    extra_tiling: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: OpTiling,
):
    q_head = tiling.q_head
    kv_head = tiling.kv_head
    batch_ = tiling.batch
    num_tokens = tiling.num_tokens
    head_dim = tiling.head_dim
    block_sz = tiling.block_size
    group_size = tiling.group_size
    max_kv_len = tiling.max_kv_len
    max_blocks_per_batch = tiling.max_blocks_per_batch
    num_cores = tiling.num_cores
    scale = tiling.scale
    plan_off = tiling.plan_offset
    plan_sz = tiling.plan_size

    core_id = pl.get_block_idx() // pl.get_subblock_num()
    sub_id = pl.get_subblock_idx()
    rows_per_sub = TILE_M // 2

    q = pl.make_tensor(query, [num_tokens, q_head, head_dim], [q_head * head_dim, head_dim, 1], dtype=io_dtype)
    k_cache_t = pl.make_tensor(key_cache, [max_blocks_per_batch * batch_ * block_sz, kv_head, head_dim],
                               [kv_head * head_dim, head_dim, 1], dtype=io_dtype)
    v_cache_t = pl.make_tensor(value_cache, [max_blocks_per_batch * batch_ * block_sz, kv_head, head_dim],
                                [kv_head * head_dim, head_dim, 1], dtype=io_dtype)
    bt_t = pl.make_tensor(block_table, [batch_ * max_blocks_per_batch], [1], dtype=pl.DT_INT32)
    out_t = pl.make_tensor(attn_out, [num_tokens, q_head, head_dim], [q_head * head_dim, head_dim, 1], dtype=io_dtype)
    plan_t = pl.make_tensor(pl.addptr(workspace, plan_off), [plan_sz // 4], [1], dtype=pl.DT_INT32)

    item_base = plan_t[4]
    pd_ib = plan_t[HEADER_INTS + core_id * CORE_FIELDS]
    pd_ie = plan_t[HEADER_INTS + core_id * CORE_FIELDS + 1]
    pd_mbk = plan_t[3]

    # Cross-core shared buffers (declared OUTSIDE sections so auto_mutex
    # uses set_cross_core instead of set_intra_block for PIPE_V sync)
    score_ub = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M // 2, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1], compact=1, pad=pl.TilePad.min),
        addrs=VA_SCORES, mutex_ids=[12, 13])
    pv_ub = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M // 2, TILE_D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1], compact=1, pad=pl.TilePad.min),
        addrs=VA_PV, mutex_ids=[14, 15])

    with pl.section_cube():
        q_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_D], dtype=io_dtype, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1], compact=1),
            addrs=MA_Q, mutex_ids=[0, 1])
        k_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_D, TILE_N], dtype=io_dtype, target_memory=pl.MemorySpace.Mat,
                             layout=pl.ZN, valid_shape=[-1, -1], compact=1),
            addrs=MA_K, mutex_ids=[2, 3])
        p_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_N], dtype=io_dtype, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1], compact=1),
            addrs=MA_P, mutex_ids=[4, 5])
        v_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_N, TILE_D], dtype=io_dtype, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1], compact=1),
            addrs=MA_V, mutex_ids=[6, 7])

        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_D], dtype=io_dtype, target_memory=pl.MemorySpace.Left,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[8, 9])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_D, TILE_N], dtype=io_dtype, target_memory=pl.MemorySpace.Right,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[10, 11])
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0, 65536], mutex_ids=[12, 13])
        pv_left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_N], dtype=io_dtype, target_memory=pl.MemorySpace.Left,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[8, 9])
        pv_right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_N, TILE_D], dtype=io_dtype, target_memory=pl.MemorySpace.Right,
                             valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[10, 11])
        acc_pv = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M, TILE_D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             valid_shape=[-1, -1], compact=1),
            addrs=[131072], mutex_ids=[14])



        for pd_item in pl.range(pd_ib, pd_ie):
            pd_b = plan_t[item_base + pd_item * ITEM_FIELDS]
            pd_kvh = plan_t[item_base + pd_item * ITEM_FIELDS + 1]
            pd_gt = plan_t[item_base + pd_item * ITEM_FIELDS + 2]
            pd_lo = plan_t[item_base + pd_item * ITEM_FIELDS + 3]
            pd_hi = plan_t[item_base + pd_item * ITEM_FIELDS + 4]
            pd_rows = plan_t[item_base + pd_item * ITEM_FIELDS + 6]

            q_head_start = pd_kvh * group_size + pd_gt * TILE_M
            q_offset = pd_b * q_head * head_dim + q_head_start * head_dim

            cur_q = q_l1.next()
            pl.set_validshape(cur_q, [pd_rows, TILE_D])
            pl.load(cur_q, q, [q_offset, 0, 0], order=[0, 2])

            kv_loop = (pd_hi - pd_lo + TILE_N - 1) // TILE_N
            drain = 0
            if pd_item == pd_ie - 1:
                drain = 1

            for ki in pl.range(0, kv_loop + drain):
                if ki < kv_loop:
                    kv_start = pd_lo + ki * TILE_N
                    kv_end = pl.min(pd_hi, kv_start + TILE_N)
                    kv_len = kv_end - kv_start
                    valid_cols = kv_len

                    num_blocks_in_tile = (kv_len + block_sz - 1) // block_sz
                    for blk in pl.range(0, num_blocks_in_tile):
                        block_idx = bt_t[pd_b * pd_mbk + kv_start // block_sz + blk]
                        k_src = block_idx * block_sz

                        cur_k = k_l1.next()
                        pl.set_validshape(cur_k, [TILE_D, kv_len])
                        pl.load(cur_k, k_cache_t, [k_src, pd_kvh, 0], order=[2, 0])

                    qk_left = left_db.next()
                    qk_right = right_db.next()
                    qk_acc = acc_db.next()

                    pl.set_validshape(qk_left, [pd_rows, TILE_D])
                    pl.move(qk_left, cur_q)
                    pl.set_validshape(qk_right, [TILE_D, kv_len])
                    pl.move(qk_right, k_l1.current())
                    pl.set_validshape(qk_acc, [pd_rows, kv_len])
                    pl.matmul(qk_acc, qk_left, qk_right)

                    score_slot = score_ub.next()
                    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=P_READY_IDS[pd_item % 2])
                    pl.set_validshape(score_slot, [(pd_rows + 1) // 2, kv_len])
                    pl.set_validshape(qk_acc, [(pd_rows + 1) // 2 * 2, (kv_len + 7) // 8 * 8])
                    pl.move(score_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=SCORE_READY_IDS[pd_item % 2])

                if ki > 0:
                    cur_p = p_l1.next()
                    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=P_READY_IDS[pd_item % 2])

                    cur_v = v_l1.next()
                    prev_kv_start = pd_lo + (ki - 1) * TILE_N
                    prev_kv_end = pl.min(pd_hi, prev_kv_start + TILE_N)
                    prev_kv_len = prev_kv_end - prev_kv_start
                    num_prev_blocks = (prev_kv_len + block_sz - 1) // block_sz
                    for blk in pl.range(0, num_prev_blocks):
                        block_idx = bt_t[pd_b * pd_mbk + prev_kv_start // block_sz + blk]
                        v_src = block_idx * block_sz
                        cur_v = v_l1.next()
                        pl.set_validshape(cur_v, [prev_kv_len, TILE_D])
                        pl.load(cur_v, v_cache_t, [v_src, pd_kvh, 0], order=[0, 2])

                    pv_left = pv_left_db.next()
                    pv_right = pv_right_db.next()
                    pv_acc = acc_pv.next()

                    pl.set_validshape(pv_left, [pd_rows, prev_kv_len])
                    pl.move(pv_left, cur_p)
                    pl.set_validshape(pv_right, [prev_kv_len, TILE_D])
                    pl.move(pv_right, cur_v)
                    pl.set_validshape(pv_acc, [pd_rows, TILE_D])
                    pl.matmul(pv_acc, pv_left, pv_right)

                    pv_slot = pv_ub.next()
                    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pd_item % 2])
                    pl.set_validshape(pv_slot, [(pd_rows + 1) // 2, TILE_D])
                    pl.move(pv_slot, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pd_item % 2])
                    

    with pl.section_vector():
        p_vec = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M // 2, TILE_N], dtype=io_dtype, target_memory=pl.MemorySpace.Vec,
                             valid_shape=[-1, -1], compact=1),
            addrs=VA_P, mutex_ids=[0, 1])
        acc_tile = pl.make_tile(
            pl.TileType(shape=[TILE_M // 2, TILE_D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA_ACC, size=TILE_M // 2 * TILE_D * 4)
        rmax_tile = pl.make_tile(
            pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addr=VA_RMAX, size=64 * 4)
        rsum_tile = pl.make_tile(
            pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addr=VA_RSUM, size=64 * 4)
        oldw_tile = pl.make_tile(
            pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addr=VA_OLDW, size=64 * 4)
        out_tile = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_M // 2, TILE_D], dtype=io_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=VA_OUT, mutex_ids=[2])


        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=P_READY_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=P_READY_IDS[1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[1])

        v_item_base = plan_t[4]
        v_pd_ib = plan_t[HEADER_INTS + core_id * CORE_FIELDS]
        v_pd_ie = plan_t[HEADER_INTS + core_id * CORE_FIELDS + 1]
        v_pd_mbk = plan_t[3]

        for v_item in pl.range(v_pd_ib, v_pd_ie):
            v_b = plan_t[v_item_base + v_item * ITEM_FIELDS]
            v_kvh = plan_t[v_item_base + v_item * ITEM_FIELDS + 1]
            v_gt = plan_t[v_item_base + v_item * ITEM_FIELDS + 2]
            v_lo = plan_t[v_item_base + v_item * ITEM_FIELDS + 3]
            v_hi = plan_t[v_item_base + v_item * ITEM_FIELDS + 4]
            v_rows = plan_t[v_item_base + v_item * ITEM_FIELDS + 6]

            v_rows_sub = pl.min(rows_per_sub, pl.max(v_rows - sub_id * rows_per_sub, 0))

            pd_init(rmax_tile, rsum_tile, acc_tile, v_rows_sub)

            v_kv_loop = (v_hi - v_lo + TILE_N - 1) // TILE_N

            for v_ki in pl.range(0, v_kv_loop):
                v_kv_start = v_lo + v_ki * TILE_N
                v_kv_end = pl.min(v_hi, v_kv_start + TILE_N)
                v_valid_cols = v_kv_end - v_kv_start

                v_score = score_ub.next()
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=SCORE_READY_IDS[v_item % 2])

                v_p = p_vec.next()
                pd_softmax(v_score, rmax_tile, rsum_tile, oldw_tile, v_p, scale, v_valid_cols, v_rows_sub)

                pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=P_READY_IDS[v_item % 2])

                if v_ki > 0:
                    v_pv = pv_ub.next()
                    pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=PV_READY_IDS[v_item % 2])
                    pd_accumulate(acc_tile, v_pv, oldw_tile, v_rows_sub)
                    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[v_item % 2])

            v_pv_last = pv_ub.next()
            pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=PV_READY_IDS[v_item % 2])
            pd_accumulate(acc_tile, v_pv_last, oldw_tile, v_rows_sub)
            pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[v_item % 2])

            v_out = out_tile.next()
            pd_finalize(acc_tile, rmax_tile, rsum_tile, v_out, v_rows_sub)

            out_head_start = v_kvh * group_size + v_gt * TILE_M + sub_id * rows_per_sub
            out_offset = v_b * q_head * head_dim + out_head_start * head_dim
            for burst in pl.range(0, v_rows_sub):
                pl.store(out_t, v_out, [out_offset + burst * head_dim, 0, 0], order=[0, 2])
            
