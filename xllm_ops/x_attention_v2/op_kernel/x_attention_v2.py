#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-split XAttention kernel (group-local, no separate combine stage).

Architecture:
  * Each core processes a complete (shared + unshared) attention task using
    a shared running state (m, l, O) — no combine phase, no GM workspace.
  * Tasks are partitioned across all cores via a strided loop over the task table.
  * Grouped-query (GQA) with group ∈ {2,4}; unshared uses a diagonal-block mask
    (merged-group) that CONTINUES the same online-softmax state.

Layouts (host prepares):
  * query  -> permuted [B, Hkv, beam, group, D]   (task rows contiguous)
  * output -> kernel writes permuted [B, Hkv, beam, group, D], host permutes back
  * unshared_k/v -> permuted [B, Hkv, beam, maxDs, D]
  * shared_k/v   -> [sum(Lb), Hkv, D] (natural)

Constraints (v1):
  * shared_m in {64, 128} (compile-time tiling key SharedM; beams_per_task =
    shared_m // group: 32/16 for M=64, 64/32 for M=128, group 2/4)
  * loaded unshared width = beams_per_task * max_ds <= 128
  * decode_step in [1,4], group in {2,4}, head_dim = 128
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField

# ================================================================
#  Constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
NEG_INF = -1e9
ML_W = 8
FLOAT_REP_SIZE = 64
D_LOOPS = TD // FLOAT_REP_SIZE
TAIL_D = TD % FLOAT_REP_SIZE
REDUCE_SIZE = 1
BLOCK_STRIDE_ND = TS >> 1 | 0x1
REPEAT_STRIDE_ND = 1

Q_F16 = TS * TD * 2
KT_F16 = TD * TKV * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB4_KV = TS_HALF * TKV * 4
VB6 = (TS_HALF + 1) * TKV * 2
VB_RED = TS_HALF * 1 * 4
VB_ML = TS_HALF * ML_W * 4

QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
P_READY_FORWARD_IDS = (4, 5, 6)
PV_READY_FORWARD_IDS = (7, 8)
PV_READY_BARKWARD_IDS = (9, 10)

MA0_Q = 0
MA1_K = Q_F16 * 2
MA2_P = MA1_K + KT_F16 * 2
MA3_V = MA2_P + P_F16 * 3

VA0 = 0
VA1 = VA0 + VB4_KV * 2
VA_GMAX0 = VA1 + VB6 * 2
VA_GMAX1 = VA_GMAX0 + VB_RED
VA_GMAX2 = VA_GMAX1 + VB_RED
VA_GSUM0 = VA_GMAX2 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
VA_GSUM2 = VA_GSUM1 + VB_RED
VA_EXPMAX0 = VA_GSUM2 + VB_RED
VA_EXPMAX1 = VA_EXPMAX0 + VB_RED
VA_EXPMAX2 = VA_EXPMAX1 + VB_RED
VA7 = VA_EXPMAX2 + VB_RED
VA8 = VA7 + VB4
VA9 = VA8 + VB4 * 2
VA10 = VA9 + VB2
VA11 = VA10 + VB_ML
VA12 = VA11 + VB_RED
VA13 = VA12 + VB_RED
VB_MASK = TS_HALF * TKV
VB_WTBL = TS * 8 * 4
VA_UMASK = VA13 + VB_ML
VA_WTBL = VA_UMASK + VB_MASK


# ================================================================
#  TilingKey: per-shared-M compile-time variants
# ================================================================
class XAttnV2TilingKey:
    """shared_m geometry (parser folds each field per concrete key):

      SharedM: 64 = 64 query rows per task (beams_per_task 32/16 for group 2/4)
               128 = 128 query rows per task (beams_per_task 64/32 for group 2/4)
    """
    SharedM = TilingKeyField(bits=1, values=[64, 128])


# ================================================================
#  Tiling
# ================================================================
@dataclass
class TaskTiling:
    hq: int
    hkv: int
    batch: int
    beam_size: int
    shared_m: int
    group: int
    unshared: int
    max_ds: int
    shared_total: int
    scale: float
    num_tokens: int
    total_cores: int
    task_count: int = 0
    beams_per_task: int = 0
    unshared_n: int = 0

    @property
    def flat_rows(self):
        return self.num_tokens * self.hq


def _select_tiling(hq, hkv, batch, beam_size, decode_step, max_ds, core_num):
    group = hq // hkv
    if group not in (2, 4) or decode_step not in (1, 2, 3, 4):
        return None
    for shared_m in (128, 64):
        bpt = shared_m // group
        if beam_size % bpt or bpt * max_ds > TKV:
            continue
        task_count = batch * hkv * (beam_size // bpt)
        if shared_m == 128 and task_count < core_num * 2:
            continue  # M=128: tasks < 2*cores, load imbalance, skip to M=64
        return TaskTiling(hq, hkv, batch, beam_size, shared_m, group, decode_step, max_ds,
                          0, 0.0, 0, core_num, beams_per_task=bpt, unshared_n=bpt * max_ds)
    return None


def _build_task_table(request_num, beam_size, hkv, shared_lengths, tiling):
    beam_groups = beam_size // tiling.beams_per_task
    tasks = []
    shared_token_start = 0
    for ridx in range(request_num):
        slen = int(shared_lengths[ridx])
        stiles = (slen + TKV - 1) // TKV
        for kh in range(hkv):
            for bg in range(beam_groups):
                bs = bg * tiling.beams_per_task
                tasks.append([shared_token_start, ridx, kh, bs, slen, stiles])
        shared_token_start += slen
    tasks.sort(key=lambda t: (-t[5], t[1], t[2], t[3]))
    return torch.tensor(tasks, dtype=torch.int32), len(tasks)


# ================================================================
#  Vector functions
# ================================================================
@pl.vector_function
def init_running_vf(o_tile, max_tile, sum_tile, rows):
    """m = -inf, l = 0, O = 0 for the task's running state."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    ureg_m = vf.unalign_reg_for_store()
    ureg_s = vf.unalign_reg_for_store()
    vninf = vf.full(NEG_INF)
    vzero = vf.full(0.0)
    for i in pl.range(rows):
        vf.store_unalign(max_tile, vninf, ureg_m, 1, post_update=True)
        vf.store_unalign(sum_tile, vzero, ureg_s, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_m, 0, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_s, 0, post_update=True)
    for i in pl.range(rows):
        for j in pl.range(D_LOOPS):
            off = i * TD + j * FLOAT_REP_SIZE
            vf.store_align(o_tile + off, vzero, preg)
        for _ in pl.range(TAIL_D):
            off = i * TD + D_LOOPS * FLOAT_REP_SIZE
            preg_tail_ex = vf.update_mask(TAIL_D, dtype=pl.DT_FP32)
            vf.store_align(o_tile + off, vzero, preg_tail_ex)


@pl.vector_function
def flash_update_basic_vf(dst, cur, pre, exp_corr, s1_size, has_tail):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(TAIL_D, dtype=pl.DT_FP32)
    for i in pl.range(s1_size):
        r = vf.load_align(exp_corr, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(D_LOOPS):
            off = i * TD + j * FLOAT_REP_SIZE
            v1 = vf.load_align(pre, off)
            v2 = vf.load_align(cur, off)
            t0 = vf.mul(r, v1, preg)
            t1 = vf.add(t0, v2, preg)
            vf.store_align(dst + off, t1, preg)
        for _ in pl.range(has_tail):
            off = i * TD + D_LOOPS * FLOAT_REP_SIZE
            v1 = vf.load_align(pre, off)
            v2 = vf.load_align(cur, off)
            t0 = vf.mul(r, v1, preg_tail)
            t1 = vf.add(t0, v2, preg_tail)
            vf.store_align(dst + off, t1, preg_tail)


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign64(input_tile, dst_tile, max_tile, max_tile_st, sum_tile,
                                           s1_size, s2_size, scale):
    ureg_max = vf.unalign_reg_for_store()
    ureg_sum = vf.unalign_reg_for_store()
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    vreg_min = vf.full(NEG_INF)
    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, scale, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_even, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_sum, 1, post_update=True)
        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all_f16, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_dst_even_f16, _ = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(dst_tile, vreg_dst_even_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign(input_tile, dst_tile, max_tile, max_tile_st, sum_tile,
                                         s1_size, s2_size, scale):
    ureg_max = vf.unalign_reg_for_store()
    ureg_sum = vf.unalign_reg_for_store()
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    vreg_min = vf.full(NEG_INF)
    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)
        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_tail)
        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)
        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_sum, 1, post_update=True)
        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(dst_tile, vreg_exp_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, s1_size, scale):
    ureg_max = vf.unalign_reg_for_store()
    ureg_sum = vf.unalign_reg_for_store()
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)
        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_all)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)
        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_sum, 1, post_update=True)
        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(dst_tile, vreg_exp_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(sum_tile, ureg_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign64(input_tile, dst_tile, max_tile,
                                        tmp_max, tmp_max_st, tmp_sum, s1_size, s2_size, scale):
    ureg_max = vf.unalign_reg_for_store()
    ureg_sum = vf.unalign_reg_for_store()
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    vreg_min = vf.full(NEG_INF)
    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, scale, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)
        vreg_exp_sum = vf.reduce_sum(vreg_exp, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_sum, vreg_exp_sum, ureg_sum, 1, post_update=True)
        vreg_exp_even_f16 = vf.astype(vreg_exp, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_dst_even_f16, _ = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(dst_tile, vreg_dst_even_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_sum, ureg_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign(input_tile, dst_tile, max_tile,
                                      tmp_max, tmp_max_st, tmp_sum, s1_size, s2_size, scale):
    ureg_max = vf.unalign_reg_for_store()
    ureg_sum = vf.unalign_reg_for_store()
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    vreg_min = vf.full(NEG_INF)
    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)
        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_tail)
        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)
        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_sum, vreg_exp_sum, ureg_sum, 1, post_update=True)
        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(dst_tile, vreg_exp_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_sum, ureg_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf(input_tile, dst_tile, max_tile,
                              tmp_max, tmp_max_st, tmp_sum, s1_size, scale):
    ureg_max = vf.unalign_reg_for_store()
    ureg_sum = vf.unalign_reg_for_store()
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)
        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_all)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)
        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_sum, vreg_exp_sum, ureg_sum, 1, post_update=True)
        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(dst_tile, vreg_exp_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_sum, ureg_sum, 0, post_update=True)


@pl.vector_function
def update_exp_sum_vf(exp_diff, max_tile, tmp_max, sum_tile, tmp_sum):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vm = vf.load_align(max_tile, 0)
    vt = vf.load_align(tmp_max, 0)
    ve = vf.exp_sub(vm, vt, preg)
    vf.store_align(exp_diff, ve, preg)
    vf.store_align(max_tile, vt, preg)
    vs = vf.load_align(sum_tile, 0)
    vts = vf.load_align(tmp_sum, 0)
    t0 = vf.mul(vs, ve, preg)
    t1 = vf.add(t0, vts, preg)
    vf.store_align(sum_tile, t1, preg)


@pl.vector_function
def normalize_store_vf(running_o, running_m, running_l, out_f16, half_s1):
    """Final: O / l -> fp16 store; reset m/l for next task.
    Uses pl.cast at tile level in the caller (o_f16 -> pl.cast -> pl.store)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vninf = vf.full(NEG_INF)
    vzero = vf.full(0.0)
    for i in pl.range(half_s1):
        lv = vf.load_align(running_l, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(D_LOOPS):
            off = i * TD + j * FLOAT_REP_SIZE
            vv_tmp = vf.load_align(running_o, off)
            vv = vf.div(vv_tmp, lv, preg)
            vf.store_align(running_o + off, vv, preg)
    ureg_m = vf.unalign_reg_for_store()
    ureg_s = vf.unalign_reg_for_store()
    for i in pl.range(half_s1):
        vf.store_unalign(running_m, vninf, ureg_m, 1, post_update=True)
        vf.store_unalign(running_l, vzero, ureg_s, 1, post_update=True)
    vf.store_unalign_post(running_m, ureg_m, 0, post_update=True)
    vf.store_unalign_post(running_l, ureg_s, 0, post_update=True)


@pl.vector_function
def gen_umask_vf(mask_tile, wtbl_w, wtbl_we, row_off, rows):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    merge_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    merge_unroll_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    row_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    temp_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    vreg_zero = vf.full(0, dtype=pl.DT_UINT16)
    vreg_one = vf.full(1, dtype=pl.DT_UINT16)
    index = vf.arange(0, dtype=pl.DT_INT32)
    index_unroll = vf.arange(64, dtype=pl.DT_INT32)
    for m in pl.range(0, rows):
        wreg = vf.load_align(wtbl_w, (row_off + m) * 8, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
        wereg = vf.load_align(wtbl_we, (row_off + m) * 8, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
        merge_bit = vf.ge(index, wreg, preg_all)
        temp_reg = vf.lt(index, wereg, preg_all)
        merge_bit = vf.and_(merge_bit, temp_reg, preg_all)
        merge_unroll_bit = vf.ge(index_unroll, wreg, preg_all)
        temp_reg = vf.lt(index_unroll, wereg, preg_all)
        merge_unroll_bit = vf.and_(merge_unroll_bit, temp_reg, preg_all)
        row_reg, temp_reg = vf.de_interleave(merge_bit, merge_unroll_bit, dtype=pl.DT_UINT16)
        mask_b16 = vf.select(vreg_zero, vreg_one, row_reg)
        vf.store_align(mask_tile + m * TKV, mask_b16, preg_all_b16, dist=pl.StoreDist.PACK)


@pl.vector_function
def process_vec1_ug_update_unalign64(input_tile, dst_tile, max_tile, tmp_max, tmp_max_st, tmp_sum,
                                     sum_tile, mask_tile, s1_size, s2_size, scale):
    """Masked softmax + running-state update, s2_size <= 64."""
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_sum = vf.unalign_reg_for_store()
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    vreg_min = vf.full(NEG_INF)
    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, scale, preg_tail)
        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_even, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_sum, vreg_exp_sum, ureg_sum, 1, post_update=True)
        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all_f16, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_dst_even_f16, vreg_dst_odd_f16 = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(dst_tile, vreg_dst_even_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_sum, ureg_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_ug_update_unalign(input_tile, dst_tile, max_tile, tmp_max, tmp_max_st, tmp_sum,
                                   sum_tile, mask_tile, s1_size, s2_size, scale):
    """Masked softmax + running-state update, 64 < s2_size <= 128."""
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_hi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_sum = vf.unalign_reg_for_store()
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
    vreg_min = vf.full(NEG_INF)
    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)
        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_tail)
        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_hi = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_hi)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)
        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_sum, vreg_exp_sum, ureg_sum, 1, post_update=True)
        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(dst_tile, vreg_exp_f16, preg_all_f16,
                       block_stride=BLOCK_STRIDE_ND, repeat_stride=REPEAT_STRIDE_ND,
                       data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, post_update=True)
    vf.store_unalign_post(tmp_sum, ureg_sum, 0, post_update=True)


# ================================================================
#  Kernel
# ================================================================
@pl.jit(arch="a5", auto_mutex=True, compile_timeout=200,
        tiling_key=XAttnV2TilingKey,
        datatype={
            "query": "io_dtype", "shared_key_block": "io_dtype", "shared_value_block": "io_dtype",
            "unshared_key_block": "io_dtype", "unshared_value_block": "io_dtype", "attn_out": "io_dtype",
        })
def x_attention_v2_kernel(
    query: pl.Ptr[pl.DT_UINT8],
    shared_key_block: pl.Ptr[pl.DT_UINT8],
    shared_value_block: pl.Ptr[pl.DT_UINT8],
    unshared_key_block: pl.Ptr[pl.DT_UINT8],
    unshared_value_block: pl.Ptr[pl.DT_UINT8],
    unshared_block_table: pl.Ptr[pl.DT_UINT8],
    shared_kv_lens: pl.Ptr[pl.DT_UINT8],
    decode_step: pl.Ptr[pl.DT_UINT8],
    task_table: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    tiling: TaskTiling,
):
    M = SharedM  # noqa: F821 — compile-time constant from tiling key
    HALF = M // 2
    group = tiling.group
    bpt = M // group
    u_ds = tiling.unshared
    max_ds = tiling.max_ds
    unshared_n = bpt * max_ds
    scale = tiling.scale
    total_cores = tiling.total_cores
    task_count = tiling.task_count
    batch = tiling.batch
    beam_size = tiling.beam_size
    hq = tiling.hq
    hkv = tiling.hkv
    shared_total = tiling.shared_total
    num_tokens = tiling.num_tokens

    core_id = pl.get_block_idx() // pl.get_subblock_num()
    sub_id = pl.get_subblock_idx()


    # Query/output permuted to [B, Hkv, beam*group, D] (task rows contiguous).
    q_perm = pl.make_tensor(query, [batch, hkv, beam_size * group, TD],
                            [hkv * beam_size * group * TD, beam_size * group * TD, TD, 1],
                            dtype=io_dtype)
    o_perm = pl.make_tensor(attn_out, [batch, hkv, beam_size * group, TD],
                            [hkv * beam_size * group * TD, beam_size * group * TD, TD, 1],
                            dtype=io_dtype)
    s_k = pl.make_tensor(shared_key_block, [shared_total, hkv, TD], [hkv * TD, TD, 1], dtype=io_dtype)
    s_v = pl.make_tensor(shared_value_block, [shared_total, hkv, TD], [hkv * TD, TD, 1], dtype=io_dtype)
    # unshared permuted [B, Hkv, beam*maxDs, TD]
    u_k = pl.make_tensor(unshared_key_block, [batch, hkv, beam_size * max_ds, TD],
                         [hkv * beam_size * max_ds * TD, beam_size * max_ds * TD, TD, 1],
                         dtype=io_dtype)
    u_v = pl.make_tensor(unshared_value_block, [batch, hkv, beam_size * max_ds, TD],
                         [hkv * beam_size * max_ds * TD, beam_size * max_ds * TD, TD, 1],
                         dtype=io_dtype)
    ubt = pl.make_tensor(unshared_block_table, [batch], [1], dtype=pl.DT_INT32)
    skv = pl.make_tensor(shared_kv_lens, [batch], [1], dtype=pl.DT_INT32)
    ds = pl.make_tensor(decode_step, [1], [1], dtype=pl.DT_INT32)
    ttab = pl.make_tensor(task_table, [1, 6], [6, 1], dtype=pl.DT_INT32)

    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS, TKV], dtype=io_dtype, target_memory=pl.MemorySpace.Mat),
        addrs=MA2_P, mutex_ids=[14, 15, 16])
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                          valid_shape=[-1, -1], compact=1, pad=pl.TilePad.min),
        addrs=VA0, mutex_ids=[17, 18])
    pv_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA8, mutex_ids=[19, 20])

    # ========== CUBE SECTION ==========
    with pl.section_cube():
        q_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Mat,
                              valid_shape=[-1, -1], compact=1),
            addrs=MA0_Q, mutex_ids=[0, 1])
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=io_dtype, target_memory=pl.MemorySpace.Mat,
                              layout=pl.ZN, valid_shape=[-1, -1], compact=1),
            addrs=MA1_K, mutex_ids=[2, 3])
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Mat,
                              valid_shape=[-1, -1], compact=1),
            addrs=MA3_V, mutex_ids=[4, 5])
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Left,
                              valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[6, 7])
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TD, TKV], dtype=io_dtype, target_memory=pl.MemorySpace.Right,
                              valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[8, 9])
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                              valid_shape=[-1, -1], compact=1),
            addrs=[0, 65536, 131072, 196608], mutex_ids=[10, 11, 12, 13])
        left2 = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TKV], dtype=io_dtype, target_memory=pl.MemorySpace.Left,
                              valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[6, 7])
        right2 = pl.make_tile_group(
            type=pl.TileType(shape=[TKV, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Right,
                              valid_shape=[-1, -1], compact=1),
            addrs=[0, 32768], mutex_ids=[8, 9])
        acc2 = pl.make_tile_group(
            type=pl.TileType(shape=[TS, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                              valid_shape=[-1, -1], compact=1),
            addrs=[0, 65536, 131072, 196608], mutex_ids=[10, 11, 12, 13])

        for task_idx in pl.range(core_id, task_count, total_cores):
            tbase = task_idx * 6
            shared_token_start = pl.getval(ttab, tbase + 0)
            request_idx = pl.getval(ttab, tbase + 1)
            kv_head = pl.getval(ttab, tbase + 2)
            beam_start = pl.getval(ttab, tbase + 3)
            shared_len = pl.getval(ttab, tbase + 4)
            shared_tiles = pl.getval(ttab, tbase + 5)
            qoff = beam_start * group
            cache_slot = pl.getval(ubt, request_idx)
            u_koff = beam_start * max_ds
            n_stages = shared_tiles + 1  # shared tiles + 1 unshared

            cur_q = q_l1_db.next()
            pl.set_validshape(cur_q, [M, TD])
            pl.load(cur_q, q_perm, [request_idx, kv_head, qoff, 0], order=[2, 3])

            # One unified pipeline over (shared tiles + unshared).  All cross-core
            # event ids are derived from the PRODUCING stage's task_id (stored in ctx),
            # so cube and vector stay in lockstep without a shared counter.
            task_id = 0
            ctx_arr = pl.struct_array(4, "Ctx", stage=0, is_shared=0, s2=0)
            for stage in pl.range(0, n_stages + 2):
                if stage < n_stages:
                    ctx_cur = ctx_arr[task_id % 4]
                    ctx_cur.stage = stage
                    ctx_cur.is_shared = 1 if stage < shared_tiles else 0
                    if ctx_cur.is_shared:
                        ctx_cur.s2 = pl.min(TKV, shared_len - stage * TKV)
                    else:
                        ctx_cur.s2 = unshared_n
                    cur_k = k_l1_db.next()
                    pl.set_validshape(cur_k, [TD, ctx_cur.s2])
                    if ctx_cur.is_shared:
                        pl.load(cur_k, s_k, [shared_token_start + stage * TKV, kv_head, 0],
                                order=[2, 0])
                    else:
                        pl.load(cur_k, u_k, [cache_slot, kv_head, u_koff, 0], order=[3, 2])
                    qk_left = left_db.next()
                    qk_right = right_db.next()
                    qk_acc = acc_db.next()
                    pl.set_validshape(qk_left, [M, TD])
                    pl.move(qk_left, cur_q)
                    pl.set_validshape(qk_right, [TD, ctx_cur.s2])
                    pl.move(qk_right, cur_k)
                    pl.set_validshape(qk_acc, [M, ctx_cur.s2])
                    pl.matmul(qk_acc, qk_left, qk_right)
                    qk_slot = qk_vec_db.next()
                    pl.system.wait_cross_core(pipe=pl.PipeType.FIX,
                                              event_id=QK_READY_BARKWARD_IDS[task_id % 2])
                    pl.set_validshape(qk_slot, [(M + 1) // 2, ctx_cur.s2])
                    pl.set_validshape(qk_acc, [(M + 1) // 2 * 2, (ctx_cur.s2 + 7) // 8 * 8])
                    pl.move(qk_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                              event_id=QK_READY_FORWARD_IDS[task_id % 2])
                pv_stage = task_id - 2
                if pv_stage >= 0 and pv_stage < n_stages:
                    ctx_pre = ctx_arr[(task_id + 2) % 4]
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1,
                                              event_id=P_READY_FORWARD_IDS[ctx_pre.stage % 3])
                    cur_v = v_l1_db.next()
                    pl.set_validshape(cur_v, [ctx_pre.s2, TD])
                    if ctx_pre.is_shared:
                        pl.load(cur_v, s_v, [shared_token_start + ctx_pre.stage * TKV, kv_head, 0],
                                order=[0, 2])
                    else:
                        pl.load(cur_v, u_v, [cache_slot, kv_head, u_koff, 0], order=[2, 3])
                    pv_left = left2.next()
                    pv_right = right2.next()
                    pv_acc = acc2.next()
                    cur_p = p_mat_db.next()
                    pl.set_validshape(cur_p, [M, ctx_pre.s2])
                    pl.set_validshape(pv_left, [M, ctx_pre.s2])
                    pl.move(pv_left, cur_p)
                    pl.set_validshape(pv_right, [ctx_pre.s2, TD])
                    pl.move(pv_right, cur_v)
                    pl.set_validshape(pv_acc, [M, TD])
                    pl.matmul(pv_acc, pv_left, pv_right)
                    pv_slot = pv_vec_db.next()
                    pl.system.wait_cross_core(pipe=pl.PipeType.FIX,
                                              event_id=PV_READY_BARKWARD_IDS[ctx_pre.stage % 2])
                    pl.set_validshape(pv_slot, [(M + 1) // 2, TD])
                    pl.set_validshape(pv_acc, [(M + 1) // 2 * 2, TD])
                    pl.move(pv_slot, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                                              event_id=PV_READY_FORWARD_IDS[ctx_pre.stage % 2])
                task_id = task_id + 1

    # ========== VECTOR SECTION ==========
    with pl.section_vector():
        tile_nz_g = pl.make_tile_group(
            type=pl.TileType(shape=[65, 128], dtype=io_dtype, target_memory=pl.MemorySpace.Vec,
                              valid_shape=[-1, -1], layout=pl.NZ),
            addrs=VA1, mutex_ids=[0, 1])
        running_o = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA7, size=VB4)
        o_f16_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=VA9, mutex_ids=[2])
        red_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                               layout=pl.DN)
        gmax_0 = pl.make_tile(red_type, addr=VA_GMAX0, size=VB_RED)
        gmax_1 = pl.make_tile(red_type, addr=VA_GMAX1, size=VB_RED)
        gmax_2 = pl.make_tile(red_type, addr=VA_GMAX2, size=VB_RED)
        global_max = (gmax_0, gmax_1, gmax_2)
        gsum_0 = pl.make_tile(red_type, addr=VA_GSUM0, size=VB_RED)
        gsum_1 = pl.make_tile(red_type, addr=VA_GSUM1, size=VB_RED)
        gsum_2 = pl.make_tile(red_type, addr=VA_GSUM2, size=VB_RED)
        global_sum = (gsum_0, gsum_1, gsum_2)
        tmp_max = pl.make_tile(red_type, addr=VA11, size=VB_RED)
        tmp_sum = pl.make_tile(red_type, addr=VA12, size=VB_RED)
        exp_max0 = pl.make_tile(red_type, addr=VA_EXPMAX0, size=VB_RED)
        exp_max1 = pl.make_tile(red_type, addr=VA_EXPMAX1, size=VB_RED)
        exp_max2 = pl.make_tile(red_type, addr=VA_EXPMAX2, size=VB_RED)
        exp_corr_db = (exp_max0, exp_max1, exp_max2)

        umask_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
            addrs=VA_UMASK, mutex_ids=[5])
        w_tbl = pl.make_tile(pl.TileType(shape=[TS, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
                             addr=VA_WTBL, size=VB_WTBL)
        we_tbl = pl.make_tile(pl.TileType(shape=[TS, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
                              addr=VA_WTBL + VB_WTBL, size=VB_WTBL)

        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[1])

        # Build the (w, we) mask window tables ONCE per launch.
        for t_r in pl.range(0, TS):
            t_w = (t_r // group) * max_ds
            pl.setval(w_tbl, t_r * 8, t_w)
            pl.setval(we_tbl, t_r * 8, t_w + u_ds)
        pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.V, event_id=6)
        pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.V, event_id=6)

        for task_idx in pl.range(core_id, task_count, total_cores):
            tbase = task_idx * 6
            request_idx = pl.getval(ttab, tbase + 1)
            kv_head = pl.getval(ttab, tbase + 2)
            beam_start = pl.getval(ttab, tbase + 3)
            shared_len = pl.getval(ttab, tbase + 4)
            shared_tiles = pl.getval(ttab, tbase + 5)
            qoff = beam_start * group
            n_stages = shared_tiles + 1

            half_s1 = HALF
            first_s1 = HALF
            if sub_id == 1:
                half_s1 = M - HALF
            state_slot = 0
            gmax = global_max[state_slot]
            gsum = global_sum[state_slot]
            init_running_vf(running_o, gmax, gsum, half_s1)

            # Unified stage loop: shared tiles (0..shared_tiles-1) + unshared (shared_tiles)
            task_id = 0
            q_count = 0
            ctx_arr = pl.struct_array(4, "VecCtx", stage=0, is_shared=0, half_s1=0, first_s1=0,
                                       s2=0, q_count=0)
            for stage in pl.range(0, n_stages + 3):
                if stage < n_stages:
                    ctx_cur = ctx_arr[task_id % 4]
                    ctx_cur.stage = stage
                    ctx_cur.is_shared = 1 if stage < shared_tiles else 0
                    ctx_cur.half_s1 = half_s1
                    ctx_cur.first_s1 = first_s1
                    ctx_cur.s2 = pl.min(TKV, shared_len - stage * TKV) if stage < shared_tiles else unshared_n
                    ctx_cur.q_count = q_count
                p_stage = task_id - 1
                if p_stage >= 0 and p_stage < n_stages:
                    ctx_p = ctx_arr[(task_id + 3) % 4]
                    p_eid = ctx_p.stage % 2
                    q_idx_p = ctx_p.q_count % 3
                    gmax_p = global_max[q_idx_p]
                    gsum_p = global_sum[q_idx_p]
                    qk_slot = qk_vec_db.next()
                    tile_nz = tile_nz_g.next()
                    pl.system.wait_cross_core(pipe=pl.PipeType.V,
                                              event_id=QK_READY_FORWARD_IDS[p_eid])
                    row_off = ctx_p.first_s1 * sub_id
                    if ctx_p.is_shared:
                        if ctx_p.stage == 0:
                            if ctx_p.s2 == 128:
                                process_vec1_nd_no_update_vf(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p,
                                                             ctx_p.half_s1, scale)
                            elif ctx_p.s2 <= 64:
                                process_vec1_nd_no_update_vf_unalign64(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p,
                                                                       ctx_p.half_s1, ctx_p.s2, scale)
                            else:
                                process_vec1_nd_no_update_vf_unalign(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p,
                                                                     ctx_p.half_s1, ctx_p.s2, scale)
                        else:
                            if ctx_p.s2 == 128:
                                process_vec1_nd_update_vf(qk_slot, tile_nz, gmax_p, tmp_max, tmp_max, tmp_sum,
                                                          ctx_p.half_s1, scale)
                            elif ctx_p.s2 <= 64:
                                process_vec1_nd_update_vf_unalign64(qk_slot, tile_nz, gmax_p, tmp_max, tmp_max, tmp_sum,
                                                                     ctx_p.half_s1, ctx_p.s2, scale)
                            else:
                                process_vec1_nd_update_vf_unalign(qk_slot, tile_nz, gmax_p, tmp_max, tmp_max, tmp_sum,
                                                                   ctx_p.half_s1, ctx_p.s2, scale)
                            update_exp_sum_vf(exp_corr_db[ctx_p.stage % 3], gmax_p, tmp_max,
                                              gsum_p, tmp_sum)
                    else:
                        mask_buf = umask_db.next()
                        pl.set_validshape(mask_buf, [ctx_p.half_s1, TKV])
                        gen_umask_vf(mask_buf, w_tbl, we_tbl, row_off, ctx_p.half_s1)
                        if ctx_p.s2 <= 64:
                            process_vec1_ug_update_unalign64(qk_slot, tile_nz, gmax_p, tmp_max, tmp_max,
                                                             tmp_sum, gsum_p, mask_buf,
                                                             ctx_p.half_s1, ctx_p.s2, scale)
                        else:
                            process_vec1_ug_update_unalign(qk_slot, tile_nz, gmax_p, tmp_max, tmp_max,
                                                           tmp_sum, gsum_p, mask_buf,
                                                           ctx_p.half_s1, ctx_p.s2, scale)
                        update_exp_sum_vf(exp_corr_db[ctx_p.stage % 3], gmax_p, tmp_max,
                                          gsum_p, tmp_sum)
                    pl.system.set_cross_core(pipe=pl.PipeType.V,
                                              event_id=QK_READY_BARKWARD_IDS[p_eid])
                    cur_p = p_mat_db.next()
                    pl.set_validshape(tile_nz, [ctx_p.half_s1, ctx_p.s2])
                    pl.insert(cur_p, tile_nz, [row_off, 0])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                                              event_id=P_READY_FORWARD_IDS[ctx_p.stage % 3])
                g_stage = task_id - 3
                if g_stage >= 0 and g_stage < n_stages:
                    ctx_gu = ctx_arr[(task_id + 1) % 4]
                    ec = exp_corr_db[ctx_gu.stage % 3]
                    pv_slot = pv_vec_db.next()
                    pl.system.wait_cross_core(pipe=pl.PipeType.V,
                                              event_id=PV_READY_FORWARD_IDS[ctx_gu.stage % 2])
                    pl.set_validshape(running_o, [ctx_gu.half_s1, TD])
                    pl.set_validshape(pv_slot, [ctx_gu.half_s1, TD])
                    has_tail = 0
                    if TAIL_D != 0:
                        has_tail = 1
                    if ctx_gu.stage == 0:
                        pl.move(running_o, pv_slot)
                    else:
                        flash_update_basic_vf(running_o, pv_slot, running_o, ec,
                                              ctx_gu.half_s1, has_tail)
                    pl.system.set_cross_core(pipe=pl.PipeType.V,
                                              event_id=PV_READY_BARKWARD_IDS[ctx_gu.stage % 2])
                    ctx_gu.stage = -1
                task_id = task_id + 1
            q_count = q_count + 1

            # ---- finalize: O / l -> fp16 -> store (permuted layout) ----
            o_out = o_f16_g.next()
            pl.set_validshape(o_out, [half_s1, TD])
            normalize_store_vf(running_o, gmax, gsum, o_out, half_s1)
            pl.cast(o_out, running_o, mode=pl.RoundMode.CAST_ROUND)
            o_row = qoff + first_s1 * sub_id
            pl.store(o_perm, o_out, [request_idx, kv_head, o_row, 0], order=[2, 3])


# ================================================================
#  Host entry
# ================================================================
def x_attention_v2(
    query: torch.Tensor,
    shared_k: torch.Tensor,
    shared_v: torch.Tensor,
    unshared_k: torch.Tensor,
    unshared_v: torch.Tensor,
    shared_kv_lens: torch.Tensor,
    decode_step: torch.Tensor,
    attn_out: torch.Tensor,
    unshared_block_table: torch.Tensor | None = None,
    shared_block_table: torch.Tensor | None = None,
    scale: float | None = None,
) -> None:
    """Task-split xLLM-style beam-decode attention, external entry point.

    Same calling convention as x_attention (see x_attention.py): writes the
    output into the CALLER-provided ``attn_out`` (no return value).

      query               [B*beam, Hq, D]                 fp16/bf16, NPU tensor
      shared_key/value    [sum(Lb), Hkv, D] contiguous    NPU tensors
      unshared_key/value  [B, beam, Hkv, maxDecodeStep, D]  NPU tensors
      shared_kv_lens      [B] int32 NPU tensor: per-batch shared KV lengths
      decode_step         [1] int32 NPU tensor: valid unshared length ([1, maxDs])
      attn_out            pre-allocated NPU output [B*beam, Hq, D] (same shape
                          and dtype as query); written in place.
      unshared_block_table [B] int32 NPU tensor or None: slot gather (None =
                          direct logical-batch addressing).
      shared_block_table  [B, maxBlocks] int32 NPU tensor or None: paged shared
                          block table (None = contiguous layout).  Paged shared
                          is NOT supported: a non-None value raises
                          NotImplementedError.
      scale               optional scale_value; default 1/sqrt(D).

    Supported profile subset (narrower than x_attention): group in {2,4},
    decode_step in [1,4], d = 128, beam divisible by 64//group or 128//group,
    with (64//group)*maxDecodeStep <= 128 (M=64) or (128//group)*maxDecodeStep
    <= 128 (M=128, requires group=4 or group=2 with maxDecodeStep<=2).  M=128
    is preferred when feasible.  Unsupported profiles raise ValueError.
    """
    if shared_block_table is not None:
        raise NotImplementedError(
            "x_attention_v2: paged shared KV (shared_block_table) not supported")
    device = query.device
    num_tokens, hq, d = query.shape
    batch = unshared_k.shape[0]
    beam = num_tokens // batch
    hkv = shared_k.shape[1]
    ds = int(decode_step.item())
    max_ds = unshared_k.shape[3]
    group = hq // hkv
    shared_lens = [int(x) for x in shared_kv_lens.cpu().tolist()]
    shared_total = shared_k.shape[0]

    if hq % hkv or group not in (2, 4) or ds not in (1, 2, 3, 4) or d != 128:
        raise ValueError(
            f"Unsupported profile for x_attention_v2: group={group} "
            f"(need 2/4), decode_step={ds} (need [1,4]), d={d} (need 128), "
            f"Hq%Hkv={hq % hkv}")
    total_cores = get_platform_info().core_num
    batch = unshared_k.shape[0]
    valid_bpt = None
    for shared_m in (128, 64):
        bpt_c = shared_m // group
        if beam % bpt_c or bpt_c * max_ds > TKV:
            continue
        task_count = batch * hkv * (beam // bpt_c)
        if shared_m == 128 and task_count < total_cores * 2:
            continue
        valid_bpt = bpt_c
        break
    if valid_bpt is None:
        raise ValueError(
            f"Unsupported profile for x_attention_v2: beam={beam} must be "
            f"divisible by beams_per_task (64//group or 128//group) and "
            f"beams_per_task*maxDecodeStep({max_ds}) <= {TKV}")
    tiling = _select_tiling(hq, hkv, batch, beam, ds, max_ds, total_cores)
    if tiling is None:
        raise ValueError("No valid tiling")
    scale_value = scale if scale is not None else 1.0 / math.sqrt(d)
    tiling.shared_total = shared_total
    tiling.scale = scale_value
    tiling.num_tokens = num_tokens
    ttab, tc = _build_task_table(batch, beam, hkv, shared_lens, tiling)
    tiling.task_count = tc
    tiling.total_cores = total_cores

    # Permute query and unshared into task-contiguous layouts.
    q_perm = (query.view(batch, beam, hkv, group, d)
              .permute(0, 2, 1, 3, 4).contiguous())
    u_k_perm = unshared_k.permute(0, 2, 1, 3, 4).contiguous()
    u_v_perm = unshared_v.permute(0, 2, 1, 3, 4).contiguous()
    o_perm = torch.empty_like(q_perm)

    ttab = ttab.to(device)
    q_perm = q_perm.to(device)
    u_k_perm = u_k_perm.to(device)
    u_v_perm = u_v_perm.to(device)
    o_perm = o_perm.to(device)

    pl_dtype = pl.DT_FP16 if query.dtype == torch.float16 else pl.DT_BF16
    dt = {k: pl_dtype for k in ("query", "shared_key_block", "shared_value_block", "unshared_key_block", "unshared_value_block", "attn_out")}
    ubt_t = unshared_block_table if unshared_block_table is not None \
        else torch.arange(batch, dtype=torch.int32, device=device)
    x_attention_v2_kernel[None, tiling.total_cores, {"SharedM": tiling.shared_m}, dt](
        q_perm, shared_k, shared_v, u_k_perm, u_v_perm,
        ubt_t, shared_kv_lens, decode_step, ttab, o_perm, tiling,
    )
    torch.npu.synchronize()

    # Permute output back to [T, Hq, D].
    attn_out.copy_(
        o_perm.view(batch, hkv, beam, group, d).permute(0, 2, 1, 3, 4).reshape(num_tokens, hq, d))


__all__ = ["x_attention_v2"]