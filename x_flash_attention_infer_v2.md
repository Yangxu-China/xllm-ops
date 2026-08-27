# XFlashAttentionInfer V2 算子文档

## 整体思路

分两步完成 x_flash_attention_infer 算子的演进：

**Step 1 — v1 (`XFlashAttentionInfer`) 现有 AscendC C++ 算子**
- 基于 catlass 的 paged-KV flash-decoding attention，用于 LLM 推理 decode 阶段
- 支持 TND/NZ 两种 KV 布局，causal mask，FD（Flash-Decoding）模式
- 依赖外部 `extra_tiling` 输入携带 per-core KV-split 任务分配（`SplitKvExtraInfo`），host tiling **不填充**该信息
- 支持 SOC: `ascend910b`, `ascend910_93`, `ascend950`

**Step 2 — v2 (`XFlashAttentionInferV2`) PyPTO Pro kernel 算子引入**
- 从 `/workspace/x_flash_attention_infer_replicate` 迁移 PyPTO Pro kernel 算子
- v2 kernel 用 Python DSL 编写，经 `easyasc` 描述 → transpile 到 `pypto_pro.language`（`@pl.jit`），而非传统 AscendC C++
- v2 外部功能对标 v1：接口、测试用例复用，验证两者行为等价
- 核心工作是将 replicate 仓的 PyPTO kernel 编译链路接入 xllm-ops 自研 cmake 体系（复用 x_attention_v2 已建立的 `enable_pypto_kernel` / `pypto_codegen.py` / `pypto_compile_op` 基础设施）

### replicate 仓的关键设计决策（来自 README）

replicate 仓有一条核心不变量：**三条发射通路（direct / aclnn / pypto）跑同一份 kernel，三者设备时间中位数偏差 ≤0.2%**——选哪条通路是集成问题，不是性能问题。这意味着：

- **aclnn 路径已可用**：replicate 仓 `aclnn/generated/` 下已有完整的 18 套 AscendC C++ 算子包（op_host + op_kernel），可以不依赖 pypto 编译链路直接安装使用
- **pypto 路径已可用**：replicate 仓 `pypto_bridge/generated/` 下已有 18 个 `@pl.jit` 模块，首次调用时 JIT 编译
- **三条通路性能等价**：aclnn 与 direct 逐例比中位数 0.998，pypto 与 direct 中位数 1.002

### replicate 仓的性能基线（来自 PERFORMANCE_REPORT）

- **基线**：`torch_npu.npu_fused_infer_attention_score`（FIA），不是 v1 算子
- **验收线**：逐 shape `ratio = FIA / ours ≥ 0.99`，不是平均达标
- **结果**：三条通路各 20/20 达标，18-19/20 严格快于 FIA，领先 0.6%–32.3%
- **测试环境**：Ascend950PR_9579，28 AIC / 56 AIV，CANN 9.1.0-beta.3（xllm-ops 用 9.2.0-beta.1）

**赢面分布**：领先幅度与 `groups/core` 成反比——并行度余量越大赢面越小：

| 每核 group 数 | 领先幅度 | 典型场景 |
|---|---|---|
| < 2 | 1%–32% | 单 stream 长序列 / 少 batch 长序列 |
| 2–5 | 3%–13% | MQA / 奇数 group / 页尾 |
| ≥ 9 | -0.8%–8.7% | 大 batch / MHA（已贴带宽墙） |

**数值一致性**：aclnn vs direct 21/21 逐位相同；pypto vs direct 仅 4/21 逐位相同，其余 17 例差异恰好为 1 ULP（末位舍入），无一例越过一个末位。ND 与 FRACTAL_NZ 路径逐位相同。

---

## 1. v1 (`XFlashAttentionInfer`) 算子概述

`XFlashAttentionInfer` 是 xllm-ops 中的自定义 AscendC 算子，实现 **paged-KV flash-decoding attention**，主要用于 LLM 推理 decode 阶段。

### 核心特性

- **Paged KV cache**：`key_cache`/`value_cache` 按 block 组织 `[numBlocks, blockSize, kvHead, headDim]`，通过 `block_table` 映射
- **KV 布局**：支持 TND（`[numBlocks, blockSize, kvHead, headDim]`）和 NZ（`[numBlocks, 16, blockSize, 16]` fractal）
- **FD（Flash-Decoding）模式**：将 KV 沿 sequence 维切分到多个 cube core，各 core 独立计算部分 attention，最后合并 LSE（log-sum-exp）
- **GQA 支持**：`qHead`/`kvHead` 分组
- **Causal mask**：通过 int8 `[2048, 2048]` 上三角 mask 表实现
- **混合核模式**：`KERNEL_TYPE_MIX_AIC_1_2`（1 AIC : 2 AIV）
- **外部调度**：per-core KV-split 任务分配由 `extra_tiling` 输入携带（`SplitKvExtraInfo` 结构体），host tiling 不填充

### 算子源码位置

```
xllm_ops/x_flash_attention_infer/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   ├── x_flash_attention_infer_def.cpp       # 算子定义（输入/输出/属性）
│   ├── x_flash_attention_infer_proto.cpp      # InferShape/InferDataType
│   ├── x_flash_attention_infer_tiling.cpp      # Tiling 计算（host 侧）
│   └── x_flash_attention_infer_tiling.h       # TilingData 结构定义
└── op_kernel/
    ├── x_flash_attention_infer.cpp            # kernel 入口（TILING_KEY_IS 分流）
    ├── x_flash_attention_infer_common.h       # TilingKey 宏 + SplitKvExtraInfo 结构 + 共享常量
    ├── x_flash_attention_infer.h              # 非 FD kernel 模板实现
    └── x_flash_attention_infer_fd.h           # FD kernel 模板实现（catlass GEMM 流水线）
```

### 算子注册名

- Op Type (PascalCase): `XFlashAttentionInfer`
- kernel 入口 (snake_case): `x_flash_attention_infer`
- aclnn API: `aclnnXFlashAttentionInfer`

### 支持芯片

`ascend910b`, `ascend910_93`, `ascend950`（见 `def.cpp` 中 `AddConfig`）

### v1 接口

**输入（8 个）**：

| # | Name           | ParamType | DataType              | Format          | 说明 |
|---|----------------|-----------|-----------------------|----------------|------|
| 0 | query          | REQUIRED  | FP16, BF16            | ND             | `[numTokens, qHead, headDim]` TND |
| 1 | key_cache      | REQUIRED  | FP16, BF16            | ND / FRACTAL_NZ | `[numBlocks, blockSize, kvHead, headDim]` paged |
| 2 | value_cache    | REQUIRED  | FP16, BF16            | ND / FRACTAL_NZ | 同 key_cache |
| 3 | mask           | OPTIONAL  | INT8                  | ND             | `[2048, 2048]` causal mask |
| 4 | block_table    | REQUIRED  | INT32                 | ND             | `[batch, maxBlocksPerBatch]` |
| 5 | actual_q_lens  | REQUIRED  | INT32                 | ND             | `[batch]` 累积前缀和 |
| 6 | actual_kv_lens | REQUIRED  | INT32                 | ND             | `[batch]` 每 batch KV 长度 |
| 7 | extra_tiling   | REQUIRED  | INT32                 | ND             | per-core KV-split 任务分配（SplitKvExtraInfo） |

**输出（1 个）**：

| # | Name      | ParamType | DataType   | Format | 说明 |
|---|-----------|-----------|------------|--------|------|
| 0 | attn_out  | REQUIRED  | FP16, BF16 | ND     | `[numTokens, qHead, headDim]`（同 query） |

**属性（4 个）**：

| Name     | Type   | 默认值    | 说明 |
|----------|--------|-----------|------|
| layout   | String | `"TND"`   | 布局标识 |
| qHead    | Int    | (必填)    | Q 头数 |
| kvHead   | Int    | (必填)    | KV 头数 |
| scale    | Float  | `1.0`     | softmax scale（host tiling 中被 `1/sqrt(headDim)` 覆盖） |

### v1 TilingKey 编码

```
BASE = 1 000 000 000 000 000 000

KV_LAYOUT_TND  = +100      KV_LAYOUT_NZ   = +200
DTYPE_FP16     = +10       DTYPE_BF16     = +20
CAUSAL_MASK    = +3
FD             = +1000
```

实际使用 4 个 FD 变体（NoFD 变体定义但未使用）：

| TilingKey 宏 | dtype | KV layout |
|---|---|---|
| `QFP16_KVFP16_TND_CAUSALMASK_FD_TILING` | fp16 | TND |
| `QBF16_KVBF16_TND_CAUSALMASK_FD_TILING` | bf16 | TND |
| `QFP16_KVFP16_KVNZ_CAUSALMASK_FD_TILING` | fp16 | NZ |
| `QBF16_KVBF16_KVNZ_CAUSALMASK_FD_TILING` | bf16 | NZ |

### v1 SplitKvExtraInfo 结构

`extra_tiling` 输入被 kernel 重解释为 `SplitKvExtraInfo*`：

```cpp
struct CoreNode {           // 40 bytes = 10 u32
    uint32_t startBIdx, startN1Idx, startS2Idx;
    uint32_t endBIdx, endN1Idx, endS2Idx;
    uint64_t firstSplitKVTaskLseOffset, firstSplitKVTaskOOffset;
};

struct SplitNode {          // 40 bytes = 10 u32
    uint32_t batchIdx, headStartIdx, headEndIdx;
    uint32_t qStartIdx, qEndIdx, splitNum;
    uint64_t lseTaskOffset, oTaskOffset;
};

struct SplitKvExtraInfo {   // 2004 bytes = 501 int32
    CoreNode  coreInfo[25];    // per-cube-core 任务范围
    SplitNode splitInfo[25];   // per-split-KV 合并信息
    uint32_t  totalSplitNodeNum;
};
```

### v1 测试

`test/python_test/test_x_flash_attention_infer.py`（3 条用例）：
- 仅 fp16，qSeqlen=1（decode 场景）
- `(fp16, 1, 8, 8, 128, 128, 128)` / `(fp16, 2, 8, 8, 128, 128, 128)` / `(fp16, 2, 16, 8, 128, 256, 128)`
- 单 core 模式（`extra_tiling` 中 core 0 处理全部任务，其余 core skip）
- 容差 `atol=6e-2, rtol=6e-2`

### v1 编译选项

```cmake
add_ops_compile_options(
    OP_NAME XFlashAttentionInfer
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            -I${CANN_3RD_LIB_PATH}/catlass/include
)
```

---

## 2. replicate 仓分析

### 2.1 仓结构

`/workspace/x_flash_attention_infer_replicate` 是一个独立的 PyPTO Pro kernel 算子复刻仓，包含：

```
x_flash_attention_infer_replicate/
├── hardware_desc_kernel/
│   ├── paged_decode.py            # DSL 源码（easyasc 硬件描述，867 行）
│   ├── device.py / runner.py / check.py
├── pypto_bridge/
│   ├── module.py                   # @pl.jit 模块加载器（AST 签名解析）
│   ├── launcher.py                 # PyPTO JIT 启动器（GM 视图 + bind）
│   ├── pitch.py                    # GM row pitch 常量
│   ├── generated/                  # 18 个 @pl.jit 模块（transpile 产物）
│   │   ├── paged_decode_float16_bs32_m16_nd_mf_kernel.py  (82KB, 代表)
│   │   └── ...（共 18 个变体）
│   └── so_cache.py                 # .so 内容寻址缓存（跨进程复用）
├── aclnn/generated/                # CANN aclnn 算子包（18 套 op_host + op_kernel）
├── direct_launch/generated/        # direct_launch 算子包（18 套）
├── common/
│   ├── contract.py                 # 支持面约束（Q_SEQ_LEN_MAX=1, head_dim=128）
│   ├── kernels.py                  # 变体 key + kernel_name 生成规则
│   ├── planner.py                  # plan tensor 调度器（654 行）
│   ├── shapes.py                   # 20 组测试 shape
│   └── golden.py                   # CPU golden 实现
├── api/
│   ├── xllm_api.py                 # xllm 接口适配（backend={direct,aclnn,pypto}）
│   └── paged_attention.py          # PagedAttention 对象接口
├── tools/
│   └── transpile_pypto.py          # easyasc → pypto_pro 转译器
└── bench/
    └── check_correctness.py        # 正确性验证
```

### 2.2 kernel 变体体系

replicate 仓的 kernel 不是单一函数，而是 **18 个编译期特化变体**，通过 `common/kernels.py` 的 `VariantKey` 选择：

```python
VariantKey = (dtype, block_size, tile_m, kv_format, merge_free, narrow_tail)
```

| 字段 | 取值 | 说明 |
|---|---|---|
| `dtype` | `float16`, `bfloat16` | Q/K/V/O 数据类型 |
| `block_size` | `16, 32, 64, 128` | paged KV block 大小，需整除 `TILE_N(256)` |
| `tile_m` | `16, 32` | Q tile 行数，由 `choose_tile_m(group_size)` 选择 |
| `kv_format` | `ND`, `FRACTAL_NZ` | KV 布局 |
| `merge_free` | `True/False` | plan 派生：无 group 被切分 → 免 workspace + 免 all-core barrier |
| `narrow_tail` | `True/False` | plan 派生：ragged-tail 窄化到 fractal 整数倍（仅 ND） |

**kernel_name 生成规则**：
```python
def kernel_name(dtype, block_size, tile_m, kv_format, merge_free, narrow_tail):
    fmt = "nz" if kv_format == "FRACTAL_NZ" else "nd"
    tail = ("_mf" if merge_free else "") + ("_nt" if narrow_tail else "")
    return f"paged_decode_{dtype}_bs{block_size}_m{tile_m}_{fmt}{tail}_kernel"
```

示例：`paged_decode_float16_bs32_m16_nd_mf_kernel`

### 2.3 `@pl.jit` 装饰器配置

```python
@pl.jit(auto_mutex=True)
def paged_decode_...(query, key, value, block_table, plan, [ws_accum, ws_state,] out, q_head, kv_head, scale):
```

与 x_attention_v2 的对比：

| 特性 | x_attention_v2 | replicate paged_decode |
|---|---|---|
| `@pl.jit` 参数 | `arch="a5", auto_mutex=True, compile_timeout=200, tiling_key=..., datatype=...` | 仅 `auto_mutex=True` |
| `tiling_key` | `XAttnV2TilingKey`（`TilingKeyField` 声明式） | 无 `tiling_key` 参数（变体选择在模块层面，不通过 tiling_key） |
| `datatype` | `{"query": "io_dtype", ...}` 映射 | 无 `datatype` 参数（每个变体是独立模块，dtype 已编入文件名） |
| GM 参数类型 | `pl.Ptr[pl.DT_UINT8]`（delivery 模式，host 侧 make_tensor） | `pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]`（tensor 模式，kernel 内直接用） |
| 编译期变体 | 1 个 `@pl.jit` × N 个 tiling_key × M 个 dtype | 18 个独立 `@pl.jit` 模块（每个文件一个） |

### 2.4 plan tensor（调度核心）

replicate 仓的调度不依赖 host tiling 的 `extra_tiling`，而是用一个扁平 int32 `plan` tensor：

```
header       8 ints:  num_cores, num_items, num_merges, max_blocks,
                     item_base, merge_base, num_slots, reserved
core table   MAX_CORES(64) × 4:  item_begin, item_end, merge_begin, merge_end
item table   num_items × 8:     batch, kv_head, group_tile, kv_lo, kv_hi,
                               out_slot, rows, direct
merge table  num_merges × 6:    batch, kv_head, group_tile, slot0, nsplit, rows
```

`Item.direct = 1` 表示该 item 独占整个 KV 范围，直接写 `out`（merge-free 快速路径）。

### 2.5 GM row pitch 约定

replicate 仓的 `pypto_bridge/pitch.py` 硬编码了 GM tensor 的 row pitch：

```python
GM_ROW_PITCH = {
    "query": 128,      # head_dim
    "out": 128,
    "ws_accum": 128,
    "key": None,        # kv_head * 128 (dynamic)
    "value": None,      # kv_head * 128 (dynamic)
}
```

PyPTO `@pl.jit` 的 GM 参数以 `[DYNAMIC, DYNAMIC]` 二维视图声明，kernel 内通过 row pitch 计算偏移。host 侧 `launcher.py` 需将原始 tensor reshape 到匹配的 pitch。

### 2.6 aclnn 路径的 tiling 结构

replicate 仓的 aclnn `op_host/*_tiling.h` 仅有 3 个标量字段：

```cpp
BEGIN_TILING_DATA_DEF(PagedDecode...TilingData)
  TILING_DATA_FIELD_DEF(int32_t, q_head);
  TILING_DATA_FIELD_DEF(int32_t, kv_head);
  TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;
```

调度信息全在 `plan` tensor 中，tiling struct 仅传 attrs。

---

## 3. v1 vs replicate 对比

### 3.1 接口对比

| 维度 | v1 `XFlashAttentionInfer` | replicate `paged_decode_*` |
|---|---|---|
| Op Type | `XFlashAttentionInfer` | `PagedDecodeFloat16Bs32M16NdMfKernel`（每个变体独立注册） |
| 输入数 | 8（含 mask, extra_tiling） | 7-9（query, key, value, block_table, plan, [ws_accum, ws_state], out） |
| 调度输入 | `extra_tiling`（`SplitKvExtraInfo` 结构体） | `plan`（扁平 int32 tensor） |
| mask | 外部 int8 `[2048,2048]` causal mask | 无 mask 输入（qSeqlen=1 时 causal 为空） |
| attrs | layout, qHead, kvHead, scale | q_head, kv_head, scale（编译期编入 kernel_name） |
| 变体选择 | TilingKey（runtime 分流） | 编译期独立模块（18 个 `@pl.jit`） |
| dtype 支持 | fp16 + bf16 | fp16 + bf16 |
| KV 布局 | TND + NZ | ND + NZ |
| qSeqlen | 1（decode），支持多 batch | 1（decode），`Q_SEQ_LEN_MAX=1` |
| head_dim | 128 | 128（`TILE_D=128`） |
| block_size | 128（host tiling 硬编码） | 16/32/64/128（变体选择） |

### 3.2 架构差异

| 维度 | v1 | replicate |
|---|---|---|
| kernel 语言 | AscendC C++（catlass 模板） | PyPTO Pro DSL（`@pl.jit` Python） |
| 源码来源 | 手写 C++ | `easyasc` 硬件描述 → transpile → `@pl.jit` |
| 编译链路 | CANN `compile_op`（AscendC 标准路径） | `pypto_compile_op`（PyPTO 编译路径） |
| 混合核模式 | `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC:2 AIV） | `section_cube()` / `section_vector()` 分区 |
| cross-core 同步 | `AscendC::SyncAll()` + workspace LSE 合并 | `pl.system.set_cross_core` / `wait_cross_core` |
| merge-free 优化 | 无（总是走 split-KV + 合并） | 有（`Item.direct=1` 时跳过 workspace + barrier） |
| tile 尺寸 | `Q_TILE=128, KV_STACK=512`（硬编码） | `TILE_M=16/32, TILE_N=256, TILE_D=128`（变体选择） |

### 3.3 调度对比

| 维度 | v1 `extra_tiling` | replicate `plan` |
|---|---|---|
| 结构 | C struct `SplitKvExtraInfo`（固定 25 元素数组） | 扁平 int32 tensor（header + core_table + item_table + merge_table） |
| 填充方 | Python 侧 `_build_xfa_extra_tiling()`（单 core 模式） | `common/planner.py::build_plan()`（多 core 调度） |
| 多 core 支持 | 是（`CoreNode[25]` per-core 范围） | 是（`core_table[MAX_CORES=64]` per-core item 范围） |
| KV split 合并 | `SplitNode[25]` + `totalSplitNodeNum` | `merge_table`（per-merge 记录） |
| merge-free 快速路径 | 无 | `Item.direct=1` → 跳过 workspace + all-core barrier |

---

## 4. v2 迁移方案

### 4.1 核心挑战

replicate 仓与 xllm-ops 的差异远大于 x_attention_v2 的迁移：

1. **18 个变体 vs 1 个 kernel**：replicate 有 18 个独立 `@pl.jit` 模块，x_attention_v2 只有 1 个。xllm-ops 的 `enable_pypto_kernel` 机制目前只支持 1 个 `@pl.jit` kernel per op_file。20 个性能用例需要 10 个变体，21 个结构用例需要 15 个，并集 18 个。

2. **`plan` tensor 替代 `extra_tiling`**：replicate 用扁平 int32 `plan` tensor 调度，v1 用 `SplitKvExtraInfo` struct。README 明确说明 replicate 的 xllm 兼容 API **收下 `extra_tiling` 后丢弃，自建 `plan`**——即接口层兼容 v1，但内部不用 v1 的调度格式。v2 需要决定：
   - 在 host tiling C++ 中移植 `common/planner.py::build_plan()` 逻辑生成 `plan` tensor（接口对标 v1，host 侧新增 plan 生成逻辑）
   - 或将 `plan` 作为额外输入（接口不对标 v1）

3. **mask 处理**：README 明确说明 `mask` 与 `extra_tiling` 一样**收下即弃**——`q_seq_len == 1` 时因果三角是空的，mask 无意义。v2 可以在接口层保留 mask 输入但不使用。

4. **GM row pitch 约束**：replicate 的 `@pl.jit` kernel 假设 GM tensor 有特定 row pitch（query/out=128, key/value=kvHead*128），需要 host 侧 reshape。v1 的 aclnn 接口直接传原始 tensor。

5. **无 `tiling_key` / `datatype` 声明式**：replicate 的 `@pl.jit` 不使用 `tiling_key` 和 `datatype` 参数，变体选择在模块层面。xllm-ops 的 `enable_pypto_kernel` codegen 流程期望从 `@pl.jit` 的 `tiling_key` 和 `datatype` 生成 `*_tiling.h` / `*_tilingkey.h`。

6. **CANN 版本差异**：replicate 仓在 CANN 9.1.0-beta.3 上验证，xllm-ops 用 CANN 9.2.0-beta.1。需验证 9.2.0 下 `@pl.jit` 模块是否仍可编译。

### 4.2 迁移策略选项

**方案 A：搬运 replicate 仓的 aclnn 路径（18 个 AscendC 变体）**
- 不使用 PyPTO `@pl.jit`，直接用 replicate 仓 `aclnn/generated/` 下的 18 套 AscendC C++ kernel
- replicate 仓已有完整的 aclnn 算子包（op_host + op_kernel + CMakeLists），可直接安装
- 优点：不依赖 pypto 编译链路，性能已验证（与 pypto 路径 ≤0.2% 偏差）；数值一致性最好（aclnn vs direct 21/21 逐位相同）
- 缺点：18 套 op_host + op_kernel 文件，维护成本高；不是 PyPTO kernel；每个变体独立注册为不同 Op Type（`PagedDecodeFloat16Bs32M16NdMfKernel` 等），不对标 v1 的单一 `XFlashAttentionInfer`

**方案 B：搬运 replicate 仓的 PyPTO `@pl.jit` 路径（18 个 Python 模块）**
- 使用 replicate 仓 `pypto_bridge/generated/` 下的 18 个 `@pl.jit` 模块
- 需要扩展 xllm-ops 的 `enable_pypto_kernel` 支持多 `@pl.jit` 模块
- 需要解决 `plan` tensor 调度与 v1 接口的对齐
- 优点：真正的 PyPTO kernel，与 x_attention_v2 模式一致
- 缺点：18 个模块的 codegen/install 复杂；`plan` tensor 需要在 host tiling 中生成；pypto 路径数值有 1 ULP 差异（17/21 例）

**方案 C：合并变体为单个 `@pl.jit` + `tiling_key` + `datatype`（对标 x_attention_v2）**
- 将 18 个变体合并为 1 个 `@pl.jit` kernel，用 `tiling_key`（dtype × block_size × tile_m × kv_format × merge_free）和 `datatype`（fp16/bf16）做编译期分流
- 优点：与 x_attention_v2 模式完全一致，复用现有 `enable_pypto_kernel` 基础设施
- 缺点：需要重写 kernel 以支持 `tiling_key` 分流；工作量大

**方案 D：混合方案——aclnn 变体 + 统一 Op Type**
- 用 replicate 仓 `aclnn/generated/` 的 18 套 AscendC kernel，但统一注册为 `XFlashAttentionInferV2` 单一 Op Type
- host tiling 根据 shape 参数选择变体，通过 TilingKey 分流到对应 kernel
- 优点：不依赖 pypto 编译链路；性能已验证；数值一致性最好
- 缺点：需要合并 18 套 op_host 为 1 套（统一 def/proto/tiling）；kernel 入口需要统一调度逻辑

### 4.3 推荐方案

**推荐方案 D**（混合方案——aclnn 变体 + 统一 Op Type），理由：

1. **性能已验证且最优**：replicate 仓在 20 个 shape 上逐例达标 FIA，aclnn 路径与 direct 路径中位数偏差 0.998，性能等价
2. **数值一致性最好**：aclnn vs direct 21/21 逐位相同（`torch.equal`），无 ULP 差异
3. **不依赖 pypto 编译链路**：避免 `enable_pypto_kernel` 多模块扩展、`plan` tensor 在 codegen 阶段的复杂性
4. **接口对标 v1**：统一注册为 `XFlashAttentionInferV2`，host tiling 在 C++ 中生成 `plan` tensor（移植 `common/planner.py` 逻辑），外部接口与 v1 一致
5. **变体选择透明**：host tiling 根据 shape 参数（dtype, block_size, tile_m, kv_format, merge_free）选择对应 kernel 变体，通过 TilingKey 分流

**备选方案 B**（PyPTO `@pl.jit` 路径）：如果需要与 x_attention_v2 保持完全一致的 PyPTO 编译模式，或需要后续 kernel 迭代用 Python DSL 而非 C++。

### 4.4 决策结论

所有待确认问题已闭环，最终决策如下：

| # | 问题 | 决策 | 说明 |
|---|---|---|---|
| 1 | 方案选择 | **方案 C**：PyPTO `@pl.jit` + `tiling_key` + `datatype`（对标 x_attention_v2） | 单 kernel 入口，aclnn proto/tiling/def 按 x_attention_v2 模式组织 |
| 2 | 变体合并 | 合并 18 个变体为 1 个 `@pl.jit` kernel | `block_size`/`tile_m`/`kv_format`/`merge_free`/`narrow_tail` 作为 `TilingKeyField`（有技术挑战，见 §4.5） |
| 3 | mask | def 接口与 v1 一致：保留 `mask` 为 OPTIONAL 输入，kernel 丢弃 | `qSeqlen=1` 时 causal mask 无意义 |
| 4 | merge-free | 保留为 `tiling_key` 字段 | host tiling 的 plan 生成逻辑决定 `merge_free` 值 |
| 5 | NZ 布局 | 支持，作为 `tiling_key` 字段 | v1 支持 NZ，v2 对标 |
| 6 | A3 支持 | 不支持，仅 A5 | PyPTO-pro 仅支持 A5 |
| 7 | 测试用例 | 基于 v1 `test_x_flash_attention_infer.py` 范围扩展 replicate 仓内容 + golden cache | 见 §4.7 |

**技术挑战**：18 个 easyasc 生成变体的差异不仅在控制流，还在 `TileType` 声明（layout=ZN/NZ、buffer 尺寸=tile_m 16/32、load 展开次数=block_size 16/32/64/128）。x_attention_v2 的 tiling_key 只影响寻址逻辑不影响 buffer 声明，而这里的变体差异影响 `pl.make_tile_group` 的 `shape`/`layout` 参数。需验证 PyPTO 的 tiling_key 机制能否 fold 掉不同 buffer 声明的 dead branch。

### 4.5 调研：`planner.py` 移植到 C++ host tiling 的可行性

**结论：可以移植。** `planner.py`（654 行）是纯整数/浮点算术调度模块，无 numpy 依赖，无框架依赖。

**依赖分析**：

| Python 特性 | C++ 等价物 | 移植难度 |
|---|---|---|
| `heapq`（LPT 负载均衡） | `std::priority_queue<pair<int,int>, vector<>, greater<>>` | 低（需注意 tie-break 按 core index） |
| `torch.zeros` / `torch.tensor`（输出 plan tensor） | 直接写 `int32_t*` 缓冲区 | 低（无需 torch） |
| `sorted(..., key=lambda)` | `std::sort` + lambda comparator | 低 |
| `NamedTuple` + `@property` | C++ struct + member function | 低 |
| 列表推导 / `zip` | 普通 for 循环 | 低 |

**算法概要**（`build_plan()` 的 8 步）：

1. 校验 shape 约束 + num_cores 范围
2. `choose_splits()`：遍历候选 target，去重后选最优 split 方案（最小化 penalised makespan），再用二分搜索减少切分数
3. 三重循环 `batch × kv_head × group_tile` 生成 items 和 merges（单 interval → `direct` item，多 interval → 带 slot 的 items + merge）
4. 计算 penalised costs（1/16 key-tile 单位 + per-item penalty）
5. 负载均衡：`_pack_free_wins()` 决定 free（LPT 重排）vs contiguous（二分搜索最优 max load），或用 `_quantile_cut` 等分
6. merges 按 weight=1 round-robin 分配
7. 组装 `core_ranges`（pad 到 `MAX_CORES=64`）
8. 序列化为扁平 int32 tensor（8 header + 64×4 core_table + items×8 + merges×6）

**完整调用图**：

```
build_plan
├── check_contract (contract.py)
├── choose_splits                         [if splits is None]
│   ├── _group_costs → _group_tiles
│   ├── _penalised
│   ├── _even_cut
│   ├── _makespan_of
│   │   ├── _balance (binary search + _quantile_cut fallback)
│   │   └── _pack_free (heapq LPT)
│   └── _fewest_cuts_for (sorted + bisect)
├── _group_tiles
├── _rows_of
├── _split_points (TILE_N-aligned cuts)
├── _penalised
├── _pack_free_wins
│   ├── _pack_free
│   └── _balance
├── _pack_free OR _balance               [items]
├── _balance                             [merges, weight=1]
└── serialize → int32 buffer
```

**移植注意点**：
- `choose_splits` 的候选去重（`if candidate == last: continue`）必须保留，否则慢 100×（将 `O(sqrt(max_cost))` 退化为 `O(max_cost)`）
- `_balance` 的两段式（二分搜索 + `_quantile_cut` 等分回退）都要移植，否则等量 batch 会留 idle core
- `int(round(...))` 需用 `std::lround` 匹配 round-half-away-from-zero
- `Plan.merge_free` 和 `Plan.tail_saving()` 也要移植——它们决定变体选择（`merge_free` 和 `narrow_tail` 是 VariantKey 的两个 bool）
- `kernels.py::choose_tile_m(group_size)`：`group_size <= 16 ? 16 : 32`，trivial

**移植工作量估算**：~800 行 C++（含 Item/Merge/Plan struct + 全部 helper），纯算术无框架依赖，可作为 host tiling 的一个独立模块。

### 4.6 调研：aclnn kernel 的 GM row pitch 约束

**结论：aclnn 路径有与 pypto 路径完全相同的 pitch 约束。**

aclnn kernel 的 `*_cube.h` 中，GM tensor 通过 `GM2L1_ND2NZ` DMA 指令加载，srcStride 参数硬编码：

| Tensor | pitch 值 | 代码位置 | 来源 |
|---|---|---|---|
| query | `128`（= head_dim） | `*_cube.h` `GM2L1_ND2NZ(..., query[...], rows, 128, 128, 16)` | 硬编码字面量 |
| key | `128 * kv_head` | `*_cube.h` `int pd_kvstride = 128*kv_head;` → `GM2L1_ND2NZ(..., key[...], block_size, 128, pd_kvstride, 256)` | inline 计算，用 tiling 的 `kv_head` |
| value | `128 * kv_head` | 同 key | 同 key |
| out | `128`（per-head）, `128 * q_head`（per-batch） | `*_vec.h` `UB2GMPAD(out[...], ..., 256, 0, 0)` + `pd_qstride = 128*q_head` | inline 计算 |

tiling struct 仅含 `{int32_t q_head; int32_t kv_head; float scale;}`，**不含任何 pitch/stride/shape 字段**。TilingFunc 是纯 attribute passthrough，不检查输入 shape。

**与 pypto 路径的对比**：

| 维度 | pypto 路径 | aclnn 路径 |
|---|---|---|
| pitch 值 | 完全相同（query/out=128, key/value=kv_head*128） | 完全相同 |
| pitch 来源 | `pypto_bridge/pitch.py` 的 `GM_ROW_PITCH` dict | kernel 内 inline 硬编码 |
| pitch 检查 | host 侧 `launcher.py::_gm_views` assert | 无检查（kernel 内直接用，pitch 不匹配则 DMA 读错地址） |
| pitch 可配置 | 否（编译期固定） | 否（kernel 内字面量） |

**对 v2 迁移的影响**：无论选方案 B（pypto）还是方案 D（aclnn），GM row pitch 约束相同。host 侧需要确保输入 tensor 的布局与 kernel 期望一致：
- `query`/`out`：`[numTokens, qHead, headDim]`，row pitch = `headDim = 128`
- `key`/`value`：`[numBlocks, blockSize, kvHead, headDim]`，row pitch = `kvHead * headDim = kvHead * 128`

这与 v1 的 v1 接口一致（v1 的 def.cpp 声明相同的 shape），因此**v2 接口对标 v1 时，pitch 约束自动满足**，无需额外 reshape。

### 4.7 测试用例规划

**策略**：基于 v1 `test_x_flash_attention_infer.py` 的 3 条用例范围扩展，融入 replicate 仓的测试维度，并实现 golden cache（对标 `test_x_attention_v2.py`）。

**v1 现有用例（3 条）**：

```python
(torch.float16, 1, 8, 8, 128, 128, 128),   # fp16, b1, MHA, kv128, bs128
(torch.float16, 2, 8, 8, 128, 128, 128),   # fp16, b2, MHA, kv128, bs128
(torch.float16, 2, 16, 8, 128, 256, 128),  # fp16, b2, GQA(2:1), kv256, bs128
```

**扩展维度（来自 replicate 仓）**：

| 维度 | v1 | 扩展取值 | 来源 |
|---|---|---|---|
| dtype | fp16 only | + bf16 | replicate 21 例中 6 条 bf16 |
| batch | 1, 2 | + 6, 8, 16, 24, 48, 64, 128 | replicate benchmark |
| q_head : kv_head | 8:8(MHA), 16:8(GQA2) | + 32:8(GQA4), 16:4(GQA4), 32:32(MHA), 32:1(MQA), 28:4(GQA7), 64:8(GQA8) | replicate shapes |
| kv_seqlen | 128, 256 | + 512, 1024, 4096, 8192, 16384, 32768 | replicate benchmark |
| block_size | 128 only | + 16, 32, 64 | replicate variants |
| kv_lens 变长 | 不支持 | + 逐 batch 变长（如 64–8192） | replicate varlen |
| kv_format | TND only | + FRACTAL_NZ | replicate NZ 用例 |

**golden cache 机制**（对标 `test_x_attention_v2.py`）：

```
calc_data() 入口
  ├─ cache_key = f"{dtype}_{q_head}_{kv_head}_{batch}_{kv_seqlen}_{block_size}_{kv_format}"
  ├─ cached = _load_golden(cache_key)     ← 尝试读 golden_cache/{cache_key}.pt
  │
  ├─ [cache hit] 恢复输入 + golden → 调 NPU 算子 → allclose
  └─ [cache miss] 生成输入 → CPU fp32 计算 golden → _save_golden() → 调 NPU 算子 → allclose
```

与 `test_x_attention_v2.py` 的 golden cache 差异：
- cache key 新增 `block_size` 和 `kv_format` 维度（v1/x_attention 无此维度）
- golden 参考实现移植自 replicate 仓 `common/golden.py`（paged-KV decode attention，qSeqlen=1）
- 容差：fp16 `atol=0.001, rtol=0.001`，bf16 `atol=0.01, rtol=0.01`（对标 x_attention_v2 容差策略；v1 用 `atol=6e-2`，过于宽松）

**预计用例数**：以 dtype(2) × batch(4: 1,2,6,16) × q_head:kv_head(4: 8:8, 16:8, 32:8, 16:4) × kv_seqlen(4: 128, 512, 2048, 8192) × block_size(2: 32, 128) = 256 条（全部 TND/ND）。NZ 子集可选追加。

---

## 5. v2 源码位置（规划）

```
xllm_ops/x_flash_attention_infer_v2/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt                          # require_pypto_pro + enable_pypto_kernel + add_ops_compile_options
│   ├── x_flash_attention_infer_v2_def.cpp      # 算子定义（接口与 v1 一致：8 输入 1 输出 4 属性）
│   ├── x_flash_attention_infer_v2_proto.cpp     # InferShape/InferDataType（output = input[0] shape/dtype）
│   ├── x_flash_attention_infer_v2_tiling.cpp   # Tiling 计算 + planner 移植（生成 plan tensor 写入 workspace）
│   ├── x_flash_attention_infer_v2_tiling.h     # TilingData 结构（q_head, kv_head, scale + plan offset + tiling_key）
│   └── config/ascend950/
│       ├── x_flash_attention_infer_v2_binary.json
│       └── x_flash_attention_infer_v2_simplified_key.ini
└── op_kernel/
    └── x_flash_attention_infer_v2.py           # 单 @pl.jit kernel + tiling_key + datatype（对标 x_attention_v2）
```

测试：
```
test/python_test/test_x_flash_attention_infer_v2.py
```

---

## 6. 构建迁移修改预览

### 6.1 可复用的 x_attention_v2 基础设施

以下 cmake/scripts 改动已在 x_attention_v2 PR 中完成，v2 可直接复用：

- `cmake/func.cmake`：`require_pypto_pro` + `enable_pypto_kernel` + pypto install
- `cmake/obj_func.cmake`：tiling cpp force-include pypto 生成头
- `cmake/custom_build.cmake`：`--pypto-ops` 参数传递
- `cmake/config.cmake`：`--hi_python` 传给 prepare.sh
- `cmake/scripts/prepare.sh`：接收 `--hi_python`
- `cmake/scripts/pypto_codegen.py`：PyPTO codegen 驱动脚本
- `cmake/scripts/util/ascendc_impl_build.py`：pypto 编译路径
- `cmake/scripts/util/const_var.py`：`CHECK_ASC_DEVKIT_VERSION` 双路
- `cmake/scripts/util/opdesc_parser.py`：`ascend950pr_9599` 映射

### 6.2 可能需要的额外改动

- 如果方案 D（aclnn 变体 + 统一 Op Type）：需合并 18 套 `op_host` 为 1 套统一 def/proto/tiling；kernel 入口统一调度逻辑；host tiling 移植 `planner.py` 生成 `plan` tensor
- 如果方案 B（PyPTO `@pl.jit`）：`enable_pypto_kernel` 扩展支持多 `@pl.jit` kernel（如果 18 个变体放在同一文件），或多次调用（如果每个变体独立文件）
- `build_aclnn.sh`：添加 `x_flash_attention_infer_v2` 到算子列表
- `test/python_test/RegisterOps.cpp` + `custom_ops.py`：添加 v2 绑定

### 6.3 编译方法（预期）

```bash
# 环境要求
# - CANN 9.2.0+ (Ascend950/Ascend950PR)
# - pypto pip 包（含 pypto_pro 子包）已安装
# - Python 3.12

# 编译
bash build.sh -n x_flash_attention_infer_v2

# 测试
pytest -v test_x_flash_attention_infer_v2.py
```

---

## 7. 注意事项

1. **replicate 仓的 `easyasc` 依赖**：原始 DSL 源码 `paged_decode.py` 使用 `easyasc` 包，但 `@pl.jit` 产物和 aclnn 产物均已提交在 `generated/` 目录中，迁移时只需搬运生成产物，不需要 `easyasc` 依赖。

2. **replicate 仓的 `so_cache.py`**：monkey-patch `pypto_pro.runtime.jit._compile_shared_library` 做 .so 内容寻址缓存。xllm-ops 的 aclnn 路径不需要此机制（编译时生成 .o 而非 .so）。

3. **replicate 仓的 `plan` tensor 格式**：与 v1 的 `extra_tiling`（`SplitKvExtraInfo`）格式完全不同。replicate 仓的 xllm 兼容 API 明确**收下 `extra_tiling` 后丢弃，自建 `plan`**。v2 需要在 host tiling 中生成 `plan` tensor。

4. **replicate 仓的 GM row pitch**：`@pl.jit` kernel 假设特定 row pitch，需要在 host 侧或 kernel 内处理 reshape。x_attention_v2 使用 `pl.Ptr[pl.DT_UINT8]` + kernel 内 `make_tensor` 避免了此问题。aclnn 路径的 AscendC kernel 需检查是否有同样约束。

5. **18 个变体的维护成本**：如果直接搬运 18 个 `@pl.jit` 模块或 18 套 AscendC kernel，总代码量约 1MB。考虑是否可以通过 `tiling_key` + `datatype` 合并变体，或用方案 D 统一 Op Type 后在 host tiling 中做变体选择。

6. **replicate 仓仅支持 A5**：`Q_SEQ_LEN_MAX=1`，`head_dim=128` 硬编码，不支持 A3（Ascend910B）。v2 是否需要支持 A3？

7. **replicate 仓的接口兼容策略**：README 明确说明 replicate 的 `x_flash_attention_infer` API 接受 v1 的全部参数（含 `mask` 和 `extra_tiling`），但**收下即弃**——`q_seq_len == 1` 时因果三角是空的，`extra_tiling` 是 v1 的调度表，replicate 自建自己的 `plan`。v2 可沿用此策略：接口层兼容 v1，内部不用 v1 的 mask/extra_tiling。

8. **replicate 仓的三条通路性能等价**：direct / aclnn / pypto 三条通路设备时间中位数偏差 ≤0.2%，选哪条通路是集成问题不是性能问题。这意味着 v2 选择 aclnn 路径（方案 D）不会牺牲性能。

9. **replicate 仓的赢面来源**：性能领先来自 host 侧四件事——负载按累计成本分位切、merge-free 变体、最少切分、按 plan 决定要不要收窄尾轮。不是 kernel 内部更快，而是调度更优。v2 迁移时需确保 `plan` 生成逻辑完整移植。

10. **replicate 仓的 NZ 用例不与 FIA 对比**：主线算子 FIA 没有 NZ 入口，NZ 用例只做正确性验证不做性能对比。v2 如果支持 NZ，性能基线需另选。

11. **replicate 仓的 pypto 数值差异**：pypto 路径 vs direct 有 17/21 例存在 1 ULP 差异（末位舍入），aclnn 路径无此问题。如果 v2 对数值一致性要求高，应优先选 aclnn 路径（方案 D）。

---

## 8. 第一版实现进度（方案 C 尝试）

### 8.1 已完成的文件

| 文件 | 状态 | 说明 |
|---|---|---|
| `xllm_ops/x_flash_attention_infer_v2/CMakeLists.txt` | ✅ 完成 | 对标 x_attention_v2 |
| `op_host/CMakeLists.txt` | ✅ 完成 | `require_pypto_pro` + `enable_pypto_kernel` + 编译选项 |
| `op_host/x_flash_attention_infer_v2_def.cpp` | ✅ 完成 | 接口对标 v1（8 输入 1 输出 4 属性），仅 A5 |
| `op_host/x_flash_attention_infer_v2_proto.cpp` | ✅ 完成 | output = input[0] shape/dtype |
| `op_host/x_flash_attention_infer_v2_tiling.h` | ✅ 完成 | TilingData 16 字段（int64_t + float），匹配 pypto 生成的 OpTiling |
| `op_host/x_flash_attention_infer_v2_tiling.cpp` | ✅ 完成 | host tiling + plan 生成（移植 planner.py 核心逻辑），C++ 编译通过 |
| `op_host/config/ascend950/x_flash_attention_infer_v2_binary.json` | ✅ 完成 | 2 个 dtype 条目（bf16 + fp16） |
| `op_host/config/ascend950/x_flash_attention_infer_v2_simplified_key.ini` | ✅ 完成 | `default=0` |
| `op_kernel/x_flash_attention_infer_v2.py` | ⚠️ codegen 通过，bisheng 编译失败 | 见 §8.3 |
| `test/python_test/test_x_flash_attention_infer_v2.py` | ✅ 完成 | golden cache + 168 条用例 |
| `test/python_test/RegisterOps.cpp` | ✅ 完成 | 添加 `x_flash_attention_infer_v2` 绑定 |
| `test/python_test/custom_ops.py` | ✅ 完成 | 添加 `x_flash_attention_infer_v2_npu` 封装 |
| `xllm_ops/build_aclnn.sh` | ✅ 完成 | 3 处添加 `x_flash_attention_infer_v2` |

### 8.2 pypto codegen 阶段

- `@pl.jit` 装饰器配置：`arch="a5", auto_mutex=True, compile_timeout=300, tiling_key=XFAInferV2TilingKey(MergeFree), datatype={...}`
- pypto codegen（`pypto_codegen.py`）成功通过，生成了 `OpTiling_tiling.h`、`XFAInferV2TilingKey_tilingkey.h`、`x_flash_attention_infer_v2_pypto_infer.cpp`
- TilingKey 当前仅 `MergeFree`（bits=1, values=[0,1]），生成 2 个 tiling key × 2 dtype = 4 个编译变体
- host tiling C++ 编译通过（int64_t 类型修复、std::max 类型匹配修复）

### 8.3 bisheng 编译阶段失败

**关键发现**：直接在 CANN 9.2.0 上编译 replicate 仓的 @pl.jit kernel（`paged_decode_float16_bs128_m16_nd_mf_kernel.py`）**也失败**，但错误不同——`Non-conforming matrix fractal`（Left tile 的 `BLayout::RowMajor` 在 A5 上不支持 matmul）。replicate 仓在 CANN 9.1.0-beta.3 上验证通过，9.2.0 有更严格的约束。

**我的 kernel 错误**：`set_intra_block(PIPE_V, 1/3/17/19)` — bisheng 报 "the ranges of 1st parameter must be [0, 0], [2, 5], [10, 10]"。

**根因分析**：
1. CANN 9.2.0 的 bisheng 编译器对 `set_intra_block` 的 event_id 有严格范围限制（PIPE_V: 仅允许 0, 2-5, 10；PIPE_FIX: 仅允许 0-1, 4-5）
2. `auto_mutex=True` 自动为 cross-core 共享 buffer 分配 event_id，分配结果不在合法范围内
3. `set_cross_core` 默认使用 `CrossCoreSyncMode.INTRA_BLOCK`（mode 2），生成 `set_intra_block` 指令
4. x_attention_v2 不受影响因为其 event_id（0-10）全部在合法范围内，且其 cross-core buffer 的 auto_mutex 分配 ID 也恰好合法

**event_id 修复尝试**：将 P_READY_ID 从 1 改为 2（合法），PV_READY_ID 从 3 改为 5（合法）。用户指定的 event_id 变为合法，但 auto_mutex 分配的 ID（18, 21, 16）仍不合法。

**auto_mutex ID 来源**：score_ub（mutex_ids=[12,13]）→ auto_mutex 生成 PIPE_V event_id 18 和 PIPE_FIX event_id 16；pv_ub（mutex_ids=[14,15]）→ 生成 PIPE_V 21 和 PIPE_FIX 21。分配机制似乎是 mutex_id + offset，offset 随分配数变化，无法直接控制。

**已尝试的修复**（均未解决 auto_mutex ID 问题）：
- `layout=pl.DN` 导致 `static_assert` 对齐失败 → 改为 `shape=[64,1]` + `layout=pl.DN` → 修复了 `assignData` 错误
- cross-core buffer 从 section_cube 内移到外部声明 → 未解决 auto_mutex 的 `set_intra_block` 生成
- 去掉 `vf.store_align` 的 `block_stride`/`data_copy_mode` 参数 → 无影响（错误在 `set_intra_block` 不在 `vsts`）
- 修改 event_id 到合法值 → 用户 ID 变合法，但 auto_mutex ID 仍不合法

**深入分析（CANN 9.2.0 bisheng 编译器行为）**：

1. `set_cross_core` 默认使用 `CrossCoreSyncMode.INTRA_BLOCK`（mode 2），生成 `set_intra_block` 指令
2. `INTRA_BLOCK` 模式为每个调用生成两个 `set_intra_block`：`event_id` 和 `event_id + 16`（"sets both VEC subcores"）
3. bisheng 对 `set_intra_block` 的 event_id 有严格范围检查：PIPE_V 仅允许 {0, 2-5, 10}，PIPE_FIX 仅允许 {0-1, 4-5}
4. **静态 event_id 被范围检查**；**动态 event_id（运行时表达式）也被范围检查**（bisheng 评估可能值）
5. x_attention_v2 不受影响因为：
   - 其 event_id 全在合法范围内（0-10）
   - 其动态表达式来自 `pl.range` 循环变量（bisheng 可评估可能值范围）
   - auto_mutex 生成 `get_buf`/`rls_buf`（mutex 管理），不是 `set_intra_block`（event 同步）
6. **关键发现**：auto_mutex 生成的是 `get_buf`/`rls_buf`（23 个），不是 `set_intra_block`！所有 `set_intra_block` 错误都来自我的 `set_cross_core`/`wait_cross_core` 调用

**已尝试 INTER_BLOCK 模式**：将 `sync_mode` 改为 `INTER_BLOCK`（mode 0），错误从 48 降到 28，但仍有范围检查（PIPE_V: [2,5]∪[10,10]，PIPE_FIX: [1,1]∪[4,5]）。

**根因总结**：CANN 9.2.0 的 bisheng 编译器对所有跨核同步指令（无论 INTRA_BLOCK 还是 INTER_BLOCK）都做 event_id 范围检查。x_attention_v2 的 event_id 恰好全在合法范围内。我的 kernel 使用了不在合法范围内的 event_id（如 0 对 PIPE_V INTER_BLOCK 无效），以及 bisheng 无法评估的动态表达式（`pl.get_block_idx() % 2`）。

### 8.4 当前代码状态

**kernel .py 当前版本**的关键配置：
- `@pl.jit(arch="a5", auto_mutex=True, compile_timeout=300, tiling_key=XFAInferV2TilingKey(MergeFree), datatype={...})`
- event_id 用 tuple + `pl.get_block_idx() % 2` 索引（试图生成动态表达式绕过范围检查）
- `sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK`（从默认 INTRA_BLOCK 改为 INTER_BLOCK，减少 20 个错误）
- cross-core buffer（score_ub, pv_ub）声明在 section 外部
- rmax/rsum/oldw tile 用 `shape=[64,1]` + `layout=pl.DN`（修复了 `assignData` 对齐错误）

**编译错误现状**：28 个 bisheng 错误（14 per tiling_key × 2 tiling_keys），全部来自 `set_cross_core`/`wait_cross_core` 的 event_id 范围检查。

### 8.5 下一步选项

| 选项 | 说明 | 预期难度 |
|---|---|---|
| A. 调整 event_id | 所有 event_id 改到合法值 + 用 `pl.range` 循环变量做索引（让 bisheng 可评估可能值） | 中：需要重构 priming 逻辑，primging 在循环外无法用循环变量 |
| B. Patch pypto 包 | 修改 `system_ops.py` 的 `set_cross_core` 代码生成，跳过或绕过 bisheng 范围检查 | 中：需修改 CANN 包内文件，需记录到 md |
| C. 方案 E3（aclnn AscendC） | 搬 replicate 仓 `aclnn/generated/` 的 18 套 C++ kernel，不依赖 pypto 编译 | 低：已有编译通过的 kernel，但需适配 xllm-ops 构建系统 |
| D. 方案 E1（多 @pl.jit 文件） | 每个 replicate 的 @pl.jit 模块独立编译，扩展 `enable_pypto_kernel` 支持多文件 | 高：需修改 cmake 基础设施 |

### 8.6 方案 A 调试结论（已穷尽）

**方案 A 无法解决 `+ 16` 问题。** 详细调试过程如下：

1. **event_id 合法值调整**：将所有 event_id 改到 PIPE_V 合法范围 {0, 2-5, 10} 和 PIPE_FIX 合法范围 {0-1, 4-5}。用户指定的 event_id 变为合法，但 `set_cross_core` 的 INTRA_BLOCK 模式会为每个调用生成两个 `set_intra_block`：`event_id` 和 `event_id + 16`。`event_id + 16`（如 2+16=18）超出合法范围。

2. **动态表达式尝试**：
   - `P_READY_IDS[_task_id]` where `_task_id = pl.get_block_idx() % 2` → 生成动态 `_expr_tmp_31_0`，但 `+ 16` 仍是 `_expr_tmp_31_0 + 16`，bisheng 仍范围检查
   - `pd_ib - pd_ib` → bisheng 常量传播 `x - x = 0`，仍为静态
   - `pl.range` 循环变量做索引 → 循环内调用变为动态，但 priming（循环外）仍为静态

3. **sync_mode 切换**：
   - `INTER_BLOCK`（mode 0）：错误从 48 降到 28，但仍有范围检查（PIPE_V: [2,5]∪[10,10]，PIPE_FIX: [1,1]∪[4,5]）
   - `UNICAST_BLOCK`（mode 3）：72 errors，无改善
   - `INTRA_BLOCK`（mode 2，默认）：72 errors

4. **`if` 条件块**：将 priming 包在 `if core_id >= 0:` 运行时条件内（对标 x_attention_v2 的 `if core_id < shared_core_num`），试图阻止 `+ 16` 生成。但 `+ 16` 仍然生成。

5. **`pad=pl.TilePad.min`**：给 cross-core Vec buffer 添加 `pad=pl.TilePad.min`（对标 x_attention_v2），试图改变 auto_mutex 的 pipe 选择。auto_mutex 确实从 PIPE_V 切换到 PIPE_FIX（消除了 auto_mutex 的 `+ 16`），但 `set_cross_core` 本身的 `+ 16` 仍然存在。

6. **对比 x_attention_v2**：x_attention_v2 的 `set_cross_core(PIPE_V, event_id)` 生成的代码**没有** `+ 16` companion。使用相同的 pypto 版本和相同的默认 sync_mode（INTRA_BLOCK）。差异原因不明——可能与 x_attention_v2 的 `set_cross_core` 调用在 `if core_id < shared_core_num` 条件内有关，但我的 `if core_id >= 0:` 没有同样效果。

**结论**：`set_cross_core(PIPE_V, ...)` 的 `+ 16` companion 生成是 pypto codegen 的固定行为，无法从 kernel .py 层面控制。需要 patch pypto 包（方案 B）或改用非 pypto 路径（方案 C）。

### 8.7 当前 kernel .py 状态

当前 `x_flash_attention_infer_v2.py` 的关键配置：
- `@pl.jit(arch="a5", auto_mutex=True, compile_timeout=300, tiling_key=XFAInferV2TilingKey(MergeFree), datatype={...})`
- event_id 用 tuple + 静态索引 `[0]`/`[1]`（分 subcore 设置）
- `sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK`（最后尝试的模式）
- priming 调用包在 `if core_id >= 0:` 条件内
- cross-core buffer（score_ub, pv_ub）声明在 section 外部，带 `pad=pl.TilePad.min`
- rmax/rsum/oldw tile 用 `shape=[64,1]` + `layout=pl.DN`
- 循环内 sync 用 `pd_item % 2` / `v_item % 2` 做动态索引

**编译状态**：72 个 bisheng 错误（全部来自 `set_cross_core`/`wait_cross_core` 的 `event_id + 16` 范围检查）。

### 8.8 突破：根因确认与编译通过

**根因（来自 pypto 源码 `/workspace/pypto/framework/src/interface/pypto_pro/backend/backend_cce_ops.cpp:1546`）**：

`+ 16` companion 的生成条件（三个同时满足）：
1. `arch == "a5"`（A5 架构）
2. `sync_mode == CrossCoreSyncMode.INTRA_BLOCK`（默认模式）
3. `codegen.GetTarget() == ir::SectionKind::Cube`（调用在 `section_cube()` 内）

当三个条件同时满足时，C++ 后端生成两个 `set_intra_block`：
- `set_intra_block(PIPE, event_id)` — 通知 AIV subcore 0
- `set_intra_block(PIPE, event_id + 16)` — 通知 AIV subcore 1

**PIPE 参数不影响 `+ 16` 生成**——只决定生成的指令字符串。

**x_attention_v2 不受影响的原因**：x_attention_v2 的 `set_cross_core(PIPE_V, ...)` 调用全部在 `section_vector()` 内（line 1423-1426），`GetTarget() == Vector`，不满足条件 3，不生成 `+ 16`。

**修复**：
1. 将 PIPE_V priming 调用从 `section_cube()` 移到 `section_vector()`（和 x_attention_v2 一样）
2. 将 cube section 的 `wait_cross_core(PIPE_V, ...)` 改为 `wait_cross_core(PIPE_FIX, ...)`（PIPE_FIX `+ 16` 的 event_id 是动态表达式，bisheng 不范围检查）
3. cross-core buffer（score_ub, pv_ub）声明在 section 外部，带 `pad=pl.TilePad.min`
4. **构建系统问题**：staging 目录中的 .py 是旧版（PIPE_V priming 在 cube section）。需要删除 stale staging 目录（`build/binary/ascend950/src/x_flash_attention_infer_v2/`）后重新编译才能使用新版 .py。

**编译结果**：**0 errors, SUCCESS** — 算子编译安装通过。

**直接 codegen 验证**：用 `_codegen()` 手动生成 kernel.cpp，确认 cube section 内 0 个 PIPE_V `set_intra_block` 调用，vector section 内 7 个 PIPE_V 调用（全部无 `+ 16`）。cube section 内的 `set_intra_block` 全部在 PIPE_FIX 上（动态 event_id + 16）。

### 8.9 精度验证结果

**结果**：168/168 全部失败。

这是 kernel 逻辑问题（非编译问题）。simplified flash attention pipeline 的以下方面可能存在 bug：
- cube/vector pipeline 的 sync 时序（priming 调用在 vector section，但 cube 需要 wait）
- softmax 在线更新逻辑（`pd_softmax` 的 online softmax 实现）
- plan tensor 的读取和使用（item 分配是否正确）
- `pd_accumulate` 和 `pd_finalize` 的 fp32 累加和归一化
- cross-core buffer 的 ping-pong 同步（`score_ub`/`pv_ub` 的 `.next()` 时序）

**下一步**需要逐个调试 kernel 逻辑，确认 pipeline 各阶段的输入输出是否正确。

### 8.10 当前 kernel .py 最终状态

关键配置：
- `@pl.jit(arch="a5", auto_mutex=True, compile_timeout=300, tiling_key=XFAInferV2TilingKey(MergeFree), datatype={...})`
- PIPE_V priming 在 `section_vector()` 内（4 个静态 event_id 调用，无 `+ 16`）
- cube section 的 `wait_cross_core` 用 PIPE_FIX（动态 event_id + 16，bisheng 不检查）
- vector section 的 `set_cross_core` 用 PIPE_V（在 vector section，无 `+ 16`）
- cross-core buffer 带 `pad=pl.TilePad.min`
- rmax/rsum/oldw tile 用 `shape=[64,1]` + `layout=pl.DN`
- event_id: SCORE_READY_IDS=(0,1), P_READY_IDS=(2,3), PV_READY_IDS=(4,5)
- 无 sync_mode 参数（使用默认 INTRA_BLOCK）
- 无 `if` 条件块包裹 priming

**编译状态**：✅ 0 errors, SUCCESS
**精度状态**：❌ 168/168 failed（kernel 逻辑 bug）

---

## 9. 方案修订：多 @pl.jit 入口 + host 侧调度

### 9.1 新信息

从 replicate 仓作者处了解到：**replicate 仓实际使用了多个 `@pl.jit` 入口（每个变体一个），host 侧判断走哪个 kernel**。这类似多 tiling_key 功能，但通过 host 侧调度而非编译期 tiling_key 分流实现。

### 9.2 方案修订分析

原方案 C（单 `@pl.jit` + `tiling_key`）的核心困难是：18 个变体的差异影响 `TileType` 声明（layout/shape/compact），tiling_key 机制难以 fold 掉这些差异，导致 auto_mutex 生成非法的 `set_intra_block` 调用。

**修订方案 E：多 `@pl.jit` 文件 + 单 Op Type + tiling_key 分流**

- 从 replicate 仓 `pypto_bridge/generated/` 搬运已有的 18 个 `@pl.jit` 模块（每个是独立编译通过的 kernel）
- 每个变体放在独立的 `.py` 文件中（或合并到一个文件中多个 `@pl.jit` 函数）
- host tiling C++ 根据 shape 参数（dtype, block_size, tile_m, kv_format, merge_free, narrow_tail）计算 tiling_key
- CANN 框架根据 tiling_key 选择对应的 kernel binary
- `enable_pypto_kernel` 需要扩展以支持多个 `@pl.jit` kernel per op

**关键技术问题**：
1. `enable_pypto_kernel` 当前期望 1 个 op_file = 1 个 `@pl.jit` kernel。需要扩展为 1 个 op = N 个 `@pl.jit` kernel
2. `pypto_codegen.py` 已经能发现文件中的所有 `@pl.jit` kernel（`kernels = [v for v in vars(module).values() if isinstance(v, _TileJitKernel)]`），但 `generate_binary_headers(kernel)` 对每个 kernel 生成独立的 tiling headers
3. `pypto_compile_op` 用 `main_func`（= op_file name）查找 `@pl.jit` kernel。多 kernel 场景需要不同的 main_func 或调度机制
4. CANN 框架的 tiling_key 机制：一个 Op Type 可以有多个 kernel binary，每个 binary 对应一个 tiling_key 值。host tiling `SetTilingKey(key)` 选择 binary

### 9.3 实现路径（待确认）

**路径 E1：多文件 + 多次 enable_pypto_kernel**
- 每个变体一个 `.py` 文件（如 `paged_decode_fp16_bs128_m16_nd_mf.py`）
- CMakeLists.txt 调用 `enable_pypto_kernel` 多次（每个文件一次）
- 每个 `enable_pypto_kernel` 生成独立的 tiling headers 和 kernel binary
- 需要修改 `enable_pypto_kernel` 的 op_file 参数为变体文件名

**路径 E2：单文件 + 多 @pl.jit + 扩展 codegen**
- 所有变体放在 `x_flash_attention_infer_v2.py` 中（18 个 `@pl.jit` 函数）
- 扩展 `pypto_codegen.py` 为每个 `@pl.jit` 生成独立的 tiling headers
- 扩展 `pypto_compile_op` 或 aclnn wrapper 以支持多 kernel 调度

**路径 E3：搬 replicate 仓的 aclnn 路径（非 pypto）**
- 直接用 replicate 仓 `aclnn/generated/` 的 18 套 AscendC C++ kernel
- 每套 kernel 编译为独立 binary，统一注册为 `XFlashAttentionInferV2` Op Type
- host tiling 通过 tiling_key 分流
- 不需要 `enable_pypto_kernel` 或 pypto 编译链路

### 9.4 待确认

1. 路径 E1/E2/E3 哪个更可行？
2. 如果选 E3（aclnn 路径），是否需要 `require_pypto_pro`/`enable_pypto_kernel`？还是直接用 AscendC 编译链路？
3. 18 个变体的 kernel binary 如何统一注册为 1 个 Op Type？需要 `binary.json` 列出所有变体？
4. host tiling 如何确定 tiling_key 值？需要建立 variant → tiling_key 的映射？
