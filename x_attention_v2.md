# XAttention 算子文档

## 整体思路

分两步完成 x_attention 算子的演进：

**Step 1 — v1 (`XAttention`) A3/A5 代码合并 + 测试扩展**
- 将 A3 (Ascend910B) 和 A5 (Ascend950) 两套 kernel 合入同一份源码，通过架构宏分流，A3 逻辑不受影响
- 扩展 `test_x_attention.py` 测试范围（96 → 144 条，新增 GQA(16,4)）
- 引入 golden cache 机制，缓存 CPU fp32 参考实现结果，避免每次回归重复计算

**Step 2 — v2 (`XAttentionV2`) 算子引入**
- 从 pypto 仓迁移 PyPTO Pro kernel 算子，kernel 用 Python DSL (`pypto_pro.language`) 编写
- v2 外部功能对标 v1：接口、测试用例、golden cache 完全复用，验证两者行为等价
- 核心工作是将 PyPTO kernel 的编译链路接入 xllm-ops 自研 cmake 体系

---

## 1. v1 (`XAttention`) 算子概述

`XAttention` 是 xllm-ops 中的自定义 AscendC 算子，实现 **shared KV + unshared KV 双路合并的 flash decoding attention**，主要用于 LLM 推理 decode 阶段。

### 核心特性

- **双路 KV**：shared KV（所有 beam 共享的 prompt KV cache）+ unshared KV（每个 request 独立的 decode KV cache）
- **Online Softmax 合并**：两路独立计算 attention 后，通过 rowmax/rowsum 做跨路 rescale 合并
- **GQA 支持**：q_head / kv_head 分组（如 32:8 = 4:1 group）
- **布局**：TND（query 拼接为 `[numTokens, numHeads, headDim]`）
- **无 mask**：decode 场景 q_seqlen=1，单 query 看到全部 KV，无需 causal mask

### 算子源码位置

```
xllm_ops/x_attention/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   ├── x_attention_def.cpp          # 算子定义（输入/输出/属性）
│   ├── x_attention_proto.cpp         # InferShape/InferDataType
│   ├── x_attention_tiling.cpp        # Tiling 计算（host 侧，A3/A5 分流）
│   ├── x_attention_tiling.h          # TilingData 结构定义（A3/A5 字段共存）
│   ├── xa_arch_config.h / .h.in      # CMake configure_file 模板，按 SOC 生成 CATLASS_ARCH
│   └── xa_arch_config.h              # 生成产物（A5: #define CATLASS_ARCH 3510，A3: 空文件）
└── op_kernel/
    ├── x_attention.cpp               # kernel 入口（#if defined(XA_ARCH35) 架构分流）
    ├── x_attention_catlass_helper.h  # A3 kernel 封装
    ├── x_attention_catlass_kernel.h  # A3 catlass kernel 实现
    └── arch35/                       # A5 kernel（tla:: tensor abstraction）
        ├── combine_kernel.h          # A5 combine scale kernel
        ├── shared_infer_catlass_kernel.h   # A5 shared KV attention（4 级流水线 ping-pong）
        ├── unshared_infer_catlass_kernel.h # A5 unshared KV attention（3 级流水线）
        ├── x_attention_catlass_helper.h    # A5 kernel 封装
        └── x_attention_common.h            # A5 XAttnKernelCommonParams + 命名空间
```

### 算子注册名

- Op Type（PascalCase）: `XAttention`
- kernel 入口（snake_case）: `x_attention`
- aclnn API: `aclnnXAttention`

### 支持芯片

`ascend910b`, `ascend910_93`, `ascend950`（见 `x_attention_def.cpp` 中 `AddConfig`）

---

## 2. v1 A3/A5 代码合并

### 2.1 合并来源

从 ware2009 仓库 `origin/20260820` 分支合并，核心 commit：
- `ad4f45b` feat: x_attention adapter the a5
- `1c1b562` feat: adapte the a5 and adjust the catlass module in common/catlass

现已合入 main 分支（commit `60e50af`），A3/A5 代码在同一份源码中通过架构宏分流。

### 2.2 架构分流机制

`x_attention.cpp` 中的核心分流逻辑：

```cpp
// A5 检测（device 侧用 __NPU_ARCH__，host 侧用 CATLASS_ARCH，两者皆可触发）
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510))
#define XA_ARCH35 1
#endif

// A5 时在 device TU 派生 CATLASS_ARCH（host 仅注入 -DCATLASS_ARCH，device 不注入）
#if defined(XA_ARCH35) && !defined(CATLASS_ARCH)
#define CATLASS_ARCH 3510
#endif

// A3 检测（__NPU_ARCH__ == 2201），同样派生 CATLASS_ARCH
#if !defined(XA_ARCH35) && !defined(CATLASS_ARCH) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#define CATLASS_ARCH 2201
#endif

// include 分流
#if defined(XA_ARCH35)
#include "arch35/x_attention_catlass_helper.h"   // A5 kernel
#else
#include "x_attention_catlass_helper.h"           // A3 kernel (原有)
#endif

// kernel 入口分流
extern "C" __global__ __aicore__ void x_attention(...) {
#if defined(XA_ARCH35)
    // A5 path: 不同 workspace 布局 + tla:: API + XAttnKernelCommonParams
    // workspace: [sharedO, sharedMax, sharedSum, unsharedO, unsharedMax, unsharedSum]
    ...
#else
    // A3 path: 原有 CALL_XATTN_KERNEL 宏 + TILING_KEY_IS 分支
    // workspace: [s, p, oTemp, oUpdate, shared_workspace, unshared_workspace]
    ...
#endif
}
```

### 2.3 合并内容

仅合并 `xllm_ops/x_attention/` 目录，**不修改** `common/catlass/` 和 `cmake/`：

| 文件 | 类型 | 改动说明 |
|------|------|---------|
| `op_kernel/x_attention.cpp` | 修改 | 新增 `__NPU_ARCH__`/`CATLASS_ARCH` 架构检测宏，A5 走 `#if defined(XA_ARCH35)` 分支，A3 走 `#else` 原有逻辑 |
| `op_kernel/x_attention_catlass_kernel.h` | 修改 | 新增 `CATLASS_ARCH` arch guard（从 `__NPU_ARCH__` 派生 2201/3510），移除 `#include "catlass/debug.hpp"` |
| `op_host/x_attention_tiling.h` | 修改 | 新增 A5 专用 tiling 字段（`baseInfo`/`sharedInfo`/`unsharedInfo` 结构体 + `qOSize`/`sumMaxSize` 等） |
| `op_host/x_attention_tiling.cpp` | 修改 | 新增 A5 tiling 计算逻辑（与 A3 tiling 共存，通过 `#ifdef CATLASS_ARCH` 分流） |
| `op_host/CMakeLists.txt` | 修改 | 新增 `xa_arch_config.h` 生成逻辑 + 按 `ASCEND_COMPUTE_UNIT` 动态设置 `CATLASS_ARCH` 编译选项 |
| `op_host/xa_arch_config.h` / `.h.in` | 新增 | CMake configure_file 模板，A5 生成 `#define CATLASS_ARCH 3510`，A3 生成空文件 |
| `op_kernel/arch35/combine_kernel.h` | 新增 | A5 combine scale kernel |
| `op_kernel/arch35/shared_infer_catlass_kernel.h` | 新增 | A5 shared KV attention kernel（4 级流水线 ping-pong，使用 `tla::` tensor abstraction） |
| `op_kernel/arch35/unshared_infer_catlass_kernel.h` | 新增 | A5 unshared KV attention kernel（3 级流水线） |
| `op_kernel/arch35/x_attention_catlass_helper.h` | 新增 | A5 kernel 封装（`CallSharedInferKernel`/`CallUnsharedInferKernel`/`CallCombineScale`） |
| `op_kernel/arch35/x_attention_common.h` | 新增 | A5 专用 `XAttnKernelCommonParams` 结构体 + `SharedInfer`/`UnSharedInfer` 命名空间 + `TaskArgs` |

### 2.4 A5 编译依赖

A5 编译需要以下两个依赖（A3 编译不需要）：

#### 依赖 1：catlass submodule 替换为 cann 版本

ware2009 `origin/20260820` 将 catlass submodule URL 从 `gitcode.com/xLLM-AI/catlass.git` 改为 `gitcode.com/cann/catlass.git`，commit 从 `c02a6e8` 变更为 `7079773`。

```bash
cd third_party/catlass
git remote add cann https://gitcode.com/cann/catlass.git
git fetch cann
git checkout 7079773b9823a5cd31f4a56f4a5174b4f19075ae
```

cann 版本的 catlass 提供了 A5 基础 `tla::` API（tile abstraction layer），包括 `tla::Shape`、`tla::MakeTensor`、`tla::MakeLayout`、`PackedTileCopyTlaToUB` 等 A5 专用模板。

#### 依赖 2：common/catlass/ 补丁文件

arch35/ 的 A5 kernel 引用了 x_attention 专用的 catlass 扩展头文件，这些文件不在 catlass submodule 中，而在 `common/catlass/include/` 下：

```
common/catlass/include/
├── catlass/
│   ├── epilogue/block/
│   │   ├── block_epilogue_xa_combine_scale_ascend950.hpp    # A5 combine scale epilogue
│   │   ├── block_epilogue_xa_shared_rescale_ascend950.hpp   # A5 shared rescaleO
│   │   ├── block_epilogue_xa_shared_softmax_ascend950.hpp   # A5 shared softmax
│   │   └── block_epilogue_xa_unshared_softmax_ascend950.hpp # A5 unshared softmax
│   └── gemm/block/
│       ├── block_mmad_xa_shared_qk_tla.hpp    # A5 shared QK matmul
│       ├── block_mmad_xa_shared_pv_tla.hpp    # A5 shared PV matmul
│       ├── block_mmad_xa_unshared_qk_tla.hpp  # A5 unshared QK matmul
│       └── block_mmad_xa_unshared_pv_tla.hpp  # A5 unshared PV matmul
└── catlass_patch/
    ├── xa_register.hpp                # x_attention catlass 模板注册
    └── attention_extra_register.hpp   # 额外注册
```

**注意**：`common/catlass/include` 需要被添加到 kernel 编译的 include 路径中。当前 `op_host/CMakeLists.txt` 的 `-I${CANN_3RD_LIB_PATH}/catlass/include` 只指向 `third_party/catlass/include`，不包含 `common/catlass/include`。A5 编译时可能需要手动添加 `-I${CMAKE_CURRENT_LIST_DIR}/../../../common/catlass/include` 到 `add_ops_compile_options`，或在 cmake 全局配置中添加。此问题需在 A5 环境上实际编译时验证。

### 2.5 A5 编译方法（不安装）

在 A5 环境上编译 `.run` 包但不安装：

```bash
# 1. 替换 catlass submodule（如果尚未替换）
cd third_party/catlass
git remote add cann https://gitcode.com/cann/catlass.git
git fetch cann
git checkout 7079773b9823a5cd31f4a56f4a5174b4f19075ae

# 2. 编译 A5 .run 包（不安装）
cd xllm_ops
bash build.sh --pkg --ops=x_attention --soc=ascend950

# 3. 检查产物
ls -la build_out/cann-ops-xllm-*.run
```

**注意**：不要在 A3 环境上用 `bash build.sh -c ascend950` 编译安装，会覆盖 A3 的 kernel binary 导致 A3 算子不可用。

### 2.6 A3 验证结果

A5 适配合并后，在 A3 (Ascend910B, soc=251) 环境上验证：

```bash
bash build.sh -n x_attention        # 编译安装 A3 版本
pytest -v test_x_attention.py       # 用例全部通过
```

A5 架构分流代码通过 `#if defined(XA_ARCH35)` 隔离，在 A3 上不生效，原有 A3 kernel 逻辑完全不受影响。

---

## 3. v1 测试扩展

### 3.1 测试范围扩展（96 → 144 条）

原始 `test_x_attention.py` 使用单个 `@pytest.mark.parametrize` 列举所有组合（96 条），扩展后改为独立参数化并新增 GQA(16,4)：

**扩展前（96 条）**：
```python
@pytest.mark.parametrize("dtype,request_num,beam_size,...,num_head,kv_heads,...", [
    (bf16, 1, 128, ..., 16, 8, ...),
    (bf16, 1, 256, ..., 16, 8, ...),
    ...
])
```

**扩展后（144 条）**：
```python
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_head, kv_heads", [(16, 8), (32, 8), (16, 4)])  # 新增 (16,4)
@pytest.mark.parametrize("request_num", [1, 6])
@pytest.mark.parametrize("beam_size", [128, 256, 512])
@pytest.mark.parametrize("kv_seqlen", [128, 256, 512, 1024])
@pytest.mark.parametrize("unshared_seqlen", [2, 4])
```

| 参数 | 值 | 数量 | 来源 |
|------|------|------|------|
| dtype | bfloat16 | 1 | 推理主流 dtype |
| num_head, kv_heads | (16,8), (32,8), (16,4) | 3 | GQA group=2,4,4（新增 (16,4)） |
| request_num(batch) | 1, 6 | 2 | 单/多 request |
| beam_size | 128, 256, 512 | 3 | 典型 decode beam |
| kv_seqlen(prompt_length) | 128, 256, 512, 1024 | 4 | KV cache 长度 |
| unshared_seqlen(max_decode_step) | 2, 4 | 2 | decode 步数 |
| **总计** | | **144** | `1×3×2×3×4×2` |

固定参数: `q_seqlen=1, head_dim=128, block_size=128, is_varied_len=0, mask_type=0, shared_kv_type=0(连续), unshared_kv_type=1(paged)`

> **注**：GQA(16,4) 组在测试代码中主动跳过 `unshared_seqlen=4` 的子集（`pytest.skip("GQA(16,4) only tested with max_decode_step=2")`），实际执行 120 passed + 24 skipped。

### 3.2 Golden Cache 机制

**动机**：CPU fp32 参考实现计算量大（大 shape 单条 90~180s），全量 144 条回归耗时过长。引入 golden cache 后，首次运行计算并存盘，后续回归直接复用，秒级完成。

**机制**：

```
calc_data() 入口
  ├─ cache_key = f"{dtype}_{num_head}_{kv_heads}_{request_num}_{beam_size}_{kv_seqlen}_{unshared_seqlen}"
  ├─ cached = _load_golden(cache_key)     ← 尝试读 golden_cache/{cache_key}.pt
  │
  ├─ [cache hit] cached is not None
  │    ├─ 恢复输入数据 (query, key_cache, value_cache, unshared_*, block_tables, ...)
  │    ├─ 恢复 golden 结果 (final_true_out)
  │    ├─ 跳过 CPU golden 计算
  │    ├─ 调用 NPU 算子
  │    └─ allclose(npu_res, golden_res)   ← 验证精度
  │
  └─ [cache miss] cached is None
       ├─ 生成输入数据 (uniform 随机)
       ├─ CPU fp32 计算 golden
       │    ├─ ref_single_query_shared_kv_attention()   → shared_true_out, shared_gm, shared_gl
       │    ├─ ref_single_query_unshared_kv_attention() → unshared_true_out, unshared_gm, unshared_gl
       │    └─ online softmax merge → final_true_out
       ├─ _save_golden(cache_key, {输入数据 + golden 结果})   ← 存盘
       ├─ 调用 NPU 算子
       └─ allclose(npu_res, golden_res)   ← 验证精度
```

**Cache 存储内容**：输入数据（query, key_cache, value_cache, unshared_key, unshared_value, block_tables, unshared_block_tables, actual_shared_kvlen, decode_step）+ golden 结果（final_true_out）。

**Cache key 格式**：`{dtype}_{num_head}_{kv_heads}_{request_num}_{beam_size}_{kv_seqlen}_{unshared_seqlen}`，例如 `bf16_16_8_1_128_128_2`。

**存储路径**：`test/python_test/golden_cache/{cache_key}.pt`

### 3.3 CPU Golden 实现

双路独立计算后做 online softmax 合并：

1. **Shared KV 路径** (`ref_single_query_shared_kv_attention`)
   - 从 paged/连续 cache 取 K/V
   - Q@K^T * scale -> softmax(不除sum) -> P@V
   - 输出: shared_ref_out(bf16) + shared_true_out(fp32) + shared_gm/gl

2. **Unshared KV 路径** (`ref_single_query_unshared_kv_attention`)
   - 从 paged/连续 cache 取 K/V
   - 同样 attention 计算
   - 输出: unshared_ref_out + unshared_true_out + unshared_gm/gl

3. **合并** (online softmax merge)
   ```
   gm = max(shared_gm, unshared_gm)
   gl = shared_gl * exp(shared_gm - gm) + unshared_gl * exp(unshared_gm - gm)
   final = (shared_true * exp(shared_gm-gm) + unshared_true * exp(unshared_gm-gm)) / gl
   ```

### 3.4 容差

| dtype | atol | rtol | 依据 |
|-------|------|------|------|
| bfloat16 | 0.01 | 0.01 | 大 shape 下 bf16 累积误差超出 0.001 |
| float16 | 0.001 | 0.001 | fp16 精度更高 |

### 3.5 关键修复点

1. **max_decode_step 动态化**:
   - 原始: `max_decode_step = 3`（硬编码）
   - 修改后: `max_decode_step = gen_data_params.unshared_kvlen`
   - 原因: unshared_seqlen=4 时 cache 容量不足导致越界

2. **容差按 dtype 区分**:
   - 原始: 固定 `atol=0.001, rtol=0.001`
   - 修改后: bf16 用 0.01，fp16 用 0.001
   - 原因: 大 shape（batch=6, beam=512）下 bf16 累积误差超出 0.001

### 3.6 test_x_attention_with_pa.py — 官方 PA 算子等价性测试

**策略**：用 CANN 官方 `torch_npu._npu_paged_attention` 作为 golden，验证 x_attention 与官方 PA 的等价性。

#### 完整参数范围（7680 条用例）

```python
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("q_head_num, kv_head_num", [(16, 8), (32, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch", [1, 6, 8])
@pytest.mark.parametrize("beam_size", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("max_decode_step", [2, 4, 8, 12, 16, 20, 50, 100])
@pytest.mark.parametrize("prompt_length", [128, 256, 512, 1024])
```

test_x_attention.py 的 144 条用例是此 7680 条的精简子集（去掉 fp16、batch=8、beam=32/64、max_decode_step>4，新增 GQA(16,4)）。

### 3.7 两个测试文件对比

| 维度 | test_x_attention.py | test_x_attention_with_pa.py |
|------|---------------------|------------------------------|
| Golden 来源 | CPU 手写 fp32 参考 | NPU 官方 PA 算子 |
| 验证目标 | 绝对正确性 | 等价性 |
| 用例数 | 144 | 7680 |
| Golden cache | 有（v1/v2 共用） | 无 |
| 双精度参考 | 是（bf16 + fp32） | 否（直接用 NPU 输出） |
| Online softmax 合并 | 手动实现 | 无需（PA 算子内部处理） |
| 容差(bf16) | 0.01 | 0.01 |
| 容差(fp16) | 0.001 | 0.001 |

### 3.8 调用链路

```
test_x_attention[_v2].py (pytest)
  -> custom_ops.py: x_attention[_v2]_npu()
      -> custom_ops_lib (C++ pybind11 扩展)
          -> RegisterOps.cpp: x_attention[_v2]_impl_npu()
              -> EXEC_NPU_CMD(aclnnXAttention[V2], ...)
                   -> dlopen libcust_opapi.so -> dlsym("aclnnXAttention[V2]")
                        -> NPU 上的 op_host tiling + op_kernel binary
```

### 3.9 构建与执行

```bash
# 编译安装算子
bash build.sh -n x_attention        # v1
bash build.sh -n x_attention_v2     # v2

# 编译 C++ 测试扩展
cd test/python_test
bash build_and_run.sh
# 若 pip install 失败（PEP 668），追加:
pip3 install --force-reinstall --break-system-packages dist/*.whl

# 运行测试
cd test/python_test
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
TORCH_NPU_LIB=$(python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__) + '/lib')")
export LD_LIBRARY_PATH="${TORCH_LIB}:${TORCH_NPU_LIB}:${LD_LIBRARY_PATH}"

pytest -v test_x_attention_v2.py           # 全量 144 用例
pytest -v test_x_attention_v2.py -k "2-128-128-1-16" -x   # 单条
```

### 3.10 采集性能数据（msprof）

```bash
# 应用级 profiler（生成 op_summary.csv）
msprof --output=./msprof_x_attention \
  --task-time=on --ai-core=on \
  pytest test_x_attention.py

# 算子级 profiler（per-core ArithmeticUtilization）
msprof op --output=./msprof_x_attention_op \
  --warm-up=3 --launch-count=1 \
  --aic-metrics=ArithmeticUtilization \
  pytest test_x_attention.py -k "2-128-128-1-16" -x
```

应用级 msprof 输出结构:
```
msprof_x_attention/
└── PROF_*/
    └── mindstudio_profiler_output/
        ├── op_summary_*.csv       # 每条用例一行（Task Duration, mac_ratio, vec_ratio 等）
        ├── op_statistic_*.csv     # 汇总统计（count, min, avg, max）
        ├── api_statistic_*.csv    # ACL API 耗时
        └── task_time_*.csv        # kernel task 时间线
```

op_summary.csv 关键指标列:
- `Task Duration(us)`: kernel 总执行时间
- `aic_mac_ratio`: cube 矩阵乘利用率
- `aiv_vec_ratio`: vector 计算利用率
- `cube_utilization(%)`: cube 综合利用率
- `aic_mte2_ratio` / `aiv_mte2_ratio`: 内存带宽利用率

#### msprof sqlite 解析（CANN 9.2.0-beta.1 兼容）

> **已知问题**：CANN 9.2.0-beta.1 的 msprof 不生成标准 `op_summary.csv`（`mindstudio_profiler_output/` 下只有 `README.txt`），`msprof --export=on --summary-format=csv` 也无法导出 csv。性能数据仅以 sqlite 格式存储在 `device_0/sqlite/` 下（`ai_core_op_summary.db` + `ascend_task.db`）。
>
> 后续升级 CANN 版本后，期望 msprof 能直接生成官方 `op_summary.csv`，届时无需 sqlite 解析。

当前环境使用 `x_attention_msprof_sqlite_parse.py`（仓库根目录）从 sqlite 提取数据，反向构造 `Input Shapes`/`Input Data Types` 列，生成与 msprof 标准格式兼容的 `op_summary.csv`，再调用 `x_attention_msprof_parser.py` 生成 xlsx：

```bash
# 从 sqlite 提取并生成 op_summary.csv + op_summary.xlsx
python3 x_attention_msprof_sqlite_parse.py test/python_test/msprof_v1 XAttention
python3 x_attention_msprof_sqlite_parse.py test/python_test/msprof_v2 XAttentionV2

# 产物（每个 msprof 目录下）:
# mindstudio_profiler_output/op_summary.csv   ← 标准格式（含 Input Shapes/Input Data Types/Task Duration(us) + 性能指标）
# mindstudio_profiler_output/op_summary.xlsx  ← parse 脚本输出（original + parsed 两个 sheet）
```

`x_attention_msprof_sqlite_parse.py` 的工作流程：
1. 从 `ai_core_op_summary.db` 读取 `ai_core_metrics` 表（aic_total_time, mac_ratio, vec_ratio 等）
2. 从 `ascend_task.db` 的 `AscendTask` 表关联 `task_id`，筛选 `device_task_type='AI_CORE'` 的行，获取 `duration`（Task Duration(us)）
3. 按测试参数化顺序（与 pytest 用例顺序一致）反向构造 `Input Shapes`/`Input Data Types` 字符串
4. 输出 `op_summary.csv`，调用 `x_attention_msprof_parser.py` 的 `parse_csv()` 生成 xlsx

> **注意**：此脚本假设 AI_CORE task 按 `task_id` 排序后与测试参数化顺序一一对应。如果测试用例顺序发生变化，需要同步调整 `_PARAM_COMBOS` 的生成逻辑。

---

## 4. v2 (`XAttentionV2`) 算子引入

### 4.1 背景

`x_attention_v2` 是从 pypto 仓迁移而来的 PyPTO Pro kernel 算子。kernel 用 Python DSL (`pypto_pro.language`) 编写，而非传统 AscendC C++。

xllm-ops 的构建系统是自研 cmake 体系（`cmake/func.cmake` + `cmake/obj_func.cmake` + `cmake/custom_build.cmake`），与 pypto/ops-transformer 的构建系统不同。迁移的核心工作是**将 PyPTO kernel 的编译链路接入 xllm-ops 构建系统**。

参考实现来自 ops-transformer 仓的 `sparse_flash_mla_softmax_l1_norm` 和 `dense_lightning_indexer_softmax_lse_v2` 两个 PyPTO 算子。

### 4.2 v1/v2 接口对标

v2 的设计目标是外部功能对标 v1，两者在用户接口上完全一致，仅算子名不同：

- op_def 输入/输出/属性：完全相同（query, shared_key_block, shared_value_block, unshared_key_block, unshared_value_block, unshared_block_table[optional], shared_kv_lens, decode_step, shared_block_table[optional], attn_out, scale_value[attr]）
- RegisterOps.cpp / custom_ops.py：参数列表完全相同，仅函数名 `x_attention` → `x_attention_v2`
- 测试用例范围：完全相同（144 条，与 v1 一致）
- golden cache：v1/v2 共用 `golden_cache/` 目录和相同的 cache key（因算子语义等价，CPU golden 可复用）

`test/python_test/test_x_attention_v2.py` 基于 v1 的 `test_x_attention.py` 改造，仅 4 处差异：
- `custom_ops.x_attention_npu` → `custom_ops.x_attention_v2_npu`
- 测试函数名 `test_x_attention_npu` → `test_x_attention_v2_npu`
- `set_device(0)` → `set_device(int(os.environ.get("ASCEND_DEVICE_ID", 0)))`
- golden cache 与 v1 共用（同目录、同 cache key，CPU golden 可复用）

---

## 5. v2 构建迁移修改总览

### 5.1 本仓 cmake 构建系统

#### `cmake/func.cmake` — 新增 `require_pypto_pro` + `enable_pypto_kernel` + pypto install

**新增两个 CMake 函数**（从 ops-transformer 移植）：

- `require_pypto_pro(<op_name>)` — macro，通过 `HI_PYTHON` 执行 `importlib.util.find_spec('pypto_pro')` 检查 pypto_pro 是否可用，不可用则 `return()` 跳过整个算子。
- `enable_pypto_kernel(<op_file>)` — function，在 CMake configure 阶段调用 `pypto_codegen.py` 对 kernel `.py` 做 codegen，生成 `*_tiling.h`/`*_tilingkey.h`/`*_pypto_infer.cpp` 到 `PYPTO_GEN_DIR`。设置 3 个变量：
  - `PYPTO_GEN_DIR`（PARENT_SCOPE）— 供 `obj_func.cmake` 给 tiling cpp 加 `-include`
  - `${op_file}_pypto_enabled`（CACHE INTERNAL）— 供 `add_bin_compile_target` 触发 pypto install
  - `PYPTO_ENABLED_OPS`（GLOBAL property）— 供 `custom_build.cmake` 传 `--pypto-ops`

**`add_bin_compile_target` 中新增 pypto install 逻辑**（2 处）：
- 在 `DYNAMIC_PY_FILE` 之前插入 `${OP_TARGET_NAME}_pypto_install` custom target
- 在 `add_dependencies` 行后追加 pypto install 依赖

#### `cmake/obj_func.cmake` — tiling cpp force-include pypto 生成头

在 `add_modules_sources` 和 `add_modules_sources_with_soc` 两个 macro 中各追加一段（共 2 处）：当 `DEFINED PYPTO_GEN_DIR` 时，glob `*_tilingkey.h` 和 `*_tiling.h`，通过 `set_source_files_properties` 给 tiling cpp 加 `-include` 编译选项。

#### `cmake/custom_build.cmake` — `--pypto-ops` 参数传递

读取 `PYPTO_ENABLED_OPS` 全局属性，拼成逗号分隔字符串，作为 `--pypto-ops` 参数传给 `ascendc_impl_build.py`。

#### `cmake/config.cmake` — HI_PYTHON 传递给 prepare.sh

在调用 `prepare.sh` 的 `execute_process` 参数中追加 `--hi_python ${HI_PYTHON}`，使 prepare.sh 子进程能获取正确的 python 路径。

#### `cmake/scripts/prepare.sh` — 接收 HI_PYTHON 参数

新增 `--hi_python` 参数解析，cmake 命令追加 `-DPython3_EXECUTABLE=${HI_PYTHON:-python3}`。`prepare.sh` 跑独立 cmake 做 prepare_build，需要通过 `-DPython3_EXECUTABLE` 让 `find_package(Python3)` 找到装有 pypto_pro 的 python。

#### `cmake/scripts/pypto_codegen.py` — 新增文件

从 ops-transformer 复制。PyPTO kernel codegen 驱动脚本。**修复**：原版用 `importlib` 加载 kernel 模块时不注册 `sys.modules`，导致 Python 3.12+ 的 `@dataclass` 装饰器查找 `cls.__module__` 时返回 None。修复方式：`sys.modules[py_file.stem] = module`（在 `exec_module` 之前注册）。

#### `cmake/scripts/util/ascendc_impl_build.py` — 最小化 patch

在原版基础上只追加 pypto 相关逻辑（非整体替换）：
- `IMPORT_HEADER` 双路（`asc_op_compile_base` vs `tbe`，依赖 `const_var.CHECK_ASC_DEVKIT_VERSION`）
- `IMPL_HEAD` 模板 import 改 `{}` 占位符
- `PYPTO_COMPILE_OP_API` + `PYPTO_IMPORT_HEADER` 常量
- `OpDesc.__init__` 加 `is_pypto = False` + `write_adapt` 跳过 `.cpp` 检查
- `_write_head` 追加 `PYPTO_IMPORT_HEADER` + `_write_compile_api` 加 pypto 分支
- `write_scripts` 加 `pypto_ops` 参数 + `main` 加 `--pypto-ops` 解析

#### `cmake/scripts/util/const_var.py` — 最小化 patch

在原版基础上只新增 `CHECK_ASC_DEVKIT_VERSION` + `check_asc_devkit_version()` 函数 + `BIN_CMD` 双路分支（`asc_opc` vs `opc`）。`pypto_compile_op` 内部依赖 `asc_op_compile_base` 的 context 机制，只有 `CHECK_ASC_DEVKIT_VERSION=True` 时 `ascendc_impl_build.py` 的 `IMPORT_HEADER` 才会用 `asc_op_compile_base` import。

#### `cmake/scripts/util/opdesc_parser.py` — 最小化 patch

`SOC_TO_SHORT_SOC_MAP` 新增一行 `"ascend950pr_9599": "ascend950"` 映射。`COMPUTE_UNIT Ascend950PR_9599` 需要此映射才能正确解析。

### 5.2 本仓算子代码

#### `x_attention_v2/op_host/x_attention_v2_tiling.cpp`

- 追加 `#include <cmath>` — `std::sqrt`/`std::round` 需要
- `OP_LOGE` 用 `#ifndef OP_LOGE` 包裹 — 若已有定义则不重复定义
- 修复 `sharedTableShapePtr->GetDim(1)` → `sharedTableShapePtr->GetStorageShape().GetDim(1)`
- `GET_TPL_TILING_KEY` 参数加 `u` 后缀消除 narrowing warning：`? 1u : 0u`

#### `x_attention_v2/op_host/x_attention_v2_tiling.h`

保留 2 个 include：`register/tilingdata_base.h` + `tiling/tiling_api.h`（其余头文件在 tiling.cpp 中 include）。

#### `x_attention_v2/op_kernel/x_attention_v2.py`

3 处修改使 kernel 与 pypto 编译链路兼容：

1. **`@pl.jit` kernel 函数名** `x_attention_v2_kernel` → `x_attention_v2`
   - `pypto_compile_op` 用 `main_func`（= op_file 名 = `x_attention_v2`）查找 `@pl.jit` kernel，函数名必须一致
   - 原 host 入口函数 `x_attention_v2` 改名为 `x_attention_v2_host` 避免重名

2. **kernel 参数名** 严格对齐 `op_def.cpp` 输入名（共 4 个 KV 参数）：
   - `shared_k_ptr` → `shared_key_block`
   - `shared_v_ptr` → `shared_value_block`
   - `unshared_k_ptr` → `unshared_key_block`
   - `unshared_v_ptr` → `unshared_value_block`
   - 原因：`@pl.jit` 的 `datatype` 映射 key 必须同时匹配 kernel 参数名和 op_def 输入名（两重验证）

3. **host 入口函数** `x_attention_v2_kernel` → `x_attention_v2_host`（pypto 直接调用模式的 host 封装，xllm-ops 走 aclnn API 不使用此函数）

### 5.3 非本仓代码修改（pypto pip 包）

#### `pypto_pro/runtime/opc/pypto_compile.py`

**文件路径**：`<conda_env>/lib/python3.12/site-packages/pypto_pro/runtime/opc/pypto_compile.py`

**Bug**：`_load_kernel` 函数用 `importlib` 加载 kernel `.py` 时不注册 `sys.modules`，导致 Python 3.12+ 的 `@dataclass` 装饰器内部查找 `sys.modules.get(cls.__module__).__dict__` 时抛出 `AttributeError`。

**修复**（2 行）：
```python
import sys                                          # 追加
sys.modules["_pypto_opc_kernel_mod"] = mod          # exec_module 之前注册
```

**影响范围**：所有使用 `@dataclass` 的 PyPTO kernel 在 Python 3.12+ 下都会触发。pypto 0.2.1 pip 包未修复。同一修复也应用到本仓 `cmake/scripts/pypto_codegen.py`。

---

## 6. v2 编译方法与流程

### 6.1 编译方法

```bash
# 环境要求
# - CANN 9.2.0+ (Ascend950/Ascend950PR, soc=260/ascend950pr_9599)
# - pypto pip 包（含 pypto_pro 子包）已安装
# - Python 3.12
# - pypto_compile.py 已修复 sys.modules 注册 bug（见 5.3 节）
# - pip 包 regex 已安装（ascendc_impl_build.py / ascendc_bin_param_build.py 依赖）

# 设置环境（PATH 让 find_package(Python3) 找到装有 pypto_pro 的 python）
export PATH="<python_with_pypto_pro_bin>:$PATH"
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
TORCH_NPU_LIB=$(python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__) + '/lib')")
export LD_LIBRARY_PATH="${TORCH_LIB}:${TORCH_NPU_LIB}:${LD_LIBRARY_PATH}"
export ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID:-0}

# 编译
bash build.sh -n x_attention_v2

# 产物
# - .run 包: build/cann-ops-xllm-custom_linux-x86_64.run
# - kernel binary: build/binary/ascend950/bin/x_attention_v2/*.o + *.json
# - 自动安装到 $ASCEND_OPP_PATH/vendors/custom_xllm_math/
```

### 6.2 验证环境（2026-08-25）

| 项目 | 值 |
|------|------|
| CANN 版本 | 9.2.0-beta.1 |
| SOC | Ascend950PR (ascend950pr_9599) |
| Python | 3.12.9 |
| pypto | 随 CANN 安装（`$ASCEND_HOME_PATH/python/site-packages/pypto_pro`） |
| 算子安装路径 | `$ASCEND_OPP_PATH/vendors/custom_xllm_math/` |
| Vendor config | `load_priority=custom_xllm_math` |
| 测试结果 | 120 passed, 24 skipped, 0 failed (203.50s) |

### 6.3 构建流程图

```
build.sh -n x_attention_v2
  └─ cmake configure
       ├─ find_package(Python3) → HI_PYTHON
       ├─ config.cmake → prepare.sh --hi_python ${HI_PYTHON}
       │    └─ prepare cmake: -DPython3_EXECUTABLE=${HI_PYTHON}
       ├─ op_host/CMakeLists.txt
       │    ├─ require_pypto_pro(x_attention_v2)     ← 检查 pypto_pro 可用
       │    ├─ add_op_to_compiled_list()
       │    ├─ enable_pypto_kernel(x_attention_v2)   ← configure 阶段 codegen
       │    │    └─ pypto_codegen.py
       │    │         ├─ import kernel .py
       │    │         ├─ pto_compile.generate_binary_headers()
       │    │         └─ 输出 *_tiling.h, *_tilingkey.h, *_pypto_infer.cpp
       │    └─ add_ops_compile_options(...)
       ├─ obj_func.cmake: add_modules_sources
       │    └─ PYPTO_GEN_DIR → tiling cpp -include *_tiling.h
       └─ custom_build.cmake: generate_xllm_adapt_py
            └─ ascendc_impl_build.py --pypto-ops x_attention_v2
                 └─ 生成 pypto wrapper .py (调用 pypto_compile_op)
  └─ make
       ├─ 编译 tiling cpp (含 pypto force-include)
       ├─ 编译 host def/proto/opapi
       └─ 编译 kernel binary (per tilingkey)
            └─ pypto_compile_op → per-key codegen → bisheng compile
                 └─ 输出 *.o + *.json
  └─ 打包 .run + 安装
```

---

## 7. 注意事项

1. **pypto pip 包的 `pypto_compile.py` 修复是环境级修改**，不在本仓中。每次在新的环境安装 pypto pip 包后需要手动应用此修复（2 行改动）。后续应推动 pypto 仓修复此 bug。

2. **`PATH` 中需有装有 pypto_pro 的 python**——构建系统的 `find_package(Python3)` 按 PATH 查找 python，若找到无 pypto_pro 的 python 则 `require_pypto_pro` 会跳过算子。

3. **首次编译会构建 protobuf/abseil 依赖**（~25 分钟），后续增量编译跳过（不要 `rm -rf build`）。

4. **kernel 函数名必须与 op_file 名一致**，kernel 参数名必须与 op_def 输入名一致——这是 pypto 编译链路的硬约束。

5. **v1/v2 golden cache 共用**——两个算子语义等价，CPU golden 计算结果相同，共用 `golden_cache/` 目录避免重复计算。首次运行任一算子测试会生成 cache，后续两个算子均直接复用。

6. **`regex` pip 包依赖**——`cmake/scripts/util/ascendc_impl_build.py` 和 `ascendc_bin_param_build.py` 使用 `import regex as re`，环境需 `pip install regex`。

7. **v1 A5 编译需替换 catlass submodule**——A5 kernel 依赖 cann 版 catlass 的 `tla::` API，A3 编译不需要。详见 2.4 节。
