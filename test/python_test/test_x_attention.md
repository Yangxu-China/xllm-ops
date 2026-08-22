# XAttention 算子测试上下文

## 1. 算子概述

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
│   ├── x_attention_tiling.cpp        # Tiling 计算（host 侧）
│   └── x_attention_tiling.h          # TilingData 结构定义
└── op_kernel/
    ├── x_attention.cpp               # kernel 入口（extern "C" void x_attention）
    ├── x_attention_catlass_helper.h  # catlass kernel 封装
    └── x_attention_catlass_kernel.h  # catlass kernel 实现（SharedFAInferKernel/UnsharedFAInferKernel/CombineScale）
```

### 算子注册名

- Op Type（PascalCase）: `XAttention`
- kernel 入口（snake_case）: `x_attention`
- aclnn API: `aclnnXAttention`

### 支持芯片

`ascend910b`, `ascend910_93`, `ascend950`（见 `x_attention_def.cpp` 中 `AddConfig`）

## 2. 测试文件

### 2.1 test_x_attention.py — CPU golden 正确性测试

**策略**：手写 CPU fp32 参考实现，验证 NPU kernel 输出的绝对正确性。

#### 调用链路

```
test_x_attention.py (pytest)
  -> custom_ops.py: x_attention_npu()
      -> custom_ops_lib (C++ pybind11 扩展)
          -> RegisterOps.cpp:64 x_attention_impl_npu()
              -> EXEC_NPU_CMD(aclnnXAttention, ...)
                   -> dlopen libcust_opapi.so -> dlsym("aclnnXAttention")
                        -> NPU 上的 op_host tiling + op_kernel binary
```

#### Golden 实现

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

#### 当前参数化范围（96 条用例）

```python
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_head, kv_heads", [(16, 8), (32, 8)])
@pytest.mark.parametrize("request_num", [1, 6])
@pytest.mark.parametrize("beam_size", [128, 256, 512])
@pytest.mark.parametrize("kv_seqlen", [128, 256, 512, 1024])
@pytest.mark.parametrize("unshared_seqlen", [2, 4])
```

| 参数 | 值 | 数量 | 来源 |
|------|------|------|------|
| dtype | bfloat16 | 1 | 推理主流 dtype |
| num_head | 16, 32 | 2 | GQA group=2,4 |
| kv_heads | 8 | 1 | 固定 |
| request_num(batch) | 1, 6 | 2 | 单/多 request |
| beam_size | 128, 256, 512 | 3 | 典型 decode beam |
| kv_seqlen(prompt_length) | 128, 256, 512, 1024 | 4 | KV cache 长度 |
| unshared_seqlen(max_decode_step) | 2, 4 | 2 | decode 步数 |
| **总计** | | **96** | |

固定参数: `q_seqlen=1, head_dim=128, block_size=128, is_varied_len=0, mask_type=0, shared_kv_type=0(连续), unshared_kv_type=1(paged)`

#### 容差

| dtype | atol | rtol | 依据 |
|-------|------|------|------|
| bfloat16 | 0.01 | 0.01 | 与 test_x_attention_with_pa.py 一致 |
| float16 | 0.001 | 0.001 | 与 test_x_attention_with_pa.py 一致 |

#### 关键修复点

1. **max_decode_step 动态化**（line 367）:
   - 原始: `max_decode_step = 3`（硬编码）
   - 修改后: `max_decode_step = gen_data_params.unshared_kvlen`
   - 原因: unshared_seqlen=4 时 cache 容量不足导致越界

2. **容差按 dtype 区分**（line 494-497）:
   - 原始: 固定 `atol=0.001, rtol=0.001`
   - 修改后: bf16 用 0.01，fp16 用 0.001
   - 原因: 大 shape（batch=6, beam=512）下 bf16 累积误差超出 0.001

### 2.2 test_x_attention_with_pa.py — 官方 PA 算子等价性测试

**策略**：用 CANN 官方 `torch_npu._npu_paged_attention` 作为 golden，验证 x_attention 与官方 PA 的等价性。

#### 核心逻辑

1. 生成 shared KV + unshared KV 数据，调用 `x_attention_npu()` 得到输出
2. 将双路 KV 数据重组为 PA 算子所需的单路 paged 格式（`pa_key_block`/`pa_block_table`/`context_lens`）
3. 调用 `torch_npu._npu_paged_attention()` 得到 golden
4. 对比两个 NPU 输出

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

test_x_attention.py 的 96 条用例是此 7680 条的精简子集（去掉 fp16、batch=8、beam=32/64、max_decode_step>4）。

#### 数据范围差异

| 数据 | test_x_attention.py | test_x_attention_with_pa.py |
|------|---------------------|------------------------------|
| query | uniform(-1, 1) | uniform(-50, 50) |
| key | uniform(-1, 1) | uniform(-500, 500) |
| value | uniform(-1, 1) | uniform(-3, 3) |

## 3. 两个测试文件对比

| 维度 | test_x_attention.py | test_x_attention_with_pa.py |
|------|---------------------|------------------------------|
| Golden 来源 | CPU 手写 fp32 参考 | NPU 官方 PA 算子 |
| 验证目标 | 绝对正确性 | 等价性 |
| 用例数 | 96 | 7680 |
| 代码行数 | 539 | 150 |
| Golden 代码量 | ~400 行（4 个参考函数） | ~60 行（数据重组 + 1 行 PA 调用） |
| 双精度参考 | 是（bf16 + fp32） | 否（直接用 NPU 输出） |
| Online softmax 合并 | 手动实现 | 无需（PA 算子内部处理） |
| 容差(bf16) | 0.01 | 0.01 |
| 容差(fp16) | 0.001 | 0.001 |

## 4. 构建与执行

### 4.1 编译安装算子

```bash
# 在仓库根目录
bash build.sh -n x_attention
# 自动编译 + 安装到 $ASCEND_OPP_PATH/vendors/custom_xllm_math/
```

### 4.2 编译 C++ 测试扩展

```bash
cd test/python_test
bash build_and_run.sh
# 若 pip install 失败（PEP 668），追加:
pip3 install --force-reinstall --break-system-packages dist/*.whl
```

### 4.3 运行测试

```bash
cd test/python_test

# 设置 LD_LIBRARY_PATH
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
TORCH_NPU_LIB=$(python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__) + '/lib')")
export LD_LIBRARY_PATH="${TORCH_LIB}:${TORCH_NPU_LIB}:${LD_LIBRARY_PATH}"

# 全量运行（96 用例）
pytest -v test_x_attention.py

# 单个用例
pytest -v test_x_attention.py -k "2-128-128-1-16" -x
```

### 4.4 采集性能数据（msprof）

```bash
# 应用级 profiler（生成 op_summary.csv）
msprof --output=./msprof_x_attention_96 \
  --task-time=on --ai-core=on \
  pytest test_x_attention.py

# 算子级 profiler（per-core ArithmeticUtilization）
msprof op --output=./msprof_x_attention_op \
  --warm-up=3 --launch-count=1 \
  --aic-metrics=ArithmeticUtilization \
  pytest test_x_attention.py -k "2-128-128-1-16" -x
```

### 4.5 性能结果文件

应用级 msprof 输出结构:
```
msprof_x_attention_96/
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

## 5. 环境信息

| 项目 | 值 |
|------|------|
| CANN 版本 | 9.1.0 |
| SOC Version | 251 (Ascend910B) |
| Python | 3.12 |
| torch_npu | 已安装 |
| 算子安装路径 | `$ASCEND_OPP_PATH/vendors/custom_xllm_math/` |
| Vendor config | `load_priority=custom_xllm_math` |

## 6. TODO

### 6.1 用 PA 算子替代 CPU golden

**背景**：当前 `test_x_attention.py` 使用手写 CPU fp32 参考实现作为 golden，计算量极大（大用例单条 90~180s），需要 golden 缓存机制才能实用。而 `test_x_attention_with_pa.py` 已验证 x_attention 与 CANN 官方 `torch_npu._npu_paged_attention` 的等价性（7680 条用例，atol=0.01），说明 PA 算子结果可作为 x_attention 的 golden。

**方案**：用 `torch_npu._npu_paged_attention()` 替代 CPU golden，将 shared+unshared 双路 KV 重组为 PA 单路 paged 格式后调用 PA 算子。

**优势**：
- NPU 上毫秒级计算，无需 golden 缓存机制
- 代码大幅简化（~60 行数据重组 + 1 行 PA 调用 vs 400 行手写 attention）
- 日常回归 96 条用例可秒级完成

**风险**：
- 独立性降低：CPU golden 是不同实现、不同硬件的独立验证，能 catches NPU 系统性 bug；PA golden 与 x_attention 同为 NPU kernel，数学实现相似，无法 catches 两个 kernel 共有的 bug
- 数据重组复杂度：需将双路 KV 拼接为 PA 的单路 paged KV cache + block_table + context_lens，重组逻辑若有 bug 会掩盖错误

**建议**：两者并存——PA golden 做日常快速回归，CPU golden（带缓存）做定期深度验证。

**参考实现**：`test_x_attention_with_pa.py:67-135` 已有完整的数据重组 + PA 调用逻辑。

## 7. A5 (Ascend950) 适配合并

### 7.1 合并来源

从 ware2009 仓库 `origin/20260820` 分支合并，核心 commit：
- `ad4f45b` feat: x_attention adapter the a5
- `1c1b562` feat: adapte the a5 and adjust the catlass module in common/catlass

### 7.2 合并内容

仅合并 `xllm_ops/x_attention/` 目录（12 个文件），**不修改** `common/catlass/` 和 `cmake/`：

| 文件 | 类型 | 改动说明 |
|------|------|---------|
| `op_kernel/x_attention.cpp` | 修改 | 新增 `__NPU_ARCH__` 架构检测宏，A5(`__NPU_ARCH__==3510`)走 `#if defined(XA_ARCH35)` 分支，A3 走 `#else` 原有逻辑 |
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

### 7.3 架构分流机制

`x_attention.cpp` 中的核心分流逻辑：

```cpp
// A5 检测
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510))
#define XA_ARCH35 1
#endif

// A3 检测（用于 catlass forwarding header dispatch）
#if !defined(XA_ARCH35) && !defined(CATLASS_ARCH) && \
    defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
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
    // A5 path: 不同的 workspace 布局 + tla:: API
    ...
#else
    // A3 path: 原有 CALL_XATTN_KERNEL 宏 + TILING_KEY_IS 分支
    ...
#endif
}
```

### 7.4 A5 编译依赖

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

这些文件已从 ware2009 `origin/20260820` 合入本仓 `common/catlass/`。

**注意**：`common/catlass/include` 需要被添加到 kernel 编译的 include 路径中。当前 `op_host/CMakeLists.txt` 的 `-I${CANN_3RD_LIB_PATH}/catlass/include` 只指向 `third_party/catlass/include`，不包含 `common/catlass/include`。A5 编译时可能需要手动添加 `-I${CMAKE_CURRENT_LIST_DIR}/../../../common/catlass/include` 到 `add_ops_compile_options`，或在 cmake 全局配置中添加。此问题需在 A5 环境上实际编译时验证。

### 7.5 A5 编译方法（不安装）

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

### 7.6 A3 验证结果

A5 适配合并后，在 A3 (Ascend910B, soc=251) 环境上验证：

```bash
bash build.sh -n x_attention        # 编译安装 A3 版本
pytest -v test_x_attention.py       # 96 条用例全部通过
```

结果：**96 passed in 9.61s** — A5 架构分流代码通过 `#if defined(XA_ARCH35)` 隔离，在 A3 上不生效，原有 A3 kernel 逻辑完全不受影响。

### 7.7 未验证项（需在 A5 环境上确认）

| 项目 | 说明 |
|------|------|
| A5 编译是否通过 | arch35/ kernel 能否在 ascend950 SOC 上编译成功 |
| `common/catlass/include` 路径解析 | `catlass_patch/xa_register.hpp` 的 include 路径是否正确解析 |
| A5 精度验证 | 96 条用例在 A5 上的 allclose 是否通过 |
| A5 性能 | msprof 采集 A5 上的 Task Duration / cube_utilization 等 |
| build.sh catlass 兼容补丁 | ware2009 `origin/20260820` 中 `build.sh` 的 `patch_catlass_compat()` 函数是否需要 |
