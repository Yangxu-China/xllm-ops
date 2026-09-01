# x_attention_v2 Group-Local 迁移状态

## 日期: 2026-09-01

## 背景

`x_attention_v2.py`（core-split 架构）从 `/workspace/xllm/xllm/python/kernels_npu/pypto/x_attention_v2.py` 迁移而来，是基于 gcw 仓 `x_attention.py` 做代码格式优化后的产物。

`x_attention_task.py`（group-local 架构）由 zhuqingbo 参考 DSL 方案实现，位于 `/workspace/pypto_zhuqingbo/python/tests/st/pypto_pro/frontend/fa/xattention/op_kernel/x_attention_task.py`。

本次将 `x_attention_task.py` 的 group-local 架构迁移到 xllm-ops 的 `x_attention_v2.py`，保留 v2 的命名约定和格式。

## 迁移内容

### 源文件
- 基础: `/workspace/pypto_zhuqingbo/python/tests/st/pypto_pro/frontend/fa/xattention/op_kernel/x_attention_task.py`（1082 行）

### 目标文件
- `/workspace/xllm-ops/xllm_ops/x_attention_v2/op_kernel/x_attention_v2.py`（1101 行）
- `/workspace/xllm-ops/xllm_ops/x_attention_v2/op_host/x_attention_v2_tiling.h`
- `/workspace/xllm-ops/xllm_ops/x_attention_v2/op_host/x_attention_v2_tiling.cpp`

### 保留的 v2 约定
1. xLLM Apache license header
2. 命名: `x_attention_v2_kernel`, `x_attention_v2`, `XAttnV2TilingKey`
3. 指针参数名: `shared_key_block`, `shared_value_block`, `unshared_key_block`, `unshared_value_block`
4. Host entry 类型注解
5. 无返回值约定（写回 caller 提供的 `attn_out`）

### 从 task.py 迁移的架构变更
1. **group-local 架构**: 每核处理完整的 shared→unshared→inline normalize，无 sync_all，无 GM workspace，无 combine 阶段
2. **task_table 调度**: host 侧构建 task_table，按 shared_tiles 降序排序，kernel 按 strided loop 遍历
3. **统一 stage loop**: shared tiles + 1 unshared stage，online softmax running state 跨 shared→unshared 延续
4. **新增 vf 函数**: `init_running_vf`, `normalize_store_vf`, `process_vec1_ug_update_*`（带 running-state update）
5. **TilingKey**: `SharedM` 字段（64/128 两个编译期变体）
6. **Host permute**: Q/K/V permute 到 `[B, Hkv, beam*group, D]` 布局，output permute 回 `[T, Hq, D]`

## C++ Tiling 一致性

### 已迁移到 C++ 的 Python host 逻辑

| Python host 逻辑 | C++ tiling 对应 | 状态 |
|---|---|---|
| `_select_tiling`（M=128/64 双路径 + 负载均衡检查） | `SelectTiling()` | ✅ |
| `_build_task_table`（shared_kv_lens 读取 + 排序） | `BuildTaskTable()` | ✅ |
| `decode_step` 运行时值读取 | `ParseShapeAndAttrs()` 中读 tensor | ✅ |
| `tiling_key` 按 SharedM 设置 | `SetTilingKey()` | ✅ |
| `scale` 属性读取 | `ParseShapeAndAttrs()` | ✅ |
| `total_cores` = platform core count | `RunTiling()` | ✅ |
| workspace 分配 | `SetWorkspaces()` | ✅ |

### TilingData 字段映射

| Python `TaskTiling` | C++ `XAttentionV2TilingData` | 类型 |
|---|---|---|
| `hq` | `hq` | int64_t |
| `hkv` | `hkv` | int64_t |
| `batch` | `batch` | int64_t |
| `beam_size` | `beam_size` | int64_t |
| `shared_m` | `shared_m` | int64_t |
| `group` | `group` | int64_t |
| `unshared` | `unshared` | int64_t |
| `max_ds` | `max_ds` | int64_t |
| `shared_total` | `shared_total` | int64_t |
| `scale` | `scale` | float |
| `num_tokens` | `num_tokens` | int64_t |
| `total_cores` | `total_cores` | int64_t |
| `task_count` | `task_count` | int64_t |
| `beams_per_task` | `beams_per_task` | int64_t |
| `unshared_n` | `unshared_n` | int64_t |

## 验证结果

### 确认1: C++ tiling 覆盖 Python host 全部逻辑 — ✅ 通过

### 确认2: JIT 模式 144 case — ✅ 通过
```
144 passed in 13.40s
```

### 确认3: built-in 模式 — ❌ 未通过

**现象**: built-in 编译安装成功，但执行 `test_x_attention_v2.py` 时 segfault。

**根因**: group-local kernel 依赖 host 侧 permute Q/K/V 到 `[B, Hkv, beam*group, D]` 布局。JIT 模式下 Python host 做了 permute，但 built-in aclnn 调用路径不做 permute。C++ tiling 虽然分配了 workspace 空间（task_table + permuted Q/K/V/O），但没有实际执行 permute 操作和 task_table 写入。

**需要补充的工作**:
1. 在 C++ tiling 中将 task_table 数据写入 workspace（通过 `GetRawTilingData` 或 workspace 机制传递给 kernel）
2. 在 kernel 中增加 permute 逻辑，或通过 host-side copy op 在 kernel 执行前做 permute
3. 考虑两种方案:
   - **方案 A**: kernel 内部做 permute（增加 GM 读取开销，但不需要额外的 host 步骤）
   - **方案 B**: 通过 aclnn 的 workspace 在 host 侧完成 permute（需要修改 op_def 或增加 copy kernel）
4. `unshared_block_table` 的默认值处理（Python 当 None 时用 `torch.arange(batch)`，C++ 需要等价逻辑）
5. output permute back（kernel 写入 permuted 布局，host 侧需要 permute 回 `[T, Hq, D]`）

## 性能对比（JIT 模式）

| 指标 | core-split (原) | group-local (迁移后) | 变化 |
|---|---|---|---|
| 总 kernel 时间 (144 cases) | 7112us | 6102us | -14.2% |
| vs built-in baseline | 7077us | 6102us | -13.8% |
| batch=6 (72 cases) | 5863us | 5115us | -12.8% |
| batch=1 (72 cases) | 1250us | 987us | -21.0% |
| 测试时间 | 13.28s | 10.98s | -17.3% |

## JIT 模式验证与性能采集流程

### 环境准备

```bash
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
TORCH_NPU_LIB=$(python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__) + '/lib')")
export LD_LIBRARY_PATH="${TORCH_LIB}:${TORCH_NPU_LIB}:${LD_LIBRARY_PATH}"
source /home/developer/Ascend/cann-9.2.0/opp/vendors/custom_xllm_math/bin/set_env.bash
export ASCEND_DEVICE_ID=0
```

### 1. 精度验证（144 case）

```bash
cd /workspace/xllm-ops/test/python_test
pytest test_x_attention_v2_jit.py -q
```

测试覆盖 144 个配置组合：
- dtype: bf16
- heads: (16,8) / (32,8) / (16,4)  → group 2/4/4
- batch: 1, 6
- beam: 128, 256, 512
- kv_seqlen (prompt): 128, 256, 512, 1024
- unshared_seqlen (maxDs): 2, 4

每条 case 执行后输出 `OP-MATRIX-V2` 日志行，含完整 shape 信息和精度 diff：
```
OP-MATRIX-V2 Hq=16 Hkv=8 d=128 b=1 beam=128 prompt=128 maxDs=2 dstep=2 dtype=torch.bfloat16: max|dev-gold|=3.9062e-03
```

### 2. 性能采集（msprof + stdout 日志）

```bash
# 采集：msprof 调用 pytest，同时保存 stdout 日志
rm -rf msprof_v2_jit
msprof --output=./msprof_v2_jit --task-time=on --ai-core=on \
  pytest test_x_attention_v2_jit.py -q -s 2>&1 | tee jit_run.log

# 验证行数匹配
grep "OP-MATRIX-V2" jit_run.log | wc -l          # 应为 144
CSV=$(ls msprof_v2_jit/PROF_*/mindstudio_profiler_output/op_summary_*.csv | head -1)
grep "x_attention" "$CSV" | wc -l                 # 应为 144
```

### 3. 合并 shape + timing（merge_prof）

```bash
# 将 stdout 日志的 shape 信息与 msprof CSV 的 timing 按行序合并
python3 x_attention_v2_jit_merge_prof.py \
  --log jit_run.log \
  --csv "$CSV" \
  -o msprof_v2_jit/merged_summary.csv
```

输出 CSV 列：`Hq, Hkv, grp, batch, beam, prompt, maxDs, dstep, dtype, diff, Task_Duration_us, aicore_time_us, aic_mte2_ratio, aic_mac_ratio, ...`

### 4. 排序 + 生成 xlsx（parser）

```bash
# 排序（按 dtype, batch, Hq, Hkv, beam, maxDs, prompt）并生成 xlsx
python3 x_attention_v2_parser.py msprof_v2_jit/merged_summary.csv
```

输出：
- `merged_summary_parsed.csv` — 排序后的 CSV
- `merged_summary.xlsx` — 两个 sheet：original（原始顺序）+ parsed（排序后）

### 相关脚本

| 脚本 | 路径 | 功能 |
|---|---|---|
| `test_x_attention_v2_jit.py` | `test/python_test/` | JIT 精度测试，输出 OP-MATRIX-V2 日志 |
| `x_attention_v2_jit_merge_prof.py` | `test/python_test/` | 合并 stdout 日志 shape + msprof CSV timing |
| `x_attention_v2_parser.py` | `test/python_test/` | 排序 + 生成 xlsx（支持 JIT merged CSV 和 built-in raw op_summary） |

### 脚本兼容性

`x_attention_v2_parser.py` 支持两种输入：
- **JIT merged CSV**（`merged_summary.csv`）：shape 已在列中（`Hq, Hkv, batch, beam, ...`），跳过提取直接排序
- **Built-in raw op_summary**（`op_summary_*.csv`）：从 `Input Shapes` / `Input Data Types` 列提取 shape 参数，排序列一致

两种模式排序后生成相同结构的 xlsx（original + parsed sheet）。

## 文件清单

| 文件 | 说明 |
|---|---|
| `op_kernel/x_attention_v2.py` | 迁移后的 group-local kernel + Python host entry |
| `op_kernel/x_attention_v2_core_split.py` | 原 core-split 版本备份 |
| `op_host/x_attention_v2_tiling.h` | C++ TilingData 定义（TaskTiling 字段） |
| `op_host/x_attention_v2_tiling.cpp` | C++ tiling 逻辑（_select_tiling + _build_task_table） |
| `op_host/x_attention_v2_def.cpp` | Op 定义（未修改，仍为原始版本） |
| `test/python_test/test_x_attention_v2_jit.py` | JIT 精度测试（含 OP-MATRIX-V2 日志输出） |
| `test/python_test/x_attention_v2_jit_merge_prof.py` | 合并 stdout shape + msprof timing |
| `test/python_test/x_attention_v2_parser.py` | 排序 + 生成 xlsx |
