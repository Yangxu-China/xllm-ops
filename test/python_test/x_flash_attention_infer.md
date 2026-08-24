# XFlashAttentionInfer 算子分析上下文

## 1. 算子概述

`XFlashAttentionInfer` 是 xllm-ops 中的自定义 AscendC 算子，实现 **推理（Decode）阶段的 Flash Attention**，支持 PagedAttention、GQA、Causal Mask，基于华为自研 catlass 模板库实现 Cube/Vector 双核流水。

### 核心特性

- **Flash Decode (FD)**：KV 序列切分到多核并行，最后 `combineScale` 归约，适合长序列 decode
- **PagedAttention**：通过 `block_table` 索引 KV cache 的物理 block，支持分页 KV cache
- **GQA**：q_head / kv_head 分组（group query attention）
- **Causal Mask**：支持 chunked causal mask，跳过全被 mask 的 KV block
- **Cube/Vector 双核流水**：MM 在 Cube、Online Softmax / Rescale O 在 Vector，跨核 flag 同步
- **PRE_LAUNCH=2 软流水**：KV 搬运与计算重叠，提前 2 个 block 预取
- **布局**：TND（query 拼接为 `[numTokens, numHeads, headDim]`），KV 支持 TND 和 FRACTAL_NZ
- **数据类型**：FP16 / BF16

### 算子源码位置

```
xllm_ops/x_flash_attention_infer/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   ├── x_flash_attention_infer_def.cpp       # 算子定义（输入/输出/属性）
│   ├── x_flash_attention_infer_proto.cpp      # InferShape/InferDataType
│   ├── x_flash_attention_infer_tiling.h       # TilingData 结构定义
│   └── x_flash_attention_infer_tiling.cpp     # Tiling 计算（host 侧）
└── op_kernel/
    ├── x_flash_attention_infer.cpp            # kernel 入口（FD 模式分发）
    ├── x_flash_attention_infer.h              # 标准模式 kernel（FAInferKernel）
    ├── x_flash_attention_infer_fd.h           # Flash Decode 模式 kernel（FAInferKernelFD）
    └── x_flash_attention_infer_common.h       # 公共定义、常量、辅助函数
```

### 算子注册名

- Op Type（PascalCase）: `XFlashAttentionInfer`
- kernel 入口（snake_case）: `x_flash_attention_infer`
- aclnn API: `aclnnXFlashAttentionInfer`

## 2. 算子接口

### 2.1 输入

| 接口 | 类型 | 格式 | 说明 |
|------|------|------|------|
| `query` | FP16/BF16, REQUIRED | ND | shape `[numTokens, qHead, embeddingSize]`，TND 布局 |
| `key_cache` | FP16/BF16, REQUIRED | ND / FRACTAL_NZ | Paged KV Cache，按 block 组织 |
| `value_cache` | FP16/BF16, REQUIRED | ND / FRACTAL_NZ | Paged KV Cache，按 block 组织 |
| `mask` | INT8, OPTIONAL | ND | Causal Mask |
| `block_table` | INT32, REQUIRED | ND | PagedAttention block 映射表 `[batch, maxNumBlocksPerBatch]` |
| `actual_q_lens` | INT32, REQUIRED | ND | 每 batch 实际 Q 序列长度（TND 下为前缀和） |
| `actual_kv_lens` | INT32, REQUIRED | ND | 每 batch 实际 KV 序列长度（TND 下为前缀和） |
| `extra_tiling` | INT32, REQUIRED | ND | FD 模式的核间切分信息 `SplitKvExtraInfo` |

### 2.2 输出

| 接口 | 类型 | 格式 | 说明 |
|------|------|------|------|
| `attn_out` | FP16/BF16, REQUIRED | ND | 输出，shape 同 query |

### 2.3 属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layout` | String | "TND" | 输入布局 |
| `qHead` | Int | - | Q head 数 |
| `kvHead` | Int | - | KV head 数 |
| `scale` | Float | 1.0 | 缩放因子（tiling 中实际用 `1/sqrt(embeddingSize)`） |

## 3. Tiling 策略

### 3.1 RunTiling 流程

`x_flash_attention_infer_tiling.cpp:172-194` — `RunTiling()` 流程：

1. 获取平台信息（cube/vec 核数）
2. `ParseInputShapeAndAttrs()` → 解析 shape、dtype、layout、mask，填充基本 tiling 字段
3. `FillSplitCoreTilingDataForJD()` — 计算 JD 模式的任务切分：
   - Q 维按 `GetQNBlockTile()` 切分（GQA group 内合并多个 Q head 到 128 行 tile）
   - S 维按 `Q_TILE_CEIL=128` 切分
   - `totalTaskNum = curQNBlockNum * curQSBlockNum * batch`
4. `SetWorkspaces()` — 分配 workspace：`s`(QK分数) + `p`(softmax) + `oTemp` + `oUpdate` + `splitLse` + `splitO`
5. `SetBlockDim(cubeCoreNum)` — 按 Cube 核数分配

### 3.2 TilingKey 编码

`x_flash_attention_infer_tiling.cpp:70-109`

```
baseKey(1e18) + KVLayout(TND=100/NZ=200) + dtype(FP16=10/BF16=20) + mask(3) + FD(1000)
```

当前 `usingFD` 恒为 `true`（`x_flash_attention_infer_tiling.h:129`），所以始终走 FD 模式。

### 3.3 关键常量

| 常量 | 值 | 说明 |
|------|------|------|
| `Q_TILE_CEIL` | 128 | Q 行 tile 大小 |
| `MAX_KV_STACK_LEN` | 512 | KV stack 最大长度 |
| `PRE_LAUNCH` | 2 | 软流水预取级数 |
| `L1_MAX_SIZE` | 524288 (512KB) | L1 buffer 最大尺寸 |
| `L1_MAX_N_NUM` | 128 | L1 N 维动态上限 |
| `WORKSPACE_BLOCK_SIZE_DB` | 128*512=65536 | workspace block 大小 |

## 4. Kernel 实现

### 4.1 入口分发

`x_flash_attention_infer.cpp:14-59` — 入口函数：

- 任务类型 `KERNEL_TYPE_MIX_AIC_1_2`（Cube:Vector = 1:2 双核混合）
- workspace 划分为 `s → p → oTemp → oUpdate → gmlse → glo`
- **只分发 FD 模式的 4 种 tiling key**（FP16/BF16 × TND/NZ × CausalMask × FD）
- 通过 `extraInfo->coreInfo[coreIdx].startBIdx != UINT32_MAX` 判断当前核是否有任务，无任务则只做 `SyncAll()`

### 4.2 两种 Kernel

#### FAInferKernel（标准模式，`x_flash_attention_infer.h`）

- **Online Softmax** Flash Attention，不切分 KV 到多核
- 任务循环：`for (taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += coreNum)` 跨核步进
- Cube 核做 QK 和 PV 的 MM（matmul），Vector 核做 Online Softmax 和 Rescale O
- **PRE_LAUNCH=2** 软件流水：提前 2 个 block 预取，实现计算与搬运重叠
- 跨核同步：`qkReady`（Cube→Vec）、`softmaxReady`（Vec→Cube）、`pvReady`（Cube→Vec）

#### FAInferKernelFD（Flash Decode 模式，`x_flash_attention_infer_fd.h`）

- **KV 维度切分到多核**（`SplitKvExtraInfo` 描述每核负责的 `[startBIdx, startN1Idx, startS2Idx]` ~ `[endBIdx, endN1Idx, endS2Idx]`）
- 每核处理完自己的 KV 分片后，将部分结果写入 `gmlse`/`glo` workspace
- 最后 `SyncAll()` + `combineScale()` 做跨核归约（合并各核的 LSE 和 O，得到最终 attention 输出）
- `isSplitKV` 标记决定是否需要写中间结果用于后续归约

### 4.3 计算流水（Cube/Vector 协同）

```
Cube:  loadQGM → MM(QK) ──set qkReady──→ MM(PV) ──set pvReady──→
Vec:                    wait qkReady → OnlineSoftmax ──set softmaxReady──→ wait pvReady → RescaleO
```

- QK MM：L1Tile `[128, 128, 128]`，Q 一次性 load 到 L1，KV 按 `blockStackNum`（MAX_KV_STACK_LEN=512 / pagedBlockSize）循环搬运
- PV MM：L1Tile `[128, 128, 256]`
- L1 动态 N 维：根据 `embedV * MAX_KV_STACK_LEN` 预留后剩余空间动态计算 `nDynNum`，上限 `L1_MAX_N_NUM=128`

### 4.4 PagedAttention 支持

通过 `block_table` 索引 KV cache 的物理 block：
- `blockMmadQK` / `blockMmadPV` 接收 `gBlockTable[blockBOffset]`，在 MM 内部完成虚拟→物理 block 地址映射
- 支持 NZ 布局（`layout::nZ` for K，`layout::zN` for V），offset 计算有专门分支（`x_flash_attention_infer.h:261-266`）

### 4.5 Causal Mask 处理

`x_flash_attention_infer.h:359-414` / `x_flash_attention_infer_fd.h:347-404`

- `noSkipKvS = min(kvSeqlen, qSBlockEnd + (kvSeqlen - qSeqlen))` — 跳过全被 mask 的 KV block
- `triUp = noSkipKvS - qSBlockSize` — mask 左上起点
- 若 `triUp >= kvSEndIdx - 1`（当前 block 完全在 mask 下方），走无 mask 快速路径
- 否则调用带 mask 的 `epilogueOnlineSoftmax`，传入 `triUp/triDown/kvSStartIdx/kvSEndIdx` 做 chunked mask

## 5. 已知问题

1. **非 FD kernel 疑似死代码**：入口 `x_flash_attention_infer.cpp` 只分发 `*_FD_TILING`，`FAInferKernel`（标准模式）未在入口被调用，且 `usingFD` 恒为 true。`FAInferKernel` 似乎是保留/过渡代码。

2. **类型不一致**：`FAInferKernel` 中 `gActualQseqlen`/`gActualKvseqlen` 声明为 `int64_t`（`x_flash_attention_infer.h:95-98`），但算子定义中这两个输入是 `DT_INT32`。`FAInferKernelFD` 中正确使用了 `int32_t`（`x_flash_attention_infer_fd.h:93-96`）。这进一步说明 `FAInferKernel` 当前未实际使用。

3. **`extra_tiling` 未在 host tiling 中填充**：`SplitKvExtraInfo`（`coreInfo[25]`/`splitInfo[25]`）是 FD 模式的核心切分信息，但 host 侧 `RunTiling()` 中没有生成它的逻辑。推测由 aclnn 封装层（`aclnnXFlashAttentionInfer`）在 host 端计算后写入 `extra_tiling` 输入，tiling kernel 中直接读取。这部分逻辑不在当前算子目录内。

4. **`FillSplitCoreTilingDataForJD` 计算的 `firstBatchTaskNum`/`totalTaskNum` 在 FD 模式下未被 kernel 使用**（FD kernel 用 `extraInfo` 而非 `totalTaskNum` 来分配任务），仅 JD 模式会用到，但 JD 路径未启用。

5. **`embeddingSizeV` 被设为等于 `embeddingSize`**（`x_flash_attention_infer_tiling.cpp:61`），不支持 QK 和 V 的 head_dim 不同。

## 6. 芯片支持分析

### 6.1 芯片型号对应关系

| 简称 | SOC 短名 | 完整型号 | 架构 | __NPU_ARCH__ |
|------|----------|---------|------|-------------|
| **A2** | `ascend910b` | Ascend910B1 | dav-c220 | 2201 |
| **A3** | `ascend910_93` | Ascend910_9391 (Ascend910C) | dav-c220 | 2201 / 3003 / 3113 |
| **A5** | `ascend950` | Ascend950PR_9599 (Ascend910_95) | dav-c310 | 3510 |

此对应关系由 `build.sh:110-115`、`packer.py:90`、`build_aclnn.sh:171/299` 多处独立印证。

### 6.2 算子注册情况

`x_flash_attention_infer_def.cpp:71-73` 注册了全部三种芯片配置：

```cpp
this->AICore().AddConfig("ascend910b");    // A2
this->AICore().AddConfig("ascend910_93");  // A3
this->AICore().AddConfig("ascend950");     // A5
```

### 6.3 A3：完全支持

A3 与 A2 共享 `dav-c220` 架构，kernel 中的 `__DAV_C220_CUBE__`/`__DAV_C220_VEC__` 宏和 `Arch::AtlasA2` 标签都直接适用，无需额外适配。catlass 库和 pto-isa 中 A2/A3 统一放在 `a2a3/` 目录。

### 6.4 A5：注册了但 kernel 缺少适配

对比同仓库的 `x_attention` 算子，A5 需要以下适配，而 `x_flash_attention_infer` **全部缺失**：

| A5 适配项 | x_attention | x_flash_attention_infer |
|----------|-------------|------------------------|
| arch guard (`XA_ARCH35` / `__NPU_ARCH__==3510`) | 有 (`x_attention.cpp:16-35`) | **无** |
| CMakeLists 注入 `-DCATLASS_ARCH=3510` | 有 (`op_host/CMakeLists.txt:29-35`) | **无** |
| arch35 专属 kernel 目录/文件 | 有 (`block_epilogue_*_ascend950.hpp`) | **无** |
| A5 专属 tiling 分支 | 有 (`x_attention_tiling.h:84`) | **无** |

更关键的是，kernel 代码（`x_flash_attention_infer.h:114-171` 和 `x_flash_attention_infer_fd.h:116-174`）大量使用 `#ifdef __DAV_C220_CUBE__` / `#ifdef __DAV_C220_VEC__` 来区分 Cube/Vector 核代码路径。A5 使用 `dav-c310` 架构，这两个宏在 A5 上**可能不会被定义**，导致 Cube 和 Vector 核的代码路径都无法编译。

### 6.5 结论

- **A3**：完全支持，与 A2 共享架构，无需额外适配
- **A5**：算子定义中注册了 `ascend950` 配置，但 kernel 代码**缺少 A5 适配**（无 arch35 guard、无 CATLASS_ARCH=3510 注入、依赖 dav-c220 专属宏）。大概率无法在 A5 上正确编译或运行，除非 CANN 工具链对 dav-c310 提供了 `__DAV_C220_CUBE__`/`__DAV_C220_VEC__` 的向后兼容宏

### 6.6 A5 适配建议

如果需要在 A5 上使用这个算子，建议参考 `x_attention` 的 A5 适配方式：

1. **添加 arch guard**：在 kernel 入口和实现中添加 `#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)` 的 A5 分支
2. **CMakeLists 注入 CATLASS_ARCH**：在 `op_host/CMakeLists.txt` 中添加 ascend950 检测逻辑，注入 `-DCATLASS_ARCH=3510`
3. **arch35 专属 epilogue**：参考 `common/catlass/include/catlass/epilogue/block/block_epilogue_xa_*_ascend950.hpp`，为 FD 模式的 OnlineSoftmax/RescaleO/CombineScale 提供 A5 特化实现
4. **验证 `__DAV_C220_CUBE__`/`__DAV_C220_VEC__` 兼容性**：确认 A5 的 dav-c310 工具链是否提供这两个宏的向后兼容，若不提供则需要替换为 `__DAV_C310_CUBE__`/`__DAV_C310_VEC__` 或其他 A5 专属宏
