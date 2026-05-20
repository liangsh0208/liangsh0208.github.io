---
created: 2026-05-06
---

# Dense 解码内核：Seesaw 调度

> 源码路径：`csrc/sm90/decode/dense/`  
> 关键文件：`traits.h`、`splitkv_mla.h`、`config.h`  
> 技术博客：`docs/20250422-new-kernel-deep-dive.md`

---

## 1. 为什么 MLA 解码是计算瓶颈

与传统 GQA/MHA 解码不同，MLA 解码是**计算瓶颈而非内存瓶颈**。

### 1.1 计算-带宽比分析

```
FLOPs ≈ 2 · h_q · s_q · s_k · (d_k + d_v)
内存访问 ≈ 2 · s_k · d_k   （KV Cache 主导）

计算-带宽比 = FLOPs / 内存 ≈ h_q · s_q · (d_k+d_v)/d_k ≈ 2·h_q·s_q
```

DeepSeek-V3 不使用 TP 进行解码，`h_q=128`，`s_q=1`（非投机解码时）：

```
计算-带宽比 ≈ 2 × 128 = 256
H800 平衡点 = TFlops / (2 × 带宽) ≈ 865 / (2 × 3.35) ≈ 129
```

256 >> 129，因此 MLA 解码是**计算瓶颈**，需要最大化 Tensor Core 利用率。

---

## 2. Seesaw 调度原理

### 2.1 FlashAttention-3 Ping-Pong 为什么不适用

FlashAttention-3 用 ping-pong 双缓冲：两个 warpgroup 轮流持有 O 矩阵，重叠 GEMM 和 softmax。但 MLA 的 `O(64×512)` 需要 **32768 个 32-bit 寄存器**，占 SM 总寄存器（65536）的 50%，无法同时放置两个 O 矩阵。

### 2.2 Seesaw（跷跷板）解决方案

将 O 矩阵**纵向切分**为 `O_L(64×256)` 和 `O_R(64×256)`，由两个 warpgroup 各持有一半：

```
WarpGroup 0 持有：O_L  [64×256]
WarpGroup 1 持有：O_R  [64×256]
```

同时将每轮的 V 块也切分为 `V_L` 和 `V_R`。

### 2.3 每轮的 11 步计算序列

以一轮处理两个 KV 块 (K₀, K₁, V₀, V₁) 为例：

| 步骤 | warpgroup | 操作 |
|------|-----------|------|
| 0 | [0] | `p₀ = Q · K₀ᵀ / scale` |
| 1 | [1] | `p₁ = Q · K₁ᵀ / scale`（与 p₀ GEMM 重叠）|
| 2 | [0] | `mp₀ = max(p₀)`, `m_new₀ = max(m, mp₀)`, `scale₀ = exp(m_new₀ - m)`, 更新 m |
| 3 | [0] | softmax: `p₀ ← exp(p₀ - m_new₀)` |
| 4 | [0] | `O_L ← O_L·scale₀ + p₀·V₀L` |
| 5 | [1] | `mp₁ = max(p₁)`, `m_new₁ = max(m, mp₁)`, `scale₁ = exp(m_new₁ - m)`, 更新 m |
| 6 | [1] | softmax: `p₁ ← exp(p₁ - m_new₁)` |
| 7 | [1] | `O_R ← O_R·(scale₀·scale₁) + p₁·V₁R` |
| 8 | [0] | `p₀ ← p₀ · scale₁`（重新缩放以补偿 step 7 中 scale₁ 的更新）|
| 9 | [1] | `O_R ← O_R + p₀·V₀R`（交叉：wg1 用 wg0 的 p₀）|
| 10 | [0] | `O_L ← O_L·scale₁ + p₁·V₁L`（交叉：wg0 用 wg1 的 p₁）|

步骤 9 和 10 就是"跷跷板"的核心——两个 warpgroup 互相使用对方持有的 p 矩阵，实现 CUDA Core（softmax）和 Tensor Core（GEMM）的真正重叠。

---

## 3. TMA 管线化

### 3.1 细粒度 TMA 分段

K 块大小为 `64×576`，等待整块 TMA 复制完成再开始 GEMM 会引入大量延迟。FlashMLA 将 K 块切分为 **9 个 64×64 的子块**，每个子块复制完成后立即开始对应的 GEMM：

```
TMA copy K[0:64]  → GEMM(Q, K[0:64])
TMA copy K[64:128] → GEMM(Q, K[64:128])
...
TMA copy K[512:576] → GEMM(Q, K[512:576])
```

实现中通过 TMA Barrier 数组实现：

```cpp
TMABarrier barriers_K0[HEAD_DIM_K/64];  // 9 个 barrier，每个保护一个子块
TMABarrier barriers_K1[HEAD_DIM_K/64];
```

### 3.2 双 K 缓冲区

共享内存中为 K 分配两个缓冲区：

```cpp
struct SharedMemoryPlan {
    array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK0;  // K 双缓冲 0
    array_aligned<InputT, cosize_v<SmemLayoutK>> smem_sK1;  // K 双缓冲 1
    array_aligned<InputT, cosize_v<SmemLayoutQ>> smem_sQ;
    array_aligned<InputT, cosize_v<SmemLayoutP0>> smem_sP0; // P 用于 V GEMM
    ...
};
```

当 GEMM 处理 sK0 时，TMA 同步加载下一个 K 块到 sK1，实现 hide memory latency。

---

## 4. WGMMA 矩阵乘法配置

Traits 类定义了四种 WGMMA 配置：

```cpp
// QK GEMM：Q 在 smem，K 在 smem（两者都从 TMA 加载）
using TiledMMA_QK_sQ = ss_op_selector<InputT, InputT, float,
    Shape<BLOCK_SIZE_M, PAGE_BLOCK_SIZE, HEAD_DIM_K>>;

// QK GEMM：Q 在寄存器，K 在 smem（Q 已加载，避免重复 TMA）
using TiledMMA_QK_rQ = rs_op_selector<InputT, InputT, float,
    Shape<BLOCK_SIZE_M, PAGE_BLOCK_SIZE, HEAD_DIM_K>>;

// PV GEMM（左半 V）：P 在寄存器，V 在 smem
using TiledMMA_PV_LocalP = rs_op_selector<InputT, InputT, float,
    Shape<BLOCK_SIZE_M, HEAD_DIM_V/2, PAGE_BLOCK_SIZE>>;

// PV GEMM（右半 V）：P 在 smem，V 在 smem（交叉访问）
using TiledMMA_PV_RemoteP = ss_op_selector<InputT, InputT, float,
    Shape<BLOCK_SIZE_M, HEAD_DIM_V/2, PAGE_BLOCK_SIZE>>;
```

"RemoteP" 表示 p 矩阵通过 shared memory 传递（跨 warpgroup 的跷跷板步骤）。

---

## 5. Split-KV 与 Tile Scheduler

### 5.1 Split-KV 原理

为充分利用多个 SM，将每个请求的 KV 序列切分为若干段，各段分配给不同 SM 并行处理。每段输出一个局部 `(O_partial, lse_partial)`，最后由 `combine` kernel 合并：

```
Request 0 (seq_len=8192):
  SM 0: 处理 token 0~2047 → (O₀, lse₀)
  SM 1: 处理 token 2048~4095 → (O₁, lse₁)
  SM 2: 处理 token 4096~6143 → (O₂, lse₂)
  SM 3: 处理 token 6144~8191 → (O₃, lse₃)
  → combine: O = Σᵢ exp(lseᵢ - lse_max) · Oᵢ / Σᵢ exp(lseᵢ - lse_max)
```

### 5.2 Tile Scheduler 元数据

`FlashMLASchedMeta` 存储调度元数据，避免每次调用都重新计算：

```python
@dataclasses.dataclass
class FlashMLASchedMeta:
    tile_scheduler_metadata: Optional[torch.Tensor]   # (num_sm_parts, DecodingSchedMetaSize), int32
    num_splits: Optional[torch.Tensor]                # (1,), int32
```

- `tile_scheduler_metadata`：描述每个 SM 负责哪些 request 的哪些 block
- `num_splits`：每个请求被分成几段（控制 `combine` kernel 的工作量）

首次调用时计算并缓存，后续相同形状的请求直接复用：

```python
if not sched_meta.have_initialized:
    sched_meta.have_initialized = True
    sched_meta.config = FlashMLASchedMeta.Config(...)
    # tile_scheduler_metadata 由 CUDA kernel 在 dense_decode_fwd 中填充
```

### 5.3 Programmatic Dependent Launch

`splitkv_mla` kernel 完成后直接在 GPU 端触发 `combine` kernel，无需 CPU 介入：

```cpp
// splitkv_mla kernel 末尾
cudaGridDependencySynchronize();  // 等待 split kernel 全部完成
// 触发 combine kernel（通过 CUDA dependency launch）
```

这消除了 kernel launch 的 CPU overhead（约 5-10μs），对高频小 batch 场景非常重要。

---

## 6. 缓存提示优化

```cpp
// TMA copy 使用 EVICT_FIRST 提示
cute::TMA::CacheHintSm90::EVICT_FIRST
```

KV Cache 数据访问一次后不再需要（streaming 访问模式），使用 EVICT_FIRST 告诉 L2 cache 尽快驱逐，避免污染 Q、P、O 的缓存行。实验显示这可以提升 L2 命中率和整体带宽。

---

## 7. 性能数据

在 H800 SXM5（SM90a，CUDA 12.8）：

| 配置 | 性能 |
|------|------|
| 内存限制（b=1, s_k=1024） | ~3000 GB/s（内存带宽利用率 ~90%）|
| 计算限制（b=128, s_k=32768） | ~660 TFlops（Tensor Core 利用率 ~76%）|
| 典型推理配置（b=64, s_k=8192）| ~450 TFlops |

旧版内核（无 Seesaw）：580 TFlops（计算限制场景）  
新版内核（Seesaw）：**660 TFlops**，提升约 13.8%。
