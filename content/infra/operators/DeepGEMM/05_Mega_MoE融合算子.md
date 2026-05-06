# Mega MoE 融合算子

> 源码路径：  
> `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` — 主 kernel  
> `deep_gemm/mega/__init__.py` — Python 封装  
> `deep_gemm/include/deep_gemm/comm/barrier.cuh` — NVLink barrier  
> `deep_gemm/include/deep_gemm/layout/sym_buffer.cuh` — 对称内存布局  
> `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` — Workspace 布局  
> `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` — Mega MoE 调度器  
> `tests/test_mega_moe.py` — 使用示例与基准

---

## 1. 设计动机

传统 MoE 推理的执行流程是**顺序**的：

```
EP Dispatch（NVLink 通信）→ Linear1（FP8 GEMM）→ SwiGLU → Linear2（FP8 GEMM）→ EP Combine（NVLink 通信）
```

每个阶段都要等前一个完成，NVLink 通信和 Tensor Core 计算**完全串行**，导致 GPU 大量空闲。

**Mega MoE** 将以上所有步骤融合进**一个 mega kernel**，在 SM 级别实现计算与通信的重叠：

- 部分 SM 负责 NVLink dispatch/combine
- 其他 SM 同时做 Linear1 + SwiGLU + Linear2 的计算
- 通过 Tensor Memory 和 SMEM 中转数据，避免 HBM 往返

---

## 2. 系统架构

```
[Rank 0 GPU]                    [Rank 1 GPU]
    │                                │
    ▼                                ▼
对称内存（PyTorch symmetric memory）
    │                                │
    ▼                                ▼
┌─────────────────────────────────────────┐
│               sm100_fp8_fp4_mega_moe     │
│                                          │
│  ┌──────────┐   ┌────────────────────┐  │
│  │ Dispatch │   │  Linear1 (FP8×FP4) │  │
│  │ threads  │──▶│  SwiGLU            │  │
│  │          │   │  Linear2 (FP8×FP4) │  │
│  └──────────┘   └────────────────────┘  │
│       ▼                   ▼             │
│  ┌──────────┐   ┌────────────────────┐  │
│  │ Combine  │◀──│  Epilogue threads  │  │
│  │ threads  │   │  (BF16 output)     │  │
│  └──────────┘   └────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## 3. 对称内存（Symmetric Memory）

对称内存是 Mega MoE 通信的基础，每个 rank 分配相同布局的内存，通过地址映射访问远端：

```cuda
// deep_gemm/include/deep_gemm/layout/sym_buffer.cuh
template <uint32_t kNumRanks = kNumMaxRanks>  // 最多 72 个 rank
struct SymBuffer {
    int64_t base;                    // 本 rank 的基地址
    int64_t offsets[kNumMaxRanks];   // 各 rank 相对本 rank 的地址偏移

    // 将本 rank 的指针映射到目标 rank 的对应位置
    CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }
};
```

Python 侧使用 PyTorch 2.9 的 `symmetric_memory` 分配：

```python
# deep_gemm/mega/__init__.py
class SymmBuffer:
    def __init__(self, group, num_experts, num_max_tokens_per_rank, num_topk,
                 hidden, intermediate_hidden, use_fp8_dispatch=True, activation='swiglu'):
        # 分配对称缓冲区
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(...)
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)

        # 将缓冲区切片为各功能区
        (self.x, self.x_sf,           # FP8 输入 token + scale
         self.topk_idx, self.topk_weights,  # 专家路由
         self.l1_acts, self.l1_acts_sf,     # Linear1 中间结果
         self.l2_acts, self.l2_acts_sf,     # Linear2 中间结果
         ) = slice_input_buffers(self.buffer)
```

---

## 4. Kernel 线程分工

```cuda
template <
    uint32_t kNumDispatchThreads,      // EP dispatch 线程（128 的倍数）
    uint32_t kNumNonEpilogueThreads,   // UMMA 计算线程（128）
    uint32_t kNumEpilogueThreads,      // 输出写回 + EP combine 线程
    ...
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(...)
```

线程职责分工：

```
thread_idx < kNumDispatchThreads:
    负责读取 topk_idx，通过 NVLink 将 token 分发到目标 rank
    在 dispatch warp 组内完成：FP8 cast + 跨 rank 写入

kNumDispatchThreads <= thread_idx < kNumDispatchThreads + kNumNonEpilogueThreads:
    负责 UMMA MMA 计算（Non-epilogue）
    读 token data → Linear1 UMMA → SwiGLU（在 TMEM 中） → Linear2 UMMA

thread_idx >= kNumDispatchThreads + kNumNonEpilogueThreads:
    负责 Epilogue：从 TMEM 读结果，dequant，通过 NVLink combine 到原始 rank
```

---

## 5. 权重布局变换

FP4 权重需要特殊的内存布局才能被 UTCCP 高效处理：

```python
# deep_gemm/mega/__init__.py

def _interleave_l1_weights(l1_weights):
    # L1 权重（gate + up 投影合并）需要交错排列
    # 从 [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...]
    # 变为标准 [gate | up] 再变为 interleaved 格式
    def interleave(t, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up   = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.stack([gate, up], dim=2).reshape(g, n, *rest)
    return interleave(l1_weights[0]), interleave(l1_weights[1])

def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # UE8M0 scale 需要特殊转置：
    # [G, MN, packed_SF_K] -> [G, MN/128, 4, 32, packed_SF_K]
    # -> transpose(2,3) -> [G, MN, packed_SF_K]
    num_groups, mn, packed_sf_k = sf.shape
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
               .transpose(2, 3)
               .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)
```

这两步变换保证了 UTCCP 硬件能在一次 issue 内读取到所有需要的 scale，不浪费内存带宽。

---

## 6. NVLink Barrier 实现

跨 rank 同步使用自定义的 NVLink barrier，基于对称内存中的原子计数器：

```cuda
// deep_gemm/include/deep_gemm/comm/barrier.cuh
template <uint32_t kNumRanks, uint32_t kNumSMs, ...>
CUTLASS_DEVICE void nvlink_barrier(
    const layout::Workspace& workspace,
    const layout::SymBuffer<kNumRanks>& sym_buffer, ...) {

    // 1. 本 rank 内 grid sync（确保所有 SM 都到达）
    grid_sync<kNumSMs>(workspace, sm_idx, thread_idx, sync_scope);

    // 2. 只有 SM 0 参与跨 rank 信令
    if (sm_idx == 0) {
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // 向所有 rank 的对称内存写入信号（NVLink 原子操作）
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);

        // 等待所有 rank 都发送信号（带 30 秒超时）
        const auto start_clock = clock64();
        while (ptx::ld_acq_sys(signal_ptr) != target) {
            if (clock64() - start_clock >= 30ll * 2000000000ll) {
                printf("NVLink barrier timeout\n");
                DG_DEVICE_ASSERT(false);
            }
        }
    }

    // 3. 再次 grid sync（等 SM 0 完成）
    grid_sync<kNumSMs>(workspace, sm_idx, thread_idx, sync_scope);
}
```

这个实现支持最多 72 个 rank，适用于 H100/B200 的 NVLink 集群。

---

## 7. Grid Sync 实现

由于 persistent kernel 跨多个 wave，需要软件实现 grid sync（CUDA cooperative groups 不适用于所有情况）：

```cuda
template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0>
CUTLASS_DEVICE void grid_sync(const layout::Workspace& workspace,
                               const uint32_t& sm_idx, const uint32_t& thread_idx,
                               const sync_scope_t& sync_scope) {
    constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();  // 本地 SM 内同步

    if (thread_idx == 0) {
        auto* count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        // SM 0 用 (kFinishSumTag - (kNumSMs-1)) 抵消，其他 SM 各加 1
        // 总和回到 kFinishSumTag 时所有 SM 到齐
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        do { new_value = ptx::ld_acq(count_ptr); }
        while (((new_value ^ old_value) & kFinishSumTag) == 0);
    }
    sync_scope();
}
```

这是一个无锁的 barrier，利用最高位（`0x80000000`）作为 flip 信号，避免多次使用时的 ABA 问题。

---

## 8. 完整使用流程

```python
import torch
import torch.distributed as dist
import deep_gemm

def run_mega_moe(rank, world_size):
    dist.init_process_group(...)
    group = dist.new_group(list(range(world_size)))

    # 参数
    num_experts = 64
    num_topk = 8
    hidden, intermediate_hidden = 7168, 2048
    num_max_tokens_per_rank = 4096

    # 1. 分配对称缓冲区（一次性，可复用）
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden
    )

    # 2. 准备权重并变换布局（只需一次）
    l1_weights = (torch.randn(num_experts//world_size, intermediate_hidden*2, hidden, device='cuda'),
                  sf_tensor_l1)  # (FP4 weights, UE8M0 scales)
    l2_weights = (torch.randn(num_experts//world_size, hidden, intermediate_hidden, device='cuda'),
                  sf_tensor_l2)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # 3. 每次推理前填充输入（可融入前驱 kernel）
    num_tokens = 1024
    buffer.x[:num_tokens].copy_(x_fp8)          # FP8 输入
    buffer.x_sf[:num_tokens].copy_(x_sf)         # 缩放因子
    buffer.topk_idx[:num_tokens].copy_(topk_idx) # expert routing
    buffer.topk_weights[:num_tokens].copy_(topk_weights)

    # 4. 执行 Mega MoE
    y = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_mega_moe(
        y, transformed_l1, transformed_l2, buffer,
        recipe=(1, 1, 32),          # (gran_m, gran_n, gran_k)
        activation='swiglu',
        fast_math=True
    )
```

---

## 9. 性能特点

| 指标 | 传统流程（串行） | Mega MoE（融合重叠） |
|------|----------------|---------------------|
| NVLink 传输 | 串行（等计算完成） | 与计算重叠 |
| HBM 访问 | 每阶段完整读写 | 中间结果留在 SMEM/TMEM |
| Activation（SwiGLU） | 独立 kernel | 融合在 MMA 之间 |
| FP8 cast（dispatch） | 独立 kernel | dispatch 线程内融合 |
| 硬件利用 | TC/NVLink 交替 | TC + NVLink 同时 |

> 注：Mega MoE 需要 SM100（B200）+ CUDA 12.9 + PyTorch 2.9，且必须多进程（multi-GPU）启动。
