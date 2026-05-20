---
created: 2026-05-06
---

# 02 Layout 布局计算

> 源文件：`csrc/kernels/layout.cu`、`deep_ep/buffer.py` (get_dispatch_layout)

---

## 1. 模块作用与设计动机

在 MoE dispatch 操作之前，必须先知道"哪些 token 需要发送给哪个 rank、哪个专家"。`get_dispatch_layout` kernel 负责从 `topk_idx` 张量出发，计算出以下路由统计信息：

| 输出张量 | 形状 | 含义 |
|---------|------|------|
| `num_tokens_per_rank` | `[num_ranks]` | 每个目标 rank 需要接收的 token 总数 |
| `num_tokens_per_rdma_rank` | `[num_rdma_ranks]` | 每个 RDMA rank 需要接收的 token 数（跨节点时使用） |
| `num_tokens_per_expert` | `[num_experts]` | 每个专家需要处理的 token 数 |
| `is_token_in_rank` | `[num_tokens, num_ranks]` | 布尔矩阵，token i 是否发往 rank j |

这些信息是后续 `notify_dispatch` kernel 和 dispatch kernel 的前置依赖，它们决定了：
- CPU 等待时的接收总量（用于解除 CPU 阻塞）
- 每个通道的发送/接收偏移量前缀和
- 接收缓冲区的分配大小

---

## 2. 核心算法：并行统计

### 2.1 Kernel 参数（`csrc/kernels/layout.cu`）

```cpp
template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(
    const topk_idx_t* topk_idx,      // [num_tokens, num_topk]，int32/int64
    int* num_tokens_per_rank,         // 输出 [num_ranks]
    int* num_tokens_per_rdma_rank,    // 输出 [num_rdma_ranks]，跨节点时非 null
    int* num_tokens_per_expert,       // 输出 [num_experts]
    bool* is_token_in_rank,           // 输出 [num_tokens, num_ranks]
    int num_tokens, int num_topk,
    int num_ranks, int num_experts)
```

编译期参数（固定为 `kNumThreads=256, kNumExpertsPerSM=4, kNumRanksPerSM=8`）：
- 每个 SM 处理 4 个专家，或 8 个 rank
- 总 SM 数 = ⌈num_experts/4⌉ + ⌈num_ranks/8⌉

### 2.2 并行分工：专家统计 SM 与 rank 统计 SM

Layout kernel 将 SM 分为两组：

**第一组 SM（专家统计）**：
```
SM_begin = 0
SM_end = ceil(num_experts / kNumExpertsPerSM)
每个 SM 负责 [expert_begin, expert_begin+4) 的专家统计
```

```cpp
// 在每个线程中，独立统计每个 token 被哪些专家选中
for (int i = thread_id; i < num_tokens; i += kNumThreads) {
    for (int j = 0; j < num_topk; ++j) {
        expert_idx = topk_idx[i * num_topk + j];
        if (expert_begin <= expert_idx and expert_idx < expert_end)
            ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin];
    }
}
__syncthreads();
// 归约：每个 expert 对所有线程求和
if (expert_begin + thread_id < expert_end) {
    int sum = 0;
    for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_expert_per_thread[i][thread_id];
    num_tokens_per_expert[expert_begin + thread_id] = sum;
}
```

**第二组 SM（rank 统计）**：
```
SM_begin = ceil(num_experts / kNumExpertsPerSM)
每个 SM 负责 [rank_begin, rank_begin+8) 的 rank 统计
```

```cpp
// 对每个 token，确定它去往哪些 rank（基于专家到 rank 的映射）
for (int i = thread_id; i < num_tokens; i += kNumThreads) {
    int is_in_rank[kNumRanksPerSM] = {0};
    for (int j = 0; j < num_topk; ++j) {
        expert_idx = topk_idx[i * num_topk + j];
        if (expert_begin <= expert_idx && expert_idx < expert_end) {
            rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
            is_in_rank[rank_idx]++;
        }
    }
    // 写入 is_token_in_rank 矩阵
    for (int j = 0; j < kNumRanksPerSM; ++j)
        is_token_in_rank[i * num_ranks + rank_begin_idx + j] = (is_in_rank[j] > 0);
    // 累计 per-rank 数量
    ...
}
```

### 2.3 关键设计：token-in-rank 语义

`is_token_in_rank[i][j] = true` 的含义是：token i **至少有一个选中的专家**位于 rank j。即使 token i 选择了 rank j 上的多个专家，`is_token_in_rank[i][j]` 也只计为 1（token 只发送一次，包含所有 topk 元数据）。

---

## 3. Python 侧接口

### 3.1 `Buffer.get_dispatch_layout`（`deep_ep/buffer.py`）

```python
def get_dispatch_layout(self,
                        topk_idx: torch.Tensor,    # [num_tokens, num_topk]，dtype=topk_idx_t
                        num_experts: int,
                        previous_event: Optional[EventOverlap] = None,
                        async_finish: bool = False,
                        allocate_on_comm_stream: bool = False
                       ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, EventOverlap]:
    """
    Returns:
        num_tokens_per_rank:      [num_ranks] int
        num_tokens_per_rdma_rank: [num_rdma_ranks] int（节点内时为 None）
        num_tokens_per_expert:    [num_experts] int
        is_token_in_rank:         [num_tokens, num_ranks] bool
        event:                    CUDA event（仅 async_finish=True 时有效）
    """
```

### 3.2 使用模式

```python
# 步骤 1：计算布局（可以与前一步的 CUDA 操作异步）
num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, \
    is_token_in_rank, event = buffer.get_dispatch_layout(
        topk_idx, num_experts,
        previous_event=prev_event,
        async_finish=True,
        allocate_on_comm_stream=True
    )

# 步骤 2：将布局结果传给 dispatch
recv_x, recv_topk_idx, recv_topk_weights, num_recv_per_expert, handle, event = \
    buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=event,
        async_finish=True
    )
```

---

## 4. 性能特性

### 4.1 共享内存使用

每个 SM 分配固定大小的 shared memory：

```cpp
__shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
// = 256 * 4 * 4 bytes = 4 KB（专家统计时）

__shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
// = 256 * 8 * 4 bytes = 8 KB（rank 统计时）
```

### 4.2 与通信 kernel 的关系

```
get_dispatch_layout
       │
       ├── num_tokens_per_rank ──────► notify_dispatch (CPU 侧 MoE 接收计数)
       ├── num_tokens_per_expert ────► notify_dispatch (expert-level 对齐计数)
       ├── is_token_in_rank ─────────► dispatch kernel (per-token 路由决策)
       └── num_tokens_per_rank (前缀和) → channel_prefix_matrix (通道分配)
```

Layout 的结果会被缓存在 `handle` 中，同一批次的反向传播可以通过传入 `handle` 参数跳过重新计算：

```python
# 前向时保存 handle
_, _, _, _, handle, _ = buffer.dispatch(x, ...)

# 反向时复用布局（combine 的 backward 就是 dispatch）
grad_x, _, event = buffer.dispatch(grad_combined_x, handle=handle, ...)
```

---

## 5. 模块依赖关系

```
layout.cu
    │
    ├── configs.cuh    — NUM_MAX_NVL_PEERS、topk_idx_t 类型定义
    ├── exception.cuh  — EP_DEVICE_ASSERT
    └── launch.cuh     — SETUP_LAUNCH_CONFIG、LAUNCH_KERNEL

buffer.py (get_dispatch_layout)
    └── deep_ep_cpp.Buffer.get_dispatch_layout()
            └── layout::get_dispatch_layout()  ← layout.cu
```

Layout 模块是整个通信流水线中唯一不依赖 NVSHMEM 的 kernel，因此在纯 NVLink 节点内场景下也必须存在。
