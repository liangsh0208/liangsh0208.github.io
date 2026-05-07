# 03 Intranode 节点内内核

> 源文件：`csrc/kernels/intranode.cu`、`csrc/kernels/buffer.cuh`、`csrc/kernels/utils.cuh`

---

## 1. 模块作用与设计动机

Intranode 内核处理**单节点内（≤ 8 GPU）的 Expert Parallelism 通信**，全程使用 NVLink 直接访问对端 GPU 内存（CUDA IPC），无需网络介入。

**使用场景**：
- 单机 8 卡 MoE 训练（EP=8）
- 多机训练中，同节点内的 NVLink 分发阶段

**设计目标**：
- 充分利用 NVLink 的 ~160 GB/s 双向带宽
- 支持 BF16 和 FP8 两种数据格式
- 通过 TMA（Tensor Memory Accelerator，Hopper 专有）进一步提升带宽利用率
- 通道化（channeled）发送，让多个 SM 并行工作

---

## 2. 节点内通信协议设计

### 2.1 总体架构：发送 SM 与接收 SM 配对

```
SM 分配规则：
  num_channels = num_sms / 2  （例如 num_sms=24 → 12 channels）
  偶数编号 SM (0, 2, 4, ...) → 发送方（Sender）
  奇数编号 SM (1, 3, 5, ...) → 接收方（Receiver）
  channel_id = sm_id / 2
```

每个 channel 对应 `num_ranks` 个（rank, channel）槽位，由 NVLink 缓冲区的一个分区承载。

### 2.2 共享内存缓冲区布局（每个 NVLink peer 的 buffer_ptrs[rank]）

```
buffer_ptrs[rank] 地址空间：
┌─────────────────────────────────┐ offset 0
│ rank_prefix_matrix              │ kNumRanks × kNumRanks × int
│ (per_rank_buffer)               │ 存储所有 rank-to-rank 的 token 数量前缀和
├─────────────────────────────────┤
│ per_expert_buffer               │ kNumRanks × num_experts_per_rank × int
├─────────────────────────────────┤
│ channel metadata (清零区域)      │
│  ┌── channel_start_offset       │ num_channels × kNumRanks × int
│  ├── channel_end_offset         │ 发送 offset 范围（负值编码区分零 token）
│  ├── channel_head_idx           │ 接收方维护的 head 指针
│  └── channel_tail_idx           │ 发送方维护的 tail 指针
├─────────────────────────────────┤
│ channel_x_buffers               │ 环形数据缓冲区
│ channel_src_idx_buffers         │ 源 token 索引
│ channel_topk_idx_buffers        │ TopK 专家索引（已本地化）
│ channel_topk_weights_buffers    │ TopK 权重
│ channel_x_scales_buffers        │ FP8 缩放因子
└─────────────────────────────────┘
```

### 2.3 队列协议：基于环形缓冲区的生产者-消费者

```
发送方（Sender SM, 偶数 block）：
1. 等待接收方队列有空位：
   num_used = tail_idx - ld_volatile(head_idx)
   while (recv_buffer_tokens - num_used < num_max_send_tokens) { spin-wait }

2. 逐 token 写入：
   slot_idx = (tail_idx++) % num_recv_buffer_tokens
   写入 channel_x_buffers[slot_idx]
   写入 channel_src_idx_buffers[slot_idx]
   写入 channel_topk_idx_buffers[slot_idx]（专家索引本地化：减去 rank 的专家起始偏移）

3. 更新 tail：
   st_release_sys_global(channel_tail_idx, new_tail)
   （release 语义确保数据写入对接收方可见）

接收方（Receiver SM, 奇数 block）：
1. 先等待 channel_start_offset 和 channel_end_offset（负值编码）
   解码得到 [token_start, token_end) 范围和总偏移

2. 循环拉取数据：
   while (tail = ld_acquire_sys_global(channel_tail_idx) == head) { spin-wait }
   UNROLLED_WARP_COPY 或 TMA 复制到最终 recv_x 张量
   更新 head_idx
```

**负值编码（区分零 token）**：
```cpp
// 发送方写入：offset = -value - 1
st_relaxed_sys_global(channel_start_offset, -prefix_sum_start - 1);
// 接收方读取：等待直到值不为 0，然后解码
total_offset = -total_offset - 1;
```
这样即使 prefix sum 为 0，也能区分"尚未写入"（值为 0）和"已写入且值为 0"（写入了 -1）。

---

## 3. notify_dispatch：Token 计数同步

在发送数据之前，必须先让所有 rank 知道它们要接收多少 token（用于 CPU 侧解除阻塞）。

### 3.1 实现逻辑（`intranode.cu`）

```cpp
template <int kNumRanks>
__global__ void notify_dispatch(
    const int* num_tokens_per_rank,
    int* moe_recv_counter_mapped,  // 映射到 CPU 可见的固定内存
    ...) {

    if (sm_id == 0) {
        // 1. 所有 NVLink peer 同步（双次 barrier）
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        // 2. 各 rank 将 num_tokens_per_rank 写入对端缓冲区
        per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];

        // 3. 再次同步，等待所有人写完
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

        // 4. 计算前缀和并通知 CPU（moe_recv_counter_mapped 是固定内存，CPU 轮询）
        local_per_rank_buffer[rank column prefix sum];
        *moe_recv_counter_mapped = prefix_sum[kNumRanks-1][rank];
    } else {
        // 其他 SM 并行计算 channel_prefix_matrix（每个通道发送的 token 数前缀和）
        for (channel_id in channels) {
            count = sum(is_token_in_rank[..][dst_rank] in channel range);
            channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
        }
        // prefix sum 计算
    }
}
```

### 3.2 barrier_block 实现（`utils.cuh`）

```cpp
template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs, int rank) {
    // 利用 atomicAdd_system + atomicSub_system 实现全局屏障：
    // 自增自己的计数器，自减对端计数器
    if (thread_id < kNumRanks) {
        atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);   // 通知对端
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);   // 递减等待计数
    }
    // 等待所有对端都通知本地
    while (!all_sync(barrier_signal_ptrs[rank][thread_id] <= 0))
        { /* spin-wait with timeout */ }
}
```

`FINISHED_SUM_TAG = 1024`，利用符号位避免虚假触发。

---

## 4. dispatch Kernel：数据传输

### 4.1 Kernel 签名

```cpp
template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch(
    int4* recv_x,         float* recv_x_scales,
    int* recv_src_idx,    topk_idx_t* recv_topk_idx,  float* recv_topk_weights,
    int* recv_channel_offset, int* send_head,
    const int4* x,        const float* x_scales,
    const topk_idx_t* topk_idx,  const float* topk_weights,
    const bool* is_token_in_rank, const int* channel_prefix_matrix,
    ...)
```

固定参数：`kNumThreads=768`（24 warps），`kNumTMABytesPerWarp=8192`（SM90 专用 shared memory）

### 4.2 Sender 路径

```cpp
if (is_sender) {
    // 每个 warp 负责一个 responsible_rank 的发送
    // 逐 token 检查 is_token_in_rank
    for (token_idx in [token_start, token_end)) {
        if (!is_token_in_rank[token_idx * kNumRanks + responsible_rank])
            continue;

        // 获取队列槽位
        dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;

        // 复制 hidden 数据（使用 UNROLLED_WARP_COPY，unroll=5，即 5×32 int4/次）
        UNROLLED_WARP_COPY(5, lane_id, hidden_int4,
            channel_x_buffers[dst_slot], x + token_idx * hidden_int4,
            __ldg, st_na_global);

        // 复制 topk 元数据（专家索引本地化）
        idx_value = topk_idx[token_idx * num_topk + lane_id];
        idx_value = (in_range) ? idx_value - recv_expert_begin : -1;  // 本地化
        channel_topk_idx_buffers[dst_slot * num_topk + lane_id] = idx_value;
    }
    // 更新 tail：release 语义保证可见性
    st_release_sys_global(channel_tail_idx, new_tail);
}
```

### 4.3 Receiver 路径（含 TMA 优化）

```cpp
if (!is_sender) {
    // 等待 channel_start/end_offset
    while (total_offset == 0) { spin-wait; }

    while (num_tokens_to_recv > 0) {
        // 等待 tail 推进
        while (head == tail) { spin-wait; }

#ifndef DISABLE_SM90_FEATURES
        // Hopper TMA 路径：利用 mbarrier 异步加载
        for (int i = 0; i < 2; ++i) {
            tma_store_wait<0>();
            tma_load_1d(tma_buffer, src + i*half_hidden, mbarrier, half_hidden_bytes);
            mbarrier_arrive_and_expect_tx(mbarrier, half_hidden_bytes);
            mbarrier_wait(mbarrier, phase);
            tma_store_1d(tma_buffer, dst + i*half_hidden, half_hidden_bytes);
        }
#else
        // Ampere 路径：ld_nc_global + st_na_global
        UNROLLED_WARP_COPY(5, lane_id, hidden_int4, recv_x_dst, buffer_src, ld_nc_global, st_na_global);
#endif
        // 更新 head
        st_relaxed_sys_global(channel_head_idx, new_head);
    }
}
```

**TMA 优化说明**：
- Hopper（SM90）上，TMA（Tensor Memory Accelerator）可异步地将 global memory 数据搬入 shared memory，不占用 CUDA core cycles
- 将 hidden 分为两半（`half_hidden_int4`），交替使用 `tma_load_1d` + `tma_store_1d` 实现流水线

---

## 5. combine Kernel：逆向聚合

Combine 是 dispatch 的逆操作，将专家计算结果发回原始 token 所在 rank。

### 5.1 调用关系

```
combine()   →   cached_notify_combine()   +   intranode::combine()
                （通知发送计划）                 （执行数据传输）
```

### 5.2 关键区别（相对 dispatch）

```python
# combine 中，handle 的矩阵角色互换
rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
#                   ↑ 注意：dispatch 保存的 recv_channel_prefix 在 combine 中不用
```

Combine 的 sender 和 receiver 逻辑与 dispatch 对称，但：
- sender 发送专家输出（而非输入 token）
- receiver 对接收到的所有专家输出加权求和（`topk_weights`）

### 5.3 bias 支持

```cpp
// combine 支持最多 2 个 bias 向量（broadcast 到所有接收 token）
void combine(... void* bias_0, void* bias_1, ...)
```

这允许将 LayerNorm bias 或其他 residual connection 融合进 combine 操作。

---

## 6. 关键 PTX 技巧汇总（`utils.cuh`）

### 6.1 UNROLLED_WARP_COPY 宏

```cpp
#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)
// 展开因子 5 时：每次循环处理 5 × 32 = 160 个 int4（=2560 字节）
// 最后处理剩余不足 160 的部分（带边界检查）
```

### 6.2 ld_nc_global：绕过 L1 缓存

```cpp
// 适用于只读一次的通信数据，避免污染 L1
template <> __device__ int4 ld_nc_global(const int4* ptr) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0,%1,%2,%3}, [%4];");
}
```

### 6.3 st_na_global：跳过 L1 写入

```cpp
// 写入目标不需要被当前 SM 后续读取，使用 L1::no_allocate 减少缓存抖动
template <> __device__ void st_na_global(const int4* ptr, const int4& value) {
    asm volatile("st.global.L1::no_allocate.v4.s32 [%0], {%1,%2,%3,%4};");
}
```

---

## 7. 模块依赖关系

```
intranode.cu
    ├── buffer.cuh      — Buffer<T>、AsymBuffer<T>、SymBuffer<T> 模板
    ├── utils.cuh       — barrier_block、UNROLLED_WARP_COPY、ld_nc_global、TMA 原语
    ├── launch.cuh      — SETUP_LAUNCH_CONFIG、LAUNCH_KERNEL、SWITCH_RANKS 宏
    ├── configs.cuh     — NUM_MAX_NVL_PEERS、topk_idx_t
    └── exception.cuh   — EP_DEVICE_ASSERT、EP_HOST_ASSERT

调用链：
  deep_ep.cpp::Buffer::intranode_dispatch()
      ├── intranode::notify_dispatch()   ← 通知 CPU token 计数
      └── intranode::dispatch()          ← 实际数据传输
  deep_ep.cpp::Buffer::intranode_combine()
      ├── intranode::cached_notify_combine()
      └── intranode::combine()
```
