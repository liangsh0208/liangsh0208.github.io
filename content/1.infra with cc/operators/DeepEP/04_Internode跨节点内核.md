---
created: 2026-05-06
---

# 04 Internode 跨节点内核

> 源文件：`csrc/kernels/internode.cu`、`csrc/kernels/ibgda_device.cuh`、`csrc/kernels/runtime.cu`

---

## 1. 模块作用与设计动机

Internode 内核处理**多机多 GPU 的 Expert Parallelism 通信**（典型场景：DeepSeek-V3 训练中 EP=16/32/64）。

**拓扑约束**：
- 每个节点内 8 个 GPU 通过 NVLink 全互联（与节点内通信相同）
- 不同节点间的相同 GPU 索引（如每台机器的 GPU0）通过 RDMA（InfiniBand IBGDA）互联
- num_ranks 必须是 8 的倍数（即 `num_ranks = num_rdma_ranks × 8`）

```
节点 0: GPU0(rank 0)  --- NVLink --- GPU1(rank 1)  ... GPU7(rank 7)
                                              |
                                           RDMA
                                              |
节点 1: GPU0(rank 8)  --- NVLink --- GPU1(rank 9)  ... GPU7(rank 15)
```

**关键设计**：跨节点通信采用"先 RDMA 到目标节点的相同 GPU index，再 NVLink 转发到正确 GPU"的两段式路由。

---

## 2. 关键数据结构

### 2.1 SourceMeta：源地址元数据

```cpp
struct SourceMeta {
    int src_rdma_rank;              // 来自哪个 RDMA rank（节点）
    int is_token_in_nvl_rank_bits;  // 位图：哪些节点内 GPU 需要这个 token
    
    __device__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = 0;
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }
    
    __device__ bool is_token_in_nvl_rank(int nvl_rank) const {
        return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
    }
};
```

`SourceMeta` 随数据一起通过 RDMA 传输，接收节点的 GPU 读取它后知道：
1. 数据来自哪个 RDMA rank
2. 需要继续转发给本节点的哪些 GPU

### 2.2 SymBuffer：对称缓冲区（节点内）

```cpp
// NVLink 侧使用 SymBuffer，send 和 recv 是分开的区域
template <typename dtype_t, bool kDecoupled = true>
struct SymBuffer {
    uint8_t* send_ptr;  // 发送区
    uint8_t* recv_ptr;  // 接收区（不同 SM 偶/奇 channel 交替使用）
    // send/recv 各占 num_channels * num_ranks * num_recv_buffer_tokens * elem_bytes
};
```

### 2.3 路由矩阵体系

跨节点场景需要两套前缀矩阵：

| 矩阵 | 形状 | 含义 |
|------|------|------|
| `rdma_channel_prefix_matrix` | `[num_channels, num_rdma_ranks]` | 每个 channel 向各 RDMA rank 发送的 token 前缀和 |
| `gbl_channel_prefix_matrix` | `[num_channels, num_ranks]` | 每个 channel 向各全局 rank 发送的 token 前缀和 |
| `recv_rdma_rank_prefix_sum` | `[num_rdma_ranks]` | 从各 RDMA rank 接收到的 token 前缀和（接收方视角） |
| `recv_gbl_rank_prefix_sum` | `[num_ranks]` | 从各全局 rank 接收到的 token 前缀和 |

---

## 3. notify_dispatch：跨节点计数同步

### 3.1 IBGDA 初始化等待

在开始实际通信之前，必须等待上一轮飞行中的 RDMA 写操作全部完成：

```cpp
// 等待所有 QP（Queue Pair）上的 WR（Work Request）完成
auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
for (int i = thread_id; i < qps_per_rdma_rank * (kNumRDMARanks - 1); i += num_threads) {
    auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % kNumRDMARanks;
    nvshmemi_ibgda_quiet(translate_dst_rdma_rank(dst_rdma_rank, nvl_rank), qp_id);
}
```

### 3.2 双层同步：NVLink 内同步 + RDMA 外同步

```cpp
// Warp 0：节点内 NVLink 同步
barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

// Warp 1（thread 32）：跨节点 RDMA 同步
if (thread_id == 32)
    nvshmem_sync_with_same_gpu_idx(rdma_team);
    // NOTES: kLowLatencyMode=false 时用 nvshmem_sync_all()
    //        kLowLatencyMode=true 时用 nvshmem_sync(cpu_rdma_team)
```

### 3.3 发送 token 计数到所有 RDMA rank

```cpp
// 使用 SymBuffer 分配发送/接收空间
auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS + num_rdma_experts + 1, kNumRDMARanks);

// 填充发送数据（本地视角的 num_tokens_per_rank 和 num_tokens_per_expert）
for (int i = 0; i < num_ranks; ++i)
    rdma_recv_num_tokens_mixed.send_buffer(i / NUM_MAX_NVL_PEERS)[i % 8] = num_tokens_per_rank[i];

// 通过 IBGDA PUT 发送到各 RDMA rank
for (int i = warp_id; i < kNumRDMARanks; i += num_warps) {
    if (i != rdma_rank) {
        nvshmemi_ibgda_put_nbi_warp<true>(
            dst_ptr_on_remote,      // 对端的接收地址
            local_send_buffer,      // 本地发送数据
            msg_size,
            translate_dst_rdma_rank(i, nvl_rank),  // 目标 PE
            qp_id, lane_id, slot_idx);
    }
}
```

---

## 4. dispatch Kernel：两段式数据传输

### 4.1 整体流程

```
本地 token → RDMA send → 目标节点相同 GPU index → NVLink 转发 → 目标 GPU
```

实际上，每个 GPU 同时承担：
- **发送者**（向其他节点的 GPU0/GPU1/...发送本地 token）
- **中继者**（接收来自其他节点的 token，通过 NVLink 转发给节点内其他 GPU）
- **接收者**（接收节点内其他 GPU 转发过来的 token）

### 4.2 RDMA 发送路径（SM 工作）

```cpp
// 每个 SM 负责向一个 RDMA rank 发送
// token 先进入 RDMA 缓冲区（打包 SourceMeta + hidden + scales）
void* rdma_send_area = rdma_buffer_ptr + channel_offset;

for (token in channel_range) {
    // 1. 构造完整消息：SourceMeta + hidden_data + topk_meta + scales
    SourceMeta meta(rdma_rank, is_token_in_nvl_ranks_for_this_token);
    memcpy(msg.meta, &meta, sizeof(SourceMeta));
    memcpy(msg.data, x + token * hidden_int4, hidden_bytes);

    // 2. RDMA PUT（非阻塞）
    nvshmemi_ibgda_put_nbi_warp(...);

    // 3. 发送完后更新 tail 信号
}
```

### 4.3 NVLink 转发路径（接收 RDMA 后分发）

当 RDMA 数据到达目标节点时，需要根据 `SourceMeta.is_token_in_nvl_rank_bits` 将数据复制给节点内对应的 GPU：

```cpp
// 读取接收到的 SourceMeta
auto meta = recv_src_meta_ptr[token_idx];

// 解包：决定转发给哪些节点内 GPU
for (int nvl_rank = 0; nvl_rank < NUM_MAX_NVL_PEERS; ++nvl_rank) {
    if (meta.is_token_in_nvl_rank(nvl_rank)) {
        // 通过 NVLink 写入目标 GPU 的接收缓冲区
        // (buffer_ptrs[nvl_rank] 是对端 GPU 内存的直接映射)
        memcpy(buffer_ptrs[nvl_rank] + nvl_recv_offset, token_data, hidden_bytes);
    }
}
```

### 4.4 Cached Dispatch 模式

当同一 batch 需要多次 dispatch（如反向传播复用 handle），第二次可跳过 notify 阶段：

```python
# 第一次：完整计算路由信息
recv_x, ..., handle, event = buffer.dispatch(x, topk_idx=..., ...)

# 第二次：复用 handle（跳过 get_dispatch_layout 和 notify）
recv_x, _, _, _, _, event = buffer.dispatch(grad_x, handle=handle, ...)
```

对应 C++ 端调用 `internode::cached_notify(...)` 而非完整的 `notify_dispatch`。

---

## 5. IBGDA 设备端封装（`ibgda_device.cuh`）

IBGDA（InfiniBand GPU Direct Async）允许 GPU 线程直接操作 InfiniBand NIC 的 QP（Queue Pair），无需 CPU 介入。

### 5.1 核心数据结构

```cpp
// 从 NVSHMEM 获取 IBGDA 状态（包含所有 QP 的门铃地址）
__device__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;  // 设备端全局变量
}

// 获取特定 PE 的 RC（Reliable Connection）QP
__device__ nvshmemi_ibgda_device_qp_t* ibgda_get_rc(int pe, int id) {
    return ibgda_get_rc_impl(ibgda_get_state(), pe, id);
}
```

### 5.2 RDMA WRITE 操作

```cpp
// Warp 协作 RDMA PUT（非阻塞）
template <bool kSignalOnCompletion>
__device__ void nvshmemi_ibgda_put_nbi_warp(
    uint64_t dst_ptr,   // 目标 PE 上的虚拟地址
    uint64_t src_ptr,   // 本地源数据地址
    size_t nbytes,      // 数据大小
    int pe,             // 目标 PE（GPU rank）
    int qp_id,          // 选择哪个 QP
    int lane_id,        // warp lane 编号
    int slot_idx        // WQ 槽位索引
) {
    // 分解为 MLX5 WQE（Work Queue Entry）格式
    // Lane 0 写 ctrl_seg（控制段）
    // 其他 lane 写 data_seg（数据段，最多 4 个 SGE）
    // 最后 ring doorbell（写 NIC 门铃寄存器触发传输）
}
```

### 5.3 RDMA 原子操作（用于 combine 信号）

```cpp
// 原子 fetch-and-add（用于更新 combine 接收计数器）
__device__ void nvshmemi_ibgda_amo_nonfetch_add(
    int* dst_ptr, int value, int pe, int qp_id) {
    // 构造 MLX5 FA（Fetch-and-Add）WQE
    // 直接写 NIC 的 SQ（Send Queue）并敲门铃
}
```

### 5.4 P2P（NVLink）vs RDMA 动态选择

同节点内的 RDMA rank 实际上通过 NVLink 连接更快，IBGDA 会自动检测并使用 P2P：

```cpp
// 检查是否有 NVLink 直连路径
auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, src_rank, dst_rank);

if (dst_p2p_ptr == 0) {
    // 使用 RDMA 传输
    nvshmemi_ibgda_put_nbi_warp(...);
} else {
    // 使用 NVLink 直接写（更快）
    UNROLLED_WARP_COPY(8, lane_id, num_int4, dst_p2p_ptr, src, ld_nc_global, st_na_global);
}
```

---

## 6. 网络配置建议

DeepEP 从 README 提供的生产配置：

### 6.1 Virtual Lane 隔离

通过 InfiniBand Virtual Lane（VL）隔离不同流量：
- Normal 内核流量：VL 0
- Low-Latency 内核流量：VL 1
- 其他系统流量：VL 2+

```bash
# 控制 NVSHMEM 使用的 VL
export NVSHMEM_IB_SL=0  # Normal 内核
export NVSHMEM_IB_SL=1  # Low-Latency 内核
```

### 6.2 自适应路由

- 重负载下启用 Adaptive Routing（均匀分配路径，消除拥塞）
- 轻负载下使用 Static Routing（减少额外延迟）

---

## 7. 模块依赖关系

```
internode.cu
    ├── buffer.cuh       — SymBuffer<T>、Buffer<T>
    ├── ibgda_device.cuh — IBGDA QP 操作、nvshmemi_ibgda_put_nbi_warp
    ├── utils.cuh        — barrier_block（NVLink 同步）、ld_nc_global
    ├── launch.cuh       — SWITCH_RDMA_RANKS、SWITCH_HIDDEN
    └── configs.cuh      — NUM_MAX_NVL_PEERS、NUM_MAX_RDMA_PEERS

runtime.cu
    └── internode::init()   ← 调用 nvshmem_init，创建 cpu_rdma_team

调用链：
  deep_ep.cpp::Buffer::internode_dispatch()
      ├── internode::notify_dispatch()   ← 双层同步 + token 计数 RDMA 发送
      └── internode::dispatch()          ← 实际双段数据传输
  deep_ep.cpp::Buffer::internode_combine()
      ├── internode::cached_notify()
      └── internode::combine()
```
