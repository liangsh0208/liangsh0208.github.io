---
created: 2026-05-06
---

# 01 Buffer 通信缓冲区

> 源文件：`deep_ep/buffer.py`、`csrc/deep_ep.hpp`、`csrc/deep_ep.cpp`、`csrc/config.hpp`、`csrc/event.hpp`

---

## 1. 模块作用与设计动机

`Buffer` 是 DeepEP 的核心抽象，封装了 MoE Expert Parallelism 所需的全部通信能力。它的职责包括：

1. **内存管理**：分配并注册 NVLink（IPC）缓冲区和 RDMA（NVSHMEM 对称内存）缓冲区
2. **初始化协调**：通过 `all_gather_object` 交换 IPC handle 和 NVSHMEM unique ID，完成跨 GPU 的内存映射建立
3. **通信调度**：向上层提供 `dispatch`、`combine`、`low_latency_dispatch`、`low_latency_combine` 等高层接口
4. **配置管理**：管理 SM 数量、chunked 传输参数（`Config`）和低延迟模式缓冲区布局（`LowLatencyLayout`）

### 设计核心：两种缓冲区并存

```
节点内通信 (NVLink)
  GPU0 ─── buffer_ptrs[0]  (cudaIpcMemHandle 或 CUmemFabricHandle)
  GPU1 ─── buffer_ptrs[1]
  ...
  GPU7 ─── buffer_ptrs[7]
  每个 GPU 可以直接访问其他 GPU 的本地缓冲区指针

跨节点通信 (RDMA)
  rdma_buffer_ptr  (nvshmem_align 分配的对称内存)
  所有节点上同 GPU index 的 rank 共享同一逻辑地址空间
```

---

## 2. 关键数据结构

### 2.1 C++ Buffer 结构（`csrc/deep_ep.hpp`）

```cpp
struct Buffer {
private:
    // 低延迟模式双缓冲 index
    int low_latency_buffer_idx = 0;
    bool low_latency_mode = false;

    // NVLink 缓冲区：最多 8 个 NVLink 对等 GPU
    int64_t num_nvl_bytes;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};  // 8 个槽位
    void** buffer_ptrs_gpu = nullptr;                   // GPU 端可访问的指针数组

    // NVSHMEM 对称内存（RDMA）
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    // 节点拓扑信息
    int rank, rdma_rank, nvl_rank;
    int num_ranks, num_rdma_ranks, num_nvl_ranks;

    // 通信专用 CUDA 流
    at::cuda::CUDAStream comm_stream;

    // CPU-GPU 握手计数器（固定内存，供 GPU 写入通知 CPU）
    volatile int* moe_recv_counter = nullptr;          // 接收 token 总数
    volatile int* moe_recv_expert_counter = nullptr;   // 每个本地专家接收数
    volatile int* moe_recv_rdma_counter = nullptr;     // RDMA rank 接收数
    // ...
};
```

### 2.2 Config 结构（`csrc/config.hpp`）

`Config` 控制 Normal 内核的分块（chunked）传输参数：

```cpp
struct Config {
    int num_sms;                         // 参与通信的 SM 数量
    int num_max_nvl_chunked_send_tokens; // 每次 NVLink 发送的最大 token 数
    int num_max_nvl_chunked_recv_tokens; // NVLink 接收缓冲区大小（必须 > send）
    int num_max_rdma_chunked_send_tokens; // 每次 RDMA 发送的最大 token 数
    int num_max_rdma_chunked_recv_tokens; // RDMA 接收缓冲区大小
};
```

**约束设计**：
- `send < recv`：确保发送方有充裕的缓冲空间，避免死锁
- `send <= recv / 2`：与 RDMA lazy head update 机制匹配

**默认配置（`buffer.py`，以 32 EP 为例）**：

```python
# Dispatch：Config(num_sms=24, nvl_send=32, nvl_recv=288, rdma_send=8, rdma_recv=128)
Config(Buffer.num_sms, 32, 288, 8, 128)
# Combine：Config(num_sms=24, nvl_send=1, nvl_recv=288, rdma_send=8, rdma_recv=128)
Config(Buffer.num_sms, 1, 288, 8, 128)
```

Combine 的 NVLink send 只需 1，因为 combine 的方向和 dispatch 相反，NVLink 带宽需求不同。

### 2.3 LowLatencyLayout 结构（`csrc/config.hpp`）

低延迟模式使用固定大小的双缓冲（Ping-Pong），布局由 `LowLatencyLayout` 在运行时计算：

```cpp
struct LowLatencyLayout {
    size_t total_bytes = 0;
    LowLatencyBuffer buffers[2];  // 奇偶双缓冲

    LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank,
                     int hidden, int num_ranks, int num_experts) {
        // 每条消息格式：
        // dispatch msg = int4(元数据) + max(hidden * bf16, hidden * fp8 + scales)
        // combine msg  = scales(per-128-channel) + hidden * bf16
        size_t num_bytes_per_dispatch_msg = sizeof(int4) + max(hidden * 2, hidden + scales * 4);
        size_t num_bytes_per_combine_msg  = scales * sizeof(nv_bfloat162) + hidden * sizeof(nv_bfloat16);

        // 总布局：signaling * 2 + send * 2 + recv * 2
    }
};
```

**RDMA 内存布局**（按地址顺序）：
```
[signaling_0][signaling_1][send_0][send_1][recv_0][recv_1]
```

- `signaling`（int 数组）：每个专家的接收计数器，GPU-to-GPU 信号
- `send`：本地构造消息的暂存区，由 GPU 填写后 RDMA put 到远端
- `recv`：远端写入的接收区（NVSHMEM 对称，对端可直接写入）

### 2.4 EventOverlap / EventHandle（`deep_ep/utils.py`、`csrc/event.hpp`）

```python
class EventOverlap:
    event: Optional[EventHandle]      # C++ CUDA event 包装
    extra_tensors: Optional[tuple]    # 防止张量在通信完成前被回收
```

```cpp
struct EventHandle {
    std::shared_ptr<torch::Event> event;
    EventHandle() {
        event = make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream());  // 在当前流上记录
    }
    void current_stream_wait() const {
        at::cuda::getCurrentCUDAStream().unwrap().wait(*event);  // 让当前流等待
    }
};
```

`EventOverlap` 支持 Python `with` 语法，方便实现异步通信-计算重叠：

```python
with event_after_dispatch:         # 在 with 块内做计算
    do_gemm_on_compute_stream()
# 退出 with 时，当前流自动等待 event，确保 dispatch 完成
```

---

## 3. Buffer 生命周期

### 3.1 初始化流程（Python 侧）

```python
# 1. 构造 C++ Buffer（分配内存）
self.runtime = deep_ep_cpp.Buffer(rank, group_size, num_nvl_bytes, num_rdma_bytes, ...)

# 2. 通过 all_gather_object 交换 IPC handle（实现 NVLink 跨进程映射）
local_device_id = self.runtime.get_local_device_id()
device_ids = all_gather_object(local_device_id)

local_ipc_handle = self.runtime.get_local_ipc_handle()
ipc_handles = all_gather_object(local_ipc_handle)

# 3. 如果需要 RDMA，交换 NVSHMEM unique ID 并初始化
if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
    root_unique_id = self.runtime.get_local_nvshmem_unique_id()  # 仅 rank 0 有效
    nvshmem_unique_ids = all_gather_object(root_unique_id)
    root_unique_id = nvshmem_unique_ids[root_rdma_rank]

# 4. 同步完成，Buffer 可用
self.runtime.sync(device_ids, ipc_handles, root_unique_id)
```

### 3.2 C++ sync 流程（`csrc/deep_ep.cpp`）

```cpp
void Buffer::sync(...) {
    // 打开 NVLink IPC handle，映射其他 GPU 的内存
    for (int i = 0; i < num_nvl_ranks; ++i) {
        if (i != nvl_rank)
            shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
    }
    // 复制指针数组到 GPU
    cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * 8, cudaMemcpyHostToDevice);

    // NVSHMEM 初始化（RDMA）
    if (root_unique_id.has_value())
        rdma_rank = internode::init(root_unique_id_val, rank, num_ranks, low_latency_mode);

    // 分配 NVSHMEM 对称内存
    rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

    available = true;
}
```

### 3.3 销毁流程

资源释放顺序必须严格遵循：
1. `cudaDeviceSynchronize()`：等待所有 GPU 操作完成
2. `intranode::barrier()`：NVLink 节点内同步
3. 关闭所有 IPC handle（`cudaIpcCloseMemHandle`）
4. `internode::barrier()`：RDMA 全局同步
5. `nvshmem_free(rdma_buffer_ptr)`
6. `nvshmem_finalize()`
7. 释放 workspace、moe_recv_counter 等固定内存

---

## 4. 关键接口说明

### 4.1 缓冲区大小计算

```python
# 获取建议的缓冲区大小
config = Buffer.get_dispatch_config(group_size)
num_nvl_bytes = config.get_nvl_buffer_size_hint(hidden_bytes, group_size)
num_rdma_bytes = config.get_rdma_buffer_size_hint(hidden_bytes, group_size)
```

`get_nvl_buffer_size_hint` 内部计算（`config.hpp`）：

```cpp
size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
    const int num_channels = num_sms / 2;
    size_t num_bytes = 0;
    // 每个通道、每个 rank 的接收缓冲区
    num_bytes += num_channels * num_nvl_ranks * (2*num_rdma_ranks + 3) * sizeof(int);     // 队列元数据
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * hidden_bytes; // 数据
    num_bytes += /* topk_idx、topk_weights、scales */ ...;
    return align_up(num_bytes, 128);
}
```

### 4.2 SM 数量控制

```python
Buffer.set_num_sms(24)  # 全局静态变量，影响所有后续 Config 实例
```

推荐值：20~24。留出足够 SM 给 GEMM 计算，实现通信-计算重叠。

### 4.3 低延迟 RDMA 大小计算

```python
num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank,  # 每个 rank 最多发送的 token 数（≤ 256）
    hidden,                            # 隐藏维度
    num_ranks,                         # EP 规模
    num_experts                        # 专家总数
)
```

低延迟模式的 RDMA 用量远大于 Normal 模式，因为需要为每个 expert × rank 预分配固定大小的槽位。

---

## 5. 模块间依赖关系

```
buffer.py (Python Buffer 类)
    │
    ├── deep_ep_cpp.Buffer        ← csrc/deep_ep.cpp 实现
    │       ├── intranode::*      ← csrc/kernels/intranode.cu
    │       ├── internode::*      ← csrc/kernels/internode.cu
    │       ├── internode_ll::*   ← csrc/kernels/internode_ll.cu
    │       └── layout::*         ← csrc/kernels/layout.cu
    │
    ├── Config                    ← csrc/config.hpp
    ├── EventOverlap              ← deep_ep/utils.py
    └── EventHandle               ← csrc/event.hpp
```

`Buffer` 是所有通信操作的入口，所有 CUDA kernel 通过 `runtime` 句柄间接调用，Python 层只与 `Buffer`、`Config`、`EventOverlap` 三个类交互。
