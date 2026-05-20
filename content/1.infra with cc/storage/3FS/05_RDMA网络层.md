---
created: 2026-05-06
---

# RDMA 网络层：IBSocket 与零拷贝传输

> 源码路径：`src/common/net/`  
> 关键文件：`ib/IBSocket.h`、`ib/RDMABuf.h`、`ib/IBDevice.h`、`ib/IBConnect.h`、`Transport.h`、`Server.h`

---

## 1. RDMA 网络层的作用与设计动机

3FS 所有核心数据传输都走 **RDMA（Remote Direct Memory Access）** 网络，支持 InfiniBand 和 RoCE（RDMA over Converged Ethernet）两种硬件。

**为什么选择 RDMA？**

传统 TCP/IP 数据传输路径：
```
用户空间 → 内核 → NIC 驱动 → 网络 → NIC 驱动 → 内核 → 用户空间
（两次内存拷贝 + 两次上下文切换）
```

RDMA 数据传输路径：
```
注册内存 → NIC 直接 DMA（绕过 CPU）→ 对端注册内存
（零拷贝 + 零 CPU 干预 + 微秒级延迟）
```

RDMA Read 是 3FS 写入流程的核心：存储节点通过 RDMA Read 直接从客户端内存拉取待写入数据，无需客户端 CPU 参与，极大降低客户端负载并提升存储节点的处理吞吐量。

---

## 2. IBSocket：InfiniBand 连接管理

`IBSocket` 是 3FS 网络层的核心类，封装了一个 IB 队列对（QP, Queue Pair）的完整生命周期：

```cpp
// src/common/net/ib/IBSocket.h
class IBSocket : public Socket, folly::MoveOnly {
 public:
  struct Config : public ConfigBase<Config> {
    // QP 参数
    CONFIG_HOT_UPDATED_ITEM(start_psn, (uint32_t)0);         // 起始包序列号
    CONFIG_HOT_UPDATED_ITEM(timeout, (uint8_t)14);            // 超时（14 ≈ 4.096ms * 2^14 = ~67s）
    CONFIG_HOT_UPDATED_ITEM(retry_cnt, (uint8_t)7);           // 重试次数
    CONFIG_HOT_UPDATED_ITEM(rnr_retry, (uint8_t)0);           // RNR 重试次数
    CONFIG_HOT_UPDATED_ITEM(max_rdma_wr, 128u);               // Send Queue 深度
    CONFIG_HOT_UPDATED_ITEM(max_rdma_wr_per_post, 32u);       // 单次 ibv_post_send 的最大 WR 数
    CONFIG_HOT_UPDATED_ITEM(max_rd_atomic, 16u);              // 未完成 RDMA Read 的最大数量

    // 收发缓冲区（用于 RPC 消息，非数据传输）
    CONFIG_HOT_UPDATED_ITEM(buf_size, (16u * 1024));          // 单个缓冲区 16KB
    CONFIG_HOT_UPDATED_ITEM(send_buf_cnt, 32u);               // 发送缓冲区数量
    CONFIG_HOT_UPDATED_ITEM(buf_ack_batch, 8u);               // ACK 批处理阈值
    CONFIG_HOT_UPDATED_ITEM(buf_signal_batch, 8u);            // 信号批处理阈值
  };

 private:
  // IB 硬件资源（每个 IBSocket 独占一个 QP）
  IBPort port_;
  std::unique_ptr<ibv_comp_channel, IBDevice::Deleter> channel_;  // 完成事件通道
  std::unique_ptr<ibv_cq, IBDevice::Deleter> cq_;                 // 完成队列（CQ）
  std::unique_ptr<ibv_qp, IBDevice::Deleter> qp_;                 // 队列对（QP）

  // 状态机
  std::atomic<State> state_ = State::INIT;  // INIT/CONNECTING/ACCEPTED/READY/CLOSE/ERROR
};
```

### 2.1 QP 状态机（IB 标准流程）

```
INIT → READY_TO_RECV（RTR）→ READY_TO_SEND（RTS）→ ERROR/CLOSE
         ↑
    qpInit() / qpReadyToRecv() / qpReadyToSend()
    通过 TCP 握手交换 QP 属性（GID、LID、QPN、PSN）
```

连接建立使用 TCP 协议交换 `IBConnectInfo`（包含 QPN、GID、LID 等），完成后切换到纯 RDMA 通信。

### 2.2 RDMA 操作

```cpp
// 单次 RDMA Read：将远端内存读入本地
CoTryTask<void> rdmaRead(const RDMARemoteBuf &remoteBuf, RDMABuf &localBuf);

// 批量 RDMA Read（减少 post_send 系统调用次数）
RDMAReqBatch batch = socket.rdmaReadBatch();
batch.add(remoteBuf1, localBuf1);
batch.add(remoteBuf2, localBuf2);
co_await batch.post();  // 一次 ibv_post_send 提交多个 WR
```

`RDMAReqBatch` 的 WR（Work Request）结构：

```cpp
struct RDMAReq {
    uint64_t raddr;       // 远端虚拟地址
    uint32_t rkey;        // 远端内存注册 key
    uint32_t localBufFirst;  // 本地缓冲区起始索引
    uint32_t localBufCnt;   // 本地分散缓冲区数量（支持 scatter-gather）
};
```

### 2.3 WR ID 编码（Work Request Identifier）

```cpp
// 利用 64-bit WR ID 的高 16 位编码请求类型，低 48 位编码数据
enum class WRType : uint16_t {
    SEND,      // 普通消息发送
    RECV,      // 接收缓冲区
    ACK,       // 接收窗口 ACK（滑动窗口流控）
    RDMA,      // RDMA 操作（中间）
    RDMA_LAST, // RDMA 批次最后一个 WR
    CLOSE,     // 关闭连接
    CHECK,     // 连接活性探测
};
```

这种编码允许在 CQ Poll 时直接从 WR ID 判断操作类型，无需额外的 context 查找。

---

## 3. RDMABuf 与 RDMARemoteBuf：内存管理

### 3.1 RDMABuf（本地注册内存）

```cpp
// src/common/net/ib/RDMABuf.h
class RDMABuf {
  // 每个 IBDevice 对应一个 ibv_mr（Memory Region）
  class Inner {
    uint8_t *ptr_;      // 内存起始地址
    size_t capacity_;   // 分配的总大小
    std::array<RDMABufMR, IBDevice::kMaxDeviceCnt> mrs_;  // 多 IB 卡支持
    bool userBuffer_;   // 是否为用户提供的外部内存（无需内部分配）
  };

  // 引用计数，支持子范围共享
  std::shared_ptr<Inner> buf_;
  uint8_t *begin_;   // 当前有效范围起始
  size_t length_;    // 当前有效范围长度

 public:
  // 子范围操作（零拷贝切片）
  RDMABuf subrange(size_t offset, size_t length) const;
  RDMABuf takeFirst(size_t length);  // 取前 n 字节并 advance

  // 获取 RDMA 远端描述符（传递给对端用于 RDMA Read/Write）
  RDMARemoteBuf toRemoteBuf() const;
};
```

`RDMABuf` 的关键特性：支持零拷贝切片，多个 `RDMABuf` 可以引用同一内存区域的不同子范围，均由 `shared_ptr<Inner>` 保证生命周期安全。

### 3.2 RDMARemoteBuf（远端内存描述符）

```cpp
class RDMARemoteBuf {
  uint64_t addr_;   // 远端虚拟地址
  uint64_t length_; // 长度
  std::array<Rkey, IBDevice::kMaxDeviceCnt> rkeys_;  // 每个 IB 设备对应的 rkey

 public:
  // 获取特定设备的 rkey（多网卡场景）
  std::optional<uint32_t> getRkey(int devId) const;

  // 子范围操作
  RDMARemoteBuf subrange(size_t offset, size_t len) const;
};
```

**多网卡支持**：同一内存区域注册到多个 IB 设备，客户端将所有 (rkey, devId) 对都包含在 `RDMARemoteBuf` 中发送给存储节点。存储节点选择本地 IB 设备对应的 rkey 进行 RDMA Read，实现**自动多路径选择**。

---

## 4. RDMABufPool：缓冲区复用

RDMA 内存注册（ibv_reg_mr）是开销较大的操作，需要 pin 物理内存页。`RDMABufPool` 维护预注册的缓冲区池：

```cpp
class RDMABufPool : public std::enable_shared_from_this<RDMABufPool> {
  size_t bufSize_;                      // 单个缓冲区大小
  folly::fibers::Semaphore sem_;        // 信号量控制并发分配数量
  std::mutex mutex_;
  std::deque<RDMABuf::Inner *> freeList_;  // 空闲缓冲区链表

 public:
  CoTask<RDMABuf> allocate(optional<Duration> timeout = nullopt);
  void deallocate(RDMABuf::Inner *buf);
};
```

`StorageOperator` 的 `rdmabufPool` 预先创建固定数量的大缓冲区，存储节点在处理读请求时从 pool 借用缓冲区，完成后归还，避免反复注册/注销内存区域。

---

## 5. Transport 层与 IO Worker

RDMA 数据传输由 `Transport` 封装，`IOWorker` 管理 epoll 事件循环：

```cpp
// src/common/net/Transport.h
class Transport {
  // 封装 IBSocket + TCP Socket 的统一接口
  // 负责：消息序列化/反序列化（Serde）、超时管理、重试
};

// src/common/net/IOWorker.h
class IOWorker {
  // 事件驱动 IO 循环
  // - epoll 监听 CQ 文件描述符（IB 完成事件通知）
  // - 分发 WC（Work Completion）到对应的 IBSocket
  // - 触发协程恢复（baton.post()）
};
```

协程与 RDMA 的集成：

```cpp
// IBSocket 内部：RDMA Read 完成后的处理流程
int onRDMAFinished(const ibv_wc &wc, Events &events) {
    auto ctx = WRId(wc.wr_id).rdmaPostCtx();  // 从 WR ID 找到等待的协程
    if (wc.status != IBV_WC_SUCCESS) {
        ctx->setError(wc.status);
    }
    ctx->finish();  // baton.post()，唤醒等待该 RDMA 操作完成的协程
    return 0;
}
```

---

## 6. IBDevice：多 IB 设备管理

```cpp
// src/common/net/ib/IBDevice.h
class IBDevice {
 public:
  static constexpr int kMaxDeviceCnt = 8;  // 最多支持 8 张 IB/RoCE 网卡

  // 全局单例，枚举系统中所有 IB 设备
  static const std::vector<std::shared_ptr<IBDevice>> &all();

  // 内存注册
  ibv_mr *regMr(void *ptr, size_t length, int flags);
  void deregMr(ibv_mr *mr);

  // 完成队列创建（每个 QP 一个 CQ）
  ibv_cq *createCQ(int cqe, ibv_comp_channel *channel);
};
```

多 IB 设备场景（生产环境每个存储节点配置 2×200Gbps InfiniBand）：
- `RDMABuf` 注册到所有可用设备，客户端 `RDMARemoteBuf` 包含所有设备的 rkey。
- 存储节点根据本地设备 ID 选择正确的 rkey，自动利用多链路带宽。

---

## 7. 滑动窗口流控（Send Buffer Management）

RDMA over InfiniBand 的 Reliable Connection（RC）模式天然提供可靠传输，但需要管理发送窗口以防止接收方缓冲区溢出：

```cpp
// 发送缓冲区（固定大小 ring buffer）
class SendBuffers : public Buffers {
    std::atomic<uint64_t> tailIdx_{0};   // 已 ACK 的位置
    std::atomic<uint64_t> frontIdx_{0};  // 下一个可用槽位
};

// 接收方通过 Immediate Data 发送 ACK，通知发送方已处理的缓冲区数量
// ImmData::ack(count) 编码在 RDMA WRITE WITH IMMEDIATE 的立即数中
```

---

## 8. 模块依赖关系

```
RDMA 网络层（src/common/net/）
  │
  ├── 依赖 ibverbs 库（Linux InfiniBand 标准接口）
  │
  ├── 依赖 Folly
  │     ├── fibers::BatchSemaphore（RDMA WR 数量限流）
  │     ├── coro::Baton（协程挂起/唤醒）
  │     └── MPMCQueue（接收缓冲区无锁队列）
  │
  ├── 依赖 EventLoop（epoll，监听 CQ 完成事件）
  │
  ├── 被 StorageOperator 依赖
  │     └── RDMA Read 拉取客户端数据（写路径）
  │     └── RDMA Write 返回数据（读路径，通过 Iov）
  │
  ├── 被 Storage ReliableForwarding 依赖
  │     └── RDMA Read 从前驱拉取 Chunk 数据（链式转发）
  │
  └── 被 Client (FUSE/Native) 依赖
        └── RDMABuf/RDMARemoteBuf 作为 IOV 注册到 IB（零拷贝读写）
```
