---
created: 2026-05-06
---

# 存储服务（Storage）：CRAQ 协议与数据恢复

> 源码路径：`src/storage/`  
> 关键文件：`service/StorageServer.h`、`service/Components.h`、`service/StorageOperator.h`、`service/ReliableForwarding.h`、`sync/ResyncWorker.h`、`aio/AioReadWorker.h`

---

## 1. 存储服务的作用与设计动机

存储服务（Storage Service）是 3FS 的数据存储核心，负责将文件 Chunk 持久化到本地 SSD，并通过 **CRAQ（Chain Replication with Apportioned Queries）** 协议实现多副本强一致性。

**为什么选择 CRAQ 而不是 Raft/Paxos？**

在全闪存（All-Flash）场景下，读取带宽是关键瓶颈。CRAQ 的 write-all/read-any 特性允许将读流量均匀分摊到副本链的所有节点，充分利用每块 SSD 的带宽。而 Raft 通常只从 Leader 读取，无法利用 Follower 的读带宽。

---

## 2. 存储服务组件架构

```cpp
// src/storage/service/Components.h
struct Components {
  // 管理器客户端（获取路由信息、发送心跳）
  folly::atomic_shared_ptr<IMgmtdClientForServer> mgmtdClient;

  // RDMA 缓冲区池（零拷贝传输）
  BufferPool rdmabufPool;

  // 目标映射（Target ID -> TargetPtr，热更新）
  AtomicallyTargetMap targetMap;

  // 本地存储目标管理（Chunk Engine 封装）
  StorageTargets storageTargets;

  // 读路径：AIO/io_uring 异步磁盘读
  AioReadWorker aioReadWorker;

  // 链式转发（CRAQ 写请求沿链传播）
  ReliableForwarding reliableForwarding;

  // 数据同步（故障恢复时 Resync）
  ResyncWorker resyncWorker;

  // 后台工作线程
  CheckWorker checkWorker;      // 数据完整性检查
  DumpWorker dumpWorker;        // Chunk 元数据导出（Resync 前的快照）
  AllocateWorker allocateWorker; // 预分配磁盘空间
  PunchHoleWorker punchHoleWorker; // 释放磁盘空洞
  SyncMetaKvWorker syncMetaKvWorker; // 元数据 KV 同步

  // 协程池（分读/写/同步/默认 4 类优先级）
  DynamicCoroutinesPool readPool;
  DynamicCoroutinesPool updatePool;
  DynamicCoroutinesPool syncPool;
  DynamicCoroutinesPool defaultPool;

  // 核心操作处理器（batchRead/write/update 等）
  StorageOperator storageOperator;
};
```

网络配置：存储服务同时监听两个端口：
- **RDMA 端口（8000）**：处理 `StorageSerde` 服务（高吞吐数据传输）。
- **TCP 端口（9000）**：处理 `Core` 服务（控制操作，独立线程池）。

---

## 3. CRAQ 写入流程

### 3.1 写入路径概述

```
Client
  │
  │── WriteReq（包含 RDMARemoteBuf，客户端内存的远端地址）
  ▼
StorageOperator::update()
  │
  ├── 1. 验证链版本（chain_version 必须匹配最新版本）
  │
  ├── 2. RDMA Read（从客户端内存拉取数据到本地缓冲区）
  │      ↑ 若客户端/前驱节点故障 → RDMA Read 超时 → 写入中止
  │
  ├── 3. 获取 Chunk 锁（同一 Chunk 的并发写串行化，Head 节点决定顺序）
  │
  ├── 4. 读取当前已提交版本 → 应用写入 → 存储为 pending 版本
  │      pending 版本号 u = committed 版本号 v + 1
  │
  ├── 5a. 若是 tail 节点：原子提交（pending → committed），发送 ACK 给前驱
  │   5b. 若非 tail 节点：将写请求转发给后继节点（ReliableForwarding）
  │
  └── 6. 收到 ACK：将 pending 替换为 committed，释放 Chunk 锁，继续传播 ACK
```

关键数据结构：

```cpp
// StorageOperator 的写操作入口
CoTryTask<UpdateRsp> update(
    ServiceRequestContext &requestCtx,
    const UpdateReq &req,
    net::IBSocket *ibSocket);  // ibSocket 用于 RDMA Read 拉取数据
```

### 3.2 可靠转发（ReliableForwarding）

写请求沿链传播时，如果后继节点暂时不可用（Resync 中、正在重启），需要自动重试：

```cpp
// src/storage/service/ReliableForwarding.h
class ReliableForwarding {
  struct Config {
    CONFIG_HOT_UPDATED_ITEM(retry_first_wait, 100_ms);
    CONFIG_HOT_UPDATED_ITEM(retry_max_wait, 1000_ms);
    CONFIG_HOT_UPDATED_ITEM(retry_total_time, 60_s);
  };

  CoTask<IOResult> forwardWithRetry(
      ServiceRequestContext &requestCtx,
      const UpdateReq &req,
      const net::RDMARemoteBuf &rdmabuf,
      const ChunkEngineUpdateJob &chunkEngineJob,
      TargetPtr &target,
      CommitIO &commitIO,
      bool allowOutdatedChainVer = true);
};
```

特殊情况处理：当节点 B 在传播过程中下线，Mgmtd 将 B 移到链尾并广播新链表。节点 A 收到新链表后将写请求重新转发给 C（新的后继节点）。C 可能还未收到新链表而拒绝请求，但 A 会持续重试，直到 C 接受。

### 3.3 读取流程

读请求可以发送到链中任意目标：

```cpp
// batchRead：批量读取，核心读取路径
CoTryTask<BatchReadRsp> batchRead(
    ServiceRequestContext &requestCtx,
    const BatchReadReq &req,
    serde::CallContext &ctx);
```

读取状态处理：
- 目标只有 **committed 版本**：直接返回给客户端。
- 目标同时有 **committed + pending 版本**：返回特殊状态码，客户端等待后重试（或请求 relaxed read 获取 pending 版本）。

注意：与标准 CRAQ 不同，3FS 不向 tail 节点发送版本查询（避免额外 RTT），而是让客户端自行决定是否接受 pending 版本。

---

## 4. 数据恢复（Resync）

### 4.1 故障后恢复流程

当存储节点重启后，其所管理的所有目标处于 `offline` 状态，需要通过 `ResyncWorker` 完成数据恢复：

```cpp
// src/storage/sync/ResyncWorker.h
class ResyncWorker {
  struct Config {
    CONFIG_ITEM(num_threads, 16ul);       // 并发恢复线程
    CONFIG_ITEM(num_channels, 1024u);     // 转发通道数
    CONFIG_HOT_UPDATED_ITEM(batch_size, 16u);
    CONFIG_HOT_UPDATED_ITEM(sync_start_timeout, 10_s);
  };

  // 对每个需要恢复的 Chain 执行 Resync
  CoTryTask<void> handleSync(VersionedChainId vChainId);
};
```

### 4.2 Resync 算法

数据恢复采用**增量同步**策略，只传输有差异的 Chunk：

**前驱节点视角（发送方）：**

```
1. 向返回节点发送 dump-chunkmeta 请求
   → 返回节点回复所有 Chunk 的 ID、链版本、committed 版本号

2. 收集本地目标上所有 Chunk 的元数据

3. 对比两份元数据，选出需要传输的 Chunk：
   ┌─────────────────────────────────────────────────────────┐
   │ 传输条件（满足其一）：                                    │
   │ a. Chunk 只存在于本地 → 需要同步到对端                   │
   │ b. 链版本本地 > 对端 → 对端有旧版本，需要覆盖             │
   │ c. 链版本相同但 committed 版本号不匹配 → 有未完成写入      │
   └─────────────────────────────────────────────────────────┘

4. 对每个需要传输的 Chunk：
   a. 获取 Chunk 锁
   b. 读取链版本、committed 版本号、Chunk 数据
   c. 发送 full-chunk-replace 写请求（全量 Chunk 覆盖）
   d. 释放 Chunk 锁

5. 发送 sync-done 消息

6. Mgmtd 收到 sync-done 后将目标状态转为 serving
```

**返回节点视角（接收方）：**

恢复期间，节点作为链的 tail 接受来自前驱的全量写请求（full-chunk-replace），直接覆盖本地数据，不需要传播 ACK（因为它是临时 tail）。

### 4.3 并发恢复

Resync 与正常 I/O 完全并发，不会中断服务：
- 普通写入继续处理，并自动将正在恢复的目标作为写入的副本之一（full-chunk-replace）。
- 读取请求不会路由到 `syncing` 状态的目标（`syncing` 目标不服务读请求）。

---

## 5. 读取优化：AIO + io_uring

磁盘读取是吞吐量的关键瓶颈，`AioReadWorker` 支持两种读取引擎：

```cpp
// src/storage/aio/AioReadWorker.h
class AioReadWorker {
  enum class IoEngine {
    libaio,    // Linux 传统异步 IO
    io_uring,  // Linux 5.1+ 新接口，零系统调用开销
    random,    // 随机选择（测试用）
  };

  struct Config {
    CONFIG_ITEM(num_threads, 32ul);   // IO 线程数
    CONFIG_ITEM(queue_size, 4096u);   // 读请求队列深度
    CONFIG_ITEM(max_events, 512u);    // 单次 poll 的最大完成事件数
    CONFIG_ITEM(enable_io_uring, true);
    CONFIG_HOT_UPDATED_ITEM(min_complete, 128u);  // 最小批处理数
  };
};
```

读取流程：
1. `StorageOperator::batchRead()` 将读请求放入 `AioReadWorker` 的队列。
2. `AioReadWorker` 线程从队列取出请求，通过 libaio 或 io_uring 提交到内核。
3. 内核完成后，读取结果通过 RDMA Write 直接写入客户端的 Iov 内存（零拷贝）。

---

## 6. 缓冲区管理：BufferPool

RDMA 操作需要预先注册内存区域（Memory Region），注册本身有开销。`BufferPool` 维护一个预注册的 RDMA 内存缓冲区池：

```cpp
// 存储服务端的 RDMA 缓冲区池
BufferPool rdmabufPool;

// 缓冲区有固定大小，从 pool 中借用/归还
CoTask<RDMABuf> allocate(optional<Duration> timeout);
void deallocate(RDMABuf::Inner *buf);
```

缓冲区大小根据请求大小配置（`post_buffer_per_bytes`），批量读取时将多个 Chunk 读取请求合并到单个大缓冲区以减少 RDMA 操作次数。

---

## 7. 写入路径优化：UpdateWorker

写操作（Update）是 I/O 密集型的（COW：读旧数据 → 修改 → 写新数据），`UpdateWorker` 为每块磁盘维护独立的写队列，避免多磁盘之间的竞争：

```cpp
// src/storage/update/UpdateWorker.h
class UpdateWorker {
  struct Config {
    CONFIG_ITEM(queue_size, 4096u);  // 每个磁盘队列深度
    CONFIG_ITEM(num_threads, 32ul);  // 前台写线程数
    CONFIG_ITEM(bg_num_threads, 8ul); // 后台回收线程数
  };

  // 每块磁盘对应一个独立队列
  // job->target()->diskIndex() 决定路由到哪个队列
  CoTask<void> enqueue(UpdateJob *job) {
    co_await queueVec_[job->target()->diskIndex()]->co_enqueue(job);
  }
};
```

---

## 8. 模块依赖关系

```
Storage Service
  │
  ├── 依赖 Mgmtd Client
  │     ├── 心跳（携带目标本地状态）
  │     └── 获取 RoutingInfo（更新 targetMap）
  │
  ├── 依赖 common/net/ib（RDMA Read/Write）
  │     └── 数据零拷贝传输（客户端 ↔ 存储节点）
  │
  ├── 依赖 Chunk Engine（Rust FFI）
  │     ├── 块分配、COW 写入
  │     └── RocksDB 元数据持久化
  │
  ├── 依赖 AioReadWorker（libaio/io_uring 磁盘读）
  │
  ├── 依赖 Storage Client（发送写请求给后继节点）
  │     └── ReliableForwarding::forwardWithRetry
  │
  └── 被 Client（FUSE/Native）依赖
        ├── batchRead：大吞吐量并行读
        └── update/write：写入 Chunk 数据
```
