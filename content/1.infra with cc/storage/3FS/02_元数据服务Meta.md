# 元数据服务（Meta）：无状态设计与事务 KV 存储

> 源码路径：`src/meta/`  
> 关键文件：`store/MetaStore.h`、`store/Inode.h`、`store/DirEntry.h`、`components/ChainAllocator.h`、`fbs/meta/Schema.h`

---

## 1. 元数据服务的作用与设计动机

元数据服务（Meta Service）是 3FS 的文件系统语义层，负责实现 POSIX 文件系统的核心操作：创建/删除/重命名文件、目录操作、权限检查等。

**核心设计原则：无状态架构**

Meta 服务本身不持有任何持久状态。所有 Inode 和目录项数据都存储在 **FoundationDB**（FDB）中，服务进程可以随时重启、横向扩展，客户端可以连接到任意一个 Meta 节点。

这种设计带来的好处：
- **运维简化**：升级 Meta 服务无需停机，可以滚动升级。
- **故障透明**：客户端超时后自动切换到其他 Meta 节点重试。
- **水平扩展**：多个 Meta 服务并发处理请求，FDB 的事务冲突检测保证一致性。

---

## 2. 核心数据结构

### 2.1 Inode（文件节点）

```cpp
// src/fbs/meta/Schema.h
struct Acl {
  SERDE_STRUCT_FIELD(uid, Uid(0));
  SERDE_STRUCT_FIELD(gid, Gid(0));
  SERDE_STRUCT_FIELD(perm, Permission(0));
  SERDE_STRUCT_FIELD(iflags, IFlags(0));  // 扩展标志，如 FS_IMMUTABLE_FL
};

struct Layout {
  // 文件数据布局，三种模式之一：
  struct Empty {};       // 空文件，还未分配 Chain
  struct ChainRange {    // 连续 Chain 范围（最常见）
    SERDE_STRUCT_FIELD(baseIndex, uint32_t(0));  // 起始 Chain 索引
    SERDE_STRUCT_FIELD(shuffle, Shuffle(0));      // 洗牌方式
    SERDE_STRUCT_FIELD(seed, uint64_t(0));        // 洗牌种子
    // chains 数组通过 baseIndex + shuffle + seed 按需计算
  };
  struct ChainList {     // 显式 Chain 列表
    SERDE_STRUCT_FIELD(chainIndexes, std::vector<uint32_t>());
  };

  SERDE_STRUCT_FIELD(tableId, ChainTableId(0));
  SERDE_STRUCT_FIELD(tableVersion, ChainTableVersion(0));
  SERDE_STRUCT_FIELD(chunkSize, uint32_t(0));   // Chunk 大小
  SERDE_STRUCT_FIELD(stripeSize, uint32_t(0));  // Stripe 宽度（并行度）
  // chains: variant<Empty, ChainRange, ChainList>
};

struct File {
  SERDE_STRUCT_FIELD(layout, Layout{});
  SERDE_STRUCT_FIELD(length, uint64_t(0));    // 文件长度
  SERDE_STRUCT_FIELD(dynStripe, uint32_t(1)); // 动态 stripe，优化小文件
};

struct Directory {
  SERDE_STRUCT_FIELD(parent, InodeId(0));     // 父目录 ID（用于循环检测）
  SERDE_STRUCT_FIELD(layout, Layout{});       // 默认布局配置
  SERDE_STRUCT_FIELD(name, std::string{});    // 目录名（用于路径重建）
};

struct Symlink {
  SERDE_STRUCT_FIELD(target, Path{});  // 符号链接目标路径
};
```

Inode 的 KV 存储格式：
- **Key**：`"INOD"` 前缀 + InodeId（小端序编码）
- **Value**：序列化后的 InodeData（包含 Acl、类型特定数据）

小端序编码的关键作用：将 inode_id 在 FDB 的 key 空间中分散分布，避免顺序 ID 造成的热点集中到单个 FDB Storage Process。

### 2.2 DirEntry（目录项）

```cpp
// 目录项 Key 格式：
// "DENT" + parent_inode_id(8 bytes) + "/" + entry_name
//
// 例如，根目录下的文件 "data.bin"：
// Key: "DENT\x01\x00\x00\x00\x00\x00\x00\x00/data.bin"
// Value: { inode_id: 42, type: File }
```

这种 key 设计使得同一目录下的所有条目形成**连续 key 范围**，目录列举（`ls`）只需一次 FDB 范围查询，无需逐条读取。

### 2.3 Layout 与 Chunk 寻址

文件被切分为等大小的 Chunk，按 stripe 方式分散到多个 Chain 上：

```
文件偏移 → Chunk Index → Chain Index
─────────────────────────────────────
  chunk_index = file_offset / chunk_size
  chain_index = chunk_index / stripe_size % stripe_size（取决于洗牌）
  chain_id    = chainTable[chain_index]
  chunk_id    = inode_id + "_" + chunk_index
```

创建文件时，Meta 服务使用**轮询 + 随机洗牌**策略分配 Chain：

```cpp
// src/meta/components/ChainAllocator.h
CoTryTask<void> allocateChainsForLayout(Layout &layout) {
  // 1. 从链表中轮询选取 stripeSize 个连续链（round-robin）
  auto chainBegin = roundRobin(chainCnt);
  // 2. 生成安全的洗牌种子
  auto seed = find_safe_seed(layout.stripeSize);
  // 3. 使用 MT19937 标准洗牌确保分布均匀
  layout.chains = Layout::ChainRange(chainBegin, STD_SHUFFLE_MT19937, *seed);
}
```

轮询保证跨文件的负载均衡，洗牌保证单文件内各 Chunk 分布均匀，避免热点。

---

## 3. MetaStore：操作工厂模式

`MetaStore` 不直接执行操作，而是作为**操作工厂**，返回封装了具体操作逻辑的 `IOperation` 对象：

```cpp
// src/meta/store/MetaStore.h
class MetaStore {
 public:
  // 每个方法返回一个操作对象（而非直接执行）
  OpPtr<StatRsp>    stat(const StatReq &req);
  OpPtr<OpenRsp>    open(OpenReq &req);
  OpPtr<CreateRsp>  tryOpen(CreateReq &req);
  OpPtr<MkdirsRsp>  mkdirs(const MkdirsReq &req);
  OpPtr<RemoveRsp>  remove(const RemoveReq &req);
  OpPtr<RenameRsp>  rename(const RenameReq &req);
  OpPtr<SyncRsp>    sync(const SyncReq &req);
  OpPtr<HardLinkRsp> hardLink(const HardLinkReq &req);
  OpPtr<SetAttrRsp>  setAttr(const SetAttrReq &req);
  // ...
};
```

`IOperation` 接口：

```cpp
template <typename Rsp>
class IOperation {
 public:
  virtual bool isReadOnly() = 0;
  // FDB 事务执行回调
  virtual CoTryTask<Rsp> run(IReadWriteTransaction &txn) = 0;
  // 事务冲突时的重试回调
  virtual void retry(const Status &error) = 0;
  // 最终完成回调
  virtual void finish(const Result<Rsp> &result) = 0;
};
```

这种设计允许将**事务重试逻辑**与**业务逻辑**分离。FDB 事务失败（冲突）时，框架自动重试，Operation 对象只需实现 `run()` 即可，无需处理重试细节。

---

## 4. 关键操作的实现原理

### 4.1 文件创建（Create/Open）

```
1. 路径解析：解析父目录路径，获取父 inode_id
2. 权限检查：验证当前用户对父目录的写权限（AclCache 加速）
3. 检查重名：查询 DirEntry，确认文件名不存在
4. 分配 InodeId：InodeIdAllocator 分配全局唯一单调递增 ID
5. 分配 Chain：ChainAllocator 按 Layout 规则分配 Chain 列表
6. FDB 写事务：
   - 写入新 Inode（Key: INOD + inode_id）
   - 写入 DirEntry（Key: DENT + parent_id + "/" + name）
7. 返回 FileHandle
```

原子性保证：步骤 6 的两次写操作在同一 FDB 事务中完成，要么全部成功，要么全部回滚。

### 4.2 重命名（Rename）

Rename 是最复杂的操作，需要处理：
- 目录循环检测（防止将目录 A 移入其子目录 B）
- 目标路径已存在时的覆盖逻辑（文件/空目录）
- 跨目录移动时父目录的 inode 变更

```cpp
// src/meta/store/ops/Rename.cc
// 关键步骤：加载所有相关祖先目录，检查没有循环路径
CoTryTask<Void> Inode::loadAncestors(
    IReadWriteTransaction &txn,
    std::vector<Inode> &ancestors,
    InodeId parent) {
  // 沿 parent 指针向上遍历，直到根目录
  // 若遇到目标目录 ID，则说明存在循环，报错
}
```

### 4.3 批量操作（BatchedOp）

多个来自不同客户端对**同一 Inode** 的操作（如多个 Close、Sync）可以合并为一个 FDB 事务，减少事务冲突：

```cpp
// src/meta/store/ops/BatchOperation.h
class BatchedOp : public Operation<Inode> {
  // 聚合对同一 inode_id 的多个操作
  void add(Waiter<SyncReq, SyncRsp> &waiter);
  void add(Waiter<CloseReq, CloseRsp> &waiter);
  void add(Waiter<CreateReq, CreateRsp> &waiter);

  CoTryTask<Inode> run(IReadWriteTransaction &txn) override;
  // 一个事务处理所有排队请求，通过 baton 通知每个 Waiter
};
```

---

## 5. 动态文件属性：文件长度的最终一致性

多客户端并发写入同一文件时，文件长度的更新面临挑战：

### 5.1 定期汇报机制

写模式打开的文件：客户端每 **5 秒**向 Meta 服务汇报本地写入的最大偏移量。若该偏移超过 Inode 中记录的长度且无并发截断操作，则更新长度。

这只保证**最终一致性**，不是精确长度。

### 5.2 精确长度（Close/Fsync 时）

文件关闭或 fsync 时，Meta 服务从 Storage 服务查询最后一个 Chunk 的精确长度：

```
文件按 stripeSize 条带分布于多条 Chain
      → 需要查询所有可能含有尾部 Chunk 的 Chain
      → 找到最大偏移的 Chunk，计算精确长度
```

**优化：动态 stripe 计数（dynStripe）**

生产环境 stripe 大小为 200，但小文件实际使用的 Chain 数远少于 200。`dynStripe` 字段从 16 开始，每次发现超出当前 stripe 范围时翻倍，避免每次 Close 都查询 200 条 Chain：

```cpp
// dynStripe 从 16 开始，按需翻倍：16 → 32 → 64 → 128 → 200
uint32_t dynStripe = 1;
```

### 5.3 任务分派（Rendezvous Hashing）

为避免多个 Meta 实例并发更新同一文件长度导致 FDB 事务冲突，Meta 服务使用**Rendezvous Hash** 将文件长度更新任务分派给特定的 Meta 实例：

```
assigned_meta = rendezvous_hash(inode_id, meta_service_list)
```

只有被选中的 Meta 实例负责更新该文件的长度，其他实例跳过，从根源上消除冲突。

---

## 6. 会话管理（SessionManager）

写模式打开的文件需要维护 **File Session**，原因：
- 检测到客户端下线时，延迟删除仍有活跃 session 的文件。
- 防止写入过程中被删除导致孤儿 Chunk（无法被 GC 回收）。

只读模式不维护 session（AI 训练中大量只读打开不应产生 Meta 负担）。

```cpp
// src/meta/components/SessionManager.h
// 定期检查客户端存活（通过 Mgmtd 的 ClientSession 信息）
// 清理已离线客户端的孤儿 session
```

---

## 7. AclCache：权限缓存

权限检查（ACL Check）在每次文件系统操作时都需要执行，是高频操作。Meta 服务维护一个 **2M 条目的 ACL 缓存**：

```cpp
// src/meta/components/AclCache.h
// AclCache 使用 inode_id 作为 key
// 缓存命中时跳过 FDB 读取，显著降低延迟
AclCache aclCache_{2 << 20 /* 2M entries */};
```

---

## 8. 模块依赖关系

```
Meta Service
  │── 依赖 FoundationDB（IKVEngine / FDBTransaction）
  │     ├── Inode CRUD
  │     ├── DirEntry CRUD
  │     └── 事务冲突检测（Serializable Snapshot Isolation）
  │
  │── 依赖 Mgmtd Client（获取 RoutingInfo 用于 ChainAllocator）
  │
  │── 依赖 Storage Client（Close/Fsync 时查询精确文件长度）
  │
  │── 内部组件：
  │     ├── InodeIdAllocator（全局单调递增 inode ID）
  │     ├── ChainAllocator（文件 Chain 分配策略）
  │     ├── SessionManager（写会话生命周期管理）
  │     ├── GcManager（垃圾 Chunk 异步回收）
  │     └── AclCache（权限缓存，减少 FDB 读）
  │
  └── 被 Client（FUSE/Native）依赖
        ├── 文件元数据操作（open/stat/rename 等）
        └── 获取文件 Layout（用于客户端自行计算 Chunk 位置）
```
