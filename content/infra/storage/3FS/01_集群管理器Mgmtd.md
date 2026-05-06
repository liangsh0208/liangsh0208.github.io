# 集群管理器（Mgmtd）：心跳、路由与链表状态机

> 源码路径：`src/mgmtd/`  
> 关键文件：`service/MgmtdService.h`、`service/MgmtdState.h`、`service/MgmtdOperator.h`、`service/RoutingInfo.h`、`service/updateChain.h`

---

## 1. Mgmtd 的作用与设计动机

集群管理器（Management Daemon，**Mgmtd**）是 3FS 集群的"大脑"，承担以下核心职责：

1. **成员管理**：维护所有存储节点和元数据节点的在线状态，通过心跳（Heartbeat）进行活性检测。
2. **链表管理**：持有全局 Chain Table，记录每个复制链（Chain）的版本号、成员组成和目标状态。
3. **配置分发**：将最新的路由信息（RoutingInfo）分发给所有服务和客户端，确保全局一致性。
4. **高可用选举**：支持多实例部署，通过 Primary 选举实现故障切换。

设计动机：Mgmtd 是唯一拥有全局视图的组件，所有链版本变更（如节点故障、数据恢复完成）都必须经过 Mgmtd 裁决并广播，避免各节点自行决策导致的一致性问题。

---

## 2. 核心数据结构

### 2.1 RoutingInfo（路由信息）

```cpp
// src/fbs/mgmtd/RoutingInfo.h
struct RoutingInfo : public serde::SerdeHelper<RoutingInfo> {
  SERDE_STRUCT_FIELD(routingInfoVersion, RoutingInfoVersion(0));
  SERDE_STRUCT_FIELD(bootstrapping, bool(false));
  SERDE_STRUCT_FIELD(nodes, NodeMap{});          // NodeId -> NodeInfo
  SERDE_STRUCT_FIELD(chainTables, ChainTableMap{});  // ChainTableId -> ChainTableVersionMap
  SERDE_STRUCT_FIELD(chains, ChainMap{});        // ChainId -> ChainInfo
  SERDE_STRUCT_FIELD(targets, TargetMap{});      // TargetId -> TargetInfo
};
```

`RoutingInfo` 是客户端和服务端获取集群拓扑的核心结构：
- `nodes`：所有已注册节点（存储节点、元数据节点）的信息。
- `chainTables`：一个集群可以拥有多张链表（Chain Table），每张表有版本历史。每张链表代表一种数据放置策略（例如在线推理 vs 离线训练可以使用不同链表，从而隔离存储资源）。
- `chains`：所有复制链的当前状态，包含其目标列表和版本。
- `targets`：所有存储目标（SSD 粒度）的状态。

### 2.2 ChainInfo（链信息）

```cpp
// 复制链核心结构（简化）
struct ChainInfo {
  ChainId chainId;
  ChainVersion chainVersion;       // 每次成员变更递增
  std::vector<ChainTargetInfo> targets;  // 有序目标列表（head -> tail）
};

// 目标公共状态
enum class PublicState {
  serving,  // 正常服务
  syncing,  // 数据恢复中（接受写，不服务读）
  waiting,  // 等待恢复（不服务读写）
  lastsrv,  // 下线，曾经是最后一个正常目标
  offline,  // 已下线（故障或介质失效）
};
```

### 2.3 MgmtdState（管理器状态）

```cpp
// src/mgmtd/service/MgmtdState.h
struct MgmtdState {
  std::shared_ptr<core::ServerEnv> env_;
  const MgmtdConfig &config_;
  MgmtdStore store_;            // 持久化到 FoundationDB
  core::UserStoreEx userStore_; // 用户认证

  CoroSynchronized<MgmtdData> data_;                // 实时路由数据（内存）
  CoroSynchronized<ClientSessionMap> clientSessionMap_; // 客户端会话
  folly::coro::Mutex writerMu_;  // 写操作序列化锁（防止并发链表修改）
};
```

`MgmtdState` 中最关键的设计是 `writerMu_`：所有修改链表的操作（故障处理、恢复完成通知）都需要持有这把协程互斥锁，确保链版本变更的原子性和序列化，避免多个 Primary 状态变更相互冲突。

---

## 3. Mgmtd 服务接口

Mgmtd 对外提供的 RPC 方法由宏展开定义：

```cpp
// src/fbs/mgmtd/MgmtdServiceDef.h
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, getPrimaryMgmtd, GetPrimaryMgmtd, 1)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, heartbeat, Heartbeat, 3)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, registerNode, RegisterNode, 4)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, getRoutingInfo, GetRoutingInfo, 5)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, setConfig, SetConfig, 6)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, getConfig, GetConfig, 7)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, setChainTable, SetChainTable, 8)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, enableNode, EnableNode, 9)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, disableNode, DisableNode, 10)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, extendClientSession, ExtendClientSession, 11)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, listClientSessions, ListClientSessions, 12)
DEFINE_SERDE_SERVICE_METHOD(Mgmtd, updateChain, UpdateChain, 24)
```

关键方法说明：
- **`heartbeat`**：存储/元数据节点定期调用，携带本地目标状态，Mgmtd 据此判断节点是否存活。
- **`getRoutingInfo`**：客户端和服务端拉取最新路由表，携带本地已知版本号，按需返回增量更新。
- **`updateChain`**：管理员命令，手动触发链状态变更（如恢复后让目标重新上线）。
- **`setChainTable`**：替换整张链表，用于集群扩缩容时重新规划数据放置。

---

## 4. 心跳机制与故障检测

### 4.1 心跳租约模型

```
Storage Node                     Mgmtd (Primary)
    │                                  │
    │──── heartbeat (interval T/2) ───►│
    │◄─── lease renewed ──────────────│
    │                                  │
    │  (如果 T 秒内未收到心跳)           │
    │                                  │── 标记节点故障
    │                                  │── 更新链表，移除故障目标
    │                                  │── 广播新路由
    │
    │  (存储节点如果 T/2 秒内无法联系 Mgmtd)
    │── 主动退出进程（防止脑裂）
```

**关键设计**：存储节点主动自杀（在无法联系 Mgmtd 的 T/2 秒后）是防止脑裂的关键机制。这确保了在网络分区场景下，孤立的存储节点不会继续服务请求，避免数据不一致。

### 4.2 存储目标状态机

目标的**公共状态**（Public State）由 Mgmtd 根据**本地状态**（Local State）和链的上下文驱动，完整状态转换规则：

| 本地状态（Local） | 当前公共状态（Public） | 前驱目标状态 | 下一公共状态 |
|---|---|---|---|
| `up-to-date` | `serving` | 任意 | `serving` |
| `up-to-date` | `syncing` | 任意 | `serving` |
| `up-to-date` | `offline` | 任意 | `waiting` |
| `online` | `waiting` | serving | `syncing` |
| `online` | `waiting` | not serving | `waiting` |
| `offline` | `serving` | 无前驱 | `lastsrv` |
| `offline` | `serving` | 有前驱 | `offline` |
| `offline` | `lastsrv` | 任意 | `lastsrv` |

**lastsrv** 是一个特殊保护状态：当链中最后一个 serving 目标下线时，该目标进入 `lastsrv` 状态而不是 `offline`。存储服务检测到自己的目标处于 `lastsrv` 时会立即退出，防止数据丢失。

### 4.3 Background 任务

Mgmtd 后台定期扫描所有链，根据心跳信息触发状态机转换：

```cpp
// src/mgmtd/background/ 目录
// - 定期扫描所有链目标的本地状态
// - 调用状态转换逻辑
// - 递增链版本号（如有变更）
// - 广播更新后的 RoutingInfo
```

---

## 5. 链表管理（Chain Table）

### 5.1 概念层次

```
Chain Table（链表）
    │── ChainTableId + ChainTableVersion
    └── [ Chain1, Chain2, ... ChainN ]
             │
             └── ChainInfo
                    │── ChainId + ChainVersion
                    └── [ Target1(head), Target2, ... TargetM(tail) ]
                                │
                                └── TargetId + PublicState
```

一个集群可以拥有多张 Chain Table，不同文件可以指定使用不同的链表。这使得可以：
- 隔离在线推理和离线训练的存储资源。
- 为不同副本数需求配置不同链表（3 副本 vs 2 副本）。

### 5.2 链表生成算法

链表的生成通过整数规划求解，目标是在节点故障时实现**均衡流量恢复**。

问题建模（Balanced Incomplete Block Design）：
- 每条链的 Head 目标故障时，其读流量重定向到同链其他目标。
- 最优方案要求：每个节点与其他每个节点的共同 peer 数量相等，使得任意单节点故障时，故障流量均匀分散到所有其他节点。

```bash
# 实际使用 Pyomo + HiGHS 求解器生成链表
python src/model/data_placement.py \
    --num_nodes 5 --replication_factor 3 \
    --min_targets_per_disk 6 --init_timelimit 600
```

最优解示意（5 节点，每节点故障时各自承载 1/5 故障流量）：

```
Chain | Head | Mid  | Tail
  1   |  B1  |  E1  |  F1
  2   |  A1  |  B2  |  D1   ← A 与 B/D 配对
  3   |  A2  |  D2  |  F2   ← A 与 D/F 配对
  ...
```
当 A 故障时，A 的读流量均匀分散到 B、C、D、E、F，每个节点只承担 1/5 的额外负载。

---

## 6. 持久化：MgmtdStore

Mgmtd 使用 FoundationDB 持久化链表和节点信息，通过 `MgmtdStore` 封装：

```cpp
// src/mgmtd/store/MgmtdStore.h
class MgmtdStore {
  // 持久化节点注册信息
  CoTryTask<void> persistNodeInfo(const flat::PersistentNodeInfo &nodeInfo);
  // 持久化链表版本
  CoTryTask<void> persistChainTable(const flat::ChainTable &table);
  // 加载所有历史数据（Primary 启动时）
  CoTryTask<MgmtdData> loadData();
};
```

持久化策略：
- 链表变更（增加/移除节点）同步写入 FoundationDB，确保 Primary 切换后新节点能恢复完整链表历史。
- 内存中的 `MgmtdData` 作为热缓存，所有读操作直接从内存返回，避免 FoundationDB 的读延迟。

---

## 7. 模块依赖关系

```
Mgmtd
  │── 依赖 FoundationDB（MgmtdStore 持久化）
  │── 依赖 common/net（RDMA + TCP RPC 框架）
  │── 依赖 common/serde（自动序列化 RoutingInfo）
  │── 被 Meta Service 依赖（拉取路由，分配 Chain）
  │── 被 Storage Service 依赖（心跳，状态同步）
  └── 被 Client 依赖（获取路由表，发现可用 Meta/Storage 节点）
```
