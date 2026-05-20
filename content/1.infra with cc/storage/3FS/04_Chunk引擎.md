---
created: 2026-05-06
---

# Chunk 引擎（Chunk Engine）：Rust 实现的物理存储管理

> 源码路径：`src/storage/chunk_engine/`  
> 主要语言：Rust  
> 关键文件：`src/core/engine.rs`、`src/alloc/allocators.rs`、`src/meta/meta_store.rs`、`src/types/chunk_meta.rs`、`src/types/constants.rs`

---

## 1. Chunk Engine 的作用与设计动机

Chunk Engine 是 3FS 存储服务的底层物理存储管理器，使用 Rust 实现，通过 C++ FFI 接口与上层 C++ 代码集成。

**为什么用 Rust 实现 Chunk Engine？**

Chunk Engine 涉及复杂的内存所有权管理（引用计数的块位置、COW 语义下的版本管理）和高并发场景。Rust 的所有权系统可以在编译期保证内存安全，避免 C++ 中常见的 Use-After-Free、双重释放等问题。`Arc<T>` 的原子引用计数在 Rust 中是零成本抽象。

**核心设计原则：**
1. **分配器（Allocator）**：纯内存操作，管理物理块的分配/回收状态。
2. **MetaStore**：持久化操作，记录分配/回收事件到 RocksDB。
3. **两者协作**：先内存分配，后持久化；先持久化回收，后内存释放。

---

## 2. 存储层次结构

```
Disk（单块 SSD，~30TB）
  └── 多个 ChunkSize 级别的 Clusters
        └── Cluster（约 120GB，对应一个物理文件）
              └── ~960 个 Groups
                    └── Group（256 个 Chunk 槽位）
                          └── Chunk（实际数据单元）
```

Chunk 大小支持 11 个级别（2 的幂次）：

```rust
// src/storage/chunk_engine/src/types/constants.rs
pub const CHUNK_SIZE_SMALL: Size = Size::kibibyte(64);    // 64KiB
pub const CHUNK_SIZE_NORMAL: Size = Size::kibibyte(512);  // 512KiB
pub const CHUNK_SIZE_LARGE: Size = Size::mebibyte(4);     // 4MiB
pub const CHUNK_SIZE_ULTRA: Size = Size::mebibyte(64);    // 64MiB
pub const CHUNK_SIZE_SHIFT: usize = 16;    // 64KiB = 2^16
pub const CHUNK_SIZE_NUMBER: usize = 11;   // 11 个级别：64KiB ~ 64MiB
```

每个 Group 包含 256 个 Chunk 槽位，使用 256-bit bitset（4 个 u64）追踪分配状态，通过 `__builtin_ctz`（Count Trailing Zeros）实现 O(1) 的空槽查找。

---

## 3. 关键数据结构

### 3.1 ChunkMeta（块元数据）

```rust
// src/storage/chunk_engine/src/types/chunk_meta.rs
#[derive(derse::Serialize, derse::Deserialize, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct ChunkMeta {
    pub pos: Position,           // 物理位置（chunk_size + cluster + group + index）
    pub chain_ver: u32,          // 所属 Chain 的版本号
    pub chunk_ver: u32,          // Chunk 的提交版本号（单调递增）
    pub len: u32,                // 数据长度
    pub checksum: u32,           // CRC32 校验和
    pub timestamp: u64,          // 最后写入时间（微秒）
    pub last_request_id: u64,    // 最后一次写请求 ID（幂等性）
    pub last_client_low: u64,    // 最后写客户端 ID（低 64 位）
    pub last_client_high: u64,   // 最后写客户端 ID（高 64 位）
    pub etag: ETag,              // 内容标识（默认为 checksum 的十六进制）
    pub uncommitted: bool,       // 是否处于 uncommitted 状态（持久化但未提交）
}
```

### 3.2 Position（物理位置编码）

`Position` 是一个紧凑的 64-bit 整数，编码了 Chunk 在物理文件系统中的完整位置：

```rust
// src/storage/chunk_engine/src/types/position.rs
#[repr(C)]
pub struct Position(pub u64);

// 位字段布局（高到低）：
// [63:40] = chunk_size 高 24 位（右移 8 位）
// [39:32] = cluster ID (8 位)
// [31: 8] = group ID  (24 位)
// [ 7: 0] = slot index in group (8 位，0~255)

impl Position {
    pub fn chunk_size(&self) -> Size { Size::new(self.0 >> 40 << 8) }
    pub fn cluster(&self) -> u8     { (self.0 >> 32) as u8 }
    pub fn group(&self) -> u32      { (self.0 as u32) >> Self::SHIFT }
    pub fn index(&self) -> u8       { self.0 as u8 }

    // 计算在物理文件中的字节偏移
    pub fn offset(&self) -> Size    { self.chunk_size() * self.0 as u32 as u64 }
}
```

这种位域设计允许仅通过一个 64-bit 整数完整定位一个 Chunk，无需额外查找。

### 3.3 Allocators（多级分配器）

11 种 Chunk 大小各自对应一个独立的 `Allocator`：

```rust
// src/storage/chunk_engine/src/alloc/allocators.rs
pub struct Allocators {
    pub vec: [Arc<Allocator>; CHUNK_SIZE_NUMBER],  // 11 个分配器
    meta_store: Arc<MetaStore>,
}

impl Allocators {
    // 按 Chunk 大小选择合适的分配器（向上取整到最近的 2 的幂次）
    pub fn select_by_size(&self, size: Size) -> Result<&Arc<Allocator>> {
        if size <= CHUNK_SIZE_SMALL {
            Ok(&self.vec[0])
        } else if size <= CHUNK_SIZE_ULTRA {
            Ok(&self.vec[size.next_power_of_two().trailing_zeros() as usize - CHUNK_SIZE_SHIFT])
        } else {
            Err(Error::InvalidArg(...))
        }
    }
}
```

---

## 4. Engine：核心状态机

```rust
// src/storage/chunk_engine/src/core/engine.rs
#[derive(Clone)]
pub struct Engine {
    pub meta_store: Arc<MetaStore>,          // RocksDB 持久化
    pub allocators: Allocators,              // 11 级物理块分配器
    pub meta_cache: Arc<LockMap<Bytes, ChunkArc>>,  // chunk_id -> ChunkMeta 缓存
    pub workers: Arc<Mutex<Vec<Worker>>>,    // 后台工作线程
    pub allow_to_allocate: Arc<AtomicBool>,  // 动态控制是否允许分配新块
    pub metrics: Arc<Metrics>,              // 监控指标
    pub prefix_len: usize,                  // chunk_id 前缀长度（用于分片 writing_list）
    pub writing_list: Arc<WritingList>,      // 正在写入中的 Chunk 追踪
}
```

### 4.1 Engine 初始化（open）

```rust
pub fn open(config: &EngineConfig) -> Result<Self> {
    // 1. 打开 RocksDB MetaStore
    let mut meta_store = MetaStore::open(&meta_config)?;

    // 2. 恢复未提交的 Chunk（断电恢复）
    let uncommitted_chunks = meta_store.occupy_uncommitted_positions()?;

    // 3. 初始化各级别 Allocator（从 RocksDB 加载 group 状态）
    let allocators = Allocators::new(&config.path, config.create, meta_store.clone())?;

    // 4. 初始化 MetaCache（LockMap，1M 容量，256 分片）
    let meta_cache = Arc::new(LockMap::with_capacity_and_shard_amount(1 << 20, 256));

    // 5. 如果有 uncommitted chunks，将其恢复到 writing_list
    //    （这些块在内存中标记为 abort = true，等待后续正确重写）
    if !uncommitted_chunks.is_empty() {
        for (chunk_id, meta, _) in &uncommitted_chunks {
            // 重新建立对这些块的引用，防止被 GC
            let chunk = allocator.reference(meta.clone(), old_chunk.is_none());
            writing_list.insert(chunk_id.clone(), WritingHolder { chunk, abort: true });
        }
        meta_store.vacate_uncommitted_positions(uncommitted_chunks)?;
    }

    Ok(engine)
}
```

断电恢复设计：若写操作在持久化 `uncommitted` 状态后断电，重启时这些块会被恢复到 `writing_list`，等待上层（CRAQ 协议层）重新提交或中止。

---

## 5. 分配器（Allocator）设计

### 5.1 三级内存结构

```
active_groups:     Map<GroupId, GroupState>  ← 当前分配热点（有空槽的 Group）
allocated_groups:  Set<GroupId>              ← 已预分配磁盘空间但全部空闲的 Group
unallocated_groups: Set<GroupId>             ← 尚未分配磁盘空间的 Group
```

分配流程（完全在内存中）：

```rust
fn allocate(clusters: &Clusters, allow_to_allocate: bool) -> Result<Position> {
    // 1. 在 active_groups 中找空槽（O(1) 位操作）
    if let Some(pos) = active_groups.find_free_slot() {
        return Ok(pos);
    }

    // 2. active_groups 为空，从 allocated_groups 取一个 Group
    if let Some(gid) = allocated_groups.pop() {
        active_groups.insert(gid, GroupState::new());
        return active_groups.find_free_slot();
    }

    // 3. allocated_groups 也空，从 unallocated_groups 取并同步分配磁盘
    if allow_to_allocate {
        let gid = unallocated_groups.pop();
        clusters.allocate(gid)?;           // fallocate() 预分配磁盘空间
        meta_store.allocate_group(gid)?;   // RocksDB 持久化
        active_groups.insert(gid, ...);
        return active_groups.find_free_slot();
    }
}
```

### 5.2 后台 allocate_thread

后台线程持续维护 `allocated_groups` 在目标范围内（min_remain ~ max_remain），提前预分配磁盘空间，确保分配时不需要同步 I/O：

```rust
pub fn allocate_groups(&self, min_remain: usize, max_remain: usize, batch_size: usize) -> usize {
    self.allocators.allocate_groups(min_remain, max_remain, batch_size, false)
}
```

### 5.3 compact_thread（碎片整理）

定期扫描 `active_groups`，将使用率低的 Group 中的 Chunk 迁移到其他 Group，释放整个 Group 的磁盘空间：

```rust
pub fn compact_groups(&self, max_reserved: u64) -> usize {
    let group_ids = self.allocators.get_allocate_tasks(max_reserved);
    // 对每个需要整理的 group_id：
    // 1. 获取 group 中所有 chunk_id
    // 2. 逐个重新分配并迁移 Chunk 数据
    // 3. 释放原 Group
}
```

---

## 6. MetaStore：RocksDB 持久化

MetaStore 维护三类 KV 映射，通过 `MetaKey` 进行前缀区分：

```rust
// src/storage/chunk_engine/src/meta/meta_store.rs
// 三类 Key：
// 1. chunk_id -> ChunkMeta  (前缀: CHUNK_META_KEY_PREFIX)
// 2. group_id -> GroupState (前缀: GROUP_STATE_KEY_PREFIX, 使用 MergeOp 原子更新)
// 3. pos -> chunk_id        (前缀: POS_KEY_PREFIX, 用于 compact_thread 的迁移)
```

**关键优化：MergeOp**

GroupState 的位图更新使用 RocksDB 的 `MergeOp` 接口：不需要先读后写，只需写入一个"标记位 N 为 0/1"的操作，RocksDB 在合适时机自动合并，避免读-改-写的额外 I/O。

```rust
// src/storage/chunk_engine/src/meta/meta_merge.rs
// MergeOp 实现：将多个位图操作合并为最终 GroupState
struct MetaMergeOp;
impl rocksdb::MergeOperator for MetaMergeOp { ... }
```

---

## 7. COW 写入流程

```
Engine::update(chunk_id, write_data) {
    // 1. 从 MetaCache 获取当前 chunk_info（含 Arc<Position>）
    let old_chunk = engine.get(&chunk_id)?;

    // 2. 分配新的物理块（纯内存操作）
    let new_pos = allocators.allocate(chunk_size, allow_to_allocate)?;
    // 此时 new_pos 是一个 Chunk，持有对新位置的引用

    // 3. 读取旧数据（若有）+ 应用写操作 → 合并后的数据写入新块
    //    （直接写入物理文件，不经过 OS 缓存）
    write_to_physical_file(new_pos.offset(), merged_data)?;

    // 4. 原子持久化：
    //    - 新 ChunkMeta（chunk_id -> new meta, pos = new_pos）
    //    - 释放旧块（old_pos -> 位图标记为空闲）
    //    - 分配新块（new_pos -> 位图标记为已用）
    meta_store.commit_write_batch([write_new_meta, release_old, mark_new_used])?;

    // 5. 更新 MetaCache
    meta_cache.insert(chunk_id, new_chunk_info);

    // 旧 Arc<Position> 在此处 drop → 旧块计数归零 → 标记可回收
}
```

COW 语义的正确性保证：
- 旧数据在新数据写入完成之前保持完整（读请求持有旧 Arc，旧块不会被回收）。
- 新数据写入物理文件后，元数据原子更新，确保没有"写了一半"的中间状态对外可见。

---

## 8. C++ FFI 接口

Chunk Engine 通过 `build.rs` 生成 C++ 可调用的接口：

```rust
// src/storage/chunk_engine/src/cxx.rs
// 暴露给 C++ 的核心接口：
pub mod ffi {
    pub struct FdAndOffset { fd: i32, offset: u64 }

    pub struct GetReq {
        chunk_id: ...,
    }

    pub struct UpdateReq {
        chunk_id: ...,
        // 写入数据的位置和大小
    }
}
```

```toml
# src/storage/chunk_engine/Cargo.toml
[dependencies]
cxx = "1"    # C++/Rust 互操作
rocksdb = ...
lockmap = ...
derse = ...  # 轻量级序列化（替代 serde，适合嵌入 no_std 场景）
```

---

## 9. 模块依赖关系

```
Chunk Engine（Rust）
  │
  ├── 依赖 RocksDB（chunk 元数据和 group 状态持久化）
  │     └── 使用 MergeOp 实现原子位图更新
  │
  ├── 依赖 NVMe SSD 文件系统（物理 Chunk 数据存储）
  │     └── fallocate() 预分配，direct I/O 写入
  │
  ├── 暴露 C++ FFI 接口
  │     └── StorageTargets（C++）通过 FFI 调用 Engine
  │
  └── 被 StorageOperator（C++）通过 ChunkEngineUpdateJob 使用
        ├── get()：查询 ChunkMeta（读路径）
        ├── update()：COW 写入（写路径）
        └── commit()：提交到 RocksDB（WAL 保证）
```
