# 客户端与 USRBIO：FUSE 与零拷贝异步 I/O

> 源码路径：`src/fuse/`、`src/lib/api/`、`hf3fs/`  
> 关键文件：`fuse/IoRing.h`、`fuse/IovTable.h`、`fuse/FuseClients.h`、`lib/api/UsrbIo.md`

---

## 1. 客户端设计概述

3FS 提供两种客户端：

| 客户端 | 接入方式 | 适用场景 | 性能特点 |
|--------|----------|---------|---------|
| **FUSE 客户端** | 标准 POSIX 文件接口 | 通用应用，低迁移成本 | 受 FUSE 限制：内存拷贝 + 锁竞争 |
| **USRBIO（Native）客户端** | 专用 C API | 性能关键应用（数据加载、KVCache）| 零拷贝 + 异步 + io_uring 风格 |

两种客户端**共存**于同一 FUSE 守护进程中：
- FUSE 客户端通过内核 FUSE 模块处理标准 VFS 操作（`open`、`stat`、`rename` 等）。
- USRBIO 客户端应用程序通过 `libhf3fs_usrbio.so` 直接与 FUSE 守护进程的共享内存通信，绕过内核 FUSE 的数据路径。

---

## 2. FUSE 客户端的局限性

FUSE（Filesystem in Userspace）的工作原理：内核将 VFS 调用转发到用户态 FUSE 守护进程，守护进程处理后返回。

**性能瓶颈：**

1. **内存拷贝**：内核不能直接访问用户态 FUSE 进程的内存，数据需要在内核缓冲区和 FUSE 守护进程之间来回拷贝。
2. **锁竞争**：Linux 内核 FUSE 模块使用全局 spin lock 保护请求队列，高并发时 lock contention 成为瓶颈（约 400K 4KiB reads/s 上限）。
3. **单次写入不支持并发**：Linux 5.x FUSE 不支持对同一文件的并发写入，需要应用层分文件写入。

```
FUSE 数据路径（读取 4KB 数据）：
用户进程 read(fd, buf, 4096)
    → 内核 VFS
    → FUSE 内核模块（spin lock 保护共享队列）
    → FUSE 守护进程（memcpy：内核缓冲区 → 用户态缓冲区）
    → StorageClient 向 Storage 节点发送 RDMA 读请求
    → 读取数据返回（memcpy：FUSE 守护进程 → 内核缓冲区）
    → 内核 VFS memcpy → 用户进程 buf
（2 次内存拷贝，至少 2 次上下文切换）
```

---

## 3. USRBIO：异步零拷贝 API

### 3.1 核心概念

**Iov（IO Vector）**：用户进程与 FUSE 守护进程共享的大块内存区域。

```
用户进程地址空间                    FUSE 守护进程地址空间
┌─────────────────────────────────────────────────────────┐
│                  共享内存（mmap 共享）                    │
│  hf3fs_iov.base → [                      data area    ] │
│                   [  block0  | block1  | block2  | ... ]  │
└─────────────────────────────────────────────────────────┘
↑ 用户写入数据（写操作）或读取结果（读操作）
↑ FUSE 守护进程注册 IB Memory Region，Storage 节点通过 RDMA 直接读写
```

**Ior（IO Ring）**：用户进程与 FUSE 守护进程共享的环形缓冲区，仿照 io_uring 的 SQE/CQE 设计。

```
共享内存布局：
┌──────────────────────────────────────────────────────────────┐
│ sqeHead(4B) │ sqeTail(4B) │ cqeHead(4B) │ cqeTail(4B)        │
│ IoArgs[N]（SQE ring，N 个槽位）                                │
│ IoCqe[N]（CQE ring，N 个槽位）                                 │
│ IoSqe[N]（SQE 索引 ring）                                     │
│ sem_t（提交信号量，通知 FUSE 守护进程）                         │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 数据结构

```cpp
// src/fuse/IoRing.h
struct IoArgs {
  uint8_t bufId[16];    // Iov 的 UUID（唯一标识共享内存区域）
  size_t bufOff;        // 在 Iov 中的偏移

  uint64_t fileIid;     // 文件的 inode ID（对应注册的 fd）
  size_t fileOff;       // 文件偏移
  uint64_t ioLen;       // I/O 长度

  const void *userdata; // 用户自定义标记（原样返回到 CQE）
};

struct IoSqe {
  int32_t index;         // IoArgs 数组的索引
  const void *userdata;
};

struct IoCqe {
  int32_t index;         // 对应的 SQE 索引
  int32_t reserved;
  int64_t result;        // 返回值（>0 成功字节数，<0 错误码）
  const void *userdata;
};
```

`IoRing` 类的核心实现：

```cpp
// src/fuse/IoRing.h
class IoRing : public std::enable_shared_from_this<IoRing> {
 public:
  // 计算 Ior 占用的共享内存大小
  static size_t bytesRequired(int entries) {
    auto n = ringMarkerSize();
    // n*4 for sqe/cqe head/tail markers
    return n * 4 + sizeof(sem_t) +
           (sizeof(IoArgs) + sizeof(IoCqe) + sizeof(IoSqe)) * (entries + 1) + 4096;
  }

  // 提交 SQE（用户进程调用）
  bool addSqe(int idx, const void *userdata) {
    auto h = sqeHead.load();
    if ((h + 1) % entries == sqeTail.load()) return false;  // 队列满
    sqeSection[h] = {idx, userdata};
    sqeHead.store((h + 1) % entries);  // 原子更新，无锁
    return true;
  }

  // 批量处理 SQE → 发送到 StorageClient（FUSE 守护进程端）
  CoTask<void> process(
      int spt, int toProc,
      storage::client::StorageClient &storageClient,
      const storage::client::IoOptions &storageIo,
      // ...
  );

 public:
  // 使用 atomic_ref 实现无锁环形缓冲区指针
  std::atomic_ref<int32_t> sqeHead;  // SQE 生产者头指针
  std::atomic_ref<int32_t> sqeTail;  // SQE 消费者尾指针
  std::atomic_ref<int32_t> cqeHead;  // CQE 生产者头指针（FUSE 守护进程写）
  std::atomic_ref<int32_t> cqeTail;  // CQE 消费者尾指针（用户进程读）
};
```

---

## 4. Iov 管理：IovTable

`IovTable` 在 FUSE 守护进程中管理所有已注册的 Iov（共享内存区域）：

```cpp
// src/fuse/IovTable.h
class IovTable {
 public:
  // 注册新 Iov：用户进程创建共享内存后，通知 FUSE 守护进程注册到 IB
  Result<pair<meta::Inode, shared_ptr<lib::ShmBuf>>> addIov(
      const char *key,
      const Path &shmPath,    // POSIX 共享内存路径（/dev/shm/...）
      pid_t pid,
      const meta::UserInfo &ui,
      folly::Executor::KeepAlive<> exec,
      storage::client::StorageClient &sc);  // 用于 IB 内存注册

  // 注销 Iov
  Result<shared_ptr<lib::ShmBuf>> rmIov(const char *key, const meta::UserInfo &ui);

  // 查询 Iov（读写时验证 buf 地址是否在有效 Iov 范围内）
  Result<meta::Inode> lookupIov(const char *key, const meta::UserInfo &ui);

  // 内部存储
  robin_hood::unordered_map<Uuid, int> shmsById;    // UUID -> slot index
  std::unique_ptr<AtomicSharedPtrTable<lib::ShmBuf>> iovs;  // slot index -> ShmBuf
};
```

**注册机制**：当用户进程调用 `hf3fs_iovcreate()` 时：
1. 创建 POSIX 共享内存（`shm_open` + `mmap`）。
2. 通过特殊路径（`/.iov/`）发起 FUSE `open` 调用，通知守护进程。
3. FUSE 守护进程将该内存区域注册到所有 IB 设备（`ibv_reg_mr`），获取 Memory Region。
4. 未来该区域内的数据可以通过 RDMA 直接传输，无需任何内存拷贝。

---

## 5. USRBIO API 完整使用流程

```c
// 1. 创建 Ior（IO 请求环）
struct hf3fs_ior ior;
hf3fs_iorcreate4(&ior,
    "/hf3fs/mount/point",  // 区分不同 3FS 挂载点
    1024,                   // 最大并发请求数
    true,                   // for_read = true
    0,                      // io_depth = 0（无批处理限制）
    0,                      // timeout = 0
    -1,                     // numa = 当前进程所在 NUMA
    0);                     // flags

// 2. 创建 Iov（共享内存区域）
struct hf3fs_iov iov;
hf3fs_iovcreate(&iov, "/hf3fs/mount/point",
    1024 * 32 * 1024 * 1024,  // 32GB 共享内存
    0,                          // block_size = 0（单块）
    -1);                        // numa = 当前进程

// 3. 注册文件描述符
int fd = open("/hf3fs/mount/point/data.bin", O_RDONLY);
int reg_fd = hf3fs_reg_fd(fd, 0);  // 返回负数 "pseudo-fd"

// 4. 提交 IO 请求（非阻塞）
for (int i = 0; i < 1024; i++) {
    hf3fs_prep_io(&ior, &iov,
        true,                        // read
        iov.base + i * 32 * 1024 * 1024,  // 数据写入 Iov 的位置
        reg_fd,                      // 注册的 fd
        i * 32 * 1024 * 1024,        // 文件偏移
        32 * 1024 * 1024,            // 读取大小（32MB）
        (void*)(uintptr_t)i);        // userdata
}

// 5. 通知 FUSE 守护进程（可选，守护进程也会定期轮询）
hf3fs_submit_ios(&ior);

// 6. 等待完成
struct hf3fs_cqe cqes[1024];
int n = hf3fs_wait_for_ios(&ior, cqes, 1024, 1024, NULL);

// 7. 处理结果（直接使用 iov.base 中的数据，零拷贝）
for (int i = 0; i < n; i++) {
    int idx = (int)(uintptr_t)cqes[i].userdata;
    void *data = iov.base + idx * 32 * 1024 * 1024;
    // 直接访问 data，无需任何拷贝
}

// 8. 清理
hf3fs_dereg_fd(fd);
close(fd);
hf3fs_iovdestroy(&iov);
hf3fs_iordestroy(&ior);
```

---

## 6. FuseClients：文件系统状态管理

`FuseClients` 维护 FUSE 守护进程的核心状态：

```cpp
// src/fuse/FuseClients.h
struct RcInode {
  // 文件的动态写入状态（per open fd）
  struct DynamicAttr {
    uint64_t written = 0;   // 已写入的最大偏移
    uint64_t synced = 0;    // 已定期同步到 Meta 的长度
    uint64_t fsynced = 0;   // 已通过 fsync/close 精确同步的长度
    flat::Uid writer = flat::Uid(0);

    uint32_t dynStripe = 1;  // 动态扩展的 stripe 数（小文件优化）
    std::optional<meta::VersionedLength> hintLength;
    std::optional<UtcTime> atime;
    std::optional<UtcTime> mtime;
  };
};
```

文件打开缓存（`open file table`）：
- 每个打开的文件保存一个 `RcInode` 对象，记录写入进度。
- 多个 fd 可以引用同一个 `RcInode`（通过 `shared_ptr` 共享）。
- 引用计数归零（所有 fd 关闭）时触发最终 sync（精确文件长度更新）。

---

## 7. FUSE 低级操作（fuse_lowlevel_ops）

FUSE 守护进程注册一套低级操作回调：

```cpp
// src/fuse/FuseOps.cc
// 关键操作：
static void hf3fs_lookup(fuse_req_t req, fuse_ino_t parent, const char *name);
static void hf3fs_getattr(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
static void hf3fs_mkdir(fuse_req_t req, fuse_ino_t parent, ...);
static void hf3fs_open(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
static void hf3fs_read(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off, ...);
static void hf3fs_write_buf(fuse_req_t req, fuse_ino_t ino, struct fuse_bufvec *bufv, ...);
static void hf3fs_fsync(fuse_req_t req, fuse_ino_t ino, int datasync, ...);
static void hf3fs_rename(fuse_req_t req, fuse_ino_t parent, ...);
```

**特殊的 IOV/IOR 虚拟目录**：

FUSE 守护进程在挂载点内虚拟出两个特殊目录：
- `/.iov/`：用于 Iov 的注册/注销（`open` 触发 `addIov`，`release` 触发 `rmIov`）。
- `/.ior/`：用于 Ior 的创建/销毁（类似 `eventfd` 机制）。

应用程序通过 `open("/.iov/uuid", ...)` 这样的标准系统调用触发 Iov 注册，与 FUSE 守护进程通信，无需任何特殊权限或内核模块。

---

## 8. Python 接口（hf3fs 包）

`hf3fs` Python 包封装了 FUSE 挂载和 USRBIO 操作：

```python
# hf3fs/fuse.py
import hf3fs

# 获取 FUSE 客户端实例
client = hf3fs.FuseClient(mount_point="/mnt/hf3fs")

# 通过 USRBIO 高性能读取
with hf3fs.IovBuffer(mount_point, size=1<<30) as iov:
    with hf3fs.IoRing(mount_point, entries=256, for_read=True) as ior:
        fd = open("/mnt/hf3fs/data.bin")
        ior.prep_io(iov, fd, offset=0, length=4096)
        ior.submit()
        results = ior.wait_for_ios(min_results=1)
```

---

## 9. 模块依赖关系

```
客户端（FUSE + USRBIO）
  │
  ├── 依赖 Meta Client（文件元数据操作）
  │     └── open/stat/rename/mkdir → Meta Service
  │
  ├── 依赖 Storage Client（数据读写）
  │     └── IoRing::process() → StorageClient::read/write
  │
  ├── 依赖 Mgmtd Client（路由信息获取）
  │     └── 发现可用 Meta/Storage 节点
  │
  ├── 依赖 RDMA 网络层（Iov 内存注册）
  │     └── IovTable::addIov → ibv_reg_mr（注册共享内存）
  │
  ├── 依赖 common/lib/Shm（POSIX 共享内存管理）
  │
  └── 暴露给用户：
        ├── libhf3fs_usrbio.so（C API）
        ├── hf3fs Python 包
        └── 标准 FUSE 挂载（POSIX 文件接口）
```
