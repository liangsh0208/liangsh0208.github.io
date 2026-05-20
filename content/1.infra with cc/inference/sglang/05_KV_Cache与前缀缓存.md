---
created: 2026-05-06
---

# 第五章：KV Cache 与前缀缓存

## 一、概述

SGLang 的核心创新之一是基于基数树 (Radix Tree) 的前缀缓存系统 —— RadixAttention。本章详细分析 KV Cache 的内存管理和前缀缓存实现。

## 二、整体架构

### 2.1 三层内存架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        第一层：RadixCache (基数树)                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  管理缓存的逻辑结构，实现前缀匹配和缓存复用                                 ││
│  │  TreeNode.key → token_ids                                               ││
│  │  TreeNode.value → indices into KVPool                                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ indices
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    第二层：ReqToTokenPool (请求映射)                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  [num_requests, max_context_len]                                        ││
│  │  每一行存储一个请求的所有 token 在 KVPool 中的位置                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ token indices
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    第三层：KVCache (实际 KV 数据)                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  MHATokenToKVPool:                                                      ││
│  │    k_buffer[layer]: [size, head_num, head_dim]                          ││
│  │    v_buffer[layer]: [size, head_num, head_dim]                          ││
│  │                                                                         ││
│  │  MLATokenToKVPool (DeepSeek):                                           ││
│  │    kv_buffer[layer]: [size, 1, kv_lora_rank + qk_rope_dim]              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## 三、RadixCache 实现

### 3.1 类结构

**文件位置**：`python/sglang/srt/mem_cache/radix_cache.py`

```python
class RadixCache(BasePrefixCache):
    """
    基于 Radix Tree 的前缀缓存实现。

    核心功能：
    1. 前缀匹配：找到最长匹配的缓存前缀
    2. 缓存插入：存储新的 KV Cache
    3. 淘汰管理：LRU/LFU/FIFO 等策略
    4. 引用计数：保护正在使用的缓存
    """

    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.eviction_policy = params.eviction_policy.lower()

        # 树结构
        self.root_node = TreeNode()

        # 统计
        self.evictable_size_ = 0       # 可淘汰的字节数
        self.protected_size_ = 0       # 受保护（正在使用）的字节数

        # 可淘汰叶子节点集合
        self.evictable_leaves: Set[TreeNode] = set()
```

### 3.2 TreeNode 结构

```python
class TreeNode:
    """Radix Tree 节点"""

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        # 树结构
        self.children: Dict[any, TreeNode] = defaultdict(TreeNode)
        self.parent: Optional[TreeNode] = None

        # 数据
        self.key: RadixKey = None             # token IDs
        self.value: Optional[torch.Tensor] = None  # KV cache indices

        # 引用计数 (防止淘汰)
        self.lock_ref: int = 0

        # 淘汰策略相关
        self.last_access_time = time.monotonic()  # LRU
        self.creation_time = time.monotonic()     # FIFO
        self.hit_count: int = 0                    # LFU
        self.priority: int = priority              # 优先级

        # 主机备份 (分离式推理)
        self.host_ref_counter: int = 0
        self.host_value: Optional[torch.Tensor] = None
        self.hash_value: Optional[List[str]] = None

    @property
    def evicted(self) -> bool:
        return self.value is None

    @property
    def backuped(self) -> bool:
        return self.host_value is not None
```

### 3.3 RadixKey 结构

```python
class RadixKey:
    """Radix Tree 的键"""

    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        self.token_ids = token_ids      # token ID 序列
        self.extra_key = extra_key      # 命名空间标签 (LoRA ID 等)
        self.is_bigram = is_bigram      # 用于 EAGLE 推测解码
```

`extra_key` 提供命名空间隔离：
- 相同 token IDs 但不同 `extra_key` 的条目独立存储
- 支持不同 LoRA 适配器的缓存隔离

## 四、前缀匹配算法

### 4.1 match_prefix 方法

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    """查找最长匹配的前缀"""

    key = RadixKey(params.key)

    # 页对齐截断 (如果 page_size > 1)
    if self.page_size != 1:
        page_aligned_len = len(key) // self.page_size * self.page_size
        key = key[:page_aligned_len]

    # 执行匹配
    value, last_node = self._match_prefix_helper(self.root_node, key)

    # 展平结果
    if value:
        device_indices = torch.cat(value)
    else:
        device_indices = torch.empty((0,), dtype=torch.int64, device='cuda')

    return MatchResult(
        device_indices=device_indices,
        last_device_node=last_node,
    )
```

### 4.2 核心匹配算法

```python
def _match_prefix_helper(
    self,
    node: TreeNode,
    key: RadixKey,
) -> Tuple[List[torch.Tensor], TreeNode]:
    """递归前缀匹配"""

    access_time = time.monotonic()
    node.last_access_time = access_time  # 更新 LRU 时间

    value = []
    child_key = self._get_child_key(key)  # 获取第一个 token/page

    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]
        child.last_access_time = access_time

        # 计算匹配长度
        prefix_len = self._key_match(child.key, key)

        if prefix_len < len(child.key):
            # 部分匹配 - 需要分裂节点
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            # 完全匹配 - 继续下一层
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key) > 0:
                child_key = self._get_child_key(key)

    return value, node
```

### 4.3 键匹配函数

```python
# page_size = 1 时
def _key_match_page_size1(key0: RadixKey, key1: RadixKey) -> int:
    """逐 token 匹配"""
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i

# page_size > 1 时
def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int) -> int:
    """按页匹配"""
    i = 0
    while i < min(len(key0), len(key1)):
        if key0.token_ids[i:i+page_size] != key1.token_ids[i:i+page_size]:
            break
        i += page_size
    return i
```

### 4.4 节点分裂

```python
def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
    """分裂节点 - 当匹配在节点中间结束"""

    # 创建新节点
    new_node = TreeNode(priority=child.priority)
    new_node.hit_count = child.hit_count
    new_node.children = {self._get_child_key(key[split_len:]): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone()

    # 更新子节点
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()

    # 更新父节点的子节点映射
    new_node.parent.children[self._get_child_key(key)] = new_node

    return new_node
```

## 五、缓存插入

### 5.1 insert 方法

```python
def insert(self, params: InsertParams) -> InsertResult:
    """插入 KV Cache"""

    key = RadixKey(params.key)
    value = params.value
    priority = params.priority

    # 页对齐
    if self.page_size != 1:
        page_aligned_len = len(key) // self.page_size * self.page_size
        key = key[:page_aligned_len]
        value = value[:page_aligned_len]

    # 执行插入
    self._insert_helper(self.root_node, key, value, priority)

    return InsertResult()
```

### 5.2 插入辅助函数

```python
def _insert_helper(
    self,
    node: TreeNode,
    key: RadixKey,
    value: torch.Tensor,
    priority: int = 0,
):
    """递归插入"""

    # 传播最大优先级
    node.priority = max(node.priority, priority)

    if len(key) == 0:
        return

    total_prefix_length = 0
    child_key = self._get_child_key(key)

    # 遍历现有节点
    while len(key) > 0 and child_key in node.children:
        node = node.children[child_key]
        prefix_len = self._key_match(node.key, key)
        total_prefix_length += prefix_len

        key = key[prefix_len:]
        value = value[prefix_len:]

        if prefix_len < len(node.key):
            # 需要分裂
            new_node = self._split_node(node.key, node, prefix_len)
            self._inc_hit_count(new_node)
            node = new_node
        else:
            self._inc_hit_count(node)

        if len(key) > 0:
            child_key = self._get_child_key(key)

    # 创建新叶子节点
    if len(key) > 0:
        new_node = TreeNode(priority=priority)
        new_node.parent = node
        new_node.key = key
        new_node.value = value.clone()
        node.children[child_key] = new_node

        # 更新统计
        self.evictable_size_ += len(key)
        self._update_leaf_status(node)
        self._update_leaf_status(new_node)

    return total_prefix_length
```

## 六、淘汰机制

### 6.1 淘汰策略

**文件位置**：`python/sglang/srt/mem_cache/evict_policy.py`

```python
class EvictionStrategy(abc.ABC):
    """淘汰策略基类"""

    @abc.abstractmethod
    def get_priority(self, node: TreeNode) -> any:
        """返回淘汰优先级 (越小越先淘汰)"""
        pass


class LRUStrategy(EvictionStrategy):
    """LRU - 最近最少使用"""

    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time  # 越旧越先淘汰


class LFUStrategy(EvictionStrategy):
    """LFU - 最少频率使用"""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)  # 频率低且旧


class FIFOStrategy(EvictionStrategy):
    """FIFO - 先进先出"""

    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time  # 越早创建越先淘汰


class PriorityStrategy(EvictionStrategy):
    """优先级策略"""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    """分段 LRU"""

    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)
```

### 6.2 淘汰执行

```python
def evict(self, params: EvictParams) -> EvictResult:
    """执行淘汰"""

    num_tokens = params.num_tokens
    leaves = list(self.evictable_leaves)

    # 构建最小堆
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node)
        for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap) > 0:
        _, node = heapq.heappop(eviction_heap)

        # 释放内存
        self.token_to_kv_pool_allocator.free(node.value)
        num_evicted += len(node.value)

        # 删除节点
        self._delete_leaf(node)

        # 父节点可能成为可淘汰叶子
        if len(node.parent.children) == 0 and node.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(node.parent)
            heapq.heappush(eviction_heap, (new_priority, node.parent))

    return EvictResult(num_tokens_evicted=num_evicted)
```

### 6.3 叶子状态更新

```python
def _update_leaf_status(self, node: TreeNode):
    """更新叶子和可淘汰状态"""

    # 不可淘汰：已淘汰或被锁定
    if node.evicted or node.lock_ref > 0:
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        return

    # 不是叶子：有未淘汰的子节点
    for child in node.children.values():
        if not child.evicted:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

    # 是可淘汰叶子
    if node not in self.evictable_leaves:
        self.evictable_leaves.add(node)
```

## 七、引用计数

### 7.1 增加引用

```python
def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
    """增加引用计数 (防止淘汰)"""

    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            # 从可淘汰变为受保护
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)

        node.lock_ref += 1
        self._update_leaf_status(node)
        node = node.parent

    return IncLockRefResult(delta=delta)
```

### 7.2 减少引用

```python
def dec_lock_ref(self, node: TreeNode, params: Optional[DecLockRefParams] = None):
    """减少引用计数 (可能变为可淘汰)"""

    delta = 0
    while node != self.root_node:
        if node.lock_ref == 1:
            # 从受保护变为可淘汰
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)

        node.lock_ref -= 1
        self._update_leaf_status(node)
        node = node.parent

    return DecLockRefResult(delta=delta)
```

## 八、内存池实现

### 8.1 ReqToTokenPool

**文件位置**：`python/sglang/srt/mem_cache/memory_pool.py`

```python
class ReqToTokenPool:
    """
    请求到 Token 位置映射池。

    每个请求分配一行，存储该请求所有 token 在 KVPool 中的位置。
    """

    def __init__(self, size: int, max_context_len: int, device: str):
        self.size = size
        self.max_context_len = max_context_len

        # [size, max_context_len] - 每行是一个请求的 token 位置
        self.req_to_token = torch.zeros(
            (size, max_context_len),
            dtype=torch.int32,
            device=device,
        )

        # 空闲槽位
        self.free_slots = list(range(size))

    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        """分配槽位给请求"""

        # 分块 prefill 可能复用已有槽位
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        need_size = len(reqs) - len(reusing)

        if need_size > len(self.free_slots):
            return None  # 内存不足

        # 分配新槽位
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1

        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        """释放槽位"""
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def write(self, req_pool_idx: int, token_positions: torch.Tensor, cache_indices: torch.Tensor):
        """写入 token 位置"""
        self.req_to_token[req_pool_idx, token_positions] = cache_indices
```

### 8.2 MHATokenToKVPool

```python
class MHATokenToKVPool(KVCache):
    """
    多头注意力的 KV Cache 池。

    标准 Transformer 模型使用。
    """

    def __init__(
        self,
        size: int,                    # token 槽位数量
        page_size: int,               # 页大小
        dtype: torch.dtype,           # 数据类型
        head_num: int,                # 注意力头数
        head_dim: int,                # 每个头的维度
        layer_num: int,               # 层数
        device: str,
    ):
        self.size = size
        self.page_size = page_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num

        # 创建缓冲区
        # 每层: [size + page_size, head_num, head_dim]
        self.k_buffer = [
            torch.zeros(
                (size + page_size, head_num, head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]

        self.v_buffer = [
            torch.zeros(
                (size + page_size, head_num, head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]

        # 存储数据指针 (用于高效操作)
        self.k_data_ptrs = torch.tensor([x.data_ptr() for x in self.k_buffer])
        self.v_data_ptrs = torch.tensor([x.data_ptr() for x in self.v_buffer])

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的 K/V 缓冲区"""
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """设置 KV 缓存"""
        # loc: 要写入的位置索引
        # cache_k, cache_v: 要写入的值
        self.k_buffer[layer.layer_id][loc] = cache_k
        self.v_buffer[layer.layer_id][loc] = cache_v
```

### 8.3 MLATokenToKVPool

DeepSeek-V2/V3 风格的 Multi-Latent Attention 缓存：

```python
class MLATokenToKVPool(KVCache):
    """
    Multi-Latent Attention 的压缩 KV Cache。

    不存储完整的 K/V，而是存储压缩的潜在表示。
    """

    def __init__(
        self,
        size: int,
        kv_lora_rank: int,           # 压缩维度
        qk_rope_head_dim: int,       # RoPE 维度
        layer_num: int,
        device: str,
    ):
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim

        # 每层: [size + page_size, 1, kv_cache_dim]
        # 更紧凑的存储
        self.kv_buffer = [
            torch.zeros(
                (size + page_size, 1, self.kv_cache_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
```

## 九、内存分配器

### 9.1 TokenToKVPoolAllocator

**文件位置**：`python/sglang/srt/mem_cache/allocator.py`

```python
class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    Token 级别的 KV Cache 分配器 (page_size = 1)。
    """

    def __init__(self, size: int, dtype: torch.dtype, device: str, ...):
        super().__init__(size, 1, dtype, device, ...)  # page_size = 1
        self.clear()

    def clear(self):
        # 槽位 0 保留给 padding token
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64, device=self.device)
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """分配指定数量的槽位"""

        # 尝试合并已释放的页面
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None  # 内存不足

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        """释放槽位"""

        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.need_sort:
                # 延迟合并
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            # 批量释放模式
            self.free_group.append(free_index)
```

### 9.2 PagedTokenToKVPoolAllocator

页对齐的分配器，使用 Triton 内核优化：

```python
class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    页对齐的 KV Cache 分配器。
    """

    def __init__(self, size: int, page_size: int, ...):
        super().__init__(size, page_size, ...)
        self.num_pages = size // page_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """页对齐分配"""

        num_pages = (need_size + self.page_size - 1) // self.page_size

        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        # 展开页索引为 token 索引
        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices[:need_size]
```

### 9.3 Triton 内核优化

```python
@triton.jit
def alloc_extend_kernel(
    pre_lens_ptr,     # 前缀长度
    seq_lens_ptr,     # 序列长度
    last_loc_ptr,     # 最后位置
    free_page_ptr,    # 空闲页
    out_indices,      # 输出索引
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    """Prefill 扩展分配内核"""

    pid = tl.program_id(0)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)

    # 计算需要分配的范围
    extend_len = seq_len - pre_len

    # 三部分：
    # 1. 填充旧的部分页
    # 2. 新的完整页
    # 3. 新的部分页


@triton.jit
def alloc_decode_kernel(...):
    """Decode 分配内核 - 每个序列分配一个 token"""
    ...
```

## 十、缓存复用示例

### 10.1 基本流程

```
请求 1: "The capital of France is Paris"
请求 2: "The capital of France is Lyon"

RadixTree:
                    ┌─────────────────────┐
                    │       Root          │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │ "The capital of     │
                    │  France is "        │
                    │  (共享前缀)          │
                    └──────────┬──────────┘
                          ┌────┴────┐
                          │         │
                    ┌─────┴─────┐ ┌──┴────┐
                    │  "Paris"  │ │"Lyon" │
                    │  (请求1)   │ │(请求2) │
                    └───────────┘ └───────┘

请求 2 只需要 Prefill "Lyon"，复用前面的 KV Cache。
```

### 10.2 性能收益

```python
# 无缓存
请求 1: Prefill 7 tokens

# 有缓存
请求 2: Prefill 1 token (复用 6 tokens)
缓存命中率: 85.7%
延迟降低: ~85%
```

## 十一、总结

### 11.1 核心设计

1. **三层架构**：RadixCache → ReqToTokenPool → KVCache
2. **前缀复用**：基数树实现高效前缀匹配
3. **延迟释放**：支持淘汰和重新使用
4. **灵活淘汰**：LRU/LFU/FIFO 等多种策略
5. **压缩存储**：MLA 用压缩表示节省内存

### 11.2 关键优化

- **节点分裂**：支持部分匹配的精确缓存
- **引用计数**：保护正在使用的缓存
- **批量操作**：减少内存碎片
- **Triton 内核**：高效的 GPU 分配

---

**上一章**：[调度系统](04_调度系统.md)

**下一章**：[模型执行](06_模型执行.md)