# 缓存与 Fingerprint 机制

> **【文档定位】** 确定性缓存与数据转换追踪系统
>
> **【版本信息】** 基于 HuggingFace datasets v4.8.5.dev0 (`src/datasets/fingerprint.py` ~450行)
>
> **【前置知识】** 哈希算法、对象序列化、Python 函数签名

---

## 模块概述

datasets 库的确定性缓存机制是其核心优势之一。通过 `Fingerprint` 系统，每次数据转换操作都会生成确定性的哈希标识，相同输入和相同操作会复用已缓存的结果，避免重复计算。

### 核心特性

| 特性 | 说明 |
|------|------|
| **确定性缓存** | 相同输入 + 相同操作 = 相同缓存 |
| **自动追踪** | map/filter 自动计算新 fingerprint |
| **分层缓存** | 支持内存缓存和磁盘缓存 |
| **缓存控制** | 灵活的缓存启用/禁用/自定义策略 |

### Fingerprint 生成公式

```
new_fingerprint = hash(
    old_fingerprint +
    function_code +
    function_kwargs +
    operation_type
)
```

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        缓存与 Fingerprint 架构                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      Fingerprint 生成流程                            │  │
│  │                                                                      │  │
│  │   原始数据集                                                          │  │
│  │       │                                                               │  │
│  │       ▼                                                               │  │
│  │   _fingerprint = "abc123"                                            │  │
│  │       │                                                               │  │
│  │       ▼ map(transform_fn, batched=True)                             │  │
│  │       │                                                               │  │
│  │   计算新 fingerprint:                                               │  │
│  │   hasher = Hasher()                                                  │  │
│  │   hasher.update("abc123")                 # 原 fingerprint          │  │
│  │   hasher.update(transform_fn.__code)      # 函数字节码              │  │
│  │   hasher.update({'batched': True})        # 参数                    │  │
│  │   new_fp = hasher.hexdigest()              # "def456"                │  │
│  │       │                                                               │  │
│  │       ▼                                                               │  │
│  │   检查缓存: ~/.cache/huggingface/datasets/.../def456/               │  │
│  │       │                                                               │  │
│  │   ┌───┴───┐                                                          │  │
│  │   │ 存在  │ 不存在                                                    │  │
│  │   ▼       ▼                                                           │  │
│  │  加载    执行转换                                                     │  │
│  │          写入缓存                                                     │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      Caching 系统组件                                 │  │
│  │                                                                      │  │
│  │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐         │  │
│  │   │ Hasher        │   │ Fingerprint   │   │ Caching       │         │  │
│  │   │ - xxhash      │   │ - 确定性哈希  │   │ - 内存缓存    │         │  │
│  │   │ - dill        │   │ - 版本控制    │   │ - 磁盘缓存    │         │  │
│  │   │ - 序列化      │   │ - 冲突检测    │   │ - 自动清理    │         │  │
│  │   └───────────────┘   └───────────────┘   └───────────────┘         │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 核心实现

### Fingerprint 生成

```python
# src/datasets/fingerprint.py

def generate_fingerprint(dataset) -> str:
    """基于 dataset 状态生成唯一哈希"""
    hasher = Hasher()
    hasher.update(dataset._fingerprint)
    hasher.update(dataset._data.table)  # 基于数据摘要
    if dataset._indices is not None:
        hasher.update(dataset._indices.table)
    return hasher.hexdigest()

def fingerprint_transform(function, *args, **kwargs):
    """
    装饰器：为 transform 方法生成 fingerprint
    新 fingerprint = hash(原 fingerprint + 函数签名 + 参数)
    """
    def _fingerprint(function):
        # 序列化函数代码和参数
        m = Hasher()
        m.update(function)  # 使用 dill 序列化函数
        m.update(kwargs)
        return m.hexdigest()
```

### Hasher 实现

```python
class Hasher:
    """基于 xxhash 的快速确定性哈希"""
    def __init__(self):
        self.m = xxhash.xxh64()

    def update(self, obj):
        """使用 dill 序列化任意 Python 对象，然后哈希"""
        s = dumps(obj, protocol=-1)  # dill 序列化
        self.m.update(s)

    def hexdigest(self) -> str:
        return self.m.hexdigest()
```

---

## 代码示例

### 基础缓存控制

```python
from datasets import load_dataset, disable_caching, enable_caching, is_caching_enabled

# 查看缓存状态
print(is_caching_enabled())  # True/False

# 禁用缓存（所有转换重新计算，不持久化）
disable_caching()

# 重新启用缓存
enable_caching()

# 缓存只影响新操作，不影响已缓存的结果
dataset = load_dataset("rajpurkar/squad", split="train")
# 即使禁用缓存，加载仍可能使用已下载的原始数据缓存
```

### 精细缓存控制

```python
from datasets import load_dataset

dataset = load_dataset("rajpurkar/squad", split="train")

# 缓存控制选项
result = dataset.map(
    my_transform,
    cache_file_name="/custom/path/to/cache.arrow",  # 指定缓存位置
    load_from_cache_file=True,                        # 优先读取缓存
    keep_in_memory=True,                              # 保持在内存（大数据慎用）
    new_fingerprint="my_custom_fp",                   # 强制指定 fingerprint
)

# 仅生成缓存，不返回结果（用于预处理）
dataset.map(
    heavy_transform,
    cache_only=True
)
```

### 环境变量控制

```python
import os

# 设置缓存目录
os.environ['HF_DATASETS_CACHE'] = '/path/to/large/disk/cache'

# 设置内存缓存阈值（字节）
os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE'] = '1000000000'  # ~1GB

# 离线模式（不使用网络，只使用缓存）
os.environ['HF_DATASETS_OFFLINE'] = '1'
```

### 查看与调试 Fingerprint

```python
from datasets import load_dataset

dataset = load_dataset("rajpurkar/squad", split="train")

# 查看当前 fingerprint
print(dataset._fingerprint)
# e.g., 'b0e5c3f4a2d1e8b9'

# 查看缓存文件位置
print(dataset.cache_files)
# [{'filename': '...', 'skip': 0}]

# 执行转换后查看新 fingerprint
transformed = dataset.map(lambda x: x)
print(transformed._fingerprint)
# 新的哈希值

# 多次执行相同的 map， fingerprint 相同（确定性缓存）
transformed2 = dataset.map(lambda x: x)
assert transformed._fingerprint == transformed2._fingerprint
```

### 强制刷新缓存

```python
# 方法1: 修改函数名或内容（改变函数签名）
def my_transform_v2(x):  # 修改函数名
    return x

dataset = dataset.map(my_transform_v2)

# 方法2: 使用 cache_file_name 指定新位置
dataset = dataset.map(my_transform, cache_file_name="/new/path.arrow")

# 方法3: 强制指定新 fingerprint
dataset = dataset.map(my_transform, new_fingerprint="forced_fp_1")

# 方法4: 暂时禁用缓存
disable_caching()
dataset = dataset.map(my_transform)
enable_caching()

# 方法5: 删除缓存文件
import shutil
shutil.rmtree(os.path.expanduser("~/.cache/huggingface/datasets/squad"))
```

### 多进程缓存

```python
# 多进程也会正确共享缓存
dataset = dataset.map(
    heavy_transform,
    num_proc=4,  # 4 进程并行
    batched=True,
    batch_size=1000
)

# 每个进程处理不同分区，但写入同一缓存文件
# fingerprint 在所有进程间保持一致
```

---

## 配置参数表

### 全局缓存函数

| 函数 | 说明 | 返回值 |
|------|------|--------|
| `enable_caching()` | 启用缓存 | None |
| `disable_caching()` | 禁用缓存 | None |
| `is_caching_enabled()` | 检查缓存是否启用 | bool |

### map/filter 缓存参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `load_from_cache_file` | bool | None | 是否从缓存文件加载 |
| `cache_file_name` | str | None | 自定义缓存文件名 |
| `keep_in_memory` | bool | False | 结果是否驻留内存 |
| `cache_only` | bool | False | 仅生成缓存不返回结果 |
| `new_fingerprint` | str | None | 强制指定新 fingerprint |

### 环境变量

| 变量 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `HF_DATASETS_CACHE` | path | `~/.cache/huggingface/datasets` | 缓存根目录 |
| `HF_DATASETS_IN_MEMORY_MAX_SIZE` | int | 0 | 内存缓存最大字节数 |
| `HF_DATASETS_OFFLINE` | bool | 0 | 离线模式 |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `dataset._fingerprint` | str | 当前 dataset 的 fingerprint |
| `dataset.cache_files` | list | 缓存文件列表 |

---

## 常见问题排查

### 缓存命中问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 修改代码后仍加载旧数据 | 相同 fingerprint 命中缓存 | 用 `cache_file_name` 或 `new_fingerprint` 强制刷新 |
| `map` 太慢 | 缓存写入磁盘 + 单进程 | 设置 `num_proc=4` 开启多进程；确认 `batched=True` |
| 缓存占满磁盘 | Arrow 文件过大 | 使用 `streaming=True` 流式处理；设置 `HF_DATASETS_CACHE` 指向大容量路径 |
| 找不到缓存文件 | 缓存目录权限或位置变更 | 用 `cache_dir` 显式指定；检查 `~/.cache/huggingface/datasets` |

### Fingerprint 相关问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 相同操作 fingerprint 不同 | 函数引用不同对象（如闭包） | 将函数定义在模块顶层；避免 lambda |
| 并行进程 fingerprint 不一致 | 多进程哈希计算差异 | 确保使用 `multiprocess` 而非 `multiprocessing` |
| 缓存文件损坏 | 写入中断或磁盘问题 | 删除损坏缓存目录，重新生成 |

### 内存相关问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 小数据也写入磁盘 | 缓存机制默认行为 | 设置 `keep_in_memory=True` |
| 内存占用持续增长 | 缓存累积 | 定期清理缓存目录；使用 `keep_in_memory=False` |

---

*文档生成于: 2026/04/20*
