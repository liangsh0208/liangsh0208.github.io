# Dataset 类与核心 API

> **【文档定位】** Dataset 核心类的 API 设计与使用方法
>
> **【版本信息】** 基于 HuggingFace datasets v4.8.5.dev0 (`src/datasets/arrow_dataset.py` ~8500行)
>
> **【前置知识】** Apache Arrow、内存映射、Python 数据类

---

## 模块概述

`Dataset` 是最核心的数据集类，基于 Apache Arrow 实现高性能数据访问。提供索引访问、批量处理、条件过滤、格式转换等完整功能。

### 核心特性

| 特性 | 说明 |
|------|------|
| **索引访问** | 支持 `dataset[0]`、`dataset[0:100]` 等丰富索引语法 |
| **批量处理** | `map()` 支持多进程并行和自动缓存 |
| **条件过滤** | `filter()` 通过布尔索引表实现零拷贝过滤 |
| **格式切换** | 支持 NumPy、PyTorch、TensorFlow、JAX、Pandas 等格式 |
| **智能缓存** | 转换操作自动缓存，相同操作复用结果 |

---

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                     Dataset Class                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  核心存储属性                                        │  │
│  │  - _data: pa.Table               (主数据表)         │  │
│  │  - _indices: Optional[pa.Table]  (行索引表)        │  │
│  │  - _fingerprint: str            (缓存指纹)        │  │
│  │  - info: DatasetInfo            (元数据)          │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  API 方法分层                                       │  │
│  │                                                     │  │
│  │  数据访问层:                                         │  │
│  │    __getitem__(), __len__(), __iter__()            │  │
│  │                                                     │  │
│  │  数据处理层:                                         │  │
│  │    map(), filter(), select(), shuffle(), sort()    │  │
│  │                                                     │  │
│  │  数据操作层:                                         │  │
│  │    add_column(), remove_columns(), rename_column() │  │
│  │    cast(), cast_column()                           │  │
│  │                                                     │  │
│  │  格式输出层:                                         │  │
│  │    set_format(), with_format(), reset_format()     │  │
│  │                                                     │  │
│  │  IO 层:                                             │  │
│  │    save_to_disk(), load_from_disk()                │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Dataset 类定义

```python
class Dataset(DatasetInfoMixin, DatasetTransformationNotAllowedError):
    """
    核心数据集类，提供内存高效的数据访问和转换能力。
    """
    def __init__(
        self,
        arrow_table: Table,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        indices_table: Optional[Table] = None,  # 用于支持行重排而不复制数据
    ):
        self._data = arrow_table        # 核心数据存储 (Table 对象)
        self._indices = indices_table   # 可选的行索引表
        self._format_type = None        # 输出格式: None/python/numpy/pandas/torch/tf/jax
        self._format_kwargs = {}
        self._format_columns = None
        self._fingerprint = None        # 缓存指纹
```

---

## 核心方法详解

### 数据访问

| 方法 | 功能 | 代码示例 |
|------|------|----------|
| `__getitem__(key)` | 索引/切片访问 | `dataset[0]` `dataset[0:10]` `dataset[[1,3,5]]` |
| `__len__()` | 返回数据集长度 | `len(dataset)` |
| `select(indices)` | 按索引选择子集 | `dataset.select(range(1000))` |

### map() 方法

```python
def map(
    self,
    function: Optional[Callable] = None,
    with_indices: bool = False,                    # 传入行索引
    with_rank: bool = False,                         # 传入进程 rank
    batched: bool = False,                           # 批量模式
    batch_size: Optional[int] = 1000,               # 批量大小
    drop_last_batch: bool = False,                  # 丢弃不完整批次
    remove_columns: Optional[Union[str, list[str]]] = None,  # 删除的列
    keep_in_memory: bool = False,                    # 结果是否驻留内存
    load_from_cache_file: Optional[bool] = None,     # 是否读缓存
    cache_file_name: Optional[str] = None,           # 缓存文件名
    writer_batch_size: Optional[int] = None,        # 写入批次大小
    features: Optional[Features] = None,              # 结果特征
    disable_nullable: bool = False,                   # 禁用 nullable 类型
    fn_kwargs: Optional[dict] = None,                # 传递给 function 的额外参数
    num_proc: Optional[int] = None,                  # 并行进程数
    desc: Optional[str] = None,                     # 进度条描述
    cache_only: bool = False,                       # 仅生成缓存
    new_fingerprint: Optional[str] = None,           # 强制指定指纹
) -> "Dataset":
    """
    Apply a function to all the elements in the table (individually or batched)
    and update the table.

    核心流程:
    1. 计算新 fingerprint = hash(old_fingerprint + function + kwargs)
    2. 检查缓存是否存在该 fingerprint 的 arrow 文件
    3. 如存在，直接加载缓存返回
    4. 如不存在，分批处理数据，写入新的 arrow 文件
    5. 新 arrow 表使用新 fingerprint 命名，保证确定性缓存
    """
```

### filter() 方法

```python
# 通过布尔索引表实现，不复制数据
def filter(
    self,
    function: Callable,                              # 返回 bool 的处理函数
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = None,
    fn_kwargs: Optional[dict] = None,
    num_proc: Optional[int] = None,
    desc: Optional[str] = None,
) -> "Dataset":
    """
    过滤示例：只保留长问题
    """
    # 创建布尔掩码，更新 _indices 表
```

### with_format() / set_format()

```python
# 设置输出格式
from datasets import load_dataset
import torch

dataset = load_dataset("rajpurkar/squad", split="train")

# set_format: 持久化设置格式
dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels'],
    device='cuda:0'
)

# with_format: 临时切换格式
with dataset.with_format('numpy'):
    batch = dataset[0:100]  # 返回 NumPy 数组

# reset_format: 恢复默认
dataset.reset_format()
```

---

## 代码示例

### 基础使用

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("rajpurkar/squad", split="train")
print(dataset)
# Dataset({
#     features: ['id', 'title', 'context', 'question', 'answers'],
#     num_rows: 87599
# })

# 访问单条数据
dataset[0]
# {
#     'id': '57317be47776f4190067007b',
#     'title': 'University_of_Notre_Dame',
#     'context': 'Architecturally, the school has...',
#     'question': 'To whom did the Virgin Mary...',
#     'answers': {'text': [...], 'answer_start': [...]}
# }

# 批量访问
dataset[0:3]  # 返回 dict of list
```

### map() 使用示例

```python
# 单条处理模式
def add_question_length(example):
    example['question_length'] = len(example['question'].split())
    return example

dataset_with_len = dataset.map(add_question_length)

# 批量处理模式（高效）
def tokenize_batch(batch):
    # batch 是 dict of list，非 list of dict
    # 如: {'text': ['a', 'b', 'c']}
    return {
        'text_ids': [[ord(c) for c in text] for text in batch['text']]
    }

dataset_batched = dataset.map(
    tokenize_batch,
    batched=True,           # 必须设置
    batch_size=1000,        # 每批大小
    num_proc=4,             # 多进程并行
    desc="Processing..."    # 进度条描述
)

# 使用 with_indices
def add_idx(example, idx):
    example['index'] = idx
    return example

dataset_with_idx = dataset.map(add_idx, with_indices=True)

# 使用 fn_kwargs 传递额外参数
def add_prefix(example, prefix=">"):
    example['prefixed'] = prefix + example['question']
    return example

dataset_prefixed = dataset.map(add_prefix, fn_kwargs={'prefix': 'Q: '})
```

### filter() 使用示例

```python
# 基础过滤
short_questions = dataset.filter(lambda x: len(x['question'].split()) < 5)

# 批量过滤（更快）
filtered = dataset.filter(
    lambda batch: [len(q.split()) < 5 for q in batch['question']],
    batched=True
)

# 复杂条件
def complex_filter(example):
    return (
        len(example['question'].split()) > 5 and
        len(example['context'].split()) < 100
    )

filtered = dataset.filter(complex_filter)
```

### select() 与 shuffle()

```python
# 选择子集（零拷贝，仅更新索引表）
train_1000 = dataset.select(range(1000))
train_odd = dataset.select(list(range(1, len(dataset), 2)))

# 随机打乱
shuffled = dataset.shuffle(seed=42)

# 打乱后选择
subset = dataset.shuffle(seed=42).select(range(1000))
```

---

## 配置参数表

### Dataset 初始化参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `arrow_table` | Table | 核心数据表（必需） |
| `info` | DatasetInfo | 数据集元数据 |
| `split` | NamedSplit | 数据集分片标识 |
| `indices_table` | Table | 行索引表（支持重排） |

### map() 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `function` | Callable | None | 处理函数 |
| `with_indices` | bool | False | 传入行索引 |
| `with_rank` | bool | False | 传入进程 rank |
| `batched` | bool | False | 批量模式 |
| `batch_size` | int | 1000 | 批量大小 |
| `drop_last_batch` | bool | False | 丢弃不完整批次 |
| `remove_columns` | str/List[str] | None | 删除的列 |
| `keep_in_memory` | bool | False | 结果驻留内存 |
| `load_from_cache_file` | bool | None | 是否读缓存 |
| `cache_file_name` | str | None | 指定缓存位置 |
| `writer_batch_size` | int | None | 写入批次大小 |
| `features` | Features | None | 结果特征类型 |
| `disable_nullable` | bool | False | 禁用 nullable |
| `fn_kwargs` | dict | None | 传递给函数的额外参数 |
| `num_proc` | int | None | 并行进程数 |
| `desc` | str | None | 进度条描述 |
| `new_fingerprint` | str | None | 强制指定指纹 |

### filter() 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `function` | Callable | - | 返回 bool 的判断函数 |
| `batched` | bool | False | 批量模式 |
| `batch_size` | int | 1000 | 批量大小 |
| `num_proc` | int | None | 并行进程数 |

---

## 常见问题排查

### map() 相关问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `map` 太慢 | 缓存写入磁盘 + 单进程 | 设置 `num_proc=4` 开启多进程；确认 `batched=True` |
| 修改代码后仍加载旧数据 | 相同 fingerprint 命中缓存 | 用 `cache_file_name` 或 `new_fingerprint` 强制刷新 |
| 多进程 `map` 卡死 | pickle 序列化失败（如 lambda） | 将函数定义在模块顶层；避免闭包捕获大对象 |
| 返回列与原列不一致 | 未正确处理 batch 结构 | 确保返回 dict 与原结构一致 |

### filter() 相关问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 过滤后数据集为空 | 过滤条件太严格 | 检查过滤逻辑，使用 `with_indices` 调试 |
| filter 后内存占用高 | 底层数据仍保留 | 使用 `save_to_disk` + `load_from_disk` 释放 |

### 索引访问问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `dataset[0]` 返回格式不对 | 未考虑 set_format 设置 | 检查 `dataset.format` 属性 |
| 切片返回非预期数量 | 边界条件问题 | 注意 Python 切片是左闭右开 |
| 负索引失效 | 不支持 | 使用 `len(dataset) - n` 代替 |

---

*文档生成于: 2026/04/20*
