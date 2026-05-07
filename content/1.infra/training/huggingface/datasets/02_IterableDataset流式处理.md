# IterableDataset 流式处理

> **【文档定位】** 流式数据集设计与分布式训练支持
>
> **【版本信息】** 基于 HuggingFace datasets v4.8.5.dev0 (`src/datasets/iterable_dataset.py` ~5400行)
>
> **【前置知识】** Python 迭代器、生成器、分布式训练

---

## 模块概述

`IterableDataset` 是流式数据集接口，适用于超大数据集或流式数据源。不同于 `Dataset` 的随机索引访问，它通过迭代器顺序读取数据，支持分片、打乱、跳过、截取等操作，特别适合 TB 级数据处理。

### 核心特性

| 特性 | 说明 |
|------|------|
| **流式读取** | 边用边下载/读取，无需等待完整数据集 |
| **内置分片** | 支持 `shard()` 方法，适配多 worker 分布式训练 |
| **状态序列化** | 支持 `state_dict()` 保存恢复迭代状态 |
| **惰性执行** | `map`、`filter` 等操作延迟到迭代时才执行 |
| **无限数据流** | 支持循环遍历 (`repeat()`) 和动态数据源 |

### 适用场景

| 场景 | 推荐模式 |
|------|----------|
| 数据集 > 物理内存 | `streaming=True` |
| 需要动态生成数据 | 自定义 `IterableDataset` |
| 多机分布式训练 | `shard()` + `state_dict()` |
| 只需要遍历一次 | `IterableDataset` |
| 需要随机访问 | `Dataset` |

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IterableDataset 架构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    ExamplesIterable 层级                        │   │
│   │                                                                  │   │
│   │   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────┐ │   │
│   │   │ 基础迭代器        │   │ 转换迭代器        │   │ 组合迭代器    │ │   │
│   │   │ - ExamplesIterable│   │ - MappedExamplesIterable      │   │ - CyclingIterable     │ │   │
│   │   │ - FilepathIterator│   │ - FilteredExamplesIterable    │   │ - RandomlyCyclingIterable│
│   │   │ - ArchiveIterator │   │ - TypedExamplesIterable       │   │ - StoppingStrategy    │ │   │
│   │   └──────────────────┘   └──────────────────┘   └──────────────┘ │   │
│   │                                                                  │   │
│   └──────────────────────────────────┬──────────────────────────────┘   │
│                                        │                                 │
│   ┌────────────────────────────────────▼──────────────────────────────┐  │
│   │                     IterableDataset 包装类                         │  │
│   │                                                                  │  │
│   │  - _ex_iterable: _BaseExamplesIterable  (实际迭代器)            │  │
│   │  - _map(): 返回新 IterableDataset (惰性包装)                    │  │
│   │  - shard(): 按 rank 分片                                         │  │
│   │  - skip() / take(): 截取数据流                                   │  │
│   │  - shuffle(): 缓冲区打乱                                          │  │
│   │  - set_epoch(): 分布式训练 epoch 同步                            │  │
│   │  - state_dict(): 保存/恢复迭代状态                               │  │
│   │                                                                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                     分布式配置 (DistributedConfig)                  │  │
│   │                                                                  │  │
│   │   - world_size: 总进程数                                         │  │
│   │   - rank: 当前进程 rank                                          │  │
│   │   - num_workers: worker 进程数                                   │  │
│   │   - worker_id: 当前 worker ID                                    │  │
│   │                                                                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## IterableDataset 类定义

```python
class IterableDataset:
    """
    流式/迭代数据集，不一次性加载所有数据到内存。
    支持分片(Sharding)、打乱(Shuffle)、跳过(Skip)、截取(Take)
    """
    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        formatting: bool = False,
        shuffling: Optional[ShufflingConfig] = None,
        distributed: Optional[DistributedConfig] = None,
        token_per_repo_id: Optional[dict[str, Union[str, bool]]] = None,
    ):
        self._ex_iterable = ex_iterable  # 惰性示例迭代器

    def _map(self, function, batched=False, batch_size=1):
        """返回新的 IterableDataset，包装 map 后的迭代器"""
        return MappedExamplesIterable(self._ex_iterable, function, ...)

    def shard(self, num_shards, index):
        """按示例分片，用于多 worker 分布式加载"""
        return ShuffledDataSourcesExamplesIterable(self._ex_iterable, ...)

    def skip(self, n):
        """跳过前 n 个示例"""
        return self._skip(n)

    def take(self, n):
        """只取前 n 个示例"""
        return self._take(n)

    def shuffle(self, buffer_size, seed=None):
        """使用缓冲区打乱数据顺序"""
        return self._shuffle(buffer_size, seed)

    def set_epoch(self, epoch):
        """设置当前 epoch，影响打乱种子"""
        self._epoch = epoch

    def state_dict(self):
        """保存当前迭代状态，用于断点续训"""
        return self._ex_iterable.state_dict()

    def load_state_dict(self, state_dict):
        """恢复迭代状态"""
        self._ex_iterable.load_state_dict(state_dict)
```

---

## 核心方法详解

### shard() - 数据分片

```python
def shard(self, num_shards: int, index: int, contiguous: bool = False) -> "IterableDataset":
    """
    将数据集分片，每个 worker 只处理属于自己的部分

    Args:
        num_shards: 总分片数
        index: 当前分片索引
        contiguous: True=连续分片, False=交错分片

    示例:
        # 4 个 worker 场景，当前是 worker 0
        dataset = iterable_dataset.shard(num_shards=4, index=0)
    """
```

### shuffle() - 流式打乱

```python
def shuffle(
    self,
    buffer_size: int,
    seed: Optional[int] = None,
) -> "IterableDataset":
    """
    使用缓冲区打乱数据顺序。

    原理:
    1. 填充 buffer_size 个样本到缓冲区
    2. 从缓冲区随机采样一个样本输出
    3. 从数据流补充新样本到缓冲区
    4. 重复直到数据流耗尽

    注意:
    - buffer_size 越大，打乱效果越好，但内存占用越大
    - 流式打乱不是全局打乱，相邻样本可能仍在 buffer_size 范围内
    """
```

### state_dict() - 状态管理

```python
# 保存/恢复迭代状态，用于断点续训
state = dataset.state_dict()  # 获取当前迭代位置
# ... 训练中断 ...
dataset.load_state_dict(state)  # 恢复到中断位置继续
```

---

## 代码示例

### 基础流式加载

```python
from datasets import load_dataset

# 流式加载（不下载完整数据集）
stream_dataset = load_dataset(
    "timm/imagenet-1k-wds",
    streaming=True,
    split="train"
)

# 得到 IterableDataset，支持迭代
for example in stream_dataset:
    image = example['jpg']      # PIL Image
    label = example['cls']
    # 处理训练...
    break

# 使用 iter + next
iterator = iter(stream_dataset)
sample1 = next(iterator)
sample2 = next(iterator)
```

### 流式分片 (多 Worker 训练)

```python
import torch.distributed as dist
from datasets import load_dataset

# 初始化分布式环境
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# 流式加载并分片
dataset = load_dataset(
    "openwebtext",
    streaming=True,
    split="train"
)

# 分片：每个 rank 只处理 1/world_size 的数据
sharded_dataset = dataset.shard(
    num_shards=world_size,
    index=rank
)

# PyTorch DataLoader 集成
dataloader = torch.utils.data.DataLoader(
    sharded_dataset,
    batch_size=32,
    num_workers=4,
    collate_fn=collate_fn
)

for batch in dataloader:
    # 训练逻辑
    pass
```

### 流式处理链

```python
# 流式 map
resized = stream_dataset.map(
    lambda x: {'jpg': x['jpg'].resize((224, 224))}
)

# 流式 filter
filtered = stream_dataset.filter(
    lambda x: x['cls'] < 100
)

# 流式打乱（需要指定 buffer_size）
shuffled = stream_dataset.shuffle(buffer_size=1000, seed=42)

# 跳过和截取
skipped = stream_dataset.skip(1000)    # 跳过前1000条
taken = stream_dataset.take(1000)       # 只取前1000条

# 链式操作
processed = (stream_dataset
    .shuffle(buffer_size=10000, seed=42)
    .filter(lambda x: x['cls'] < 100)
    .map(lambda x: {'jpg': x['jpg'].resize((224, 224))})
    .skip(1000)
    .take(10000))
```

### 批量流式处理

```python
# 批量 map
def tokenize_batch(batch):
    # batch 仍是 dict of iterables
    texts = batch['text']
    return {'tokens': [t.split() for t in texts]}

tokenized = stream_dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=100
)

# 配合 PyTorch DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    tokenized,
    batch_size=32,
    collate_fn=lambda x: {
        'text': [i['text'] for i in x],
        'tokens': [i['tokens'] for i in x]
    }
)
```

### 分布式训练状态管理

```python
# epoch 同步
def train_epoch(dataset, epoch):
    dataset.set_epoch(epoch)
    for batch in dataset:
        # 训练...
        pass

# 断点续训# 保存检查点checkpoint = {#     'model_state': model.state_dict(),#     'epoch': current_epoch,#     'dataset_state': dataset.state_dict()  # IterableDataset 状态# }

torch.save(checkpoint, 'checkpoint.pt')

# 恢复训练
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state'])
dataset.load_state_dict(checkpoint['dataset_state'])
train_epoch(dataset, checkpoint['epoch'])
```

---

## 配置参数表

### IterableDataset 初始化参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `ex_iterable` | _BaseExamplesIterable | 底层示例迭代器 |
| `info` | DatasetInfo | 数据集元数据 |
| `split` | NamedSplit | 分片标识 |
| `formatting` | bool | 是否启用格式化 |
| `shuffling` | ShufflingConfig | 全局打乱配置 |
| `distributed` | DistributedConfig | 分布式配置 |

### shard() 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `num_shards` | int | 总分片数 |
| `index` | int | 当前分片索引 |
| `contiguous` | bool | 是否连续分片 |

### shuffle() 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `buffer_size` | int | 打乱缓冲区大小 |
| `seed` | int/None | 随机种子 |

### map() 参数 (流式)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `function` | Callable | - | 处理函数 |
| `batched` | bool | False | 批量模式 |
| `batch_size` | int | 1 | 批量大小 |
| `drop_last_batch` | bool | False | 丢弃不完整批次 |
| `remove_columns` | List[str] | None | 删除的列 |

---

## 常见问题排查

### 流式读取问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 流式加载失败 | 网络问题或数据集不存在 | 检查 path 和 HuggingFace Hub 访问权限 |
| 数据顺序不随机 | shuffle buffer_size 太小 | 增大 `buffer_size` 参数 |
| 每个 epoch 顺序相同 | 未调用 `set_epoch()` | 在每个 epoch 前调用 `dataset.set_epoch(epoch)` |

### 分布式训练问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 多 worker 数据重复 | 未正确分片 | 使用 `shard(num_shards=world_size, index=rank)` |
| 状态恢复后数据不一致 | state_dict 未正确保存 | 确保保存完整的和 |
| DataLoader 集成失败 | IterableDataset 长度未知 | 不需要，直接传入 DataLoader 即可 |

### 性能问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 流式读取慢 | 网络 I/O 瓶颈 | 增大 `batch_size`；预取数据 |
| 内存占用增长 | shuffle buffer 累积 | 控制 `buffer_size`；定期清空 |
| map/filter 变慢 | 单线程处理 | 使用 PyTorch DataLoader 的 `num_workers` |

---

*文档生成于: 2026/04/20*
