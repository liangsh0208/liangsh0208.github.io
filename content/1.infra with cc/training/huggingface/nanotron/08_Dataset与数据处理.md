---
created: 2026-05-06
---

# Nanotron: Dataset与数据处理

---

## 顶部信息栏

| 属性 | 值 |
|------|-----|
| **文档版本** | v1.0 |
| **创建日期** | 2026-04-20 |
| **代码路径** | `src/nanotron/data/` |
| **核心文件** | `dataloader.py`, `clm_collator.py`, `samplers.py`, `nanoset.py` |
| **关键类/函数** | `CausalLMDataEngine`, `DataCollatorForCLM`, `get_train_dataloader`, `Nanoset` |

---

## 模块概述

Nanotron的数据处理模块负责高效加载和预处理大规模训练数据。核心设计目标包括：

1. **无限数据流支持**：支持训练和评估的数据迭代
2. **并行感知**：与TP/PP/DP并行策略协同工作
3. **多阶段数据切换**：支持课程学习(Curriculum Learning)
4. **高效内存使用**：支持pin_memory、worker预取等优化
5. **HuggingFace集成**：与Datasets库无缝集成

### 重点特性

- **TensorPointer机制**：Pipeline并行中，非数据rank使用指针而非实际数据
- **Context Parallelism支持**：数据按CP size切片
- **文档级掩码**：支持SFT场景下的position_ids和label_mask
- **Nanoset**：高性能二进制数据集格式支持

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Data Pipeline Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐   │
│  │   Data Source    │      │   Data Source    │      │   Data Source    │   │
│  │  (HuggingFace    │      │  (Nanoset Binary │      │  (Dummy Generator)│   │
│  │   Datasets)      │      │    Files)        │      │                  │   │
│  └────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘   │
│           │                         │                         │              │
│           ▼                         ▼                         ▼              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    get_train_dataloader()                                │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  DistributedSampler / SkipBatchSampler                           │  │ │
│  │  │  - DP-aware sharding                                             │  │ │
│  │  │  - Resume support (consumed_train_samples)                       │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │                              │                                         │ │
│  │                              ▼                                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  DataCollatorForCLM / DataCollatorForCLMWithPositionIds          │  │ │
│  │  │  - Sequence length padding (seq_len + 1)                         │  │ │
│  │  │  - Input_ids: [0:-1], Label_ids: [1:]                           │  │ │
│  │  │  - Position IDs for document boundary                            │  │ │
│  │  │  - Context Parallel slicing                                      │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │              sanity_check_dataloader()                                 │ │
│  │  - DP/TP/PP sanity checks                                            │  │ │
│  │  - Tensor to GPU (non_blocking)                                      │ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐              │
│         ▼                          ▼                          ▼              │
│  ┌────────────┐            ┌────────────┐            ┌────────────┐        │
│  │  PP Rank 0 │            │  PP Rank 1 │            │  PP Rank N │        │
│  │ (Input)    │            │ (Hidden)   │            │ (Output)   │        │
│  │            │            │            │            │            │        │
│  │ input_ids  │──TP/CP──▶│ TensorPtr  │──TP/CP──▶│ label_ids  │        │
│  │ position_ids│          │            │           │ label_mask │        │
│  └────────────┘            └────────────┘            └────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心实现详解

### 1. DataLoader构建流程

```python
# File: src/nanotron/data/dataloader.py

def get_train_dataloader(
    train_dataset: "datasets.Dataset",
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,          # 输入所在的PP rank
    output_pp_rank: int,        # 输出/label所在的PP rank
    micro_batch_size: int,
    consumed_train_samples: int,# 断点续训用
    dataloader_num_workers: int,
    seed_worker: int,
    use_position_ids: bool = True,
) -> DataLoader:
```

**关键逻辑：**

1. **PP-aware数据分发**：
   - 只有input_pp_rank和output_pp_rank需要实际数据
   - 其他PP rank使用`EmptyInfiniteDataset` + `TensorPointer`

```python
# dataloader.py lines 311-332
if dist.get_rank(parallel_context.pp_pg) in [input_pp_rank, output_pp_rank]:
    # 实际数据转换为numpy格式
    train_dataset = train_dataset.with_format(type="numpy", columns=["input_ids"])
else:
    # 非数据rank：空数据集，collator会返回TensorPointer
    train_dataset = EmptyInfiniteDataset(length=dataset_length)
    dataloader_num_workers = 0
```

2. **分布式采样器**：

```python
# samplers.py lines 63-111
def get_sampler(
    dl_ranks_size: int,        # DP world size
    dl_rank: int,              # DP rank
    train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
    consumed_train_samples: int,
    micro_batch_size: Optional[int] = None,
    drop_last: Optional[bool] = True,
    shuffle: bool = False,
) -> Optional[torch.utils.data.Sampler]:
```

### 2. DataCollatorForCLM详解

```python
# File: src/nanotron/data/clm_collator.py

@dataclasses.dataclass
class DataCollatorForCLM:
    """
    CLM (Causal Language Modeling)数据整理器
    
    核心功能：
    1. 序列拼接：input_ids (seq_len+1) -> input_ids (seq_len) + label_ids (seq_len)
    2. 位置偏移：labels是input_ids左移一位
    3. PP感知：非数据rank返回TensorPointer
    4. CP切片：按context_parallel_size切分序列
    """
    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) 
                 -> Dict[str, Union[torch.Tensor, TensorPointer]]:
```

**数据转换示例：**

```
输入 (batch, seq_len+1):
  [198, 50, 30, 12532, 3589, 198, 51, 30, ..., 0, 1780, 314]

输出:
  - input_ids (PP=input_pp_rank):  [198, 50, 30, 12532, 3589, 198, 51, 30, ..., 0, 1780]
                                   (丢弃最后一个token)
  
  - label_ids (PP=output_pp_rank): [50, 30, 12532, 3589, 198, 51, 30, ..., 0, 1780, 314]
                                   (丢弃第一个token，整体左移)
  
  - input_mask: 全True
  
  - label_mask: 基于position_ids生成，文档边界处mask为False
```

**Context Parallelism切片：**

```python
# clm_collator.py lines 71-76
cp_rank, cp_size = dist.get_rank(self.parallel_context.cp_pg), \
                   self.parallel_context.context_parallel_size
local_slice = slice(
    cp_rank * self.sequence_length // cp_size, 
    (cp_rank + 1) * self.sequence_length // cp_size
)
result["input_ids"] = result["input_ids"][:, local_slice]  # (b, s/cp_size)
```

### 3. DataCollatorForCLMWithPositionIds (SFT场景)

```python
# clm_collator.py lines 137-282

@dataclasses.dataclass
class DataCollatorForCLMWithPositionIds:
    """
    支持打包序列(packed sequences)的DataCollator
    
    用于SFT场景，多个短序列打包成长序列，通过position_ids标记边界
    """
    use_doc_masking: bool = True  # 启用文档级掩码
    
    def __call__(self, examples):
        # 输入包含positions字段，标识每个token在文档中的位置
        position_ids = np.vstack([examples[i]["positions"] for i in range(len(examples))])
        
        # 创建label_mask: position_ids == 0的位置(文档开头)对应的上一个token要mask
        # 因为label_ids已左移，所以找到position_ids == 0的地方
        zeros = position_ids == 0
        result["label_mask"] &= ~zeros  # mask掉跨文档的预测目标
```

**position_ids和label_mask关系示例：**

```
input_ids:     [198, 50, 30, 12532, 3589, 198, 51, 30, 30618, ...]
position_ids:  [ 0,   1,  2,   3,    4,   5,   6,  7,   8,    ...]
                  ↑文档A                    ↑ 位置0重新开始，表示文档B开始

label_ids:     [50, 30, 12532, 3589, 198, 51, 30, 30618, ...]
               (左移一位，对应input_ids的预测目标)

label_mask:    [True, True, True, True, True, True, True, True, ...]
                                  ↑ 位置5的198是文档B的第一个token的输入
                                     但它预测的是51(文档B的第一个token)
                                     所以label_mask[5]应该为False
```

### 4. Nanoset高性能数据集

```python
# File: src/nanotron/data/nanoset.py

class Nanoset(torch.utils.data.Dataset):
    """
    Nanotron高性能二进制数据集
    
    特性:
    - 使用datatrove的内存映射格式(.ds文件)
    - 支持多数据集加权混合
    - 支持32-bit和64-bit token存储
    - position_ids预计算或动态生成
    """
    def __init__(
        self,
        dataset_folders: List[str],           # 数据集文件夹列表
        sequence_length: int,
        token_size: int,                       # 2或4字节
        train_split_num_samples: int,         # 总训练样本数
        dataset_weights: Union[List[float], None] = None,  # 各数据集权重
        random_seed: int = 1234,
        eos_token_id: int = None,
        return_positions: bool = True,        # 返回position_ids
    ):
```

**数据索引构建算法：**

```python
# nanoset.py lines 226-261 (Numba加速)

@jit(nopython=True, cache=True)
def build_nanoset_index_helper(
    n_samples: int, 
    weights: np.ndarray, 
    dataset_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于权重构建数据集索引的贪心算法
    
    原理：维护每个数据集的"欠采样误差"，每次选择误差最大的数据集
    """
    current_samples = np.zeros((len(weights),), dtype="long")
    
    for sample_idx in range(n_samples):
        sample_idx_float = max(sample_idx, 1.0)
        
        # 计算每个数据集的采样误差
        # error = weight * expected - actual
        errors = weights * sample_idx_float - current_samples
        max_error_index = np.argmax(errors)
        
        # 选择误差最大的数据集
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index] % dataset_sizes[max_error_index]
        current_samples[max_error_index] += 1
```

### 5. Dummy数据生成器

```python
# dataloader.py lines 100-227

def dummy_infinite_data_generator(
    micro_batch_size: int,
    sequence_length: int,
    input_pp_rank: int,
    output_pp_rank: int,
    vocab_size: int,
    seed: int,
    parallel_context: ParallelContext,
    use_position_ids: bool = False,
    cp_pg: Optional[dist.ProcessGroup] = None,
):
    """
    无限随机数据生成器 - 用于性能测试和benchmark
    
    特性:
    - 针对不同PP rank返回实际数据或TensorPointer
    - 支持position_ids模拟packing场景
    - CP-aware切片
    """
    def data_generator():
        generator = torch.Generator(device="cuda")
        # TP/PP同步的随机种子确保数据一致性
        generator.manual_seed(
            seed * (1 + dist.get_rank(parallel_context.dp_pg)) * 
            (1 + dist.get_rank(parallel_context.pp_pg))
        )
        
        while True:
            yield {
                "input_ids": torch.randint(..., generator=generator) 
                    if dist.get_rank(parallel_context.pp_pg) == input_pp_rank 
                    else TensorPointer(group_rank=input_pp_rank),
                "label_ids": torch.randint(..., generator=generator)
                    if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                    else TensorPointer(group_rank=output_pp_rank),
            }
```

### 6. 多阶段数据切换

```python
# File: run_train.py lines 312-391

def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    """
    支持多阶段数据切换
    
    使用场景:
    1. 课程学习：从短序列逐渐切换到长序列
    2. 数据混合策略变化：不同训练阶段使用不同数据配比
    3. 领域切换：从通用语料到专业领域
    """
    # 按start_training_step排序的数据阶段
    # data_stages = [
    #   {"name": "warmup", "start_training_step": 1, "data": DataArgs(...)},
    #   {"name": "main", "start_training_step": 10000, "data": DataArgs(...)},
    # ]
    
    current_stage = trainer.metadata.data_stages[stage_idx]
    
    # 在训练过程中切换dataloader
    if stage.start_training_step == trainer.iteration_step:
        self.current_dataloader = sanity_check_dataloader(
            dataloader=dataloader, 
            parallel_context=self.parallel_context
        )
```

---

## 配置示例

### HuggingFace Datasets配置

```yaml
data_stages:
  - name: Stable Training Stage
    start_training_step: 1
    data:
      dataset:
        hf_dataset_or_datasets: 
          - "HuggingFaceFW/fineweb-edu"
          - "bigcode/the-stack-v2"
        hf_dataset_config_name: "CC-MAIN-2024-10"  # 可选
        hf_dataset_splits: "train"
        dataset_processing_num_proc_per_process: 4
        dataset_overwrite_cache: false
        text_column_name: "text"
      seed: 42
      num_loading_workers: 2
```

### Nanoset二进制数据集配置

```yaml
data_stages:
  - name: High Quality Data Stage
    start_training_step: 1
    data:
      dataset:
        dataset_folder:
          - "/data/fineweb-edu-tokenized"
          - "/data/slimpajama-tokenized"
        dataset_weights: [0.7, 0.3]  # 70% / 30% 混合
        token_size_in_bytes: 4  # 或2，取决于词汇表大小
        return_positions: true
        shuffle_files: true
        # 自动从metadata文件读取:
        # tokenizer_name, vocab_size
      seed: 42
      num_loading_workers: 4
```

### 多阶段课程学习配置

```yaml
tokens:
  sequence_length: 4096  # 基础序列长度
  
data_stages:
  # 阶段1: 短序列warmup (1-5000步)
  - name: warmup_2048
    start_training_step: 1
    sequence_length: 2048  # 覆盖全局seq_len
    data:
      dataset:
        hf_dataset_or_datasets: "HuggingFaceFW/fineweb-edu"
      seed: 42
      num_loading_workers: 2
      
  # 阶段2: 完整序列训练 (5001-100000步)  
  - name: main_training
    start_training_step: 5001
    data:
      dataset:
        dataset_folder: "/data/full_dataset"
        dataset_weights: null
      seed: 42
      num_loading_workers: 4
```

### SFT数据集配置

```yaml
model:
  model_config:
    _use_doc_masking: true  # 必须启用文档掩码
    
data_stages:
  - name: SFT Stage
    start_training_step: 1
    data:
      dataset:
        hf_dataset_or_datasets: "trl-lib/ultrachat"
        sft_dataloader: true  # 使用SFT专用处理流程
        debug_max_samples: 1000  # 可选：调试时限制样本数
      seed: 42
      num_loading_workers: 2
```

---

## 常见问题

### Q1: 断点续训时数据从哪里恢复？

**A**: 通过`consumed_train_samples`参数控制跳过已训练样本：

```python
# samplers.py lines 32-60
class SkipBatchSampler(BatchSampler):
    """跳过已消费的批次"""
    def __init__(self, batch_sampler: BatchSampler, skip_batches: int, dp_size: int):
        # 注意：skip_batches是全局数量，每个rank跳过 skip_batches // dp_size
        self.skip_batches = skip_batches // dp_size
```

### Q2: 如何在PP并行中处理数据分布？

**A**: 只有输入和输出PP rank持有实际数据，中间rank使用`TensorPointer`：

```python
# 数据分发逻辑
current_pp_rank = dist.get_rank(parallel_context.pp_pg)
if current_pp_rank not in [input_pp_rank, output_pp_rank]:
    # 非数据rank：返回空数据集，collator返回TensorPointer
    return {
        "input_ids": TensorPointer(group_rank=input_pp_rank),
        "label_ids": TensorPointer(group_rank=output_pp_rank),
    }
```

### Q3: position_ids和label_mask的作用是什么？

**A**: 用于处理打包序列(packed sequences)场景：

- **position_ids**: 标记每个token在其所属文档中的位置
- **label_mask**: 标记哪些位置应该计算loss（防止跨文档预测）

当position_ids从0重新开始，表示新文档开始，对应位置的label_mask为False。

### Q4: DataLoader worker数量如何设置？

**A**: 
- 一般设置为`4`或`num_loading_workers=2`
- 考虑数据读取I/O瓶颈，如果是本地SSD可以增大
- 如果是网络存储(NFS/S3)，建议较小值避免连接数过多

```yaml
data_stages:
  - data:
      num_loading_workers: 4  # 每个进程的工作线程数
      dataloader_pin_memory: true  # 启用pin_memory加速CPU->GPU传输
```

### Q5: Nanoset的metadata文件格式是什么？

**A**: metadata文件应命名为`*.metadata`，格式如下：

```
tokenizer_name|token_size_in_bytes
```

示例：
```
meta-llama/Llama-3.2-1B|4
```

---

## 参考资料

1. **HuggingFace Datasets**: https://huggingface.co/docs/datasets/
2. **PyTorch DataLoader**: https://pytorch.org/docs/stable/data.html
3. **datatrove**: https://github.com/huggingface/datatrove (Nanoset底层格式)
4. **Megatron-LM数据加载**: https://github.com/NVIDIA/Megatron-LM

---

## 附录：数据流完整代码示例

```python
# 完整的数据加载流程示例

from nanotron.data.dataloader import get_train_dataloader
from nanotron.data.processing import clm_process, get_datasets
from transformers import AutoTokenizer

# 1. 加载和预处理数据集
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
raw_dataset = get_datasets(
    hf_dataset_or_datasets="HuggingFaceFW/fineweb-edu",
    splits="train"
)["train"]

train_dataset = clm_process(
    raw_dataset=raw_dataset,
    tokenizer=tokenizer,
    text_column_name="text",
    dataset_processing_num_proc_per_process=4,
    sequence_length=4096
)

# 2. 创建DataLoader
dataloader = get_train_dataloader(
    train_dataset=train_dataset,
    sequence_length=4096,
    parallel_context=parallel_context,
    input_pp_rank=0,
    output_pp_rank=parallel_context.pp_pg.size() - 1,
    micro_batch_size=2,
    consumed_train_samples=0,
    dataloader_num_workers=2,
    seed_worker=42,
    use_position_ids=isinstance(model_config, Qwen2Config)
)

# 3. 迭代训练
for batch in dataloader:
    # batch包含:
    # - input_ids: (micro_batch_size, seq_len/cp_size) 或 TensorPointer
    # - input_mask: 同上
    # - label_ids: (micro_batch_size, seq_len/cp_size) 或 TensorPointer  
    # - label_mask: 同上
    # - position_ids: 可选，用于SFT
    pass
```
