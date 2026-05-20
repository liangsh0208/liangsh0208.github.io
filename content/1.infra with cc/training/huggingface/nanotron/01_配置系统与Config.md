---
created: 2026-05-06
---

# Nanotron 配置系统与 Config 详解

> 【源码定位】`src/nanotron/config/` 目录 | `config.py` 主配置文件 | `parallelism_config.py` 并行配置 | `models_config.py` 模型配置
> 【阅读建议】结合 `examples/config_tiny_llama.py` 示例阅读
> 【前置知识】Python Dataclass、PyTorch 分布式训练基础

---

## 1. 模块概述

Nanotron 采用声明式配置系统，基于 Python Dataclass 实现类型安全、可序列化的配置管理。所有配置均可通过 YAML 或 Python 代码定义，支持配置验证、继承和多阶段数据切换。

### 关键设计决策【重点】

| 设计决策 | 实现方案 | 优势 |
|---------|---------|------|
| **配置即代码** | Python Dataclass 定义 | 类型检查、IDE自动补全、版本控制友好 |
| **YAML互操作** | 自动序列化/反序列化 | 便于人工编辑和自动化生成 |
| **分层配置** | 按功能域拆分为子配置类 | 关注点分离，易于维护 |
| **配置验证** | `__post_init__` 钩子 | 启动时捕获配置错误，避免运行时故障 |
| **多阶段数据** | `data_stages` 支持 | 支持课程学习、退火等复杂训练策略 |

---

## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Nanotron Config System                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Config (Root)                                │  │
│  │                     src/nanotron/config/config.py                     │  │
│  │  ┌──────────────┬─────────────────────────────────────────────────┐  │  │
│  │  │   Section    │              Config Class                       │  │  │
│  │  ├──────────────┼─────────────────────────────────────────────────┤  │  │
│  │  │   general    │  GeneralArgs(project, run, seed, step)         │  │  │
│  │  │ parallelism  │  ParallelismArgs(tp, pp, dp, pp_engine, tp_mode) │  │  │
│  │  │    model     │  ModelArgs(model_config, init_method, dtype)   │  │  │
│  │  │   tokens     │  TokensArgs(seq_len, train_steps, mbs, accum)  │  │  │
│  │  │  optimizer   │  OptimizerArgs(zero_stage, lr_scheduler, ...)  │  │  │
│  │  │    data      │  DataArgs(dataset: HF/Nanoset/Dummy)           │  │  │
│  │  │ data_stages  │  List[DatasetStageArgs](多阶段数据配置)         │  │  │
│  │  │ checkpoints  │  CheckpointsArgs(path, interval, resume_path)   │  │  │
│  │  │   logging    │  LoggingArgs(log_level, iteration_interval)      │  │  │
│  │  │  profiler    │  ProfilerArgs(export_path, schedule)           │  │  │
│  │  └──────────────┴─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Config Creation Flow                               │  │
│  │                                                                       │  │
│  │   YAML File  ──→  yaml.load()  ──→  dict  ──→  get_config_from_dict │  │
│  │      ↑                                               │                │  │
│  │      │                                               ↓                │  │
│  │   save_as_yaml()                                 Config Object       │  │
│  │      ↑                                               │                │  │
│  │      │                                               ↙                │  │
│  │   Python Code  ──→  Config(...)  ──→  __post_init__验证              │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Model Config Hierarchy                             │  │
│  │                                                                       │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │   │  LlamaConfig │  │  Qwen2Config │  │Starcoder2Config│             │  │
│  │   │              │  │              │  │              │               │  │
│  │   │ • hidden_size│  │ • hidden_size│  │ • hidden_size│                 │  │
│  │   │ • num_layers │  │ • num_layers │  │ • n_layer    │                 │  │
│  │   │ • num_heads  │  │ • num_heads  │  │ • n_head     │                 │  │
│  │   │ • rope_theta │  │ • moe_config │  │ • multi_query│                 │  │
│  │   │ • tie_word_  │  │ • flex_attn  │  │ • global_attn│               │  │
│  │   │   embeddings │  │   _mask      │  │   _layers    │                 │  │
│  │   └──────────────┘  └──────────────┘  └──────────────┘               │  │
│  │                                                                         │  │
│  │   Union Type: NanotronConfigs = LlamaConfig | Qwen2Config | ...       │  │
│  │                                                                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心概念与实现

### 3.1 根配置类 Config

```python
# src/nanotron/config/config.py

@dataclass
class Config:
    """Main configuration class - 所有配置的根容器"""
    general: GeneralArgs
    parallelism: ParallelismArgs
    model: ModelArgs
    tokenizer: Optional[TokenizerArgs] = None
    checkpoints: Optional[CheckpointsArgs] = None
    logging: Optional[LoggingArgs] = None
    tokens: Optional[TokensArgs] = None
    optimizer: Optional[OptimizerArgs] = None
    data_stages: Optional[List[DatasetStageArgs]] = None
    profiler: Optional[ProfilerArgs] = None
    lighteval: Optional[LightEvalConfig] = None
    s3_upload: Optional[S3UploadArgs] = None

    def __post_init__(self):
        """配置验证与派生值计算"""
        # 验证 profiling steps 不超过 train_steps
        if self.profiler is not None:
            total_profiling_steps = self.profiler.skip_first + \
                self.profiler.repeat * (self.profiler.wait + 
                                       self.profiler.warmup + 
                                       self.profiler.active)
            assert self.tokens.train_steps >= total_profiling_steps
        
        # 自动计算 lr_decay_steps
        if self.optimizer is not None:
            if self.optimizer.learning_rate_scheduler.lr_decay_steps is None:
                self.optimizer.learning_rate_scheduler.lr_decay_steps = (
                    self.tokens.train_steps - 
                    self.optimizer.learning_rate_scheduler.lr_warmup_steps
                )
        
        # 数据阶段验证
        if self.data_stages is not None:
            # 必须有从 step=1 开始的阶段
            assert any(s.start_training_step == 1 for s in self.data_stages)
            # 阶段必须按 start_training_step 排序
            self.data_stages = sorted(
                self.data_stages, 
                key=lambda s: s.start_training_step
            )
    
    @property
    def global_batch_size(self):
        """计算全局batch size"""
        return (self.tokens.micro_batch_size * 
                self.tokens.batch_accumulation_per_replica * 
                self.parallelism.dp)
    
    @property
    def global_batch_size_in_tokens(self):
        """计算全局batch size的token数"""
        return self.global_batch_size * self.tokens.sequence_length
```

### 3.2 并行配置 ParallelismArgs

```python
# src/nanotron/config/parallelism_config.py

@dataclass
class ParallelismArgs:
    """
    3D并行核心配置
    
    约束条件: TP × PP × DP × CP × EP = WORLD_SIZE
    """
    dp: int  # Data Parallel size
    pp: int  # Pipeline Parallel size  
    tp: int  # Tensor Parallel size
    
    # 可选扩展并行
    expert_parallel_size: int = 1  # Expert Parallel (MoE)
    context_parallel_size: int = 1  # Context Parallel
    
    # Pipeline Engine 选择
    pp_engine: Optional[PipelineEngine] = None
    # TP 通信模式
    tp_mode: Optional[TensorParallelLinearMode] = None
    tp_linear_async_communication: Optional[bool] = None
    
    # 内存优化
    recompute_layer: bool = False  # 激活检查点
    tp_recompute_allgather: bool = True  # AllGather重计算

    def __post_init__(self):
        # 设置保守默认值
        if self.pp_engine is None:
            self.pp_engine = AllForwardAllBackwardPipelineEngine()
        if self.tp_mode is None:
            self.tp_mode = TensorParallelLinearMode.ALL_REDUCE
```

### 3.3 模型配置 - 以 LlamaConfig 为例

```python
# src/nanotron/config/models_config.py

@dataclass
class LlamaConfig:
    """Llama 模型架构配置"""
    # Tokenizer
    vocab_size: int = 32000
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # 模型架构
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # GQA
    
    # 位置编码
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    
    # 归一化
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    
    # 优化
    tie_word_embeddings: bool = False
    _attn_implementation: str = "flash_attention_2"
    
    # Z-loss 正则化 (for MoE stability)
    z_loss_enabled: bool = False
    z_loss_coefficient: float = 0.0001
    
    def __post_init__(self):
        # 默认 MHA -> GQA -> MQA 兼容
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        
        # 验证注意力实现
        assert self._attn_implementation in ALL_ATTENTION_FUNCTIONS
```

### 3.4 优化器配置

```python
# src/nanotron/config/config.py

@dataclass
class OptimizerArgs:
    """优化器与 ZeRO 配置"""
    optimizer_factory: Union[SGDOptimizerArgs, AdamWOptimizerArgs]
    zero_stage: int  # ZeRO stage: 0 = 禁用, 1 = 优化器状态分片
    weight_decay: float
    clip_grad: Optional[float]  # 梯度裁剪阈值
    accumulate_grad_in_fp32: bool  # FP32梯度累加
    learning_rate_scheduler: LRSchedulerArgs
    
    # 支持按参数名排除weight decay
    weight_decay_exclude_named_params: Optional[List[str]] = None

@dataclass 
class AdamWOptimizerArgs:
    adam_eps: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    torch_adam_is_fused: bool = True  # 使用 fused AdamW
    name: str = "adamW"

@dataclass
class LRSchedulerArgs:
    learning_rate: float
    lr_warmup_steps: int = 0
    lr_warmup_style: str = "linear"  # "linear" | "constant"
    lr_decay_style: str = "cosine"   # "linear" | "cosine" | "1-sqrt"
    lr_decay_steps: Optional[int] = None
    lr_decay_starting_step: Optional[int] = None
    min_decay_lr: float = None
```

### 3.5 多阶段数据配置

```python
# src/nanotron/config/config.py

@dataclass
class DatasetStageArgs:
    """
    支持课程学习、数据退火等复杂训练策略
    
    Example:
        Stage 1: 通用预训练 (step 1-10000)
        Stage 2: 领域数据精调 (step 10001-15000)
        Stage 3: 高质量数据退火 (step 15001-20000)
    """
    name: str
    start_training_step: int
    data: DataArgs

@dataclass
class DataArgs:
    """数据源配置 - 支持多种数据集类型"""
    dataset: Optional[Union[
        PretrainDatasetsArgs,   # HuggingFace datasets
        NanosetDatasetsArgs,    # 二进制tokenized数据
        SFTDatasetsArgs          # 监督微调数据
    ]]
    seed: Optional[int]
    num_loading_workers: int = 1
```

### 3.6 YAML配置示例

```yaml
# Generated from config_tiny_llama.py

general:
  project: "debug"
  run: "tiny_llama_%date_%jobid"
  seed: 42

parallelism:
  dp: 2
  pp: 2
  tp: 2
  pp_engine: "1f1b"
  tp_mode: "reduce_scatter"
  tp_linear_async_communication: true
  recompute_layer: false

model:
  model_config:
    is_llama_config: true
    hidden_size: 16
    intermediate_size: 64
    num_hidden_layers: 2
    num_attention_heads: 4
    num_key_value_heads: 4
    vocab_size: 256
    max_position_embeddings: 256
    tie_word_embeddings: true
  init_method:
    std: 0.025
  dtype: "bfloat16"

tokens:
  sequence_length: 256
  train_steps: 15
  micro_batch_size: 2
  batch_accumulation_per_replica: 1

optimizer:
  zero_stage: 0
  weight_decay: 0.01
  clip_grad: 1.0
  accumulate_grad_in_fp32: true
  learning_rate_scheduler:
    learning_rate: 0.0003
    lr_warmup_steps: 2
    lr_warmup_style: "linear"
    lr_decay_style: "cosine"
    min_decay_lr: 0.00001
  optimizer_factory:
    name: "adamW"
    adam_eps: 1.0e-08
    adam_beta1: 0.9
    adam_beta2: 0.95
    torch_adam_is_fused: true

data_stages:
  - name: "Stable Training Stage"
    start_training_step: 1
    data:
      dataset:
        hf_dataset_or_datasets: "stas/openwebtext-10k"
        text_column_name: "text"
      seed: 42
  - name: "Annealing Phase"
    start_training_step: 10
    data:
      dataset:
        hf_dataset_or_datasets: "stas/openwebtext-10k"
        text_column_name: "text"
      seed: 42
```

---

## 4. 配置参数表

### 4.1 GeneralArgs

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| `project` | str | required | 项目名称，用于日志分组 |
| `run` | str | "%date_%jobid" | 运行标识，支持%date和%jobid占位符 |
| `seed` | int | 42 | 随机种子 |
| `step` | int | None | 从检查点恢复时的起始step |
| `consumed_train_samples` | int | None | 已消耗的样本数 |
| `ignore_sanity_checks` | bool | True | 是否跳过训练前检查 |

### 4.2 ParallelismArgs

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| `dp` | int | required | 数据并行度 |
| `pp` | int | required | 流水线并行度 |
| `tp` | int | required | 张量并行度 |
| `pp_engine` | str/Engine | "afab" | "1f1b"或"afab" |
| `tp_mode` | str | "all_reduce" | "all_reduce"或"reduce_scatter" |
| `tp_linear_async_communication` | bool | False | TP层异步通信 |
| `recompute_layer` | bool | False | 激活检查点 |
| `context_parallel_size` | int | 1 | 上下文并行度 |
| `expert_parallel_size` | int | 1 | 专家并行度(MoE) |

### 4.3 LlamaConfig 关键参数

| 参数名 | 类型 | 推荐值(7B) | 说明 |
|-------|------|-----------|------|
| `hidden_size` | int | 4096 | 隐藏层维度 |
| `intermediate_size` | int | 11008 | FFN中间层维度 |
| `num_hidden_layers` | int | 32 | Transformer层数 |
| `num_attention_heads` | int | 32 | 注意力头数 |
| `num_key_value_heads` | int | 32/8/1 | GQA配置：32=MHA, 8=GQA, 1=MQA |
| `vocab_size` | int | 32000+ | 词表大小(会被padding到TP倍数) |
| `max_position_embeddings` | int | 4096 | 最大序列长度 |
| `tie_word_embeddings` | bool | False | 是否共享输入/输出嵌入 |

### 4.4 OptimizerArgs

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| `zero_stage` | int | 0 | 0=禁用, 1=优化器状态分片 |
| `weight_decay` | float | 0.01 | 权重衰减系数 |
| `clip_grad` | float | None | 梯度裁剪阈值 |
| `accumulate_grad_in_fp32` | bool | True | FP32梯度累加 |
| `learning_rate_scheduler` | LRSchedulerArgs | required | 学习率调度配置 |

---

## 5. 常见问题与排查

### 5.1 配置加载错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `dacite.exceptions.MissingValueError` | 缺少必需配置项 | 检查所有 `required` 字段是否提供 |
| `ValueError: num_attention_heads must be divisible by tp` | TP分割失败 | 确保 num_heads % tp == 0 |
| `AssertionError: You must have a training stage starting at step 1` | 数据阶段配置错误 | 添加 start_training_step=1 的阶段 |
| `TypeError: Cannot cast str to PipelineEngine` | pp_engine格式错误 | 使用字符串 "1f1b" 或 "afab" |

### 5.2 配置验证失败

```python
# 常见验证错误及修复

# 错误1: TP/PP/DP 乘积不等于 WORLD_SIZE
# 修复: 计算并调整并行度
world_size = 8
# 方案1: tp=2, pp=2, dp=2  -> 2*2*2 = 8 ✓
# 方案2: tp=1, pp=4, dp=2  -> 1*4*2 = 8 ✓

# 错误2: 词表大小不能被TP整除
# 修复: Nanotron会自动padding，或手动设置
model_config.vocab_size = 32000  # 会被padding到 TP*make_vocab_size_divisible_by 的倍数

# 错误3: 微批次数量不足 (1F1B)
# 修复: 确保 batch_accumulation_per_replica >= pp - 1
```

### 5.3 配置调试技巧

```python
# 打印完整配置
config.print_config_details()

# 导出YAML检查
config.save_as_yaml("debug_config.yaml")

# 计算模型参数量
def calculate_params(config):
    m = config.model.model_config
    vocab = m.vocab_size
    h = m.hidden_size
    inter = m.intermediate_size
    layers = m.num_hidden_layers
    
    # 嵌入 + 各层(ffn + attn)
    params = vocab * h * (2 if m.tie_word_embeddings else 1)
    params += layers * (3 * h * inter + 4 * h * h)
    return params
```

---

## 6. 参考资料

1. [Python Dataclasses Documentation](https://docs.python.org/3/library/dataclasses.html)
2. [dacite - Complex type casting for dataclasses](https://github.com/konradhalas/dacite)
3. [Megatron-LM Configuration](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py)
4. [HuggingFace Transformers Training Arguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
