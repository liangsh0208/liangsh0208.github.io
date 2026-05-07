# 训练配方 Recipes

> 本文档详细分析预定义训练配方的设计与使用，包括预训练/SFT/PEFT 配方模板和自定义配方开发指南。

---

## 目录

1. [概述](#概述)
2. [配方结构](#配方结构)
3. [通用配方模板](#通用配方模板)
4. [Llama 系列配方](#llama-系列配方)
5. [长序列配置](#长序列配置)
6. [低精度配置](#低精度配置)
7. [VLM 配置](#vlm-配置)
8. [配方命名规范](#配方命名规范)
9. [使用示例](#使用示例)
10. [自定义配方开发](#自定义配方开发)
11. [相关模块](#相关模块)
12. [总结](#总结)

---

## 概述

Megatron-Bridge 提供了丰富的预定义训练配方（Recipes），覆盖主流开源模型的预训练、微调和 PEFT 配置。这些配方经过优化和验证，可直接用于生产环境。

**源码位置**: `src/megatron/bridge/recipes/`

## 配方结构

```
recipes/
├── __init__.py
├── common.py                 # 通用配方基类
├── llama/
│   ├── llama2.py
│   └── llama3.py             # Llama 系列配方
├── qwen/
│   ├── qwen2.py
│   ├── qwen3.py
│   ├── qwen3_moe.py
│   └── qwen3_next.py
├── deepseek/
│   ├── deepseek_v2.py
│   └── deepseek_v3.py
├── gemma/
│   ├── gemma2.py
│   └── gemma3.py
├── ministries/
│   └── ministral3.py
├── olmoe/
│   └── olmoe_7b.py
├── nemotronh/
│   └── nemotronh.py
├── gpt/
│   ├── gpt3_175b.py
│   └── vanilla_gpt.py
└── utils/
    ├── finetune_utils.py
    ├── optimizer_utils.py
    └── tokenizer_utils.py
```

## 通用配方模板

### _pretrain_common

```python
def _pretrain_common() -> ConfigContainer:
    """
    创建预训练基础配置模板。
    
    特点：
    - 使用 Mock 数据（可通过 blend 参数指定真实数据）
    - BF16混合精度
    - 分布式优化器
    - 梯度/参数重叠
    - Transformer Engine 实现
    
    必须设置：
    - cfg.model
    - cfg.tokenizer.tokenizer_model
    """
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=None,
        max_lr=3e-4,
        min_lr=3e-5,
    )
    
    cfg = ConfigContainer(
        model=None,  # 必须设置
        train=TrainingConfig(
            train_iters=300000,
            global_batch_size=32,
            micro_batch_size=2,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
        ),
        mixed_precision="bf16_mixed",
        ...
    )
    return cfg
```

### _sft_common

```python
def _sft_common() -> ConfigContainer:
    """
    创建全参数微调（SFT）基础配置模板。
    
    特点：
    - 使用 SQuAD 数据集（可通过 dataset 参数指定）
    - 较低学习率（5e-6）
    - Packed Sequence 启用
    - 较短训练迭代（1000）
    - 支持 pretrained_checkpoint 加载
    
    与预训练的区别：
    - 更低学习率
    - 更小批次
    - 支持 Packed Sequence
    - 无 PEFT
    """
```

### _peft_common

```python
def _peft_common() -> ConfigContainer:
    """
    创建 PEFT（LoRA/DoRA）微调基础配置模板。
    
    特点：
    - 更高学习率（1e-4）
    - LoRA 默认配置（dim=32, alpha=32）
    - 目标模块：linear_qkv, linear_proj, linear_fc1, linear_fc2
    
    与全参数 SFT 的区别：
    - 更高学习率
    - LoRA 适配器
    - 禁用分布式优化器
    """
```

## Llama 系列配方

### 预训练配置

```python
# Llama 3 8B 预训练
def llama3_8b_pretrain_config() -> ConfigContainer:
    cfg = _pretrain_common()
    
    # 模型配置
    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B").to_megatron_provider(load_weights=False)
    
    # 并行配置: TP=1, PP=1, CP=2
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 2
    
    # Transformer Engine
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    
    return cfg

# Llama 3 70B 预训练
def llama3_70b_pretrain_config() -> ConfigContainer:
    cfg = _pretrain_common()
    
    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-70B").to_megatron_provider(load_weights=False)
    
    # 并行配置: TP=4, PP=4, VPP=5, CP=2, SP=True
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.context_parallel_size = 2
    cfg.model.sequence_parallel = True
    
    # Communication Overlap
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    )
    
    return cfg
```

### SFT 配置

```python
def llama3_8b_sft_config() -> ConfigContainer:
    cfg = _sft_common()
    
    hf_path = "meta-llama/Meta-Llama-3-8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = hf_path
    
    # 并行配置: TP=2
    cfg.model.tensor_model_parallel_size = 2
    
    # Packed Sequence
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    
    return cfg
```

### PEFT 配置

```python
def llama3_8b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    cfg = _peft_common()
    
    hf_path = "meta-llama/Meta-Llama-3-8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    
    # PEFT 配置
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg
    
    # 8B模型使用 dim=8, alpha=16
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
    
    # 并行配置: TP=1
    cfg.model.tensor_model_parallel_size = 1
    
    return cfg
```

## 长序列配置

```python
# Llama 3 8B 128K 序列
def llama3_8b_128k_pretrain_config() -> ConfigContainer:
    cfg = _pretrain_common()
    
    cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B").to_megatron_provider(load_weights=False)
    
    # 并行配置: TP=4, PP=2, CP=8, SP=True
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 131072  # 128K
    
    return cfg
```

## 低精度配置

```python
def llama3_8b_low_precision_pretrain_config(mixed_precision_recipe: str) -> ConfigContainer:
    """
    低精度预训练配置。
    
    支持的精度方案：
    - "bf16_with_mxfp8_mixed"
    - "bf16_with_fp8_current_scaling_mixed"
    - "bf16_with_nvfp4_mixed"
    """
    precision_config = get_mixed_precision_config(mixed_precision_recipe)
    
    if mixed_precision_recipe == "bf16_with_nvfp4_mixed":
        precision_config.first_last_layers_bf16 = True
        precision_config.num_layers_at_start_in_bf16 = 0
        precision_config.num_layers_at_end_in_bf16 = 4
    
    cfg = _pretrain_common()
    cfg.mixed_precision = precision_config
    
    return cfg
```

## VLM 配置

```python
def _sft_common_vlm() -> ConfigContainer:
    """
    VLM SFT 配置模板。
    
    特点：
    - 使用 HFDatasetConversationProvider
    - NullTokenizer
    - DDP 不重叠
    - 支持 freeze 选项
    """
    cfg = _sft_common()
    
    # VLM 特定配置
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 2
    
    # DDP 不重叠
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    
    # VLM 数据集
    cfg.dataset = HFDatasetConversationProvider(
        seq_length=4096,
        hf_processor_path=None,
        maker_name="make_cord_v2_dataset",
    )
    
    # NullTokenizer
    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="NullTokenizer",
        vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
    )
    
    return cfg
```

## 配方命名规范

```
{model}_{size}_{mode}_config()

例如：
- llama3_8b_pretrain_config()
- llama3_70b_sft_config()
- llama31_405b_peft_config()
- llama3_8b_128k_pretrain_config()
- llama3_8b_low_precision_pretrain_config()
```

## 使用示例

### 预训练

```python
from megatron.bridge.recipes.llama.llama3 import llama3_8b_pretrain_config
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.forward_step_func_types import forward_step

# 获取配置
cfg = llama3_8b_pretrain_config()

# 设置数据路径
cfg.dataset.blend = [("/path/to/data", 1.0)]

# 运行预训练
pretrain(cfg, forward_step)
```

### 全参数微调

```python
from megatron.bridge.recipes.llama.llama3 import llama3_8b_sft_config

cfg = llama3_8b_sft_config()
cfg.checkpoint.pretrained_checkpoint = "/path/to/pretrained"
cfg.dataset.dataset_root = "/path/to/finetune_data"

pretrain(cfg, forward_step)
```

### PEFT微调

```python
from megatron.bridge.recipes.llama.llama3 import llama3_8b_peft_config

cfg = llama3_8b_peft_config(peft_scheme="lora")
cfg.checkpoint.pretrained_checkpoint = "/path/to/pretrained"
cfg.dataset.dataset_root = "/path/to/finetune_data"

# 自定义 LoRA 配置
cfg.peft.dim = 16
cfg.peft.alpha = 32
cfg.peft.target_modules = ["linear_qkv", "linear_proj"]

pretrain(cfg, forward_step)
```

## 自定义配方开发

### 基础配方模板

```python
from megatron.bridge.training.config import ConfigContainer, TrainingConfig
from megatron.bridge.recipes.common import distributed_fused_adam_with_cosine_annealing

def my_custom_pretrain_config(
    model_name: str,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    sequence_length: int = 2048,
) -> ConfigContainer:
    """
    自定义预训练配方。
    
    Args:
        model_name: HuggingFace 模型名称
        tensor_parallel: 张量并行大小
        pipeline_parallel: 流水线并行大小
        sequence_length: 序列长度
    """
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    
    # 创建优化器和调度器配置
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        max_lr=1e-4,
        min_lr=1e-5,
    )
    
    # 加载模型
    bridge = AutoBridge.from_hf_pretrained(model_name)
    model_provider = bridge.to_megatron_provider(load_weights=False)
    
    # 设置并行配置
    model_provider.tensor_model_parallel_size = tensor_parallel
    model_provider.pipeline_model_parallel_size = pipeline_parallel
    model_provider.sequence_length = sequence_length
    
    # 构建配置容器
    cfg = ConfigContainer(
        model=model_provider,
        train=TrainingConfig(
            train_iters=100000,
            global_batch_size=256,
            micro_batch_size=4,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        # ... 其他配置
    )
    
    return cfg
```

### 配方最佳实践

1. **模块化设计**: 将通用配置抽取为独立函数
2. **参数化**: 关键参数通过函数参数暴露
3. **默认值**: 使用经过验证的默认超参
4. **文档**: 添加完整的文档字符串说明

### 配方验证

```python
def validate_recipe_config(cfg: ConfigContainer) -> None:
    """验证配方配置的有效性。"""
    assert cfg.train.train_iters > 0
    assert cfg.optimizer.lr > 0
    assert cfg.model.tensor_model_parallel_size >= 1
    assert cfg.model.pipeline_model_parallel_size >= 1
    
    # 检查并行配置与批大小兼容
    total_parallel = (
        cfg.model.tensor_model_parallel_size 
        * cfg.model.pipeline_model_parallel_size
    )
    assert cfg.train.global_batch_size % total_parallel == 0
```

### 失败恢复指南

训练中断后的恢复策略：

```python
# 设置检查点加载路径
cfg.checkpoint.load = "/path/to/checkpoints"

# 配置自动恢复（内建）
cfg.checkpoint.load_optim = True     # 恢复优化器状态
cfg.checkpoint.load_rng = True         # 恢复 RNG 状态
cfg.checkpoint.finetune = False      # 正常恢复（非微调模式）

# 手动恢复训练
def resume_training(checkpoint_dir: str):
    """从检查点恢复训练。"""
    # 加载最新检查点
    cfg.checkpoint.load = checkpoint_dir
    
    # 恢复运行时配置
    runtime_config_update(cfg)
    
    # 继续训练（pretrain 会自动加载检查点）
    pretrain(cfg, forward_step)
```

### 超参调优建议

| 模型大小 | 学习率 | 全局批次 | 预热步数 | 序列长度 | TP |
|---------|--------|---------|---------|---------|----|
| 1B-3B | 3e-4 | 256-512 | 500 | 2048-4096 | 1-2 |
| 7B-13B | 2e-4 | 512-1024 | 1000 | 4096-8192 | 2-4 |
| 30B-70B | 1e-4 | 1024-2048 | 2000 | 8192-16384 | 4-8 |
| 100B+ | 5e-5 | 2048-4096 | 5000 | 16384-32768 | 8-16 |

调优原则：
1. **学习率**: 模型越大，学习率越小
2. **批次大小**: 使用更大的全局批次配合更小的学习率
3. **序列长度**: 长序列需要更大的 TP 和更小的 TP通信开销
4. **并行策略**: PP 优先用于层数多的模型，TP 优先用于隐藏层维度大的模型

### 多模型对比速查

| 模型 | 参数量 | 层数 | 隐藏层 | 注意力头 | 推荐 TP | 推荐 PP |
|------|--------|------|--------|---------|--------|---------|
| Llama 3.2 1B | 1.24B | 16 | 2048 | 32 | 1 | 1 |
| Llama 3.1 8B | 8.03B | 32 | 4096 | 32 | 1-2 | 1-2 |
| Llama 3.1 70B | 70.6B | 80 | 8192 | 64 | 4-8 | 4-8 |
| Llama 3.1 405B | 405B | 126 | 16384 | 128 | 8-16 | 8-16 |
| Qwen3 4B | 3.83B | 36 | 2560 | 40 | 1-2 | 1-2 |
| Qwen3 30B(A3B) | 30.5B | 64 | 5120 | 128 | 4 | 4 |
| DeepSeek-V3 | 671B | 61 | 7168 | 128 | 8 | 8 |

---

## 相关模块

| 模块 | 关系说明 |
|------|---------|
| [配置系统](05_配置系统.md) | 配方返回预配置的 ConfigContainer |
| [AutoBridge](02_AutoBridge自动桥接.md) | 配方使用 AutoBridge 加载预训练模型 |
| [ModelProvider](03_ModelProvider模式.md) | 配方设置 Provider 的并行配置 |
| [训练框架](04_训练框架.md) | 配方输出传入 `pretrain()` 函数 |
| [数据处理](07_数据处理系统.md) | 配方设置数据集的 blend 和 seq_length |
| [PEFT](06_PEFT参数高效微调.md) | `_peft_common()` 提供 PEFT 微调模板 |

---

## 总结

Megatron-Bridge 的配方系统通过：

1. **通用模板**: `_pretrain_common`, `_sft_common`, `_peft_common` 提供基础配置
2. **模型特定优化**: 针对不同模型尺寸的并行策略
3. **场景覆盖**: 预训练、SFT、PEFT 三种训练模式
4. **最佳实践**: 经过验证的超参和并行配置
5. **可扩展性**: 支持自定义配方开发

提供开箱即用的训练配置，大幅降低分布式训练的上手门槛。