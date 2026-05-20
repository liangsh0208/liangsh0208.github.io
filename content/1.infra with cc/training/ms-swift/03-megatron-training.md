---
created: 2026-05-09
---

# Megatron-SWIFT 训练全流程

Megatron-SWIFT 是 ms-swift 中独立的**大规模分布式训练子系统**，通过 `mcore-bridge` 桥接将 HuggingFace transformers 模型格式转换为 Megatron-LM 的并行格式。它使用独立的 CLI 入口 `megatron`，目标是让用户在使用 Megatron 并行（TP/PP/SP/CP/EP/VPP）时获得与 transformers 路径一致的开发体验。

---

## 1. 设计理念与架构

### 1.1 为什么需要 Megatron-SWIFT

| 场景 | transformers | Megatron-LM |
|-----|-------------|-------------|
| 单卡/单机 SFT | 足够 | 过重 |
| 长序列+大模型 (>70B) | 显存不足 | **TP/PP/SP 扩展**
| MoE 模型训练 | 慢 | **EP/ETP 加速**
| 大规模预训练 | 不具备 | **完整并行栈** |

Megatron-SWIFT 的设计原则：**将 Megatron 的大规模并行能力与 ms-swift 的模型生态/易用性结合**。

### 1.2 独立子系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Megatron-SWIFT 子系统                               │
│  ─────────────────────────────────────────────────────────────────────  │
│  CLI Entry: megatron (独立进程)                                          │
│  │                                                                       │
│  ├─ arguments/          # Megarton 参数体系                             │
│  │   ├─ megatron_base_args.py    (MegatronBaseArguments)                │
│  │   ├─ sft_args.py              (MegatronSftArguments)                  │
│  │   ├─ rlhf_args.py            (MegatronRLHFArguments)                │
│  │   ├─ pretrain_args.py        (MegatronPretrainArguments)            │
│  │   └─ export_args.py          (MegatronExportArguments)              │
│  │                                                                       │
│  ├─ pipelines/           # 训练管道                                    │
│  │   ├─ train/sft.py             (MegatronSft)                         │
│  │   ├─ train/rlhf.py           (MegatronRLHF)                         │
│  │   └─ train/pretrain.py        (MegatronPretrain)                     │
│  │                                                                       │
│  ├─ trainers/              # Megatron Trainer                            │
│  │   ├─ base.py                 (BaseMegatronTrainer)                  │
│  │   ├─ trainer.py              (MegatronTrainer)                       │
│  │   ├─ grpo_trainer.py         (MegatronGRPOTrainer)                 │
│  │   ├─ dpo_trainer.py          (MegatronDPOTrainer)                  │
│  │   ├─ kto_trainer.py          (MegatronKTOTrainer)                  │
│  │   ├─ gkd_trainer.py          (MegatronGKDTrainer)                  │
│  │   ├─ reward_trainer.py       (MegatronRewardTrainer)               │
│  │   ├─ embedding_trainer.py    (MegatronEmbeddingTrainer)            │
│  │   └─ reranker_trainer.py      (MegatronRerankerTrainer)            │
│  │                                                                       │
│  ├─ model/                 # 模型转换与并行化                            │
│  │   └─ utils.py                (get_mcore_model, prepare_mcore_model) │
│  │                                                                       │
│  ├─ utils/                 # Megatron 工具                              │
│  │   ├─ convert_utils.py         (HF ↔ Megatron 转换)                  │
│  │   ├─ parallel_utils.py        (并行组管理)                            │
│  │   ├─ megatron_lm_utils.py     (Megatron 初始化)                     │
│  │   └─ patcher.py              (补丁)                                  │
│  │                                                                       │
│  ├─ callbacks/           # Megatron 回调                               │
│  │   ├─ tensorboard.py          # TensorBoard                           │
│  │   ├─ wandb.py                # Weights & Biases                    │
│  │   └─ swanlab.py              # SwanLab                              │
│  │                                                                       │
│  │  mcore-bridge           # 外部依赖: transformers → Megatron 转换     │
│  │  megatron-core         # 外部依赖: NVIDIA Megatron-LM              │
│  └──────────────────────────────────────────────────────────────────────  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 与 transformers 路径的差异

| 方面 | transformers 路径 | Megatron 路径 |
|-----|------------------|---------------|
| CLI | `swift sft` | `megatron sft` |
| 参数类 | `SftArguments` | `MegatronSftArguments` |
| Trainer | `Seq2SeqTrainer` | `MegatronTrainer` |
| 模型加载 | `AutoModel.from_pretrained()` | `meta` 设备 + `mcore-bridge` |
| DeepSpeed | 原生支持 | 不使用 |
| 并行 | DDP/FSDP/Deepspeed ZeRO | TP/PP/SP/CP/EP/VPP |
| 序列并行 | Ulysses/Ring Attention | Megatron 原生 SP |
| RLHF | `SwiftRLHF` + `GRPOTrainer` | `MegatronRLHF` + `MegatronGRPOTrainer` |

---

## 2. 全流程调用链

### 2.1 CPT 训练流程

```bash
megatron pt \
    --model Qwen/Qwen3-4B \
    --dataset 'AI-ModelScope/example_text_pretrain' \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --sequence_parallel \
    --train_iters 10000
```

### 2.2 SFT 训练流程

```bash
megatron sft \
    --model Qwen/Qwen3-4B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --tensor_model_parallel_size 2 \
    --sequence_parallel \
    --train_iters 1000
```

### 2.3 RLHF 训练流程（GRPO）

```bash
megatron rlhf \
    --model Qwen/Qwen3-4B-Instruct \
    --rlhf_type grpo \
    --dataset rlhf_dataset \
    --reward_funcs accuracy format \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2  # MoE
```

### 2.4 调用链全景图

```
CLI: megatron sft/rlhf/pt
    │
    ▼
swift/megatron/cli/main.py  (同结构，ROUTE_MAPPING 指向 megatron 子模块)
    │
    ▼
MegatronSft / MegatronRLHF / MegatronPretrain (Pipeline)
    │
    ├── 1. Meta 模型加载 (不加载实际权重)
    │       with torch.device('meta'):
    │           model, processor = args.get_model_processor(return_dummy_model=True)
    │
    ├── 2. 初始化 Megatron (parallel groups / communicators)
    │       initialize_tp_communicators()
    │       init_process_group()
    │
    ├── 3. Template 初始化
    │       template = args.get_template(processor)
    │       template.use_megatron = True
    │
    ├── 4. 数据集加载 (复用 swift/dataset/)
    │       train_dataset, val_dataset = load_dataset(...)
    │
    ├── 5. 构建 Megatron 模型 (mcore-bridge)
    │       get_mcore_model(template, args)
    │           └── mcore_bridge 将 transformers config → Megatron model
    │
    ├── 6. 权重转换 (HF checkpoint → Megatron format)
    │       load_mcore_checkpoint(args.mcore_model, model, ...)
    │
    ├── 7. 构建 Optimizer (Megatron 原生)
    │       get_megatron_optimizer(model, optimizer_config)
    │
    ├── 8. Trainer 训练
    │       trainer.train(train_dataset, val_dataset)
    │
    └── 9. 保存检查点 (Megatron format)
            save_mcore_checkpoint(args.output_dir, model, ...)
```

---

## 3. 参数体系

### 3.1 MegatronBaseArguments

**文件**: `swift/megatron/arguments/megatron_base_args.py`

```python
@dataclass
class MegatronBaseArguments(MegatronArguments, BaseArguments):
    """
    继承自两个基类:
    - MegatronArguments: Megatron-LM 原生参数 (TP/PP/SP/CP/EP/...)
    - BaseArguments: ms-swift 基础参数 (model/dataset/template/...)
    """
    
    def __post_init__(self):
        # 关键: 将 sequence_parallel_size 映射到 Megatron 的 context_parallel_size
        self.sequence_parallel_size = self.context_parallel_size
        if self.packing:
            self.padding_free = True
        BaseArguments.__post_init__(self)
        # Megatron 使用 seq_length 而非 max_length
        self.seq_length = self.packing_length or self.max_length
        self._init_megatron_args()  # 初始化 Megatron 特定参数
```

### 3.2 并行策略参数

```python
# MegatronArguments (来自 megatron.training.arguments)
class MegatronArguments:
    # Tensor Parallel
    tensor_model_parallel_size: int = 1
    
    # Pipeline Parallel
    pipeline_model_parallel_size: int = 1
    
    # Sequence Parallel (Megatron 原生)
    sequence_parallel: bool = False
    
    # Context Parallel
    context_parallel_size: int = 1
    
    # Expert Parallel (MoE)
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    
    # Virtual Pipeline Parallel
    num_layers_per_virtual_pipeline_stage: int = None
    
    # MoE
    num_experts: int = None
    moe_router_topk: int = 2
    moe_aux_loss_coeff: float = 0.01
```

### 3.3 参数映射示例

```
ms-swift 参数             Megatron 参数
─────────────────────────────────────────
--tensor_model_parallel_size     tensor_model_parallel_size
--pipeline_model_parallel_size   pipeline_model_parallel_size
--sequence_parallel              sequence_parallel (bool)
--context_parallel_size          context_parallel_size
--expert_model_parallel_size     expert_model_parallel_size
--train_iters                    train_iters
--lr                             lr
--min_lr                         min_lr
--lr_decay_iters                 lr_decay_iters
--lr_warmup_iters                lr_warmup_fraction * train_iters
```

---

## 4. 核心流程详解

### 4.1 MegatronSft Pipeline

**文件**: `swift/megatron/pipelines/train/sft.py`

```python
class MegatronSft(SwiftSft):
    args_class = MegatronSftArguments
    
    def __init__(self, args):
        # 1. 华为昇腾 NPU 适配
        if is_torch_npu_available():
            patch_mindspeed_te_cp_implementation(megatron_args)
            repatch(megatron_args)
        
        # 2. 初始化输出目录
        self._init_output_dir()  # 包含 init_process_group()
        
        # 3. meta 设备加载模型元信息
        with torch.device('meta'):
            self.model, self.processor = args.get_model_processor(
                return_dummy_model=True)
        
        # 4. Template 初始化
        self._prepare_template()
        self.template.use_megatron = True
        
        # 5. 保存参数
        args.save_args(args.output_dir)
    
    def run(self):
        # 6. 加载数据集
        train_dataset, val_dataset = self._prepare_dataset()
        
        # 7. 初始化 Megatron 迭代器
        args.init_iters(train_dataset, val_dataset)
        
        # 8. 准备 Trainer
        trainer = self.prepare_trainer()
        
        # 9. 训练
        trainer.train(train_dataset, val_dataset)
        
        # 10. 清理
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def prepare_trainer(self):
        if args.task_type == 'embedding':
            return MegatronEmbeddingTrainer(args, self.template)
        elif args.task_type in {'reranker', 'generative_reranker'}:
            return MegatronRerankerTrainer(args, self.template)
        else:
            return MegatronTrainer(args, self.template)
```

### 4.2 模型初始化 (meta → mcore)

```python
# swift/megatron/model/utils.py

def get_mcore_model(template, args):
    """通过 mcore-bridge 构建 Megatron 并行模型"""
    # 1. 从 transformers config 推断 Megatron config
    mcore_config = convert_hf_config_to_mcore(args.model.config, args)
    
    # 2. 构建 Megatron 模型
    model = build_mcore_model(mcore_config, args)
    
    # 3. 加载权重
    if args.mcore_model:
        load_mcore_checkpoint(args.mcore_model, model)
    elif args.model:
        # HF checkpoint → Megatron format
        convert_hf_to_megatron(args.model, model)
    
    return model
```

### 4.3 BaseMegatronTrainer

**文件**: `swift/megatron/trainers/base.py`

```python
class BaseMegatronTrainer(ABC):
    def __init__(self, args, template):
        # 1. Router Replay 补丁（MoE）
        if args.router_replay_mode != 'disabled':
            apply_router_replay_patch()
        
        # 2. 准备模型 (mcore-bridge)
        self.prepare_model()
        
        # 3. 对齐 padding_free
        if template.padding_free != args.padding_free:
            template.padding_free = args.padding_free
        
        # 4. 构建 Optimizer + Scheduler
        self.optimizer, self.opt_param_scheduler = self.get_optimizer_and_scheduler()
        
        # 5. 构建 Data Collator
        self.data_collator = self._get_data_collator()
        
        # 6. 初始化训练状态
        self.state = TrainerState(max_steps=args.train_iters)
    
    def get_optimizer_and_scheduler(self):
        """使用 Megatron 原生 Optimizer"""
        optimizer_config = OptimizerConfig(
            optimizer='adam',
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            ...
        )
        optimizer = get_megatron_optimizer(optimizer_config, self.model)
        scheduler = get_optimizer_param_scheduler(optimizer, self.args)
        return optimizer, scheduler
    
    def train(self, train_dataset, val_dataset):
        """Megatron 训练循环"""
        for iteration in range(self.state.max_steps):
            # 1. 获取 batch (考虑 CP/PP)
            batch = self.get_batch(train_dataset)
            
            # 2. Megatron forward-backward
            loss = self.forward_backward_func(
                self.model, batch, ...)
            
            # 3. Optimizer step
            self.optimizer.step()
            self.opt_param_scheduler.step()
            
            # 4. 日志与 checkpoint
            if iteration % self.args.log_interval == 0:
                self.log_metrics()
            if iteration % self.args.save_interval == 0:
                save_mcore_checkpoint(...)
```

### 4.4 MegatronTrainer 损失计算

**文件**: `swift/megatron/trainers/trainer.py`

```python
class MegatronTrainer(BaseMegatronTrainer):
    def loss_func(self, output_tensor, *, labels, loss_scale=None, channels=None, packed_seq_params=None):
        # 1. 计算 per-token 损失
        losses = output_tensor.float()
        loss_mask = labels != -100
        
        # 2. DFT (Dynamic Fine-Tuning) 损失
        if args.enable_dft_loss:
            losses = losses * torch.exp(-losses.detach())
        
        # 3. 损失加权
        if loss_scale is not None:
            losses = losses * loss_scale
        
        # 4. 聚合
        loss = torch.cat([
            torch.sum(losses * loss_mask).view(1),
            loss_mask.sum().view(1)
        ])
        
        # 5. 跨数据并行组 all_reduce
        reporting_loss = loss.detach().clone()
        dist.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
        
        lm_loss = loss[0]
        local_num_tokens = loss[1].detach().clone().to(torch.int)
        metrics = {'loss': reporting_loss}
        
        # 6. Channel Loss (多通道损失)
        if args.enable_channel_loss:
            metrics.update(self._compute_channel_loss(losses, loss_mask, channels))
        
        return (lm_loss, local_num_tokens, metrics)
```

---

## 5. RLHF 训练

### 5.1 MegatronRLHF Pipeline

**文件**: `swift/megatron/pipelines/train/rlhf.py`

```python
class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    
    def prepare_trainer(self):
        trainer_mapping = {
            'dpo': 'MegatronDPOTrainer',
            'gkd': 'MegatronGKDTrainer',
            'grpo': 'MegatronGRPOTrainer',
            'kto': 'MegatronKTOTrainer',
            'rm': 'MegatronRewardTrainer',
        }
        trainer_cls = getattr(module, trainer_mapping[args.rlhf_type])
        
        kwargs = {}
        if args.rlhf_type in ('grpo', 'gkd'):
            kwargs['vllm_client'] = self._prepare_vllm_client()
        
        return trainer_cls(args, self.template, **kwargs)
    
    def _prepare_vllm_client(self):
        # GRPO/GKD 需要 vLLM 生成 completions
        if self.args.vllm_mode != 'server':
            return None
        if not getattr(self.args, 'use_vllm', False):
            return None
        
        # 在 last_rank 上连接 vLLM 服务
        if is_last_rank():
            vllm_client = VLLMClient(
                base_urls=self.args.vllm_server_base_url,
                hosts=self.args.vllm_server_host,
                server_ports=self.args.vllm_server_port,
                ...
            )
        return vllm_client
```

### 5.2 MegatronGRPOTrainer

**文件**: `swift/megatron/trainers/grpo_trainer.py`

```python
class MegatronGRPOTrainer(BaseMegatronTrainer):
    def loss_func(self, output_tensor, *, labels, old_logprobs, advantages, ...):
        # 1. 计算当前策略的 logprobs
        logprobs = self.get_per_token_logps(output_tensor, labels)
        
        # 2. 计算比率
        ratio = torch.exp(logprobs - old_logprobs)
        
        # 3. Clipping
        clipped_ratio = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon)
        
        # 4. GRPO 损失
        policy_loss = -torch.min(advantages * ratio, advantages * clipped_ratio).mean()
        
        # 5. KL 惩罚
        with torch.no_grad():
            ref_logprobs = self.ref_model(...)
        kl_loss = (logprobs - ref_logprobs).mean()
        
        loss = policy_loss + self.args.beta * kl_loss
        return loss, {'loss': loss.detach(), 'kl': kl_loss.detach()}
```

---

## 6. LoRA 在大规模并行下的适配

**文件**: `swift/megatron/trainers/base.py` (第13行)

```python
from mcore_bridge import LoraParallelLinear
```

Megatron-SWIFT 支持 LoRA 训练，且针对 MoE 模型有显著加速：

- `lora_parallel_linear` 将 LoRA 层融入 Megatron 的 `ColumnParallelLinear` / `RowParallelLinear`
- MoE 模型下，LoRA + EP (Expert Parallel) 可大幅减少显存占用并加速训练

```bash
# Megatron + LoRA + MoE + EP
megatron sft \
    --model deepseek-ai/DeepSeek-V3 \
    --tuner_type lora \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --sequence_parallel
```

---

## 7. 检查点与转换

### 7.1 Megatron 检查点格式

Megatron 检查点包含：
- `model_rng.pt`: 模型权重 + 随机状态
- `optim.pt`: 优化器状态
- `metadata.pt`: 迭代次数、训练状态

### 7.2 格式转换

**文件**: `swift/megatron/utils/convert_utils.py`

```python
def convert_hf_to_megatron(hf_model_path, megatron_model_path, args):
    """将 HuggingFace 检查点转换为 Megatron 格式"""
    # 1. 加载 HF 检查点
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    
    # 2. 按 TP/PP 切分权重
    for tp_rank in range(args.tensor_model_parallel_size):
        for pp_rank in range(args.pipeline_model_parallel_size):
            # 按并行策略切分权重矩阵
            sharded_state_dict = shard_weights_for_tp_pp(hf_model.state_dict(), tp_rank, pp_rank, args)
            
            # 3. 保存切分后的检查点
            save_dir = os.path.join(megatron_model_path, f'tp{tp_rank}_pp{pp_rank}')
            torch.save(sharded_state_dict, os.path.join(save_dir, 'model_optim_rng.pt'))

def convert_megatron_to_hf(megatron_model_path, hf_model_path, args):
    """将 Megatron 检查点转回 HuggingFace 格式"""
    # 1. 收集各 rank 的权重
    state_dict = {}
    for tp_rank in range(args.tensor_model_parallel_size):
        for pp_rank in range(args.pipeline_model_parallel_size):
            rank_state = load_rank_checkpoint(megatron_model_path, tp_rank, pp_rank)
            # 拼接被切分的权重
            state_dict = merge_sharded_weights(state_dict, rank_state)
    
    # 2. 保存为 HF 格式
    hf_model = AutoModelForCausalLM.from_pretrained(args.model)
    hf_model.load_state_dict(state_dict)
    hf_model.save_pretrained(hf_model_path)
```

---

## 8. 回调系统

**文件**: `swift/megatron/callbacks/`

Megatron 训练器的回调系统独立于 transformers 路径：

```python
class MegatronCallback:
    def on_train_begin(self, args, state, control, **kwargs): pass
    def on_train_end(self, args, state, control, **kwargs): pass
    def on_step_begin(self, args, state, control, **kwargs): pass
    def on_step_end(self, args, state, control, **kwargs): pass
    def on_save(self, args, state, control, **kwargs): pass

# 内置回调
callbacks_map = {
    'tensorboard': MegatronTensorBoardCallback,
    'wandb': MegatronWandbCallback,
    'swanlab': MegatronSwanlabCallback,
}
```

---

## 9. 关键代码路径索引

| 功能 | 关键文件/函数 |
|-----|-------------|
| Megatron CLI | `swift/cli/_megatron/main.py` |
| SFT Pipeline | `swift/megatron/pipelines/train/sft.py::MegatronSft` |
| RLHF Pipeline | `swift/megatron/pipelines/train/rlhf.py::MegatronRLHF` |
| 基础参数 | `swift/megatron/arguments/megatron_base_args.py::MegatronBaseArguments` |
| SFT 参数 | `swift/megatron/arguments/sft_args.py::MegatronSftArguments` |
| RLHF 参数 | `swift/megatron/arguments/rlhf_args.py::MegatronRLHFArguments` |
| Trainer 基类 | `swift/megatron/trainers/base.py::BaseMegatronTrainer` |
| 标准 Trainer | `swift/megatron/trainers/trainer.py::MegatronTrainer` |
| GRPO Trainer | `swift/megatron/trainers/grpo_trainer.py::MegatronGRPOTrainer` |
| DPO Trainer | `swift/megatron/trainers/dpo_trainer.py::MegatronDPOTrainer` |
| KTO Trainer | `swift/megatron/trainers/kto_trainer.py::MegatronKTOTrainer` |
| Reward Trainer | `swift/megatron/trainers/reward_trainer.py::MegatronRewardTrainer` |
| Embedding Trainer | `swift/megatron/trainers/embedding_trainer.py::MegatronEmbeddingTrainer` |
| Reranker Trainer | `swift/megatron/trainers/reranker_trainer.py::MegatronRerankerTrainer` |
| 模型构建 | `swift/megatron/model/utils.py::get_mcore_model()` |
| 权重转换 | `swift/megatron/utils/convert_utils.py` |
| 检查点保存 | `swift/megatron/utils/utils.py::save_mcore_checkpoint()` |
| 检查点加载 | `swift/megatron/utils/utils.py::load_mcore_checkpoint()` |
| 并行工具 | `swift/megatron/utils/parallel_utils.py` |
| NPU 补丁 | `swift/megatron/utils/patcher.py` |
| 回调映射 | `swift/megatron/callbacks/mapping.py` |
| LoRA 并行 | `mcore_bridge.LoraParallelLinear` (外部依赖) |
| Router Replay | `swift/megatron/utils/router_replay_utils.py` |
