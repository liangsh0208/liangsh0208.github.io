# Transformers 训练全流程解析

ms-swift 基于 HuggingFace `transformers` 的训练流程是其最核心的执行路径，覆盖 SFT、Pretrain、RLHF 等几乎所有任务。本文档从 CLI 入口逐层拆解到 Trainer 训练循环，梳理完整的数据流和调用链。

---

## 1. 整体流程概览

```
User Command
    │
    ▼
CLI Entry (swift/cli/main.py)
    │
    ▼
Pipeline Init (SwiftPipeline.__init__)
    ├── Arguments Parsing (dataclass)
    ├── Seed Setting
    └── main() -> run()
    │
    ▼
Pipeline Execution
    ├── 1. Model + Processor Loading
    ├── 2. Template Initialization
    ├── 3. Tuner Preparation (LoRA/Adapter/Full)
    ├── 4. Dataset Loading & Encoding
    ├── 5. Trainer Factory Routing
    └── 6. Trainer.train()
    │
    ▼
Training Loop
    ├── DataLoader (with custom collator)
    ├── Forward + Loss Computation
    ├── Backward + Optimizer Step
    ├── Checkpoint Saving
    └── Evaluation
```

---

## 2. CLI 入口层

### 2.1 命令分发

**文件**: `swift/cli/main.py`

```python
ROUTE_MAPPING = {
    'pt': 'swift.cli.pt',       # 预训练
    'sft': 'swift.cli.sft',     # 监督微调
    'rlhf': 'swift.cli.rlhf',   # 对齐训练
    'infer': 'swift.cli.infer', # 推理
    'deploy': 'swift.cli.deploy',
    'export': 'swift.cli.export',
    'eval': 'swift.cli.eval',
}

def cli_main():
    method_name = sys.argv[1].replace('_', '-')
    file_path = importlib.util.find_spec(ROUTE_MAPPING[method_name]).origin
    
    # 自动分布式包装
    torchrun_args = get_torchrun_args()  # 读取 NPROC_PER_NODE, NNODES
    if torchrun_args is not None:
        args = [python, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    else:
        args = [python, file_path, *argv]
    subprocess.run(args)
```

### 2.2 参数解析

**文件**: `swift/cli/sft.py`, `swift/cli/rlhf.py`

以 `swift sft` 为例：

```python
from swift.pipelines import sft_main

if __name__ == '__main__':
    sft_main()
```

实际调用链：`sft_main()` → `SwiftSft(args).main()`

---

## 3. Pipeline 编排层

### 3.1 SwiftPipeline 基类

**文件**: `swift/pipelines/base.py`

```python
class SwiftPipeline(ABC, ProcessorMixin):
    args_class = BaseArguments

    def __init__(self, args=None):
        self.args = self._parse_args(args)
        self._set_seed()

    def main(self):
        logger.info('Start time: ...')
        result = self.run()
        logger.info('End time: ...')
        return result

    @abstractmethod
    def run(self):
        pass
```

**核心职责**:
- 参数解析（`dataclass` + `argparse`）
- 全局随机种子设置（考虑 rank 偏移）
- 生命周期管理（开始/结束时间记录）

### 3.2 SwiftSft — SFT/Pretrain 管道

**文件**: `swift/pipelines/train/sft.py`

```python
class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = SftArguments

    def __init__(self, args=None):
        super().__init__(args)
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_flash_ckpt()

    def run(self):
        train_dataset, val_dataset = self._prepare_dataset()
        self.train_msg = self._get_train_msg()
        trainer = self._prepare_trainer(train_dataset, val_dataset)
        trainer.train()
        return self.train_msg
```

#### Step 1: 模型与 Processor 加载

```python
def _prepare_model_tokenizer(self, **kwargs):
    args = self.args
    self.model, self.processor = args.get_model_processor(**kwargs)
    # 序列并行准备
    if args.sequence_parallel_size > 1:
        sequence_parallel.prepare(
            args.sequence_parallel_size, model=self.model, 
            tokenizer=self.processor, padding_free=args.padding_free)
    # 记录模型信息
    logger.info(f'model_info: {self.model.model_info}')
    self._prepare_generation_config()
```

**关键调用链**:
```
SftArguments.get_model_processor()
    └── swift/arguments/base_args.py::BaseArguments.get_model_processor()
        └── swift/model/register.py::get_model_info_meta()  # 推断 model_type
        └── swift/model/register.py::ModelLoader.load()     # 实际加载
```

#### Step 2: Template 初始化

```python
def _prepare_template(self):
    args = self.args
    template = args.get_template(self.processor)
    template.set_mode('train')
    if template.use_model:
        template.model = self.model
    # 检查 packing/padding_free 兼容性
    if (args.padding_free or args.packing) and not template.support_padding_free:
        raise ValueError(f'Template `{args.template}` does not support padding free or packing.')
    self.template = template
```

Template 的 `mode` 决定了其行为：
- `train`: 训练模式，`encode()` 返回 labels
- `rlhf`: RLHF 模式，`encode()` 返回 chosen/rejected pairs
- `kto`: KTO 模式，`encode()` 返回 KL pairs
- `transformers`/`vllm`/...: 推理模式

#### Step 3: 数据集加载与编码

```python
def _prepare_dataset(self):
    args = self.args
    # 1. 加载原始数据集
    train_dataset, val_dataset = self._get_dataset()
    # 2. 编码处理 (Template.encode)
    train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset, pre_process=True)
    # 3. 后处理 (Lazy/Packing/SP)
    datasets = self._post_process_datasets([train_dataset, val_dataset])
    return datasets
```

**关键路径**:
```
load_dataset(args.dataset)
    └── swift/dataset/loader.py::DatasetLoader._load_repo_dataset()  # HuggingFace/ModelScope Hub
    └── swift/dataset/loader.py::DatasetLoader._load_dataset_path()  # 本地文件
_encode_dataset()
    └── swift/dataset/preprocessor/core.py::EncodePreprocessor
        └── template.encode()  # 核心编码
_post_process_datasets()
    └── LazyLLMDataset / PackingDataset / IterablePackingDataset
```

#### Step 4: Tuner 准备

```python
# TunerMixin (swift/pipelines/train/tuner.py)
class TunerMixin:
    def prepare_model(self, args, model, task_type=None):
        if args.tuner_type == 'full':
            return model  # 全参数微调，无需额外处理
        
        # LoRA / Adapter / Prompt Tuning / ...
        tuner = tuners_map[args.tuner_type]
        model = tuner.prepare_model(model, args)
        return model
```

**LoRA 准备详细流程**:
```
prepare_model(args, model)
    ├── get_target_modules(args, model)     # 解析 target_modules (all-linear -> 实际模块)
    ├── get_modules_to_save(args, model)    # 全量更新的模块
    ├── 构建 LoRAConfig
    ├── Swift.prepare_model(model, config)  # swift/tuners/base.py
    │   └── 检测是否已包装为 SwiftModel
    │   └── 如果不是: SwiftModel(base_model, config)
    │       └── _prepare_model() → 实际 patch LoRA 层
    └── apply_liger(model_type)  # 若启用 Liger-Kernel
```

#### Step 5: Trainer 路由

```python
def _prepare_trainer(self, train_dataset, val_dataset):
    args = self.args
    # TrainerFactory 按 task_type 路由
    trainer_cls = TrainerFactory.get_trainer_cls(args)
    training_args = TrainerFactory.get_training_args(args)
    
    trainer = trainer_cls(
        model=self.model,
        args=training_args,
        template=self.template,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    return trainer
```

`TrainerFactory` 路由表（`swift/trainers/trainer_factory.py`）:

```python
TRAINER_MAPPING = {
    'causal_lm': 'swift.trainers.Seq2SeqTrainer',      # SFT / Pretrain
    'seq_cls': 'swift.trainers.Trainer',                 # 序列分类
    'embedding': 'swift.trainers.EmbeddingTrainer',
    'reranker': 'swift.trainers.RerankerTrainer',
    'dpo': 'swift.rlhf_trainers.DPOTrainer',
    'grpo': 'swift.rlhf_trainers.GRPOTrainer',
    'ppo': 'swift.rlhf_trainers.PPOTrainer',
    'kto': 'swift.rlhf_trainers.KTOTrainer',
    ...
}
```

---

## 4. 训练循环层

### 4.1 Seq2SeqTrainer — Causal LM 训练核心

**文件**: `swift/trainers/seq2seq_trainer.py`

```python
class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.predict_with_generate:
            # 评估时使用推理引擎生成回复
            self.infer_engine = TransformersEngine(self.model, template=self.template)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels', None)
        
        # 1. 前向传播
        outputs = model(**inputs)
        
        # 2. MoE 辅助损失
        if getattr(outputs, 'aux_loss', None):
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        
        # 3. 损失计算 (per-token 或 standard)
        if labels is not None:
            loss = self._compute_custom_loss(outputs, labels, ...)
        else:
            loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss
```

**关键特性**:
- **per-token loss**: 支持对非 padding 位置逐 token 加权
- **MoE aux loss**: 自动检测并累加 Mixture-of-Experts 的路由辅助损失
- **predict_with_generate**: 评估阶段使用 `TransformersEngine` 生成回复，计算 ROUGE/BLEU
- **sequence parallel**: 训练前通过 `sequence_parallel.prepare_inputs()` 调整输入

### 4.2 SwiftMixin — 公共能力注入

**文件**: `swift/trainers/mixin.py` (~61KB)

所有 Swift Trainer 通过多重继承注入 `SwiftMixin`：

```python
class SwiftMixin:
    def __init__(self, model, args, template, train_dataset, eval_dataset=None, **kwargs):
        # 1. 数据 collator 构建
        data_collator = self._get_data_collator(args, template)
        
        # 2. 自定义指标初始化
        self.custom_metrics = {'train': defaultdict(...), 'eval': defaultdict(...)}
        
        # 3. 模型元信息挂载
        self.model_meta = model.model_meta
        self.model_info = model.model_info
        
        # 4. 调用 HF Trainer __init__
        super().__init__(model=model, args=args, data_collator=data_collator, ...)
        
        # 5. 回调注册 (tensorboard, wandb, swanlab, flash ckpt)
        self._add_callbacks()
        
        # 6. 梯度检查点修复
        self._fix_gradient_checkpointing()
        
        # 7. 生成配置更新 (EOS token)
        update_generation_config_eos_token(self.model.generation_config, self.template)
```

**Mixin 核心能力矩阵**:

| 能力 | 方法 | 说明 |
|-----|------|------|
| Data Collator | `_get_data_collator()` | 从 Template 自动构建 mode-aware collator |
| 自定义指标 | `MeanMetric` | 训练/评估过程中实时计算自定义指标 |
| DeepSpeed 补丁 | `_patch_deepspeed_load_checkpoint()` | 修复 resume 加载 |
| Flash Checkpoint | `_add_callbacks()` | DLRover 集成，快速保存 |
| Zero3 gather | `_fix_zero3_gather_all_parameters()` | DeepSpeed ZeRO3 参数聚合 |
| 序列并行 | 序列并行 Hook | Ulysses/Ring Attention 注入 |
| 梯度检查点 | `_fix_gradient_checkpointing()` | 兼容新旧版本 |

### 4.3 损失计算流程

```python
def compute_loss(self, model, inputs, ...):
    # 1. 提取 labels 和特殊字段
    labels = inputs.pop('labels', None)
    loss_scale = inputs.pop('loss_scale', None)
    compute_loss_func = inputs.pop('compute_loss_func', None)
    
    # 2. 前向传播
    outputs = model(**inputs)
    
    # 3. 自定义损失函数
    if compute_loss_func is not None:
        loss = compute_loss_func(outputs.logits, labels, ...)
    elif loss_scale is not None:
        loss = per_token_loss_func(outputs.logits, labels, loss_scale)
    elif labels is not None:
        # 默认: CrossEntropy
        loss = F.cross_entropy(outputs.logits, labels)
    
    # 4. MoE 辅助损失
    if hasattr(outputs, 'aux_loss'):
        loss += outputs.aux_loss * router_aux_loss_coef
    
    return loss
```

---

## 5. RLHF 管道特殊流程

### 5.1 SwiftRLHF 多模型加载

**文件**: `swift/pipelines/train/rlhf.py`

RLHF 训练需要加载多个模型：

```
policy model (可训练) ─────────────────┐
    │                                    │
ref model (冻结, 计算 KL)               │
    │                                    ├──▶ GRPOTrainer
reward model(s) (冻结, 打分)            │
    │                                    │
teacher model (冻结, GKD 用)            │
    │                                    │
vllm_client (可选, 异步生成) ───────────┘
```

```python
class SwiftRLHF(SwiftSft):
    def _prepare_model_tokenizer(self):
        # 1. 加载 ref/value/teacher 模型
        for key in ['ref', 'value', 'teacher']:
            model, processor = self._prepare_single_model(key, ...)
            setattr(self, f'{key}_model', model)
        
        # 2. 加载 reward 模型(s)
        for reward_model_path in rms:
            model, processor = self._prepare_single_model('reward', ...)
            self.reward_model.append(model)
        
        # 3. 加载 policy 模型 (基类 SwiftSft 已处理)
```

### 5.2 RLHF Trainer 初始化

以 GRPO 为例：

```python
class GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer):
    def __init__(self, model, ref_model, reward_model, reward_funcs, ...):
        # 1. 算法参数准备
        self._prepare_algorithm_params()
        
        # 2. 调用父类初始化
        super().__init__(model, ref_model, ...)
        
        # 3. rollout 准备 (vLLM / transformers)
        self.prepare_rollout()
        
        # 4. reward 准备
        self._prepare_rewards(reward_funcs, reward_model)
        
        # 5. 评估配置检查
        if self.args.eval_strategy != 'no':
            assert len(self.eval_dataset) >= total_eval_batch_size
```

---

## 6. 关键参数配置体系

### 6.1 参数继承体系

```
BaseArguments
    ├── SftArguments ────▶ SwiftSft
    │   ├── PretrainArguments
    │   └── EmbeddingArguments
    ├── RLHFArguments ───▶ SwiftRLHF
    │   ├── DPOArguments
    │   ├── GRPOArguments
    │   └── PPOArguments
    ├── InferArguments ──▶ SwiftInfer
    └── ExportArguments ─▶ SwiftExport
```

### 6.2 训练关键参数

| 参数类别 | 代表参数 | 说明 |
|---------|---------|------|
| 模型 | `--model`, `--model_type`, `--torch_dtype` | 模型选择与精度 |
| 数据 | `--dataset`, `--max_length`, `--truncation_strategy` | 数据集与长度控制 |
| Tuner | `--tuner_type`, `--target_modules`, `--lora_rank` | 微调方法配置 |
| 训练 | `--learning_rate`, `--batch_size`, `--num_train_epochs` | 基本训练超参 |
| 分布式 | `--deepspeed`, `--sequence_parallel_size` | 并行策略 |
| 优化 | `--use_liger_kernel`, `--use_unsloth`, `--enable_dft_loss` | 加速与增强 |

---

## 7. 关键代码路径索引

| 流程阶段 | 关键文件/函数 |
|---------|-------------|
| CLI 入口 | `swift/cli/main.py::cli_main()` |
| SFT Pipeline | `swift/pipelines/train/sft.py::SwiftSft` |
| RLHF Pipeline | `swift/pipelines/train/rlhf.py::SwiftRLHF` |
| Pipeline 基类 | `swift/pipelines/base.py::SwiftPipeline` |
| Tuner 混合 | `swift/pipelines/train/tuner.py::TunerMixin` |
| Trainer 工厂 | `swift/trainers/trainer_factory.py::TrainerFactory` |
| Seq2SeqTrainer | `swift/trainers/seq2seq_trainer.py::Seq2SeqTrainer` |
| SwiftMixin | `swift/trainers/mixin.py::SwiftMixin` |
| 参数基类 | `swift/arguments/base_args.py::BaseArguments` |
| SFT 参数 | `swift/arguments/sft_args.py::SftArguments` |
| RLHF 参数 | `swift/arguments/rlhf_args.py::RLHFArguments` |
| 数据集加载 | `swift/dataset/loader.py::load_dataset()` |
| 数据集编码 | `swift/dataset/preprocessor/core.py::EncodePreprocessor` |
| Template 编码 | `swift/template/base.py::Template.encode()` |
| Data Collator | `swift/template/base.py::Template.data_collator()` |
