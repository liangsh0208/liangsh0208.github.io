---
created: 2026-05-09
---

# ms-swift 训练系统：基础训练器与 Mixin 架构

ms-swift 的训练系统不从零构建 Trainer，而是在 HuggingFace `transformers.Trainer` 和 `trl` 基类之上，通过 **Mixin 多重继承** 注入框架特有的公共能力。这种设计使得标准 SFT、预训练、Embedding、Reranker 以及各类 RLHF 算法都能共享同一套基础设施。

---

## 1. 训练器继承体系

```
                         HuggingFace transformers
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    HfSeq2SeqTrainer          HfTrainer              trl.*Trainer
    (causal_lm SFT)           (seq_cls/embedding)   (DPO/GRPO/PPO)
              │                     │                     │
              └────────────┬────────┴────────┬────────────┘
                           │                   │
                           ▼                   ▼
                  ┌──────────────┐    ┌──────────────┐
                  │  SwiftMixin  │    │ DataLoaderMixin│
                  │  (公共能力注入) │    │ (数据加载扩展) │
                  └──────────────┘    └──────────────┘
                           │                   │
                           └─────────┬─────────┘
                                     │
                                     ▼
          ┌──────────────┬──────────┴──────────┬──────────────┐
          ▼              ▼                     ▼              ▼
    Seq2SeqTrainer    Trainer      RLHFTrainerMixin    RolloutTrainerMixin
    (SFT/Pretrain)   (SeqCls)         (DPO/GKD)           (GRPO/PPO)
          │              │                 │                 │
          ▼              ▼                 ▼                 ▼
    EmbeddingTrainer  RerankerTrainer   DPOTrainer       GRPOTrainer
    RerankerTrainer                                    PPOTrainer
                                                       KTOTrainer
                                                       ORPOTrainer
                                                       CPOTrainer
                                                       RewardTrainer
```

---

## 2. TrainerFactory — 训练器路由

**文件**: `swift/trainers/trainer_factory.py`

`TrainerFactory` 是训练器实例化的**统一入口**，根据参数中的 `task_type` 或 `rlhf_type` 自动路由到正确的 Trainer 和 TrainingArguments 类：

```python
class TrainerFactory:
    TRAINER_MAPPING = {
        # 标准训练任务
        'causal_lm': 'swift.trainers.Seq2SeqTrainer',
        'seq_cls': 'swift.trainers.Trainer',
        'embedding': 'swift.trainers.EmbeddingTrainer',
        'reranker': 'swift.trainers.RerankerTrainer',
        'generative_reranker': 'swift.trainers.RerankerTrainer',
        # RLHF 任务
        'dpo': 'swift.rlhf_trainers.DPOTrainer',
        'orpo': 'swift.rlhf_trainers.ORPOTrainer',
        'kto': 'swift.rlhf_trainers.KTOTrainer',
        'cpo': 'swift.rlhf_trainers.CPOTrainer',
        'rm': 'swift.rlhf_trainers.RewardTrainer',
        'ppo': 'swift.rlhf_trainers.PPOTrainer',
        'grpo': 'swift.rlhf_trainers.GRPOTrainer',
        'gkd': 'swift.rlhf_trainers.GKDTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        'causal_lm': 'swift.trainers.Seq2SeqTrainingArguments',
        'seq_cls': 'swift.trainers.TrainingArguments',
        # ... 与 TRAINER_MAPPING 一一对应
    }

    @classmethod
    def get_trainer_cls(cls, args):
        train_method = args.rlhf_type if hasattr(args, 'rlhf_type') else args.task_type
        module_path, class_name = cls.TRAINER_MAPPING[train_method].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
```

**路由逻辑**: `args.rlhf_type`（如果存在，优先级更高）→ 否则用 `args.task_type`。

---

## 3. SwiftMixin — 公共能力注入器

**文件**: `swift/trainers/mixin.py` (~61KB)

`SwiftMixin` 是所有 Swift Trainer 的**核心能力层**，在 Trainer `__init__` 中先于 HF Trainer 执行，负责数据 collator 构建、回调注册、DeepSpeed 补丁、Flash Checkpoint 等。

### 3.1 初始化流程

```python
class SwiftMixin:
    def __init__(self, model, args, template, train_dataset, eval_dataset=None, **kwargs):
        # 1. 数据集长度检查（IterableDataset 限制）
        if not hasattr(train_dataset, '__len__') and args.dataloader_num_workers > 1:
            args.dataloader_num_workers = 1

        # 2. 绑定 Template
        self.template = template
        self.is_encoder_decoder = template.is_encoder_decoder
        self.padding_free = template.padding_free
        self.task_type = template.task_type

        # 3. 自定义优化器回调
        self.optimizer_callback = optimizers_map[args.optimizer or 'default'](args, self)

        # 4. 自定义指标容器
        self.custom_metrics = {
            'train': collections.defaultdict(lambda: MeanMetric(nan_value=None, device=args.device)),
            'eval': collections.defaultdict(lambda: MeanMetric(nan_value=None, device=args.device))
        }

        # 5. 构建 data collator
        data_collator = self._get_data_collator(args, template)

        # 6. 创建损失/评估指标
        kwargs.update(self.create_loss_and_eval_metric(args))

        # 7. 兼容 transformers 5.0 tokenizer → processing_class 重命名
        tokenizer_key = 'processing_class' if 'processing_class' in inspect.signature(HfTrainer.__init__).parameters else 'tokenizer'
        kwargs[tokenizer_key] = template.tokenizer

        # 8. 调用 HF Trainer __init__
        super().__init__(model=model, args=args, data_collator=data_collator, 
                         train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)

        # 9. 梯度累积修正 (transformers 5.0)
        if version.parse(transformers.__version__) >= version.parse('5.0.0'):
            self.accelerator.gradient_state.plugin_kwargs['num_steps'] = 1

        # 10. 添加回调
        self._add_callbacks()

        # 11. 梯度检查点修复
        self._fix_gradient_checkpointing()

        # 12. 生成配置更新
        update_generation_config_eos_token(self.model.generation_config, self.template)
```

### 3.2 Data Collator 构建

```python
def _get_data_collator(self, args, template):
    # LongLoRA: 固定 padding 到 max_length
    padding_to = template.max_length if args.tuner_type == 'longlora' else None
    return partial(template.data_collator, padding_to=padding_to)
```

Template 的 `data_collator()` 会根据 `task_type`（`causal_lm`/`rlhf`/`kto`/...）和 `mode`（`train`）自动选择正确的 collator 实现。

### 3.3 回调系统

```python
def _add_callbacks(self):
    for callback in self.args.callbacks:
        self.add_callback(callbacks_map[callback](self.args, self))
```

内置回调映射（`swift/callbacks/`）:

| 回调名 | 功能 |
|-------|------|
| `tensorboard` | TensorBoard 日志 |
| `wandb` | Weights & Biases 日志 |
| `swanlab` | SwanLab 实验跟踪 |
| `flash_checkpoint` | 快速检查点（DLRover 集成）|
| `activation_cpu_offload` | FSDP 激活值卸载 |

### 3.4 DeepSpeed 补丁

```python
@contextmanager
def _patch_deepspeed_load_checkpoint(self):
    # 修复 resume_from_checkpoint + resume_only_model 组合时的 DeepSpeed 加载失败
    origin = trainer.deepspeed_load_checkpoint
    def patched(*args, **kwargs):
        try:
            return origin(*args, **kwargs)
        except Exception as e:
            logger.warning(f'Failed to call deepspeed_load_checkpoint: {e}')
    trainer.deepspeed_load_checkpoint = patched
    yield
    trainer.deepspeed_load_checkpoint = origin
```

### 3.5 Flash Checkpoint

```python
# 启用条件
def _prepare_flash_ckpt(self):
    if self.args.use_flash_ckpt:
        import dlrover.trainer.torch.flash_checkpoint.hf_trainer
        # 自动注册 Flash Checkpointer
```

Flash Checkpoint 通过 DLRover 的异步持久化机制，将 checkpoint 保存耗时从数分钟降低到秒级。

### 3.6 序列并行集成

```python
def _prepare_inputs(self, inputs):
    if self.template.sequence_parallel_size > 1:
        sequence_parallel.prepare_inputs(inputs)
    ...
```

SequenceParallel 的 Hook 在 Trainer 的 `_prepare_inputs` 阶段自动注入，调整 input_ids/attention_mask/position_ids 的分布。

### 3.7 自定义指标

```python
class MeanMetric:
    """支持 nan 值的均值指标"""
    def update(self, value):
        if not math.isnan(value):
            self.sum += value
            self.count += 1
    def compute(self):
        return self.sum / self.count if self.count > 0 else float('nan')
```

在 `compute_loss()` 中实时更新：
```python
self.custom_metrics['train']['my_metric'].update(some_value)
```

---

## 4. Seq2SeqTrainer — Causal LM 训练核心

**文件**: `swift/trainers/seq2seq_trainer.py`

专门用于因果语言模型（Causal LM）的训练，覆盖 SFT、Pretrain、Embedding 等任务。

### 4.1 初始化

```python
class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        
        # 评估时使用 TransformersEngine 做生成
        if self.args.predict_with_generate:
            self.infer_engine = TransformersEngine(
                self.model, template=self.template, 
                max_batch_size=self.args.per_device_eval_batch_size)
```

### 4.2 输入准备

```python
def _prepare_inputs(self, inputs):
    inputs = super()._prepare_inputs(inputs)
    
    # 序列并行准备
    if self.template.sequence_parallel_size > 1:
        sequence_parallel.prepare_inputs(inputs)
    
    # logits_to_keep 优化 (transformers 5.0+)
    use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
    if use_logits_to_keep:
        self.prepare_logits_to_keep(inputs)
    
    # MoE 辅助损失
    base_model = self.template.get_base_model(self.model)
    forward_params = inspect.signature(base_model.forward).parameters
    if self.model.model_info.is_moe_model:
        inputs['output_router_logits'] = True
    
    # 绑定自定义 loss 函数
    inputs['compute_loss_func'] = self.compute_loss_func
    return inputs
```

### 4.3 损失计算

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.pop('labels', None)
    compute_loss_func = inputs.pop('compute_loss_func', None)
    loss_scale = inputs.pop('loss_scale', None)
    text_position_ids = inputs.pop('text_position_ids', None)
    
    # 前向传播
    outputs = model(**inputs)
    
    # MoE 辅助损失记录
    if getattr(outputs, 'aux_loss', None):
        self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
    
    # 自定义损失计算
    if labels is not None and (compute_loss_func or loss_scale or self.template.sequence_parallel_size > 1):
        # per-token 损失
        loss = self._custom_loss(outputs.logits, labels, loss_scale, compute_loss_func, text_position_ids)
    else:
        # 标准损失
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
    
    # MoE aux loss 累加
    if getattr(outputs, 'aux_loss', None):
        loss += outputs.aux_loss
    
    return (loss, outputs) if return_outputs else loss
```

### 4.4 评估阶段 predict_with_generate

```python
def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
    if not self.args.predict_with_generate or prediction_loss_only:
        # 标准 loss 计算
        return super().prediction_step(...)
    
    # 使用 TransformersEngine 生成回复
    data_list = inputs['_data']
    labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
    
    with unwrap_model_for_generation(self.model_wrapped, self.accelerator):
        resp_list = self.infer_engine.infer(data_list, RequestConfig(max_tokens=...))
    
    # 计算 ROUGE/BLEU 等生成指标
    ...
```

---

## 5. 其他基础 Trainer

### 5.1 Trainer — 序列分类/通用任务

**文件**: `swift/trainers/trainer.py`

```python
class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    # 用于 seq_cls / 通用分类任务
    # 继承 SwiftMixin 获得所有公共能力
```

### 5.2 EmbeddingTrainer

**文件**: `swift/trainers/embedding_trainer.py`

```python
class EmbeddingTrainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    # 专门用于 Embedding 模型训练
    # 使用 contrastive/sentence-transformers 损失
```

### 5.3 RerankerTrainer

**文件**: `swift/trainers/reranker_trainer.py`

```python
class RerankerTrainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    # 重排序模型训练
    # 支持 pointwise / pairwise 损失
```

---

## 6. DataLoaderMixin — 数据加载扩展

**文件**: `swift/trainers/mixin.py` 或独立模块

提供分布式场景下的数据加载增强：

```python
class DataLoaderMixin:
    def get_train_dataloader(self):
        # 返回自定义 DataLoader，支持：
        # - sequence parallel 采样器
        # - packing 模式调整
        # - 多进程 worker 种子
```

---

## 7. 关键代码路径索引

| 功能 | 关键文件/函数 |
|-----|-------------|
| Trainer 路由工厂 | `swift/trainers/trainer_factory.py::TrainerFactory` |
| SwiftMixin 核心 | `swift/trainers/mixin.py::SwiftMixin` |
| Seq2SeqTrainer | `swift/trainers/seq2seq_trainer.py::Seq2SeqTrainer` |
| Trainer (seq_cls) | `swift/trainers/trainer.py::Trainer` |
| EmbeddingTrainer | `swift/trainers/embedding_trainer.py::EmbeddingTrainer` |
| RerankerTrainer | `swift/trainers/reranker_trainer.py::RerankerTrainer` |
| 训练参数基类 | `swift/trainers/arguments.py::TrainingArguments` |
| Seq2Seq 参数 | `swift/trainers/arguments.py::Seq2SeqTrainingArguments` |
| 回调注册 | `swift/callbacks/` 目录 |
| Flash Checkpoint | `swift/callbacks/flash_checkpoint.py` |
| DeepSpeed 补丁 | `swift/trainers/mixin.py::_patch_deepspeed_load_checkpoint` |
| 梯度检查点修复 | `swift/trainers/mixin.py::_fix_gradient_checkpointing` |
| 自定义指标 | `swift/metrics.py::MeanMetric` |
| DataLoader 扩展 | `swift/dataloader/` 目录 |
