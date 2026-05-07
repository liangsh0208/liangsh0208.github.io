# Nanotron: Trainer训练循环

---

## 顶部信息栏

| 属性 | 值 |
|------|-----|
| **文档版本** | v1.0 |
| **创建日期** | 2026-04-20 |
| **代码路径** | `src/nanotron/trainer.py`, `run_train.py` |
| **核心文件** | `trainer.py`, `helpers.py`, `dataloader.py` |
| **关键类/函数** | `DistributedTrainer`, `training_step()`, `train()`, `init_optimizer_and_grad_accumulator` |

---

## 模块概述

Nanotron的`DistributedTrainer`是整个训练系统的核心控制器，负责：

1. **训练生命周期管理**：初始化、训练、验证、保存/加载检查点
2. **混合并行协调**：TP/PP/DP/CP的组合管理
3. **优化器和调度器管理**：AdamW、学习率调度、梯度裁剪
4. **日志和监控**：吞吐量计算、内存监控、WandB集成
5. **分布式检查点**：支持断点续训和分布式参数保存

### 重点特性

- **Pipeline Engine抽象**：1F1B等流水线调度策略
- **梯度累积与同步**：FP32梯度累积、DDP/Zero优化器支持
- **Tied Weight处理**：embedding共享、跨组梯度同步
- **多阶段数据切换**：训练过程中的数据策略变化

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        DistributedTrainer Architecture                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                         __init__() 初始化流程                                   ││
│  │                                                                                  ││
│  │  1. ParallelContext ──▶ TP/PP/DP/CP/EP进程组创建                                ││
│  │         │                                                                        ││
│  │  2. init_model() ────▶ 构建模型 + 加载检查点 + 权重复制                           ││
│  │         │                                                                        ││
│  │  3. init_optimizer_and_grad_accumulator()                                         ││
│  │         │    ├── NamedOptimizer (按参数分组)                                     ││
│  │         │    ├── FP32GradientAccumulator (可选)                                  ││
│  │         │    └── ZeroDistributedOptimizer (可选)                                  ││
│  │         │                                                                        ││
│  │  4. lr_scheduler_builder() ──▶ lambda函数：warmup + decay                        ││
│  │         │                                                                        ││
│  │  5. 加载检查点 (resume) ──▶ optimizer/lr_scheduler/随机状态                       ││
│  │                                                                                  ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                    │                                                 │
│                                    ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                           train() 主训练循环                                     ││
│  │                                                                                  ││
│  │  for iteration_step in range(initial_step, final_step):                         ││
│  │      │                                                                          ││
│  │      ├── _update_dataloader_based_on_training_stages()  # 多阶段数据切换         ││
│  │      │                                                                          ││
│  │      ├── training_step()                                                        ││
│  │      │       │                                                                  ││
│  │      │       ├── pipeline_engine.train_batch_iter()  # PP forward/backward       ││
│  │      │       │       ├── forward pass (1F1B调度)                                ││
│  │      │       │       ├── backward pass (梯度累积)                                ││
│  │      │       │       └── loss scaling                                            ││
│  │      │       │                                                                  ││
│  │      │       ├── sync_gradients_across_dp()        # DP梯度同步                   ││
│  │      │       ├── sync_tied_weights_gradients()     # 共享权重梯度同步              ││
│  │      │       ├── clip_grad_norm()                  # 梯度裁剪                     ││
│  │      │       ├── optimizer.step()                  # 参数更新                     ││
│  │      │       └── lr_scheduler.step()               # 学习率更新                   ││
│  │      │                                                                          ││
│  │      ├── train_step_logs()       # 日志记录                                      ││
│  │      └── save_checkpoint()       # 定期保存                                      ││
│  │                                                                                  ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 训练步骤详细时序图

```
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Data   │    │  PP    │    │  PP    │    │  PP    │    │ Grad   │    │ Optim  │
│ Loader │    │ Rank 0 │    │ Rank 1 │    │ Rank N │    │ Sync   │    │ Step   │
└───┬────┘    └────┬───┘    └────┬───┘    └────┬───┘    └────┬───┘    └────┬───┘
    │              │              │              │              │              │
    │              │              │              │              │              │
    │  micro_batch │              │              │              │              │
    │─────────────▶│   (多次，按  │              │              │              │
    │   (x n_micro │   n_micro_   │    通过P2P   │              │              │
    │    batches)  │   batches)   │    传递      │              │              │
    │              │              │              │              │              │
    │              │ forward      │ forward      │   forward    │              │
    │              │ + backward   │ + backward   │  + backward  │              │
    │              │              │              │              │              │
    │              │              │   Gradient   │              │              │
    │              │              │   Accumulator│              │              │
    │              │              │   (FP32)     │              │              │
    │              │              │              │              │              │
    │              │              │             gradient        │              │
    │              │              │─────────────┼─────────────▶│              │
    │              │              │           all_reduce       │              │
    │              │              │           (avg op)         │              │
    │              │              │              │              │              │
    │              │   Tied Weights Sync         │              │              │
    │              │◀─────────────┼─────────────┼──────────────┼──────────────│
    │              │  (embedding共享)            │              │              │
    │              │              │              │              │              │
    │              │              │              │ gradient clipping            │
    │              │              │              │              │             │
    │              │              │              │              │ optimizer.step()│
    │              │              │              │              │   (update)   │
    │              │              │              │              │              │
    │              │              │              │              │ lr_scheduler.│
    │              │              │              │              │ step()       │
    │              │              │              │              │              │
    ▼              ▼              ▼              ▼              ▼              ▼
```

---

## 核心实现详解

### 1. run_train.py - 训练入口

```python
# File: run_train.py

def main():
    args = get_args()
    config_file = args.config_file

    # 1. 创建DistributedTrainer实例
    #    - 读取配置
    #    - 初始化模型、优化器、调度器
    #    - 创建ParallelContext
    trainer = DistributedTrainer(config_file)
    
    # 2. 创建/恢复dataloader
    dataloader = get_dataloader(trainer, args.sanity_check_dataloadeer_interval)
    
    # 3. 开始训练
    trainer.train(dataloader)

if __name__ == "__main__":
    # 启动方式:
    # torchrun --nproc_per_node=8 run_train.py --config-file config.yaml
    main()
```

**启动脚本示例：**

```bash
#!/bin/bash
# 8卡单机训练 - TinyLlama
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 重要：优化分布式性能

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=6000 \
    run_train.py \
    --config-file examples/config_tiny_llama.yaml
```

### 2. DistributedTrainer.__init__详解

```python
# trainer.py lines 131-297

class DistributedTrainer:
    def __init__(
        self,
        config_or_config_file: Union[Config, str],
        config_class: Type[Config] = Config,
        model_config_class: Optional[Type] = None,
        model_class: Type[NanotronModel] = None,
    ):
        # ========== 阶段1: 配置解析 ==========
        if isinstance(config_or_config_file, str):
            self.config = get_config_from_file(config_path=config_or_config_file)
        else:
            self.config = config_or_config_file
            
        # ========== 阶段2: 并行上下文初始化 ==========
        self.parallel_context = ParallelContext(
            tensor_parallel_size=self.config.parallelism.tp,
            pipeline_parallel_size=self.config.parallelism.pp,
            data_parallel_size=self.config.parallelism.dp,
            expert_parallel_size=self.config.parallelism.expert_parallel_size,
            context_parallel_size=self.config.parallelism.context_parallel_size,
        )
        
        # ========== 阶段3: 模型初始化 ==========
        tp_rank = dist.get_rank(self.parallel_context.tp_pg)
        set_random_seed(self.config.general.seed + tp_rank)  # TP rank不同seed
        
        self.model = self.init_model()  # 构建模型
        self.unwrapped_model = self.model.module if isinstance(self.model, DDP) else self.model
        
        # ========== 阶段4: 优化器初始化 ==========
        self.optimizer, self.grad_accumulator = init_optimizer_and_grad_accumulator(
            parametrization_method=parametrization_method,
            model=self.model,
            optimizer_args=self.config.optimizer,
            parallel_context=self.parallel_context,
        )
        
        # 从检查点恢复优化器状态
        if self.init_checkpoint_path is not None and self.config.checkpoints.load_optimizer:
            load_optimizer(optimizer=self.optimizer, ...)
        
        # ========== 阶段5: 学习率调度器 ==========
        self.lr_scheduler = lr_scheduler_builder(
            optimizer=self.optimizer,
            lr_scheduler_args=self.config.optimizer.learning_rate_scheduler,
            total_training_steps=self.config.tokens.train_steps,
        )
        
        if self.init_checkpoint_path is not None and self.config.checkpoints.load_lr_scheduler:
            load_lr_scheduler(lr_scheduler=self.lr_scheduler, ...)
```

### 3. training_step完整流程

```python
# trainer.py lines 604-731

def training_step(
    self, 
    dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]
) -> Tuple[Iterable[Dict], Optional[torch.Tensor]]:
    """
    单步训练流程
    
    返回: (outputs, loss_avg, z_loss_avg, tbi_logs)
    """
    # ===== 前置检查 =====
    before_tbi_sanity_checks(
        self.config, self.parallel_context, self.unwrapped_model, 
        self.grad_accumulator, self.lr_scheduler
    )
    
    # ===== 1. 前向/后向传播 (Pipeline) =====
    # 生成n_micro_batches_per_batch个micro-batch
    micro_batches = (next(dataloader) for _ in range(self.n_micro_batches_per_batch))
    
    with torch.profiler.record_function("train_batch_iter"):
        outputs = self.pipeline_engine.train_batch_iter(
            model=self.model,
            pg=self.parallel_context.pp_pg,
            batch=micro_batches,
            nb_microbatches=self.n_micro_batches_per_batch,
            grad_accumulator=self.grad_accumulator,  # 可选FP32累积
        )
    
    # ===== 2. 等待DDP梯度AllReduce完成 =====
    if isinstance(self.model, DistributedDataParallel) and self.grad_accumulator is not None:
        # DDP会触发fp32梯度的allreduce，需要等待完成
        for handle in self.grad_accumulator.fp32_grads_allreduce_handle:
            handle.wait()
    
    # ===== 3. 梯度同步 (非DDP场景) =====
    if not isinstance(self.model, DistributedDataParallel):
        # 手动DP梯度同步
        sync_gradients_across_dp(
            module=self.model,
            dp_pg=self.parallel_context.dp_pg,
            reduce_op=dist.ReduceOp.AVG,
            grad_accumulator=self.grad_accumulator,
        )
    
    # ===== 4. Tied Weights梯度同步 =====
    sync_tied_weights_gradients(
        module=self.unwrapped_model,
        parallel_context=self.parallel_context,
        grad_accumulator=self.grad_accumulator,
    )
    
    # ===== 5. 梯度裁剪 =====
    if self.config.optimizer.clip_grad is not None:
        named_parameters = [
            (name, param) for name, param in self.unwrapped_model.get_named_params_with_correct_tied()
            if param.requires_grad
        ]
        self.grad_norm_unclipped = clip_grad_norm(
            mp_pg=self.parallel_context.mp_pg,
            named_parameters=named_parameters,
            grad_accumulator=self.grad_accumulator,
            max_norm=self.config.optimizer.clip_grad,  # 默认1.0
        )
    
    # ===== 6. DP-CP间loss同步 (异步) =====
    if isinstance(outputs[0]["loss"], torch.Tensor):
        loss_avg = torch.stack([output["loss"] for output in outputs]).sum()
        handle = dist.all_reduce(
            loss_avg, 
            group=self.parallel_context.dp_cp_pg, 
            async_op=True,  # 异步与优化器重叠
            op=dist.ReduceOp.AVG
        )
    
    # ===== 7. 优化器前检查 =====
    before_optim_step_sanity_checks(...)
    tbi_logs = self.unwrapped_model.get_tbi_logs(non_blocking=True)
    
    # ===== 8. 参数更新 =====
    self.optimizer.step()
    self.optimizer.zero_grad()
    
    # ===== 9. 学习率更新 =====
    self.lr_scheduler.step()
    
    # 等待异步loss同步完成
    if handle is not None:
        handle.wait()
    
    return outputs, loss_avg, z_loss_avg, tbi_logs
```

### 4. Pipeline Engine训练迭代

```python
# trainer.py lines 614-623

self.pipeline_engine: PipelineEngine = self.config.parallelism.pp_engine

# PipelineEngine.train_batch_iter() 由配置决定
# 目前支持: "1f1b" (One Forward One Backward)

outputs = self.pipeline_engine.train_batch_iter(
    model=self.model,
    pg=self.parallel_context.pp_pg,          # PP进程组
    batch=micro_batches_generator,           # micro-batch生成器
    nb_microbatches=self.n_micro_batches_per_batch,  # 梯度累积步数
    grad_accumulator=self.grad_accumulator,  # 用于FP32梯度累积
)
```

**Pipeline调度流程：**

```
1F1B (One Forward One Backward)调度
适用于PP > 1的场景

时间线:
────▶

Stage 0: F0    F1    F2    F3    B3    B2    B1    B0
              │     │     │     │
Stage 1:      F0    F1    F2    F3    B3    B2    B1    B0
                    │     │     │
Stage 2:            F0    F1    F2    F3    B3    B2    B1    B0
                          │     │
Stage 3:                  F0    F1    F2    F3    B3    B2    B1    B0

F = Forward, B = Backward
数字表示micro-batch编号
```

### 5. 优化器与梯度累积

```python
# helpers.py lines 315-464

def init_optimizer_and_grad_accumulator(
    parametrization_method: ParametrizationMethod,
    model: Union[nn.Module, DistributedDataParallel],
    optimizer_args: OptimizerArgs,
    parallel_context: ParallelContext,
) -> Tuple[BaseOptimizer, GradientAccumulator]:
    """
    构建优化器链：
    AdamW -> NamedOptimizer -> (FP32GradientAccumulator) -> (ZeroDistributedOptimizer)
    """
    
    # 1. 基础优化器 - AdamW
    def basic_optimizer_builder(param_groups):
        if optimizer_args.optimizer_factory.name == "adamW":
            optimizer = lambda pg: torch.optim.AdamW(
                pg,
                lr=optimizer_args.learning_rate_scheduler.learning_rate,
                weight_decay=optimizer_args.weight_decay,
                eps=optimizer_args.optimizer_factory.adam_eps,
                betas=(optimizer_args.optimizer_factory.adam_beta1, 
                       optimizer_args.optimizer_factory.adam_beta2),
                fused=optimizer_args.optimizer_factory.torch_adam_is_fused,
            )
        return NamedOptimizer(named_params_or_groups=named_param_groups, optimizer_builder=optimizer)
    
    # 2. FSDP-style FP32梯度累积 (可选)
    if optimizer_args.accumulate_grad_in_fp32:
        def grad_optimizer_builder(named_param_groups):
            return OptimizerFromGradientAccumulator(
                gradient_accumulator_builder=lambda: FP32GradientAccumulator(...),
                optimizer_builder=basic_optimizer_builder,
            )
        optimizer_builder = grad_optimizer_builder
    
    # 3. Zero优化器 (可选)
    if optimizer_args.zero_stage > 0:
        optimizer = ZeroDistributedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=optimizer_builder,
            dp_pg=parallel_context.dp_pg,
        )
```

### 6. 梯度同步策略

```python
# src/nanotron/parallel/data_parallel/utils.py

@torch.profiler.record_function("sync_gradients_across_dp")
def sync_gradients_across_dp(
    module: nn.Module,
    dp_pg: dist.ProcessGroup,
    reduce_op: dist.ReduceOp,
    grad_accumulator: Optional[GradientAccumulator],
    reduce_scatter: bool = False,  # Zero优化器使用
):
    """
    DP梯度同步策略：
    1. 有grad_accumulator: 使用其优化的allreduce
    2. 无grad_accumulator: 直接allreduce每个参数的grad
    """
    if grad_accumulator is not None:
        grad_accumulator.sync_gradients_across_dp(dp_pg=dp_pg, reduce_op=reduce_op)
    else:
        for name, param in module.named_parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=reduce_op, group=dp_pg)
```

### 7. Tied Weights处理

```python
# trainer.py lines 1306-1339

def mark_tied_parameters(model, parallel_context, parallel_config):
    """
    标记并处理共享参数（如embedding与lm_head）
    
    处理策略:
    1. Embedding-LM Head权重绑定
    2. 跨TP的未分片参数同步
    3. 创建专门的process group用于梯度reduce
    """
    # 1. Embedding-LM Head绑定
    embeddings_lm_head_tied_names = model.get_embeddings_lm_head_tied_names()
    if len(embeddings_lm_head_tied_names) > 0:
        shared_embeddings = [
            (target, (target_global_rank,)) for target in embeddings_lm_head_tied_names
        ]
        tie_parameters(
            root_module=model, 
            ties=shared_embeddings, 
            parallel_context=parallel_context,
            reduce_op=dist.ReduceOp.SUM
        )
    
    # 2. 未分片参数作为tied处理（如LayerNorm）
    mark_unsharded_params_as_tied_across_tp(model, parallel_context, parallel_config)
```

### 8. 学习率调度器

```python
# helpers.py lines 79-183

def lr_scheduler_builder(optimizer, lr_scheduler_args, total_training_steps):
    """
    学习率调度: warmup -> constant(可选) -> decay -> constant(可选)
    
    支持:
    - warmup_style: linear / constant
    - lr_decay_style: linear / cosine / 1-sqrt
    """
    def lr_lambda(current_step, initial_lr):
        # Phase 1: Warmup (lr从0到initial_lr)
        if current_step <= lr_warmup_steps:
            if lr_warmup_style == "linear":
                return initial_lr * current_step / max(lr_warmup_steps, 1)
            
        # Phase 2: Constant (保持initial_lr)
        elif current_step < lr_decay_starting_step:
            return initial_lr
            
        # Phase 3: Decay
        elif current_step < lr_decay_starting_step + lr_decay_steps:
            if lr_decay_style == "cosine":
                return min_decay_lr + (initial_lr - min_decay_lr) * 
                       (1 + cos(pi * (current_step - lr_decay_starting_step) / lr_decay_steps)) / 2
            elif lr_decay_style == "linear":
                return min_decay_lr + (initial_lr - min_decay_lr) * 
                       (lr_decay_steps - (current_step - lr_decay_starting_step)) / lr_decay_steps
                       
        # Phase 4: 最终constant (lr = min_decay_lr)
        else:
            return min_decay_lr
    
    return LambdaLR(optimizer.get_base_optimizer(), lr_lambda=lr_lambdas)
```

### 9. 验证/评估步骤

```python
# trainer.py lines 733-739

def validation_step(self, dataloader):
    """
    验证步骤 - 只进行前向传播，不更新参数
    
    注意: 不执行梯度累积
    """
    outputs = self.pipeline_engine.validate_batch_iter(
        model=self.model,
        batch=(next(dataloader) for _ in range(self.limit_val_batches)),
        nb_microbatches=self.limit_val_batches,
    )
    return outputs
```

### 10. 训练日志记录

```python
# trainer.py lines 741-969

def train_step_logs(self, outputs, loss_avg, z_loss_avg, tbi_logs):
    """
    记录训练指标:
    - 基本指标: loss, lr, iteration_time, throughput
    - 性能指标: model_tflops, tokens/sec
    - 内存指标: CUDA memory, CPU memory
    - 详细指标: 每层的tbi_logs (可选)
    """
    # 计算吞吐量
    tokens_per_sec = (
        self.global_batch_size * self.sequence_length / 
        (elapsed_time_per_iteration_ms / 1000)
    )
    
    # 估算FLOPS
    model_tflops, hardware_tflops = self.unwrapped_model.get_flops_per_sec(
        iteration_time_in_sec=elapsed_time_per_iteration_ms / 1000,
        sequence_length=self.sequence_length,
        global_batch_size=self.global_batch_size,
    )
    
    # 记录到TensorBoard / WandB
    if should_log_to_wandb:
        wandb.log({...})
    
    # Benchmark模式: 第4步后退出并记录结果
    if os.environ.get("NANOTRON_BENCHMARK", "0") == "1" and self.iteration_step == 4:
        log_throughput(...)
        exit(0)
```

### 11. Profiler支持

```python
# helpers.py lines 498-523

def get_profiler(config: Config):
    """
    PyTorch Profiler配置
    
    示例输出: Chrome trace文件，可用chrome://tracing查看
    """
    if config.profiler is not None and dist.get_rank() == 0:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=config.profiler.wait,      # 等待步数
                warmup=config.profiler.warmup,  # warmup步数  
                active=config.profiler.active,  # 活跃记录步数
                repeat=config.profiler.repeat,  # 重复次数
                skip_first=config.profiler.skip_first,  # 跳过前N步
            ),
            on_trace_ready=tensorboard_trace_handler(
                config.profiler.profiler_export_path / ...
            ),
            record_shapes=config.profiler.record_shapes,    # 记录张量形状
            profile_memory=config.profiler.profile_memory,  # 记录内存分配
            with_stack=config.profiler.with_stack,          # 记录调用栈
        )
    else:
        prof = contextlib.nullcontext()  # 空context manager
    
    return prof
```

**Profiler配置示例：**

```yaml
profiler:
  profiler_export_path: ./tb_logs/
  wait: 1
  warmup: 1
  active: 3
  repeat: 2
  skip_first: 5
  record_shapes: true
  profile_memory: true
  with_stack: true
  export_chrome_trace: true
```

---

## 配置示例

### 基本训练配置

```yaml
tokens:
  sequence_length: 4096
  train_steps: 100000
  micro_batch_size: 2
  batch_accumulation_per_replica: 4  # 梯度累积步数
  
optimizer:
  accumulate_grad_in_fp32: true   # FP32梯度累积
  clip_grad: 1.0                  # 梯度裁剪阈值
  
  learning_rate_scheduler:
    learning_rate: 0.0003
    lr_warmup_steps: 2000
    lr_warmup_style: linear
    lr_decay_steps: 98000
    lr_decay_style: cosine
    min_decay_lr: 0.000001
    
  optimizer_factory:
    name: adamW
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    torch_adam_is_fused: true
    
  weight_decay: 0.01
  zero_stage: 0  # ZeRO优化器级别 (0=禁用, 1/2/3)
```

### 启用Profiler配置

```yaml
profiler:
  profiler_export_path: ./profiler_logs
  wait: 1        # 第1步等待
  warmup: 1      # 第2步warmup
  active: 3      # 第3-5步记录
  repeat: 1      # 重复1次
  skip_first: 5  # 跳过前5步
  record_shapes: true
  profile_memory: true
  with_stack: true
  export_chrome_trace: true
```

### WandB集成配置

```yaml
general:
  project: "nanotron-pretraining"
  run: "llama-7b-test"
  
# 自动记录关键指标到WandB:
# 1. consumed_tokens
# 2. lm_loss, z_loss
# 3. learning_rate
# 4. tokens_per_sec, model_tflops
# 5. CUDA memory usage
```

---

## 常见问题

### Q1: 训练突然OOM，如何排查？

**A**: Nanotron在训练初期(前5步)会打印内存快照：

```python
if self.iteration_step < self.initial_iter_step + 5:
    log_memory(logger=logger, msg="Before train_batch_iter")
    
# ... training logic ...

if self.iteration_step < self.initial_iter_step + 5:
    log_memory(logger=logger, msg="After train_batch_iter")
```

建议操作：
1. 减小micro_batch_size
2. 增大gradient_accumulation保持global_batch_size
3. 启用activation checkpointing: `parallelism.recompute_layer: true`
4. 启用ZeRO: `optimizer.zero_stage: 1`

### Q2: 如何设置合适的batch_accumulation？

**A**: 
```
global_batch_size = micro_batch_size 
                     * batch_accumulation_per_replica 
                     * parallelism.dp
```

通常目标global batch size约4-8M tokens (Llama标准)：
- 序列长度4096，目标GBS=1024 samples
- 8 GPUs (DP=2, TP=4), micro_batch_size=2
- 需要accumulation = 1024 / (2 * 2 * 2) = 64

### Q3: 断点续训后学习率从哪里开始？

**A**: 学习率调度器会从第`last_train_step + 1`步继续：

```python
if init_checkpoint_path is not None and load_lr_scheduler:
    load_lr_scheduler(lr_scheduler, ...)
```

学习率由当前step决定，遵循warmup -> decay曲线。

### Q4: 如何启用benchmark模式测试吞吐量？

**A**:
```bash
export NANOTRON_BENCHMARK=1
torchrun ... run_train.py --config-file config.yaml
```

训练会在第4步后自动退出，输出CSV格式的benchmark结果。

### Q5: 为什么训练时GPU利用率不稳定？

**A**: 可能原因：
1. DataLoader瓶颈 - 增大num_loading_workers
2. 检查点保存阻塞 - 检查存储系统I/O性能
3. 日志记录过于频繁 - 增大`iteration_step_info_interval`
4. PP Bubble - 使用更大batch或调整PP stage数

### Q6: 训练过程中如何动态切换数据阶段？

**A**: 配置多阶段数据：

```yaml
data_stages:
  - name: warmup
    start_training_step: 1
    data: {...}
  - name: main
    start_training_step: 5000  # 第5000步自动切换
    data: {...}
```

---

## 参考资料

1. **PyTorch Distributed**: https://pytorch.org/docs/stable/distributed.html
2. **Pipeline Parallelism (1F1B)**: https://arxiv.org/abs/2104.04473 (Megatron-LM)
3. **ZeRO**: https://arxiv.org/abs/1910.02054 (Zero Redundancy Optimizer)
4. **Learning Rate Scheduling**: https://huggingface.co/docs/transformers/perf_train_gpu_one#learning-rate

---

## 附录：核心训练循环伪代码

```python
# 完整的训练循环流程

trainer = DistributedTrainer(config_file)
dataloader = get_dataloader(trainer)

# 主训练循环
for step in range(initial_step, final_step + 1):
    # 1. 获取batch数据
    batch = next(dataloader)
    
    # 2. Pipeline前向/后向
    outputs = pipeline_engine.train_batch_iter(
        model, [batch, batch, ...]  # n_micro_batches
    )
    loss = sum([out["loss"] for out in outputs])
    
    # 3. 梯度同步
    if DDP:
        grad_accumulator.fp32_grads_allreduce_handle.wait()
    else:
        sync_gradients_across_dp(model, dp_pg)
    
    # 4. Tied weights同步
    sync_tied_weights_gradients(model, parallel_context)
    
    # 5. 梯度裁剪
    if grad_clip:
        clip_grad_norm(parameters, max_norm=1.0)
    
    # 6. 优化器步骤
    optimizer.step()
    optimizer.zero_grad()
    
    # 7. 学习率更新
    lr_scheduler.step()
    
    # 8. 日志记录
    if step % log_interval == 0:
        log_metrics(loss, lr, throughput, memory)
    
    # 9. 保存检查点
    if step % checkpoint_interval == 0:
        save_checkpoint(model, optimizer, lr_scheduler)
```
