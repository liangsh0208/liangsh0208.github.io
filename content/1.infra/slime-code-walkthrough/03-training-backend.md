---
title: "Slime 代码走读 — 03. 训练后端：MegatronTrainRayActor → train_actor → train_one_step"
date: 2026-06-15T12:00:00+08:00
draft: false
---

> 本文档走读 Megatron 训练后端的核心链路：`MegatronTrainRayActor` 生命周期 → `train_actor()` → `train()` → `train_one_step()` → `forward_backward_func()` → `loss_function()` 的完整调用链。

---

## 一、MegatronTrainRayActor 生命周期

```
RayTrainGroup.__init__()
  ├── TrainRayActor.remote(world_size, rank)      # 创建 actor
  └── actor.init.remote(args, role, ...)           # 初始化 Megatron
        ├── monkey_patch_torch_dist()            # 替换 torch.distributed
        ├── super().init()                        # NCCL + GLOO init_process_group
        ├── init(args)                           # Megatron initialize_distributed()
        ├── AutoConfig + AutoTokenizer           # 串行读取 HF config
        └── initialize_model_and_optimizer()     # get_model() + optimizer + checkpoint load
```

### 1.1 `monkey_patch_torch_dist()`

```python
from slime.utils.reloadable_process_group import (
    destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
)
```

**作用**：替换 `torch.distributed` 的进程组实现，支持 **进程组可销毁 + 重载**。

- 当 `--offload_train` 时，训练进程释放 GPU 显存，NCCL 进程组必须销毁
- `torch_memory_saver.pause()` 后，模型权重被挪到系统内存
- `torch_memory_saver.resume()` 后，进程组需要重新初始化
- monkey patch 让 `dist.init_process_group()` 可以被调用多次

### 1.2 `init(args)` — Megatron 初始化

```python
def init(args):
    from megatron.training.initialize import initialize_megatron
    initialize_megatron(...)
    
    # set random seed
    _set_random_seed(args.seed)
    
    # initialize distributed (TP/PP/CP/DP)
    _initialize_distributed(args)
    
    # model parallel manual seed for each rank
    tensor_parallel.model_parallel_cuda_manual_seed(seed)
```

### 1.3 `initialize_model_and_optimizer()`

```python
def initialize_model_and_optimizer(args, role="actor"):
    model = get_model(wrap_model_provider_with_freeze(...), ModelType.encoder_or_decoder)
    # Megatron DDP wrapper
    
    optimizer = get_megatron_optimizer(config=config, model_chunks=model)
    
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    # LR schedule: warmup → constant → decay
    
    iteration, _ = load_checkpoint(model, optimizer, opt_param_scheduler, ...)
    
    opt_param_scheduler.step(increment=iteration * args.global_batch_size)
    
    return model, optimizer, opt_param_scheduler, iteration
```

**训练参数计算**：
```python
args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
```

### 1.4 `TensorBackuper` — 多模型权重切换

```python
self.weights_backuper = TensorBackuper.create(
    source_getter=lambda: named_params_and_buffers(self.args, self.model),
    single_tag=None if args.enable_weights_backuper else "actor",
)

# 初始化时备份多个模型：
self.weights_backuper.backup("actor")      # 当前训练模型
self.weights_backuper.backup("ref")        # 参考模型（计算 KL）
self.weights_backuper.backup("old_actor")  # 上一轮的 actor（计算 importance ratio）
```

**模型切换**（GRPO 训练的核心）：
```python
def _switch_model(self, target_tag):
    self.weights_backuper.restore(target_tag)
    self._active_model_tag = target_tag

# 在 train 中切换：
self._switch_model("ref")       # 计算 ref_log_probs
self._switch_model("actor")     # 计算 current log_probs + 训练
```

---

## 二、`train()` 方法：`MegatronTrainRayActor.train()`

```python
def train(self, rollout_id, rollout_data_ref):
    if self.args.offload_train:
        self.wake_up()  # torch_memory_saver.resume()

    # ① Ray Object Store 取数据 + CPU→GPU
    rollout_data = self._get_rollout_data(rollout_data_ref)

    if self.role == "critic":
        return self.train_critic(rollout_id, rollout_data)
    else:
        return self.train_actor(rollout_id, rollout_data)
```

### 2.1 `_get_rollout_data()` — Ray→GPU 数据传输

```python
def _get_rollout_data(self, rollout_data_ref):
    rollout_data = process_rollout_data(
        self.args, rollout_data_ref,
        dp_rank, dp_size,
    )
    
    # CPU → GPU（提前 transfer，减少 train_wait）
    rollout_data["tokens"] = [torch.tensor(t, dtype=torch.long, device=cuda) for t in rollout_data["tokens"]]
    rollout_data["loss_masks"] = [torch.tensor(t, dtype=torch.int, device=cuda) for t in rollout_data["loss_masks"]]
    
    # slice rollout_log_probs with CP context
    rollout_data["rollout_log_probs"] = [
        torch.tensor(slice_log_prob_with_cp(log_prob, total_length, response_length, ...), device=cuda)
        for log_prob, total_length, response_length in zip(...)
    ]
    
    return rollout_data
```

### 2.2 `train_critic()` — Critic 训练流程

```python
def train_critic(self, rollout_id, rollout_data):
    # ① 创建 data iterator
    data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
    
    # ② forward_only 计算 values
    rollout_data.update(
        forward_only(get_values, self.args, self.model, data_iterator, num_microbatches)
    )
    
    # ③ 同步 actor 和 critic 数据（PP 广播）
    sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)
    
    # ④ 计算 advantages
    compute_advantages_and_returns(self.args, rollout_data)
    
    # ⑤ 训练 value head
    self.args.loss_type = "value_loss"
    train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler, data_iterator, num_microbatches)
```

### 2.3 `train_actor()` — Actor 训练流程（核心）

```python
def train_actor(self, rollout_id, rollout_data):
    data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

    with inverse_timer("train_wait"), timer("train"):
        
        # ========== Phase 1: Compute log_probs ==========
        if self.args.compute_advantages_and_returns:
            
            # 1.1 ref model log_probs
            if "ref" in self.weights_backuper.backup_tags:
                self._switch_model("ref")
                rollout_data.update(
                    self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")
                )  # key: "ref_log_probs"
            
            # 1.2 teacher log_probs (OPD)
            if "teacher" in self.weights_backuper.backup_tags:
                self._switch_model("teacher")
                rollout_data.update(
                    self.compute_log_prob(data_iterator, num_microbatches, store_prefix="teacher_")
                )  # key: "teacher_log_probs"
            
            # 1.3 current actor log_probs (作为 old_log_probs 用于 PPO 比率)
            self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
            if not self.args.use_rollout_logprobs:
                rollout_data.update(
                    self.compute_log_prob(data_iterator, num_microbatches, store_prefix="")
                )  # key: "log_probs"
            
            # 1.4 sync actor-critic data across ranks
            if self.args.use_critic:
                sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)
            
            # 1.5 compute advantages and returns
            self._switch_model("actor")
            compute_advantages_and_returns(self.args, rollout_data)

        # ========== Phase 2: Log rollout data ==========
        log_rollout_data(rollout_id, self.args, rollout_data)

        # ========== Phase 3: Train policy ==========
        with timer("actor_train"):
            train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler, data_iterator, num_microbatches)
        
        # ========== Phase 4: Post-processing ==========
        self.weights_backuper.backup("actor")  # 保存最新 actor 权重
        
        # 定期更新 ref model（可选）
        if self.args.ref_update_interval and (rollout_id + 1) % self.args.ref_update_interval == 0:
            self.weights_backuper.backup("ref")
        
        log_perf_data(rollout_id, self.args)
```

---

## 三、`compute_log_prob()` — 前向计算 log_probs

```python
def compute_log_prob(self, data_iterator, num_microbatches, store_prefix=""):
    with timer(f"{store_prefix}log_probs"):
        return forward_only(
            get_log_probs_and_entropy,   # loss function（只 forward，不求 backward）
            self.args,
            self.model,                   # 当前激活的模型（actor/ref/teacher）
            data_iterator,
            num_microbatches,
            store_prefix=store_prefix,
        )
```

---

## 四、`train()` — Megatron 训练循环：`slime/backends/megatron_utils/model.py`

```python
def train(rollout_id, model, optimizer, opt_param_scheduler, data_iterator, num_microbatches):
    for iterator in data_iterator:
        iterator.reset()  # 重置 microbatch 偏移
    
    # 训练模式
    for model_module in model:
        model_module.train()
    
    # Megatron pipeline parallelism 配置
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    
    # 梯度累积同步设置
    if args.overlap_grad_reduce:
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
    config.finalize_model_grads_func = finalize_model_grads
    
    # N 个 training steps（每 rollout）
    for step_id in range(len(num_microbatches)):
        loss_dict, grad_norm = train_one_step(
            args, rollout_id, step_id,
            data_iterator, model, optimizer, opt_param_scheduler,
            num_microbatches[step_id],
        )
        
        # logger.info 打印 train/* 指标
        if is_main_rank():
            accumulated_step_id = rollout_id * len(num_microbatches) + step_id
            log_dict = {
                f"train/{key}": val.mean().item() for key, val in loss_dict.items()
            }
            log_dict["train/grad_norm"] = grad_norm
            log_dict["train/lr-pg_0"] = opt_param_scheduler.get_lr(optimizer.param_groups[0])
            logger.info(f"step {accumulated_step_id}: {log_dict}")
```

---

## 五、`train_one_step()` — 单步训练（核心）

```python
def train_one_step(args, rollout_id, step_id, data_iterator, model, optimizer, opt_param_scheduler, num_microbatches):
    # ① Zero grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()
    
    # ② Forward + Backward（Megatron pipeline engine）
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,        # 见下文
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=False,
    )
    
    # ③ 梯度检查
    if args.check_for_nan_in_loss_and_grad:
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
    
    # ④ 优化器 step
    if valid_step:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        opt_param_scheduler.step(increment=args.global_batch_size)
    
    # ⑤ 释放 grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()
    
    # ⑥ 收集 loss
    if is_pipeline_last_stage():
        # Allreduce 到 DP group
        torch.distributed.all_reduce(values, group=mpu.get_data_parallel_group())
        
        loss_reduced = {}
        values = values.tolist()
        num_samples_or_tokens = values[0]  # num_tokens if per_token_loss else num_samples
        for key, value in zip(keys, values[1:]):
            loss_reduced[key] = value * cp_size / num_samples_or_tokens
        
        return loss_reduced, grad_norm
    return {}, grad_norm
```

---

## 六、`forward_step()` — 前向 + Loss 计算

```python
def forward_step(data_iterator, model, return_schedule_plan=False):
    # ① 获取 microbatch
    batch = get_batch(data_iterator, [
        "tokens", "log_probs", "ref_log_probs", "values",
        "advantages", "returns", "rollout_log_probs",
        "teacher_log_probs", "loss_masks", ...
    ], args.data_pad_size_multiplier, args.qkv_format, args.allgather_cp)
    
    # ② 模型前向
    output_tensor = model(
        input_ids=batch["tokens"],
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=batch["packed_seq_params"],
        loss_mask=batch["full_loss_masks"],
    )
    
    # ③ 返回 loss function（Megatron pipeline engine 会自动调用）
    return output_tensor, partial(loss_function, args, batch, num_microbatches)
```

**Megatron Pipeline Engine 的工作方式**：

1. `forward_backward_func` 是 Megatron 的 pipeline 引擎
2. 它调用 `forward_step()`, 得到 `output_tensor` 和 `loss_func`
3. 如果是 `forward_only=False`，pipeline engine 先执行所有 microbatch 的 forward，然后 backward
4. `loss_func` 被 pipeline engine 在合适的 stage 调用，计算 loss 并触发 backward

---

## 七、调用链完整总结

```
RayTrainGroup.async_train(rollout_id, rollout_data_ref)
  └── for actor in self._actor_handlers:
        actor.train.remote(rollout_id, rollout_data_ref)
            └── MegatronTrainRayActor.train(rollout_id, rollout_data_ref)
                ├── if offload_train: wake_up() [resume from CPU]
                ├── _get_rollout_data()
                │     ├── process_rollout_data() [Ray→CPU]
                │     ├── tokens/long_tensor() [CPU→GPU]
                │     └── slice_log_prob_with_cp() [CP 切回]
                └── if role == "actor": train_actor()
                      ├── get_data_iterator() [slime/backends/megatron_utils/data.py]
                      ├── compute_log_prob() [forward_only → get_log_probs_and_entropy]
                      │     └── forward_only()
                      │           ├── forward_step() → get_batch() → model()
                      │           └── get_log_probs_and_entropy() → calculate_log_probs_and_entropy()
                      ├── forward_only(get_values) [critic]
                      ├── sync_actor_critic_data() [PP broadcast]
                      ├── compute_advantages_and_returns() [GRPO/GAE]
                      ├── log_rollout_data() [聚合 rollout 指标]
                      └── train() [Megatron 训练循环]
                            ├── for step_id in range(num_steps_per_rollout):
                            │     train_one_step()
                            │         ├── zero_grad
                            │         ├── forward_backward_func()
                            │         │     ├── forward_step() → get_batch() → model()
                            │         │     └── loss_function() → policy_loss_function()
                            │         ├── optimizer.step()
                            │         ├── opt_param_scheduler.step()
                            │         └── log dict [train/* 指标]
                            └── logger.info(...)
```

---

## 八、关键数据流图解

```
[Ray Actor: RolloutManager]
  └── rollout_data_ref = Box(ray.put({
        "tokens": [tensor, tensor, ...],          # 来自 data_source
        "rewards": [1.0, 0.0, ...],
        "loss_masks": [mask, mask, ...],
        "rollout_log_probs": [tensor, tensor, ...],
    }))
         │
         │ ray.get(rollout_data_ref)
         ▼
[Ray Actor: MegatronTrainRayActor rank=0]
  └── _get_rollout_data(rollout_data_ref)
        │
        ├── process_rollout_data()    # Ray Object Store → CPU dict
        ├── CPU → GPU transfer       # 每个 tensor 调用 .to(device)
        └── slice_log_prob_with_cp() # CP rank 恢复 sequence
        │
        ▼ rollout_data (GPU memory)
        ┌──────────────────────────────────────────────────────┐
        │  key            │ value（per sample list）            │
        ├──────────────────────────────────────────────────────┤
        │  tokens         │ [torch.LongTensor]                  │
        │  unconcat_tokens│ [torch.LongTensor] （CP 前完整 seq） │
        │  full_loss_masks│ [torch.LongTensor]                  │
        │  packed_seq_params│ PackedSeqParams (cu_seqlens)      │
        │  log_probs      │ [torch.FloatTensor] (old policy)     │
        │  ref_log_probs  │ [torch.FloatTensor] (ref model)     │
        │  rollout_log_probs│ [torch.FloatTensor] (sglang logp) │
        │  advantages     │ [torch.FloatTensor] (GRPO=returns) │
        │  returns        │ [torch.FloatTensor]                 │
        │  values         │ [torch.FloatTensor] (critic)        │
        │  rewards        │ [float]                              │
        │  response_lengths│ [int]                               │
        │  total_lengths  │ [int]                                │
        └──────────────────────────────────────────────────────┘
        │
        ▼ get_data_iterator(self.args, self.model, rollout_data)
        │
        DataIterator (microbatch 迭代器)
          ├── rollout_data (共享引用)
          ├── micro_batch_size: int
          ├── micro_batch_indices: list[list[int]] (dynamic batch 的索引)
          └── offset: int
```
