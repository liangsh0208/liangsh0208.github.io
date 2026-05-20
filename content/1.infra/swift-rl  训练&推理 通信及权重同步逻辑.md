---
created: 2026-05-19
tags:
  - infra
  - RL算法
---

以下是 `ms-swift` 仓库中 RL（GRPO/PPO/DPO/GKD 等）训练权重与推理权重的更新逻辑分析。核心代码集中在 `swift/rlhf_trainers/` 目录下。

---

## 1. RL 训练代码入口

`swift` 中各 RL 算法的 Trainer 统一由 `trainer_factory.py` 映射，核心文件如下：

|算法|文件|类名|
|---|---|---|
|GRPO|`swift/rlhf_trainers/grpo_trainer.py`|`GRPOTrainer`|
|PPO|`swift/rlhf_trainers/ppo_trainer.py`|`PPOTrainer`|
|DPO|`swift/rlhf_trainers/dpo_trainer.py`|`DPOTrainer`|
|GKD|`swift/rlhf_trainers/gkd_trainer.py`|`GKDTrainer`|
|公共 Mixin|`swift/rlhf_trainers/rlhf_mixin.py`|`RLHFTrainerMixin`|
|Rollout Mixin|`swift/rlhf_trainers/rollout_mixin.py`|`RolloutTrainerMixin`|

---

## 2. 训练权重（Actor/Policy）的更新逻辑

### 2.1 基于 `transformers.Trainer` 的标准路径

ms-swift 中绝大多数 RL Trainer **没有重写** `optimizer.step()` 和 `loss.backward()`，而是依赖 `transformers.Trainer` 的 `_inner_training_loop` 完成参数更新。各子类仅在 `training_step` 中做薄封装：

**[swift/rlhf_trainers/grpo_trainer.py:1847-1852](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/grpo_trainer.py#L1847)**

```python
def training_step(self, model: nn.Module, inputs: DataType, num_items_in_batch=None) -> torch.Tensor:
    if self.args.async_generate:
        # Wait for the eval rollout to complete
        while not self.is_async_generate_eval_rollout_done():
            time.sleep(0.1)
    return super().training_step(model, inputs, num_items_in_batch)
```

**[swift/rlhf_trainers/dpo_trainer.py:460-462](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/dpo_trainer.py#L460)**

```python
def training_step(self, model, inputs, *args, **kwargs):
    with self.template.forward_context(self.model, inputs):
        return super().training_step(model, inputs, *args, **kwargs)
```

### 2.2 优化器的创建

优化器由 `swift` 层接管，通过 `optimizer_callback` 创建：

**[swift/trainers/mixin.py:992-1002](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/trainers/mixin.py#L992)**

```python
def create_optimizer_and_scheduler(self, num_training_steps: int):
    self.optimizer_callback.create_optimizer_and_scheduler(num_training_steps)

def create_optimizer(self, model=None):
    self.optimizer = self.optimizer_callback.create_optimizer(model=model)
    ...
```

### 2.3 基于 Megatron 的显式训练循环

如果使用 Megatron 训练（如 Megatron-GRPO），则有独立的循环和显式的 `optimizer.step()`：

**[swift/megatron/trainers/base.py:877](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/megatron/trainers/base.py#L877)**

```python
update_successful, grad_norm, _ = self.optimizer.step()
```

---

## 3. 推理/参考权重（Ref Model）的加载与更新

### 3.1 Ref Model 的初始化

`RLHFTrainerMixin` 接收 `ref_model` 参数，并根据并行策略（DeepSpeed / FSDP / 普通）进行包装：

**[swift/rlhf_trainers/rlhf_mixin.py:21-56](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/rlhf_mixin.py#L21)**

```python
def __init__(self, model=None, ref_model=None, *_args, **kwargs):
    self.ref_model = ref_model
    ...
    if ref_model is not None:
        if self.is_deepspeed_enabled:
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        elif self.is_fsdp_enabled:
            self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
```

### 3.2 Ref Model 与训练模型的权重软同步（Sync）

在 GRPO 等算法中，支持**周期性将当前 policy（训练模型）的权重按比例混合进 ref model**：

**触发条件**（`SyncRefModelCallback` 回调）： **[swift/rlhf_trainers/rollout_mixin.py:69-79](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/rollout_mixin.py#L69)**

```python
class SyncRefModelCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer.ref_model is None:
            return
        if state.global_step % args.ref_model_sync_steps != 0:
            return
        self.trainer._sync_ref_model_weights(args.ref_model_mixup_alpha)
```

**核心混合逻辑**： **[swift/rlhf_trainers/rollout_mixin.py:782-820](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/rollout_mixin.py#L782)**

```python
@torch.no_grad()
def _sync_ref_model_weights(self, alpha: float) -> None:
    policy = self.accelerator.unwrap_model(self.model)
    ref = self.accelerator.unwrap_model(self.ref_model)
    ...
    def _mix_inplace(ref_p: torch.Tensor, pol_p: torch.Tensor) -> None:
        ref_p.data.mul_(1.0 - alpha).add_(pol_p.data, alpha=alpha)
    ...
```

混合公式为：

```text
ref = (1 - alpha) * ref + alpha * policy
```

该回调在 GRPO 中注册： **[swift/rlhf_trainers/grpo_trainer.py:143-144](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/grpo_trainer.py#L143)**

```python
if args.sync_ref_model:
    self.add_callback(SyncRefModelCallback(self))
```

### 3.3 DPO 中 Ref Model 的“回退”逻辑

DPO 在不传入独立 `ref_model` 时（如 LoRA 场景），会通过 `disable_adapter()` 在当前模型上计算参考 log prob：

**[swift/rlhf_trainers/dpo_trainer.py:213-222](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/dpo_trainer.py#L213)**

```python
def compute_ref_log_probs(self, batch):
    if self.ref_model is None:
        with self.null_ref_context():
            ref_model_output = self.concatenated_forward(self.model, batch, is_ref_model=True)
    else:
        ref_model_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
```

---

## 4. 训练权重 -> vLLM 推理引擎的同步

在 GRPO/GKD 使用 vLLM 做异步 rollout 时，**训练好的 policy 权重需要显式同步到 vLLM engine**，这是“训练权重 -> 推理权重”的关键链路：

**入口**： **[swift/rlhf_trainers/rollout_mixin.py:439-458](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/rollout_mixin.py#L439)**

```python
def _move_model_to_vllm(self, skip_async_check=False):
    """Synchronize model weights to vLLM engine"""
    ...
    if tuner_type == 'full' or (not self.base_sync_done or args.sleep_level == 2) or not self.rollout_enable_lora:
        self._move_full_model_to_vllm()
    else:
        self._move_adapter_to_vllm()
```

### 4.1 全量参数同步

**[swift/rlhf_trainers/rollout_mixin.py:740-781](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/rollout_mixin.py#L740)**

```python
def _move_full_model_to_vllm(self):
    with gather_if_zero3(parameters):
        if should_merge:
            with patch_lora_merge(self.model, parameter_group):
                self.model.merge_adapter()
        try:
            state_dict = self._collect_state_dict_for_vllm(parameter_group, parameter_group_no_lora)
            self._load_state_dict_to_vllm(state_dict)
        finally:
            if should_merge:
                with patch_lora_unmerge(self.model):
                    self.model.unmerge_adapter()
```

### 4.2 LoRA Adapter 同步

**[swift/rlhf_trainers/rollout_mixin.py:460-520](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/rollout_mixin.py#L460)**

```python
def _move_adapter_to_vllm(self):
    """Transfer LoRA adapter weights to vLLM engine"""
    lora_params = OrderedDict()
    ...
    if self.vllm_mode == 'server' and self.accelerator.is_main_process:
        self.vllm_client.update_adapter_param(peft_config, lora_params)
    elif self.vllm_mode == 'colocate':
        self.engine.engine.add_lora(lora_request)
```

### 4.3 PEFT 前缀处理

在同步过程中会去掉 `base_model.model.` 前缀以适配 vLLM：

**[swift/rlhf_trainers/rollout_mixin.py:380-381](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/rollout_mixin.py#L380)**

```python
if name.startswith('base_model'):
    name = name.replace('base_model.', '')
```

---

## 5. 角色关系总览

|角色|代码体现|说明|
|---|---|---|
|**Actor / Policy**|`self.model`|真正被优化器更新的训练模型|
|**Ref Model**|`self.ref_model`|参考模型，可通过 `_sync_ref_model_weights` 软同步 policy 权重|
|**Base Model**|参数名前缀 `base_model.model.`|PEFT 场景下的底层模型|
|**Teacher Model**|`self.teacher_model`|仅在 GKD 中使用|
|**vLLM Engine**|`self.engine` / `self.vllm_client`|外部推理引擎，通过 `_move_model_to_vllm()` 接收 actor 权重|

---

## 6. 总结

- **训练权重更新**：基于 `transformers.Trainer` 的标准循环（`loss.backward()` + `optimizer.step()`），优化器由 `swift` 层的 `optimizer_callback` 创建和调度。
- **Ref Model 更新**：初始化时独立加载（或复用当前模型）；在 GRPO 中若开启 `sync_ref_model`，则通过 `_sync_ref_model_weights()` 周期性地将 policy 权重按 `alpha` 比例混合进 ref model，实现软同步。
- **推理权重同步**：在 vLLM rollout 场景下，通过 `_move_model_to_vllm()` 将训练模型的**全量参数**或**LoRA adapter**显式同步到 vLLM 推理引擎，完成训练到推理的权重更新。






# 权重同步到vll 逻辑流程

以下是权重同步到 vLLM 以及 vLLM 加载权重的**完整代码链路**详解。

---

## 一、触发入口：`training_step` 后会调用 `_fast_infer`

在 GRPO 训练流程中，每次变种训练后需要生成 rollout 时，会先检查是否需要同步权重：

**[rollout_mixin.py:928-930](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/rollout_mixin.py#L928)**

```python
if self.state.global_step != self._last_loaded_step or args.sleep_level == 2:
    self._move_model_to_vllm()
    self._last_loaded_step = self.state.global_step
```

只有当 `global_step` 变化时（即真正更新完权重后），才会触发同步，避免 grad accumulation 中间步骤的无意义加载。

---

## 二、主调度函数：`_move_model_to_vllm`

**[rollout_mixin.py:376-394](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/rollout_mixin.py#L376)**

```python
def _move_model_to_vllm(self, skip_async_check=False):
    tuner_type = args.tuner_type
    if tuner_type == 'full' or (not self.base_sync_done or args.sleep_level == 2) or not self.rollout_enable_lora:
        self._move_full_model_to_vllm()
    else:
        self._move_adapter_to_vllm()
```

这里决定走两条路：

- **Full 训练 / 首次同步 / sleep_level=2**：走全量参数同步 `_move_full_model_to_vllm`
- **LoRA 训练且 `vllm_enable_lora=True`**：走 Adapter 同步 `_move_adapter_to_vllm`

---

## 三、链路 A：全量模型权重同步

### 3.1 `_move_full_model_to_vllm` —— 管理 gather + merge 生命周期

**[rollout_mixin.py:652-692](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/rollout_mixin.py#L652)**

```python
def _move_full_model_to_vllm(self):
    is_peft = is_peft_model(self.model)
    should_merge = is_peft and not self._is_fsdp2 and not self.rollout_enable_lora
    
    gather_if_zero3 = get_gather_if_zero3_context(self)
    
    for i, parameter_group in enumerate(self.parameter_groups):
        parameter_group_no_lora = self.parameter_groups_no_lora[i]
        
        if not self._is_fsdp2:
            parameters = [
                parameter for name, parameter in self.model.named_parameters()
                if not parameter_group or name in parameter_group
            ]
        
        with gather_if_zero3(parameters):
            if should_merge:
                with patch_lora_merge(self.model, parameter_group):
                    self.model.merge_adapter()
            try:
                state_dict = self._collect_state_dict_for_vllm(parameter_group, parameter_group_no_lora)
                self._load_state_dict_to_vllm(state_dict)
            finally:
                if should_merge:
                    with patch_lora_unmerge(self.model):
                        self.model.unmerge_adapter()
```

**关键细节：**

- **分 batch 同步**：按 `parameter_groups`（在 `split_batches()` 中按 layer 切分）逐个 batch 处理，避免一次性 OOM
- **DeepSpeed Zero3 gather**：`gather_if_zero3` 把分散在多个 rank 上的参数 gather 到完整张量
- **LoRA merge**：如果是 LoRA 训练且 vLLM 不支持 LoRA，会先 `merge_adapter()` 把 LoRA 权重合进 base，传完后再 `unmerge_adapter()` 恢复
- **`finally`** 保证了即使 `_load_state_dict_to_vllm` 报错，也会恢复模型状态

### 3.2 `_collect_state_dict_for_vllm` —— 收集并清理参数名

**[rollout_mixin.py:598-650](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/rollout_mixin.py#L598)**

```python
def _collect_state_dict_for_vllm(self, parameter_group=None, parameter_group_no_lora=None):
    is_peft = is_peft_model(self.model)
    should_merge_lora = self._is_fsdp2 and is_peft and not self.rollout_enable_lora
    
    raw_state_dict = {}
    if self._is_fsdp2:
        # FSDP2: 必须用 state_dict()，named_parameters 返回的是 sharded 值
        for name, param in self.model.state_dict().items():
            if parameter_group and name not in parameter_group:
                continue
            if hasattr(param, 'full_tensor'):
                param = param.full_tensor()  # 从 DTensor 转成完整 Tensor
            raw_state_dict[name] = param
    else:
        # DeepSpeed: 用 named_parameters + param.data
        for name, param in self.model.named_parameters():
            if parameter_group and name not in parameter_group:
                continue
            raw_state_dict[name] = param.data
    
    # 清理 base_model.model. 前缀、去掉 .base_layer 后缀、过滤 adapter 参数
    state_dict = self._process_state_dict_for_vllm(raw_state_dict, is_peft, keep_lora_weights=should_merge_lora)
    
    # FSDP2 + LoRA: 在 tensor 层面手动合并 LoRA（避免 merge/unmerge 在 DTensor 上失效）
    if should_merge_lora:
        state_dict = self._merge_lora_into_state_dict(state_dict)
    
    # 用 parameter_group_no_lora 过滤出实际需要传给 vLLM 的参数
    if parameter_group_no_lora:
        state_dict = {k: v for k, v in state_dict.items() if k in parameter_group_no_lora}
    
    return state_dict
```

### 3.3 `_process_state_dict_for_vllm` —— 参数名清洗 + DTensor 转换

**[rollout_mixin.py:484-527](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/rollout_mixin.py#L484)**

```python
def _process_state_dict_for_vllm(self, state_dict, is_peft, keep_lora_weights=False):
    processed = {}
    for name, param in state_dict.items():
        clean_name = name.removeprefix('base_model.model.')
        if not self.rollout_enable_lora:
            clean_name = clean_name.replace('.base_layer', '')
        if is_peft and not keep_lora_weights and self.model.prefix in clean_name:
            continue  # 跳过 LoRA adapter 参数（除非需要保留用于后面的 tensor merge）
        if 'original_module' in clean_name:
            continue
        clean_name = self._fix_param_name_to_vllm(clean_name)
        clean_name = clean_name.replace('modules_to_save.default.', '')
        if hasattr(param, 'full_tensor'):  # FSDP2 DTensor -> Tensor
            param = param.full_tensor()
        processed[clean_name] = param
    return processed
```

### 3.4 `_load_state_dict_to_vllm` —— 最终加载到 vLLM Engine

**[rollout_mixin.py:456-475](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/rollout_mixin.py#L456)**

```python
def _load_state_dict_to_vllm(self, state_dict):
    if self.vllm_mode == 'server' and self.accelerator.is_main_process:
        use_flatten = getattr(self.args, 'enable_flattened_weight_sync', True)
        if use_flatten:
            bucket_size_mb = int(os.environ.get('SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE', 512))
            named_params = list(state_dict.items())
            parameter_buckets = _create_parameter_buckets(named_params, bucket_size_mb=bucket_size_mb)
            for bucket in parameter_buckets:
                _process_bucket_with_flattened_tensor(self, bucket)
        else:
            for name, param in state_dict.items():
                self.vllm_client.update_named_param(name, param)
    elif self.vllm_mode == 'colocate':
        llm_model = self.engine.inner_model
        patch_vllm_moe_model_weight_loader(llm_model)
        llm_model.load_weights(state_dict.items())
```

这里区分了两种模式：

#### Colocate 模式（本地 vLLM Engine）

**[rollout_mixin.py:471-474](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/rollout_mixin.py#L471)**

```python
llm_model = self.engine.inner_model
patch_vllm_moe_model_weight_loader(llm_model)
llm_model.load_weights(state_dict.items())
```

这就是你当前选中行 `engine` 所在的代码块。实际调用的是 vLLM 原生的 `llm_model.load_weights()` 来加载权重。

#### Server 模式（远端 vLLM Server）

- **flattened sync（默认）**：将参数分成多个 bucket，每个 bucket flatten 成一个长张量，通过 `_process_bucket_with_flattened_tensor` 发送给 vLLM client
- **非 flatten**：逐个 `update_named_param(name, param)` 发送

---

## 四、链路 B：LoRA Adapter 权重同步

### 4.1 `_move_adapter_to_vllm`

**[rollout_mixin.py:396-454](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/rollout_mixin.py#L396)**

```python
def _move_adapter_to_vllm(self):
    lora_params = OrderedDict()
    
    for i, parameter_group in enumerate(self.parameter_groups):
        with gather_if_zero3(parameters), patch_lora_merge(self.model, parameter_group):
            if not self._is_fsdp2:
                self.model.merge_adapter()
            cur_lora_params = get_peft_model_state_dict(self.model, state_dict)
            cur_lora_params = {
                name: param.full_tensor().detach() if hasattr(param, 'full_tensor') else param.detach()
                for name, param in cur_lora_params.items()
            }
            lora_params.update(cur_lora_params)
            if not self._is_fsdp2:
                with patch_lora_unmerge(self.model):
                    self.model.unmerge_adapter()
    
    if self.vllm_mode == 'server' and self.accelerator.is_main_process:
        # Flattened 传输或使用 update_adapter_param
        if use_flatten:
            self.vllm_client.update_adapter_flattened_param(peft_config, metadatas, flattened_tensor)
        else:
            self.vllm_client.update_adapter_param(peft_config, lora_params)
    elif self.vllm_mode == 'colocate':
        lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = TensorLoRARequest(
            lora_name=f'{lora_int_id}',
            lora_int_id=lora_int_id,
            lora_path='dummy_lora_path',  # 实际路径在本地 colocate 模式下不需要
            peft_config=asdict(peft_config),
            lora_tensors=lora_params,
        )
        self.engine.engine.add_lora(lora_request)
```

**关键细节：**

- 调用 `get_peft_model_state_dict()` 只收集 LoRA A/B 矩阵的权重
- 同样分 batch 处理 + gather_if_zero3
- Colocate 模式下通过 `self.engine.engine.add_lora(lora_request)` 动态注册 adapter
- Flattened 模式下类似全量参数，先把所有 LoRA weight flatten 再传

---

## 五、FSDP2 下的特殊 LoRA 合并

当使用 FSDP2 + LoRA + 非 vLLM LoRA 模式时，`merge_adapter()` 在 DTensor 上可能出错，所以 ms-swift 选择在**手动的 tensor 层面合并**：

### 5.1 `_merge_lora_into_state_dict`

**[rollout_mixin.py:529-596](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/rollout_mixin.py#L529)**

```python
def _merge_lora_into_state_dict(self, state_dict):
    merged = {}
    lora_keys = set()
    
    for name, module in self.model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base_name = name.removeprefix('base_model.model.')
        weight_key = f'{base_name}.weight'
        
        # 从 state_dict 中取出 LoRA A/B 权重
        lora_a_key = f'{base_name}.lora_A.{active_adapter}.weight'
        lora_b_key = f'{base_name}.lora_B.{active_adapter}.weight'
        
        lora_A = state_dict[lora_a_key]
        lora_B = state_dict[lora_b_key]
        scaling = module.scaling[active_adapter]
        
        # 计算 LoRA delta: delta = lora_B @ lora_A * scaling
        delta_weight = (lora_B @ lora_A) * scaling
        merged[weight_key] = base_weight + delta_weight.to(base_weight.dtype)
        lora_keys.add(lora_a_key)
        lora_keys.add(lora_b_key)
    
    # 丢弃 LoRA adapter 参数，只保留合并后的 base 权重
    for key, value in state_dict.items():
        if key in lora_keys or self.model.prefix in key:
            continue
        merged[key] = value
    
    return merged
```

---

## 六、完整调用链路总结

```
training_step 训练完成
    │
    ▼
_fast_infer() ──► global_step != last_loaded_step ?
    │
    ▼
_move_model_to_vllm()
    │
    ├── Full / 无 LoRA ──► _move_full_model_to_vllm()
    │                        │
    │                        ├── split_batches() 切分参数组
    │                        │
    │                        ├── gather_if_zero3() gather DeepSpeed Zero3 参数
    │                        │
    │                        ├── patch_lora_merge + merge_adapter()（如需 merge）
    │                        │
    │                        ├── _collect_state_dict_for_vllm()
    │                        │      ├── model.state_dict() / named_parameters()
    │                        │      ├── _process_state_dict_for_vllm() 清理前缀
    │                        │      └── _merge_lora_into_state_dict()（FSDP2 场景）
    │                        │
    │                        ├── _load_state_dict_to_vllm()
    │                        │      ├── Colocate: self.engine.inner_model.load_weights()
    │                        │      └── Server: self.vllm_client.update_named_param()
    │                        │
    │                        └── patch_lora_unmerge + unmerge_adapter()（恢复）
    │
    └── LoRA + vLLM LoRA ──► _move_adapter_to_vllm()
                               │
                               ├── get_peft_model_state_dict() 收集 LoRA A/B
                               │
                               ├── _load_state_dict_to_vllm() 的 adapter 版本
                               │      ├── Colocate: self.engine.engine.add_lora()
                               │      └── Server: self.vllm_client.update_adapter_param()
                               │
                               └── _load_state_dict_to_vllm() 加载到 vLLM
```

---

## 七、vLLM 加载权重的关键函数点

|模式|加载目标|实际 vLLM API|
|---|---|---|
|Colocate Full|`self.engine.inner_model`|`llm_model.load_weights(state_dict.items())`|
|Colocate LoRA|`self.engine.engine`|`engine.add_lora(lora_request)`|
|Server Full|`self.vllm_client`|`update_named_param()` / `_process_bucket_with_flattened_tensor()`|
|Server LoRA|`self.vllm_client`|`update_adapter_param()` / `update_adapter_flattened_param()`|

---

## 八、一张图理解 Colocate 模式下的核心流程

```
Training Model (actor)              vLLM Engine (推理)
       │                                  │
       │ _collect_state_dict_for_vllm()  │
       │ ──► 收集 + 清洗参数名             │
       │                                  │
       │ _load_state_dict_to_vllm()      │
       │ ──► llm_model.load_weights() ─►│ 替换 vLLM 内部模型权重
       │                                  │
       │ _rollout() ─────────────────────►│ 用新权重生成 rollout
```

如果你还想去看看 vLLM 侧 `load_weights` 到底做了什么（比如参数名映射、weight loader 等），vLLM 源码一般在 `vllm/model_executor/models/` 下对应的模型实现里，每个模型有自己继承的 `_load_weights_impl` 或者 weight loader dict。ms-swift 这里还通过 `patch_vllm_moe_model_weight_loader()` 对 MoE 模型的 weight loader 做了补丁兼容。






# vllm 独立推理任务是如何与训练任务链接通信的


在 `server` 模式下，训练任务与独立的 vLLM rollout 推理服务之间通过 **`VLLMClient`** 进行交互。整个链路分为**连接建立**、**控制面通信（HTTP）**、**数据面通信（NCCL broadcast）**三个层次。

---

## 1. 整体架构

```
独立的 vLLM Server 进程(s)          训练进程 (Training)
       │                                    │
       │  HTTP REST API (/health, /infer)   │  控制面：推理请求、参数配置
       │◄─────────────────────────────────►│
       │                                    │
       │  PyNcclCommunicator (NCCL)         │  数据面：权重张量广播
       │◄──────── broadcast ───────────────►│
```

---

## 2. VLLMClient 的创建与连接建立

### 2.1 训练参数解析时创建 Client

**[swift/arguments/rlhf_args.py:434-448](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/arguments/rlhf_args.py#L434)**

```python
def _init_external_vllm(self):
    from swift.rlhf_trainers import VLLMClient
    self.vllm_client = VLLMClient(
        base_urls=self.vllm_server_base_url,
        hosts=self.vllm_server_host,
        server_ports=self.vllm_server_port,
        group_ports=self.vllm_server_group_port,
        connection_timeout=self.vllm_server_timeout)
    self.vllm_client.close_communicator()
    self.vllm_client.init_communicator(device=get_current_device())
```

训练启动时，只要配置了 `vllm_server_host` 或 `vllm_server_base_url`，就会实例化 `VLLMClient`。

### 2.2 检查 Server 是否就绪（HTTP 轮询）

**[swift/rlhf_trainers/vllm_client.py:82-115](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/vllm_client.py#L82)**

```python
def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
    def check_single_server(i):
        start_time = time.time()
        url = f'{self.base_urls[i]}/health/'
        while True:
            try:
                response = requests.get(url, timeout=retry_interval)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            if time.time() - start_time >= total_timeout:
                return
            time.sleep(retry_interval)
```

`VLLMClient` 初始化时会向每个 vLLM Server 的 `/health/` 端点轮询发送 HTTP GET，直到确认服务可用或超时。

### 2.3 初始化 NCCL 通信组（权重广播通道）

**[swift/rlhf_trainers/vllm_client.py:182-209](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/vllm_client.py#L182)**

```python
def init_communicator(self, device=0):
    for i in range(self.num_servers):
        # 1. 查询 vLLM server 的 world_size（TP group 大小）
        response = self.sessions[i].get(f'{self.base_urls[i]}/get_world_size/')
        vllm_world_size = response.json()['world_size']
        
        world_size = vllm_world_size + 1   # +1 把训练进程纳入 group
        rank = vllm_world_size              # 训练进程在 group 中 rank 最大
        
        # 2. 通知 server 创建 communicator（server 在 group 中 rank=0）
        self.sessions[i].post(
            f'{self.base_urls[i]}/init_communicator/',
            json={'host': bind_host, 'port': self.group_ports[i], 'world_size': world_size})
        
        # 3. 训练侧也创建无状态进程组 + PyNcclCommunicator
        pg = StatelessProcessGroup.create(
            host=self.hosts[i], port=self.group_ports[i], rank=rank, world_size=world_size)
        comm = PyNcclCommunicator(pg, device=device)
        self.pynccl_comms.append(comm)
```

这里建立了**数据面**：

- 训练进程作为 **client**，但在 NCCL group 中扮演 **rank = vllm_world_size** 的角色（即最后一个 rank）
- vLLM server 内的 worker 进程作为 **rank 0 ~ world_size-1**
- 后续权重同步时，训练进程作为 **src（广播源）**，通过 `comm.broadcast()` 把张量推送给 vLLM server

---

## 3. VLLMClient 如何传给 Trainer

**[swift/pipelines/train/rlhf.py:227-228](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/pipelines/train/rlhf.py#L227)**

```python
def _get_trainer_kwargs(self):
    if self.args.rlhf_type in ['grpo', 'gkd']:
        trainer_kwargs['vllm_client'] = self.args.vllm_client
```

在 `GRPOTrainer` / `GKDTrainer` 的初始化中：

**[swift/rlhf_trainers/grpo_trainer.py:104](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/grpo_trainer.py#L104)**

```python
self.vllm_client = kwargs.pop('vllm_client', None)
```

于是 `RolloutTrainerMixin` 中可以通过 `self.vllm_client` 访问。

---

## 4. 控制面通信：推理请求（HTTP POST）

当 Trainer 需要执行 rollout 生成时，调用 `_server_rollout()`：

**[swift/rlhf_trainers/rollout_mixin.py:995-1045](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/rollout_mixin.py#L995)**

```python
def _server_rollout(self, inputs, request_config, is_global_inputs):
    # 把请求 gather 到主进程
    if self.accelerator.is_main_process:
        all_outputs = self._engine_infer(infer_requests=all_requests, request_config=request_config)
```

**[swift/rlhf_trainers/rollout_mixin.py:1078-1080](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/rollout_mixin.py#L1078)**

```python
def _engine_infer(self, infer_requests, request_config):
    if self.vllm_mode == 'server':
        res = self.vllm_client.infer([asdict(req) for req in infer_requests],
                                      asdict(request_config), use_tqdm=use_tqdm)
```

**实际发送 HTTP 请求**（`vllm_client.py`）：

**[swift/rlhf_trainers/vllm_client.py:117-180](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/vllm_client.py#L117)**

```python
def infer(self, infer_requests, request_config=None, ...):
    # 把请求按 server 数量切分，每个 server 处理一部分
    n = len(infer_requests)
    chunk_size = (n + self.num_servers - 1) // self.num_servers
    chunks = [infer_requests[i:i + chunk_size] for i in range(0, n, chunk_size)]
    
    def process_chunk(i, chunk):
        response = self.sessions[i].post(
            f'{self.base_urls[i]}/infer/',
            json={
                'infer_requests': chunk,
                'request_config': request_config,
                ...
            })
        ...
    
    # 多线程并发请求所有 servers
    with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
        futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
```

这里 `sessions` 是 `requests.Session` 池，通过 HTTP POST 到 `/infer/` 端点发送生成请求。

---

## 5. 数据面通信：权重同步（NCCL Broadcast）

### 5.1 更新单个命名参数

**[swift/rlhf_trainers/vllm_client.py:213-250](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/vllm_client.py#L213)**

```python
def update_named_param(self, name: str, weights: torch.Tensor):
    # 1. 通过 HTTP 通知 server：准备接收哪个参数，dtype/shape 是什么
    response = self.sessions[i].post(
        f'{self.base_urls[i]}/update_named_param/',
        json={'name': name, 'dtype': str(weights.dtype), 'shape': tuple(weights.shape)})
    
    # 2. 通过 PyNcclCommunicator 广播实际张量数据
    synchronize()
    self.pynccl_comms[i].broadcast(
        weights,
        src=self.pynccl_comms[i].rank,  # 训练进程是 src
        stream=getattr(get_torch_device(), 'current_stream', lambda: None)())
    synchronize()
    self.pynccl_comms[i].group.barrier()
```

### 5.2 更新 Flattened 参数（batch 同步）

**[swift/rlhf_trainers/vllm_client.py:366-407](vscode-webview://07v4i72muef2f5knnok30o1cohipc97rsv31rjtlng5bm9mj1209/swift/rlhf_trainers/vllm_client.py#L366)**

```python
def update_flattened_params(self, metadatas, flattened_tensor):
    # 先发 metadata（HTTP JSON）
    response = self.sessions[i].post(
        f'{self.base_urls[i]}/update_flattened_params/',
        json={'metadatas': metadatas})
    
    # 再广播 flatten 后的长张量（NCCL）
    self.pynccl_comms[i].broadcast(flattened_tensor, src=...)
```

---

## 6. 交互通信流程总结

|环节|协议/方式|关键代码|
|---|---|---|
|**发现与就绪检查**|HTTP GET `/health/`|`VLLMClient.check_server()`|
|**建立数据通道**|HTTP POST `/init_communicator/` + NCCL `StatelessProcessGroup`|`VLLMClient.init_communicator()`|
|**推理请求**|HTTP POST `/infer/`|`VLLMClient.infer()`|
|**同步单个参数**|HTTP POST `/update_named_param/` + NCCL `broadcast`|`VLLMClient.update_named_param()`|
|**同步批量参数**|HTTP POST `/update_flattened_params/` + NCCL `broadcast`|`VLLMClient.update_flattened_params()`|
|**同步 LoRA Adapter**|HTTP POST `/update_adapter_param/` + NCCL `broadcast`|`VLLMClient.update_adapter_param()`|
|**重置 KV Cache**|HTTP POST `/reset_prefix_cache/`|`VLLMClient.reset_prefix_cache()`|
|**查询引擎类型**|HTTP POST `/get_engine_type/`|`VLLMClient.get_engine_type()`|

---

## 7. 一句话概括

训练任务通过 **`VLLMClient`** 与独立 vLLM Server 交互：

- **控制面**走 **HTTP REST API**（`requests.Session`），负责推理请求下发、参数元数据通知、服务状态检查；
- **数据面**走 **NCCL broadcast**（`PyNcclCommunicator`），负责把训练好的**大权重张量**以零拷贝方式高速广播到 vLLM Server 的各 worker 进程。

`vllm_client` 在训练参数解析阶段创建，经 `rlhf.py` 管道传给 `GRPOTrainer`/`GKDTrainer`，最终由 `RolloutTrainerMixin` 在每次权重同步和 rollout 推理时调用。