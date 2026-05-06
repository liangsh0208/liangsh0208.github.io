# 训练引擎（Engine）模块

**路径**：`verl/workers/engine/`

训练引擎封装了模型的前向传播、反向传播、优化器更新以及模型状态的序列化/反序列化，对外提供统一的 `BaseEngine` 接口，支持多种并行训练后端。

---

## 1. `BaseEngine` 抽象接口

**文件**：`verl/workers/engine/base.py`

```python
class BaseEngine:
    """所有训练引擎的抽象基类"""

    def initialize(self):
        """初始化模型、优化器、学习率调度器"""

    @property
    @abstractmethod
    def is_param_offload_enabled(self) -> bool:
        """是否启用参数 CPU offload"""

    @property
    @abstractmethod
    def is_optimizer_offload_enabled(self) -> bool:
        """是否启用优化器状态 CPU offload"""

    def train_mode(self, **kwargs) -> ContextManager:
        """进入训练模式的上下文管理器（加载参数到 GPU）"""

    def infer_mode(self, **kwargs) -> ContextManager:
        """进入推理模式的上下文管理器"""

    def compute_log_prob(self, data: DataProto) -> DataProto:
        """前向传播，计算 log_prob"""

    def update(self, data: DataProto) -> DataProto:
        """前向 + 反向 + 优化器更新，返回 metrics"""

    def state_dict(self) -> dict:
        """获取模型权重（用于保存检查点或同步给 Rollout）"""

    def load_state_dict(self, state_dict: dict):
        """加载权重"""
```

---

## 2. FSDP Engine

**文件**：`verl/workers/engine/fsdp/transformer_impl.py`

PyTorch FSDP（Fully Sharded Data Parallel）是 verl 的默认训练后端，将模型参数、梯度、优化器状态均匀分片到所有 DP rank。

### 2.1 FSDP2 配置

```python
class FSDPEngine(BaseEngine):
    def initialize(self):
        # 1. 加载 HuggingFace 模型
        model = AutoModelForCausalLM.from_pretrained(
            config.model.path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # 2. 应用 FSDP2 分片
        model = apply_fsdp2(
            model,
            mixed_precision=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,  # 梯度 reduce 用 FP32
            ),
            cpu_offload=CPUOffloadPolicy(
                offload_params=config.cpu_offload.param,
                offload_grad=config.cpu_offload.grad,
            ),
            wrap_policy=get_fsdp_wrap_policy(model),  # 按 transformer layer 切分
        )

        # 3. 初始化优化器（在 FSDP 分片后）
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
```

### 2.2 Activation Offload（激活值 CPU Offload）

```python
# 对于超长序列，激活值可能占满显存
if config.activation_offload:
    enable_activation_offloading(model)
    # forward 时激活值存到 CPU，backward 时按需搬回 GPU
```

### 2.3 前向传播

```python
def compute_log_prob(self, data: DataProto) -> DataProto:
    with self.infer_mode():
        # 可选：去掉 padding（remove_padding）
        if self.use_remove_padding:
            hidden_states, indices = unpad_input(data.batch["input_ids"], data.batch["attention_mask"])

        output = self.model(
            input_ids=data.batch["input_ids"],
            attention_mask=data.batch["attention_mask"],
            position_ids=data.batch.get("position_ids"),
        )
        # 从 logits 计算每个 token 的 log 概率
        log_probs = logprobs_from_logits(output.logits, data.batch["responses"])

    return DataProto.from_dict({"old_log_probs": log_probs})
```

### 2.4 反向传播与参数更新

```python
def update(self, data: DataProto) -> DataProto:
    self.optimizer.zero_grad()
    total_loss = 0.0

    # 将一个大 batch 拆成若干 mini-batch（梯度累积）
    for micro_batch in split_micro_batches(data, self.config.ppo_mini_batch_size):
        with self.train_mode():
            output = self.model(**micro_batch)
            logits = output.logits

        # 计算 PPO clip loss
        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
        loss = ppo_clip_loss(
            old_log_probs=micro_batch["old_log_probs"],
            log_probs=log_probs,
            advantages=micro_batch["advantages"],
            response_mask=micro_batch["response_mask"],
        )
        loss.backward()
        total_loss += loss.item()

    # 梯度裁剪
    fsdp2_clip_grad_norm_(self.model, max_norm=self.config.max_grad_norm)
    self.optimizer.step()
    self.lr_scheduler.step()
```

### 2.5 LoRA 支持

```python
if config.lora.rank > 0:
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        target_modules=config.lora.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    # 只更新 LoRA 参数，基础模型参数冻结
```

LoRA 模式下，Reference Policy 就是关闭 LoRA 的 Actor（不用单独部署 RefPolicy Worker）。

### 2.6 动态 Batch Size

```python
if config.use_dynamic_bsz:
    # 根据每条序列的 token 数量，动态打包 micro-batch
    # 目标：每个 micro-batch 的总 token 数接近 ppo_max_token_len_per_gpu
    micro_batches = prepare_dynamic_batch(
        data,
        max_token_len=config.ppo_max_token_len_per_gpu * world_size,
    )
```

---

## 3. Megatron Engine

**文件**：`verl/workers/engine/megatron/transformer_impl.py`

Megatron-LM 支持张量并行（TP）和流水线并行（PP），适合超大模型（70B+）。

### 3.1 并行维度

```
Model Parallel Group:
  ┌──────────────────────────────────────────┐
  │  DP=2, TP=4, PP=2                        │
  │                                          │
  │  ┌────────────────┐  ┌────────────────┐  │
  │  │  PP Stage 0    │  │  PP Stage 1    │  │
  │  │  Layer 0-15    │  │  Layer 16-31   │  │
  │  │                │  │                │  │
  │  │ TP: 4 GPUs     │  │ TP: 4 GPUs     │  │
  │  └────────────────┘  └────────────────┘  │
  │           ×2 (DP)                        │
  └──────────────────────────────────────────┘
  总 GPU = DP × TP × PP = 2 × 4 × 2 = 16
```

### 3.2 Megatron-Core 集成

**文件**：`verl/models/mcore/`

verl 通过 Bridge 层将 HuggingFace 权重格式转换为 Megatron-Core 格式：

```python
class MCoreBridge:
    def __init__(self, hf_model_path, mcore_config):
        # 将 HuggingFace 模型转换为 Megatron-Core 格式
        self.mcore_model = MegatronModel(mcore_config)
        self.weight_converter = HF2MCoreWeightConverter(hf_model_path)
        self.weight_converter.load_weights(self.mcore_model)
```

### 3.3 PP 推理特殊处理

流水线并行推理时，每个 PP stage 只有该 stage 的参数：

```python
def prepare_for_rollout(self):
    """推理前：将当前 PP stage 的参数广播给所有 PP rank（让每个 rank 有完整模型）"""
    broadcast_params_across_pp_ranks(self.model)

def cleanup_after_rollout(self):
    """推理后：释放非本 stage 的参数，恢复 PP 分片"""
    free_non_pp_params(self.model)
```

### 3.4 MTP（Multi-Token Prediction）

**文件**：`verl/models/mcore/mtp_patch.py`

MTP 让模型同时预测下一个 token 和未来多个 token（类似 Medusa），提高训练效率：

```python
class MTPPatch:
    """为 Megatron 模型添加多 token 预测头"""
    def forward(self, hidden_states):
        # 主预测头
        logits = self.lm_head(hidden_states)
        # 额外预测头（预测 t+2, t+3, ...）
        extra_logits = [head(hidden_states) for head in self.extra_heads]
        return logits, extra_logits
```

---

## 4. MindSpeed Engine（华为昇腾）

**文件**：`verl/workers/engine/mindspeed/`

适配华为 Atlas 系列 NPU（Ascend）：
- 使用 CANN（华为 AI 计算框架）代替 CUDA
- HCCL 代替 NCCL 做通信
- 通过 `npu_flash_attn_utils.py` 适配 FlashAttention

---

## 5. AutoModel Engine

**文件**：`verl/workers/engine/automodel/`

自动根据配置选择合适的 Engine 后端：

```python
def create_engine(engine_config: EngineConfig) -> BaseEngine:
    if engine_config.backend == "fsdp":
        return FSDPEngine(engine_config)
    elif engine_config.backend == "megatron":
        return MegatronEngine(engine_config)
    elif engine_config.backend == "mindspeed":
        return MindSpeedEngine(engine_config)
    elif engine_config.backend == "torchtitan":
        return TorchTitanEngine(engine_config)
```

---

## 6. TorchTitan Engine

**文件**：`verl/workers/engine/torchtitan/`

Meta 的 TorchTitan 分布式训练框架集成，支持 FSDP2 + Tensor Parallel 的组合。

---

## 7. VeOmni Engine

**文件**：`verl/workers/engine/veomni/`

字节跳动内部多模态模型训练引擎，支持视觉-语言模型（VLM）的 RL 训练。

---

## 8. Diffusers Engine

**文件**：`verl/workers/engine/fsdp/diffusers_impl.py`

支持扩散模型（如 Stable Diffusion）的 RL 训练（用于图像生成质量优化）。

---

## 9. 关键工程优化

### 9.1 参数 CPU Offload

```python
# 训练时将非当前 micro-batch 的参数卸载到 CPU
if config.cpu_offload.param:
    offload_fsdp_model_to_cpu(model)

# 计算前再加载到 GPU
with load_fsdp_model_to_gpu(model):
    output = model(...)
```

### 9.2 优化器状态 Offload

```python
# 优化器 Adam 的 m1/m2 占显存大量空间
# 反向传播时加载到 GPU，更新完后卸载到 CPU
if config.cpu_offload.optimizer:
    offload_fsdp_optimizer(optimizer)
    # ...
    load_fsdp_optimizer(optimizer)
    optimizer.step()
    offload_fsdp_optimizer(optimizer)
```

### 9.3 Prefix Grouper（前缀共享）

当多条 prompt 有共同前缀时（如系统提示词），可以只计算一次前缀的 KV cache，减少 FLOPs：

```python
class PrefixGrouper:
    """将共享前缀的样本分组，前缀只做一次前向传播"""
    def group_by_prefix(self, batch):
        groups = defaultdict(list)
        for i, sample in enumerate(batch):
            prefix_hash = hash(sample.prefix)
            groups[prefix_hash].append(i)
        return groups
```

### 9.4 序列长度均衡（`seqlen_balancing.py`）

在 DP 训练中，不同 rank 的序列长度不均匀会导致 GPU 利用率不平衡：

```python
# 在 batch_balance=True 时，重新排列 batch 中的样本
# 使每个 DP rank 分到的总 token 数尽量相等
balanced_batch = get_seqlen_balanced_partitions(batch, num_partitions=dp_size)
```
