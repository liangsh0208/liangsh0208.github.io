# Actor / Critic Worker 模块

**路径**：`verl/workers/actor/`、`verl/workers/critic/`

---

## 1. 整体架构

Actor 和 Critic Worker 是训练阶段的核心计算单元，运行在 Ray actor 进程上，每个进程绑定一个 GPU。

```
ActorRollout Worker（每个 GPU 一个进程）
├── Engine（FSDP / Megatron）   ← 负责参数分片、前向、后向、优化器
├── Rollout（vLLM / SGLang）    ← 负责推理生成（Hybrid Engine 模式）
└── Sharding Manager            ← 在训练分片格式和推理格式之间转换权重
```

---

## 2. Worker 基类（`workers/actor/base.py`）

```python
class BasePPOActor:
    """Actor / Ref 的抽象基类，定义了前向计算和参数更新的接口"""

    def __init__(self, config: ActorConfig):
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """计算当前策略对给定 token 序列的 log 概率"""

    @abstractmethod
    def update_policy(self, data: DataProto) -> DataProto:
        """执行一步策略梯度更新（PPO clip loss 反向传播）"""
```

---

## 3. `DataParallelPPOActor`（FSDP Actor）

**文件**：`verl/workers/actor/dp_actor.py`

> **注**：该类已标记为 `@deprecated`（v0.8.0 移除），新代码请使用 `workers/engine/fsdp/` 中的 Engine 实现。

#### 3.1 核心功能

```python
class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config: ActorConfig, actor_module: nn.Module, optimizer=None):
        # optimizer=None 时为 Reference Policy（只做推理）
        self.use_remove_padding = config.use_remove_padding  # 去掉 padding 加速
        self.use_ulysses_sp = config.ulysses_sequence_parallel_size > 1  # Ulysses 序列并行
        self.use_dynamic_bsz = config.use_dynamic_bsz     # 动态 batch size（按 token 数量打包）
        self.use_prefix_grouper = config.use_prefix_grouper  # 前缀共享优化
```

#### 3.2 Remove Padding 优化

当 batch 中序列长度差异较大时，padding 会浪费大量计算。`use_remove_padding=True` 时：

```python
# 在 attention 计算前去掉 padding
hidden_states, indices, cu_seqlens = unpad_input(hidden_states, attention_mask)
# 计算后再还原 padding
output = pad_input(output, indices, batch_size, max_seqlen)
```

#### 3.3 Ulysses 序列并行

对超长序列（如 32K+），Ulysses 将序列维度切分到多个 GPU：

```python
if self.use_ulysses_sp:
    input_ids = ulysses_pad_and_slice_inputs(input_ids, sp_size)
    output = gather_outputs_and_unpad(output, sp_size)
```

---

## 4. `MegatronPPOActor`

**文件**：`verl/workers/actor/megatron_actor.py`

使用 Megatron-LM 的张量并行（TP）+ 流水线并行（PP）后端：
- TP：将矩阵按列/行切分到多个 GPU，每层内并行
- PP：将模型按层切分，不同层在不同 GPU 上，流水线执行
- PP 推理特殊处理：推理前需要将所有 PP stage 的参数广播到所有 GPU，推理完成后释放非本 stage 参数

---

## 5. Engine Workers（新版实现）

**文件**：`verl/workers/engine_workers.py`、`verl/workers/fsdp_workers.py`、`verl/workers/megatron_workers.py`

新版 Worker 架构将 Actor、Rollout、Reference 的逻辑统一到 `EngineWorker`，通过 `role` 参数区分行为：

```python
class EngineWorker(Worker):
    """统一的 Worker，根据 role 决定行为"""

    def __init__(self, config, role: str):
        # role in ["actor_rollout", "actor_rollout_ref", "ref_policy"]
        self.engine = create_engine(config.engine)          # FSDP / Megatron Engine
        self.rollout = create_rollout(config.rollout)       # vLLM / SGLang Rollout
        self.sharding_manager = create_sharding_manager(...)
```

`TrainingWorkerConfig` 统一了各 Role 的配置：
```python
@dataclass
class TrainingWorkerConfig:
    model_type: str          # "policy" / "value_model"
    model_config: HFModelConfig
    engine_config: EngineConfig
    optimizer_config: OptimizerConfig
    checkpoint_config: CheckpointConfig
```

---

## 6. Sharding Manager

**文件**：`verl/workers/sharding_manager/`

训练时模型参数被 FSDP/Megatron 分片到多 GPU；推理时 vLLM/SGLang 需要完整的权重。Sharding Manager 负责这两种格式之间的转换。

```python
class BaseShardingManager:
    def __enter__(self):
        """进入推理模式：将 FSDP 分片参数合并 → 加载到 vLLM"""
        self._pre_enter_hook()    # offload optimizer 到 CPU 释放显存
        self._load_to_rollout()   # 将参数同步到 vLLM engine

    def __exit__(self, ...):
        """退出推理模式：恢复 FSDP 分片状态"""
        self._post_exit_hook()    # reload optimizer 到 GPU

class FSDPUlyssesShardingManager(BaseShardingManager):
    """FSDP + Ulysses 序列并行的 Sharding Manager"""
```

---

## 7. `ActorConfig` 关键参数

```yaml
actor:
  # 训练配置
  ppo_epochs: 1                    # 每次 rollout 后更新几轮
  ppo_max_token_len_per_gpu: 8192  # 每个 GPU 最大 token 数（动态 batch）
  loss_agg_mode: "token_mean"      # loss 聚合方式
  loss_scale_factor: 1.0

  # 优化器
  optim:
    lr: 1e-6
    weight_decay: 0.0
    max_grad_norm: 1.0

  # 精度
  fsdp_config:
    param_dtype: bf16
    grad_dtype: fp32

  # 高级特性
  use_remove_padding: true         # 去掉 padding
  use_dynamic_bsz: true            # 动态 batch size
  ulysses_sequence_parallel_size: 1
  use_prefix_grouper: false        # 共享前缀优化

  # LoRA
  lora:
    rank: 64
    target_modules: ["q_proj", "v_proj"]
```

---

## 8. Critic Worker

**文件**：`verl/workers/critic/`

Critic 与 Actor 结构类似，但只做价值估计（输出标量），不做生成：

```python
class BasePPOCritic:
    def compute_values(self, data: DataProto) -> DataProto:
        """
        输入：prompts + responses（token ids）
        输出：每个 token 位置的价值估计 V(s_t)，shape (bs, seq_len)
        """

    def update_critic(self, data: DataProto) -> DataProto:
        """
        MSE Loss：L = mean((V(s_t) - returns_t)^2) * response_mask
        """
```

`DataParallelPPOCritic`（FSDP）和 `MegatronPPOCritic` 分别对应两种训练后端，接口完全相同。

### Critic 值头（Value Head）

Critic 在 LLM backbone 之上添加一个线性层，将 hidden state 映射为标量：

```python
class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        output = F.tanh(output)
        return self.summary(output).squeeze(-1)  # (bs, seq_len)
```

---

## 9. 关键 WorkerGroup 方法调用链

```
Driver                          ActorRollout WorkerGroup
  │                                │
  ├─ wg.generate_sequences(batch) ──→ worker.generate_sequences()
  │                                      └─ rollout.generate(batch)
  │                                         [vLLM 推理]
  │
  ├─ wg.compute_log_prob(batch) ──→ worker.compute_log_prob()
  │                                      └─ engine.forward(batch)
  │                                         [FSDP 前向]
  │
  └─ wg.update_actor(batch) ────→ worker.update_policy()
                                       └─ engine.backward_and_update()
                                          [FSDP 反向 + Adam]
```
