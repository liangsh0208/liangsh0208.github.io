# 05_DeepSpeed与FSDP集成

> **【源码定位】** `/Users/danchen/Documents/1.RL_fw/huggingface/accelerate/src/accelerate/utils/deepspeed.py`, `utils/fsdp_utils.py`
>
> **【阅读建议】** 本章涉及第三方框架集成，建议在理解基础 Accelerator 后阅读
>
> **【前置知识】** DeepSpeed ZeRO、PyTorch FSDP、混合精度训练

---

## 模块概述

Accelerate 通过 Plugin 系统无缝集成 DeepSpeed 和 FSDP，让用户能够以统一的方式使用这些强大的分布式训练框架，而无需修改业务代码。

### 关键设计决策【重点】

| 设计决策 | 说明 | 技术实现 |
|---------|------|---------|
| **统一接口** | 相同的 prepare() 接口自动适配不同框架 | Plugin 系统封装框架细节 |
| **配置透传** | 支持原生 DeepSpeed/FSDP 配置 | 配置文件 JSON/YAML 透传 |
| **自动包装** | 自动调用框架的模型包装逻辑 | `deepspeed.initialize()` / `FSDP()` |
| **状态一致** | 保持 AcceleratorState 与框架状态同步 | 自定义 backward 覆盖 |
| **检查点兼容** | 支持框架原生的检查点格式 | `get_state_dict()` 适配 |

---

## ASCII 架构图

### 1. DeepSpeed 集成架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DeepSpeed 集成架构                                  │
└─────────────────────────────────────────────────────────────────────────────┘

用户代码
─────────
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)


                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Accelerator.prepare()                            │
│                                                                             │
│  检测到 distributed_type == DistributedType.DEEPSPEED                       │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────────┐                                 │
│                    │ DeepSpeedEngine     │                                 │
│                    │   ┌─────────────┐   │                                 │
│                    │   │ Model       │   │                                 │
│                    │   │ (Wrapped)   │   │                                 │
│                    │   └──────┬──────┘   │                                 │
│                    │          │          │                                 │
│                    │   ┌──────▼──────┐   │                                 │
│                    │   │ Optimizer   │   │                                 │
│                    │   │ (ZeRO Mod)  │   │                                 │
│                    │   └─────────────┘   │                                 │
│                    │                     │                                 │
│                    │  deepspeed.initialize()                               │
│                    │  ├─ zero_optimization                               │
│                    │  ├─ fp16/bf16                                      │
│                    │  └─ gradient_clipping                              │
│                    └─────────────────────┘                                 │
│                              │                                              │
└──────────────────────────────┼─────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         训练时的特殊处理                                      │
│                                                                             │
│  accelerator.backward(loss)                                                  │
│         │                                                                   │
│         └──► if distributed_type == DEEPSPEED:                              │
│                 model.backward(loss)  # DeepSpeed 特殊 backward             │
│                 model.step()                                                  │
│              else:                                                            │
│                 loss.backward()                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


DeepSpeed ZeRO 阶段对比:

ZeRO Stage 1:                     ZeRO Stage 2:                     ZeRO Stage 3:
──────────────                      ──────────────                      ──────────────
┌───────────┐                       ┌───────────┐                       ┌───────────┐
│ Optimizer │                       │ Optimizer │                       │ Optimizer │
│  States   │  分片                │  States   │  分片                │  States   │  分片
├───────────┤  ─────►              ├───────────┤  ─────►              ├───────────┤  ─────►
│ Gradients │  不分片               │ Gradients │  分片                │ Gradients │  分片
├───────────┤                      ├───────────┤  ─────►              ├───────────┤  ─────►
│ Parameters│  不分片               │ Parameters│  不分片               │ Parameters│  分片
└───────────┘                      └───────────┘                      └───────────┘  ─────►
                                                                         │
                                                                    ├───────────┤
                                                                    │ Activation│
                                                                    │  Memory   │
                                                                    └───────────┘

节省显存:                            更省显存:                          最省显存:
~4x                                 ~8x                                 与数据并行度相关
```

### 2. FSDP 集成架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FSDP 集成架构                                   │
└─────────────────────────────────────────────────────────────────────────────┘

用户代码
─────────
fsdp_plugin = FullyShardedDataParallelPlugin(...)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


准备阶段
─────────
model = FSDP(
    model,
    auto_wrap_policy=fsdp_plugin.auto_wrap_policy,
    mixed_precision=fsdp_plugin.mixed_precision,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=fsdp_plugin.limit_all_gathers,
    ...
)


FSDP 分片策略图解:

FULL_SHARD (FULL_SHARDING)          SHARD_GRAD_OP
─────────────────────────          ──────────────

Forward:                            Forward:
  GPU0: [A][B][ ]                     GPU0: [A][B]
  GPU1: [ ][B][C]                     GPU1: [A][B]  参数全复制
  GPU2: [ ][ ][C]                     GPU2: [B][C]
                                        │
Backward:                           Backward:
  All-gather 参数                      │
  Compute grad                        Compute grad
  Reduce-scatter grad                 All-reduce grad  梯度分片
  │
  GPU0: [dA][ ][ ]                     GPU0: [A][dB]
  GPU1: [ ][dB][ ]                     GPU1: [A][dB]
  GPU2: [ ][ ][dC]                     GPU2: [dB][C]

参数分片 + 梯度分片                   参数全复制 + 梯度分片
最省显存                              类似 DeepSpeed Stage 2
```

### 3. Plugin 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Plugin 系统架构                                     │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │   Accelerator   │
                         │   (门面类)       │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ DeepSpeedPlugin │ │   FSDPPlugin     │ │TorchDynamoPlugin│
    ├─────────────────┤ ├─────────────────┤ ├─────────────────┤
    │─ zero_stage     │ │─ sharding_strategy│ │─ dynamo_backend │
    │─ offload_device │ │─ auto_wrap_policy │ │─ mode           │
    │─ gradient_accum │ │─ backward_prefetch│ └─────────────────┘
    │─ ...            │ │─ state_dict_type  │
    └─────────────────┘ └─────────────────┘
             │                   │
             ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐
    │ DeepSpeedEngine │ │  FSDP Model      │
    │ ├─ model        │ │ ├─ full params  │
    │ ├─ optimizer    │ │ ├─ sharded params│
    │ ├─ lr_scheduler │ │ ├─ grad         │
    │ └─ config       │ │ └─ ...          │
    └─────────────────┘ └─────────────────┘
```

---

## 核心代码片段

### 表意伪代码：DeepSpeed 集成

```python
# ============================================================
# DeepSpeedPlugin 配置
# ============================================================
class DeepSpeedPlugin:
    """DeepSpeed 配置封装"""

    def __init__(
        self,
        zero_stage: int = 2,                    # ZeRO 阶段 0/1/2/3
        gradient_accumulation_steps: int = 1,
        offload_optimizer_device: str = "none",  # "none"/"cpu"/"nvme"
        offload_param_device: str = "none",       # "none"/"cpu"/"nvme"
        nvme_path: str = "/local_nvme",
        deepspeed_config: dict = None,        # 完整配置字典
    ):
        # 构建 DeepSpeed 配置
        self.config = deepspeed_config or self._build_config(
            zero_stage=zero_stage,
            gradient_accumulation_steps=gradient_accumulation_steps,
            offload_optimizer_device=offload_optimizer_device,
            offload_param_device=offload_param_device,
            nvme_path=nvme_path,
        )

    def _build_config(self, **kwargs) -> dict:
        """构建 DeepSpeed 配置字典"""
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {
                "stage": kwargs["zero_stage"],
                "offload_optimizer": {
                    "device": kwargs["offload_optimizer_device"],
                    "pin_memory": True,
                } if kwargs["offload_optimizer_device"] != "none" else None,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
            },
            "fp16": {
                "enabled": kwargs.get("fp16", False),
            },
            "bf16": {
                "enabled": kwargs.get("bf16", False),
            },
        }


# ============================================================
# DeepSpeed 模型准备
# ============================================================
def _prepare_deepspeed(self, model, optimizer, scheduler):
    """
    准备 DeepSpeed 引擎

    这会将 model/optimizer/scheduler 包装为 DeepSpeedEngine
    """
    import deepspeed

    # 构建 DeepSpeed 配置
    config = self.deepspeed_plugin.config

    # 调用 DeepSpeed 初始化
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=config,
        lr_scheduler=scheduler,
    )

    # DeepSpeedEngine 接管了 optimizer.step()
    # 包装 optimizer 保持接口一致
    if optimizer is not None:
        optimizer = AcceleratedOptimizer(optimizer, device_placement=True)

    return model, optimizer, dataloader, lr_scheduler


# ============================================================
# DeepSpeed Backward 处理
# ============================================================
def backward(self, loss, **kwargs):
    """
    DeepSpeed 特殊 backward 处理
    """
    if self.distributed_type == DistributedType.DEEPSPEED:
        # DeepSpeed 内部处理梯度缩放和反向传播
        self.deepspeed_engine.backward(loss, **kwargs)

        # DeepSpeed 在 backward 中自动执行 optimizer.step()
        # 所以外部不需要再调用 optimizer.step()
    else:
        # 标准 PyTorch backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)
```

### 表意伪代码：FSDP 集成

```python
# ============================================================
# FSDPPlugin 配置
# ============================================================
class FullyShardedDataParallelPlugin:
    """FSDP 配置封装"""

    def __init__(
        self,
        sharding_strategy: str = "FULL_SHARD",
        # FULL_SHARD / SHARD_GRAD_OP / NO_SHARD

        auto_wrap_policy: str = "TRANSFORMER_BASED_WRAP",
        # TRANSFORMER_BASED_WRAP / SIZE_BASED_WRAP / NO_WRAP

        transformer_layer_cls_to_wrap: list = None,
        # 如 ["BertLayer", "GPT2Block"]

        mixed_precision_policy: str = "DEFAULT",
        # DEFAULT / FP16 / BF16 / FP32

        backward_prefetch: str = "BACKWARD_PRE",
        # BACKWARD_PRE / BACKWARD_POST

        state_dict_type: str = "FULL_STATE_DICT",
        # FULL_STATE_DICT / SHARDED_STATE_DICT / LOCAL_STATE_DICT

        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        sync_module_states: bool = True,
    ):
        self.sharding_strategy = self._get_sharding_strategy(sharding_strategy)
        self.auto_wrap_policy = self._get_auto_wrap_policy(auto_wrap_policy)
        self.transformer_layer_cls_to_wrap = transformer_layer_cls_to_wrap
        self.mixed_precision_policy = self._get_mixed_precision(mixed_precision_policy)
        self.backward_prefetch = backward_prefetch
        self.state_dict_type = state_dict_type

    def _get_sharding_strategy(self, strategy: str):
        from torch.distributed.fsdp import ShardingStrategy
        return ShardingStrategy[strategy]

    def _get_auto_wrap_policy(self, policy: str):
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
            size_based_auto_wrap_policy,
        )

        if policy == "TRANSFORMER_BASED_WRAP":
            return transformer_auto_wrap_policy(
                module_cls=set(self.transformer_layer_cls_to_wrap)
            )
        elif policy == "SIZE_BASED_WRAP":
            return size_based_auto_wrap_policy(
                min_num_params=1e6,  # 1M 参数
            )
        return None


# ============================================================
# FSDP 模型准备
# ============================================================
def _prepare_fsdp(self, model, optimizer, dataloader, scheduler):
    """
    包装模型为 FSDP
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    # FSDP 包装
    model = FSDP(
        model,
        sharding_strategy=self.fsdp_plugin.sharding_strategy,
        auto_wrap_policy=self.fsdp_plugin.auto_wrap_policy,
        mixed_precision=self.fsdp_plugin.mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=self.fsdp_plugin.limit_all_gathers,
        use_orig_params=self.fsdp_plugin.use_orig_params,
        sync_module_states=self.fsdp_plugin.sync_module_states,
    )

    return model, optimizer, dataloader, scheduler


# ============================================================
# FSDP 检查点处理
# ============================================================
def get_state_dict_for_fsdp(model, unwrap=False):
    """
    获取 FSDP 模型的 state_dict

    FSDP 有两种保存模式：
    1. FULL_STATE_DICT: 完整模型（单卡保存，需要 gather）
    2. SHARDED_STATE_DICT: 分片模型（每卡保存自己的分片）
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    if unwrap:
        # 获取完整模型参数（只在 rank0 有全部参数）
        with FSDP.state_dict_type(
            model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
        return state_dict
    else:
        # 获取分片参数（每卡保存）
        return model.state_dict()
```

### 可运行代码：DeepSpeed 完整示例

```python
"""
DeepSpeed 集成完整示例
运行：accelerate launch --use_deepspeed deepspeed_demo.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator, DeepSpeedPlugin

# ============ 1. 配置 DeepSpeed ============
deep_plugin = DeepSpeedPlugin(
    zero_stage=2,                          # ZeRO Stage 2
    gradient_accumulation_steps=4,
    offload_optimizer_device="none",       # 不卸载优化器
    # offload_optimizer_device="cpu",       # 可选：卸载到 CPU
)

# ============ 2. 初始化 Accelerator ============
accelerator = Accelerator(
    mixed_precision="bf16",
    deepspeed_plugin=deep_plugin,
)

accelerator.print(f"DeepSpeed 配置完成，进程 {accelerator.process_index}")

# ============ 3. 创建模型和数据 ============
class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
        )

    def forward(self, x):
        return self.net(x)

model = DemoModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

dataset = TensorDataset(
    torch.randn(1000, 1000),
    torch.randint(0, 10, (1000,))
)
dataloader = DataLoader(dataset, batch_size=32)

# ============ 4. 准备阶段 ============
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

accelerator.print("模型已准备完成")

# ============ 5. 训练循环 ============
model.train()
for epoch in range(2):
    total_loss = 0

    for step, (batch_x, batch_y) in enumerate(dataloader):
        outputs = model(batch_x)
        loss = nn.functional.cross_entropy(outputs, batch_y)

        # DeepSpeed 特殊 backward（自动处理梯度缩放和同步）
        accelerator.backward(loss)

        # 注意：使用 DeepSpeed 时，不需要显式调用 optimizer.step()
        # 因为在 backward 中会自动调用
        # 但为了兼容性，Accelerate 包装后仍会保留 optimizer.step()
        optimizer.step()

        total_loss += loss.item()

    accelerator.print(f"Epoch {epoch}: avg_loss = {total_loss / len(dataloader):.4f}")

    # 保存检查点
    if accelerator.is_main_process:
        accelerator.save_state(f"checkpoint_epoch_{epoch}")

accelerator.print("训练完成")

# ============ 6. 模型保存 ============
# DeepSpeed 保存需要 unwrap
unwrapped_model = accelerator.unwrap_model(model)
state_dict = accelerator.get_state_dict(model)

if accelerator.is_main_process:
    torch.save(state_dict, "final_model.pt")
    accelerator.print("模型已保存")
```

### 可运行代码：FSDP 完整示例

```python
"""
FSDP 集成完整示例
运行：accelerate launch --use_fsdp fsdp_demo.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import FalconDecorator

# ============ 1. 配置 FSDP ============
fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",
    auto_wrap_policy="TRANSFORMER_BASED_WRAP",
    transformer_layer_cls_to_wrap=["DemoBlock"],  # 你的 Transformer Block 类名
    mixed_precision_policy="BF16",
    state_dict_type="FULL_STATE_DICT",  # 或 "SHARDED_STATE_DICT"
)

# ============ 2. 初始化 Accelerator ============
accelerator = Accelerator(
    mixed_precision="bf16",
    fsdp_plugin=fsdp_plugin,
)

accelerator.print(f"FSDP 配置完成，进程 {accelerator.process_index}")

# ============ 3. 创建模型 ============
class DemoBlock(nn.Module):
    """模拟 Transformer Block"""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Self attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x

class DemoModel(nn.Module):
    def __init__(self, dim=512, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            DemoBlock(dim) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))

model = DemoModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

dataset = TensorDataset(
    torch.randn(1000, 32, 512),  # (batch, seq_len, dim)
    torch.randint(0, 10, (1000,))
)
dataloader = DataLoader(dataset, batch_size=32)

# ============ 4. 准备阶段 ============
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

accelerator.print("模型已准备完成")
accelerator.print(f"模型类型: {type(model)}")

# ============ 5. 训练循环 ============
model.train()
for epoch in range(2):
    total_loss = 0

    for step, (batch_x, batch_y) in enumerate(dataloader):
        outputs = model(batch_x)
        loss = nn.functional.cross_entropy(outputs, batch_y)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    accelerator.print(f"Epoch {epoch}: avg_loss = {total_loss / len(dataloader):.4f}")

# ============ 6. 保存模型 ============
# FSDP 保存需要注意 state_dict_type

# 方法1：完整 state_dict（只在 rank0 有完整参数）
if accelerator.distributed_type == "FSDP":
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.api import FSDP

    # 设置保存配置
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )

# 获取 state_dict
state_dict = accelerator.get_state_dict(model)

if accelerator.is_main_process:
    torch.save(state_dict, "fsdp_model.pt")
    accelerator.print("模型已保存")

accelerator.print("训练完成")
```

### 可运行代码：框架对比

```python
"""
DeepSpeed vs FSDP 速度对比测试
运行：accelerate launch --num_processes 4 benchmark.py
"""
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator, DeepSpeedPlugin

def benchmark_training(backend="ddp", num_steps=100):
    """对比不同后端性能"""

    print(f"\n{'='*60}")
    print(f"Backend: {backend}")
    print(f"{'='*60}")

    # 配置
    if backend == "deepspeed":
        plugin = DeepSpeedPlugin(zero_stage=2)
        accelerator = Accelerator(deepspeed_plugin=plugin)
    else:
        accelerator = Accelerator()

    # 模型
    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 数据
    data = torch.randn(64, 4096).to(accelerator.device)
    target = torch.randn(64, 4096).to(accelerator.device)

    model, optimizer = accelerator.prepare(model, optimizer)

    # 热身
    for _ in range(10):
        loss = (model(data) - target).pow(2).mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    # 计时
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_steps):
        loss = (model(data) - target).pow(2).mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    if accelerator.is_main_process:
        print(f"进程数: {accelerator.num_processes}")
        print(f"总时间: {elapsed:.2f}s")
        print(f"每步时间: {elapsed/num_steps*1000:.2f}ms")

    return elapsed

# 运行测试（可选）
# benchmark_training("ddp")
# benchmark_training("deepspeed")
```

---

## 配置参数表

### DeepSpeedPlugin 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `zero_stage` | int | 2 | ZeRO 阶段 (0/1/2/3) |
| `gradient_accumulation_steps` | int | 1 | 梯度累积步数 |
| `offload_optimizer_device` | str | "none" | 优化器状态卸载位置 |
| `offload_param_device` | str | "none" | 参数卸载位置 |
| `nvme_path` | str | "/local_nvme" | NVMe 卸载路径 |
| `gradient_clipping` | float | 1.0 | 梯度裁剪阈值 |

### FSDPPlugin 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `sharding_strategy` | str | "FULL_SHARD" | 分片策略 |
| `auto_wrap_policy` | str | "TRANSFORMER_BASED_WRAP" | 自动包装策略 |
| `transformer_layer_cls_to_wrap` | list | None | 要包装的层类名 |
| `mixed_precision_policy` | str | "DEFAULT" | 混合精度策略 |
| `backward_prefetch` | str | "BACKWARD_PRE" | 反向预取策略 |
| `state_dict_type` | str | "FULL_STATE_DICT" | 状态字典类型 |
| `limit_all_gathers` | bool | True | 限制 all-gather |

### 启动命令对比

| 框架 | 启动命令 |
|------|---------|
| DDP | `accelerate launch train.py` |
| DeepSpeed | `accelerate launch --use_deepspeed train.py` |
| DeepSpeed (配置文件) | `accelerate launch --use_deepspeed --deepspeed_config_path config.json train.py` |
| FSDP | `accelerate launch --use_fsdp train.py` |
| FSDP (完整参数) | `accelerate launch --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap BertLayer train.py` |

---

## 常见问题排查

### Q1: DeepSpeed 配置不起作用

```python
# 错误：通过代码配置但未生效
accelerator = Accelerator()
# 应该在 Accelerator 初始化时传递插件
deep_plugin = DeepSpeedPlugin(zero_stage=3)
accelerator = Accelerator(deepspeed_plugin=deep_plugin)

# 或者使用配置文件
# accelerate config 交互式配置 DeepSpeed
# accelerate launch train.py
```

### Q2: FSDP 模型保存失败

```python
# 错误：直接保存 FSDP 模型
model.save_pretrained("save_dir")  # 报错！

# 正确：先解包或转换为完整 state_dict
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.api import FSDP

# 设置保存模式
FSDP.set_state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
)

# 获取 state_dict
state_dict = model.state_dict()

# 保存（只在主进程）
if accelerator.is_main_process:
    torch.save(state_dict, "model.pt")
```

### Q3: DeepSpeed 和 FSDP 如何选择？

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| 超大模型 (>40B) | DeepSpeed ZeRO-3 | 支持 NVMe 卸载 |
| 中等模型 (7B-40B) | FSDP | PyTorch 原生，更稳定 |
| 需要 Pipeline Parallel | DeepSpeed | 支持 pipeline |
| 需要 Tensor Parallel | Megatron-LM | 更强的并行能力 |
| 快速实验 | DDP/Accelerate | 配置简单 |

### Q4: ZeRO-3 内存仍不足

```python
# 开启 NVMe 卸载
deep_plugin = DeepSpeedPlugin(
    zero_stage=3,
    offload_optimizer_device="nvme",  # CPU -> NVMe
    offload_param_device="nvme",
    nvme_path="/path/to/fast_nvme",
)

# 或者使用 DeepSpeed 的配置文件
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
        },
        "stage3_gather_16bit_weights_on_model_save": True,
    }
}
```

### Q5: FSDP 显存碎片化

```python
# 在 prepare 前设置内存分配策略
import torch
# 使用 CUDA 内存池
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# 或者在代码中
accelerator = Accelerator()
if accelerator.distributed_type == DistributedType.FSDP:
    torch.cuda.empty_cache()
```

---

## 与其他文档的交叉引用

| 内容 | 引用文档 |
|------|---------|
| Accelerator API 详情 | [01_Accelerator核心API.md](01_Accelerator核心API.md) |
| 状态管理 | [02_状态管理层.md](02_状态管理层.md) |
| 分布式通信 | [03_分布式训练与Parallel.md](03_分布式训练与Parallel.md) |
| 启动配置 | [07_实战配置指南.md](07_实战配置指南.md) |
| 大模型加载 | [04_大模型加载与Hook系统.md](04_大模型加载与Hook系统.md) |
| 整体架构 | [00_整体架构与设计理念.md](1.infra%20with%20cc/training/huggingface/accelerate/00_整体架构与设计理念.md) |
| 源码文件速查 | [README.md](1.infra%20with%20cc/training/huggingface/accelerate/README.md) |
