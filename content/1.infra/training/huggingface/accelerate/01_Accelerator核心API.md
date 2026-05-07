# 01_Accelerator核心API

> **【源码定位】** `/Users/danchen/Documents/1.RL_fw/huggingface/accelerate/src/accelerate/accelerator.py`
>
> **【阅读建议】** 理解本章节后可直接编写 Accelerate 代码，是最实用的入门文档
>
> **【前置知识】** PyTorch nn.Module/Optimizer/DataLoader、Python 解包操作

---

## 模块概述

`Accelerator` 是 Accelerate 库的核心门面类（Facade Pattern），官方宣称 **"整个 API 就在这一个类里"**。它封装了所有分布式训练细节，为用户提供统一的训练接口。

### 关键设计决策【重点】

| 设计决策 | 说明 | 代码体现 |
|---------|------|---------|
| **统一入口** | 所有操作通过 Accelerator 实例进行 | `accelerator.prepare()` / `accelerator.backward()` |
| **透明包装** | 自动将模型/优化器/数据加载器包装为分布式版本 | `_prepare_model()` 自动处理 DDP/FSDP/DeepSpeed |
| **设备自动管理** | 无需手动 `.to(device)` | `device_placement=True` 默认启用 |
| **上下文感知** | 自动处理混合精度、梯度累积上下文 | `autocast()` 自动调用 |

---

## ASCII 架构图

### 1. Accelerator 核心方法流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Accelerator 核心方法流程                            │
└─────────────────────────────────────────────────────────────────────────────┘

prepare(model, optimizer, dataloader, scheduler)
    │
    ├─> _prepare_model(model) ─────────────┐
    │                                       │
    │   ┌───────────────────────────────────┴───────────────────────────┐
    │   │                      模型包装策略                            │
    │   ├───────────────────────────────────────────────────────────┤
    │   │  if distributed_type == NO:                                 │
    │   │      return model                 # 单卡：直接返回          │
    │   │  elif distributed_type == MULTI_GPU:                        │
    │   │      return DistributedDataParallel(model)                 │
    │   │  elif distributed_type == FSDP:                             │
    │   │      return FullyShardedDataParallel(model)                │
    │   │  elif distributed_type == DEEPSPEED:                        │
    │   │      return DeepSpeedEngine(model)                         │
    │   │  elif device.type == 'xla':                                │
    │   │      return xmp.DistributedDataParallel(model)             │
    │   └───────────────────────────────────────────────────────────┘
    │
    ├─> _prepare_optimizer(optimizer) ─────┐
    │                                       │
    │   ┌─────────────────────────────┐     │
    │   │  return AcceleratedOptimizer │     │
    │   │  - 处理梯度缩放              │     │
    │   │  - 处理梯度累积              │     │
    │   │  - 处理 clipping             │     │
    │   └─────────────────────────────┘     │
    │
    ├─> _prepare_dataloader(dataloader) ───┐
    │                                       │
    │   ┌─────────────────────────────┐     │
    │   │  return AcceleratedDataLoader│     │
    │   │  - 自动添加分布式采样器      │     │
    │   │  - 自动 batch 切分           │     │
    │   └─────────────────────────────┘     │
    │
    └─> _prepare_scheduler(scheduler) ─────┐
                                           │
        ┌─────────────────────────────┐    │
        │  return AcceleratedScheduler│    │
        │  - 根据梯度累积调整 step    │    │
        │  - 处理 warmup              │    │
        └─────────────────────────────┘    │


backward(loss)
    │
    ├─> 如果 mixed_precision == "fp16":
    │       scaler.scale(loss).backward()
    │   否则如果 mixed_precision == "bf16":
    │       loss.backward()  # BF16 不需要 scaler
    │   否则:
    │       loss.backward()
    │
    ├─> DeepSpeed 特殊处理
    │       if distributed_type == DEEPSPEED:
    │           model.backward(loss)
    │
    ├─> 梯度同步时机检测
    │       if gradient_state.sync_gradients:
    │           wait_for_everyone()
    │
    └─> TPU 特殊处理
            if distributed_type == XLA:
                xm.mark_step()


gather_for_metrics(tensor/tuple)
    │
    ├─> gather(tensor)          # 从所有进程收集
    │
    ├─> 处理 padding（最后一个 batch 可能不均）
    │       if len(batch) 不整除 num_processes:
    │           截断 padding 数据
    │
    └─> 只在主进程处理完整数据


save_state(output_dir)
    │
    ├─> 保存模型状态
    │       unwrap_model(model).save_pretrained()
    │
    ├─> 保存优化器状态
    │       torch.save(optimizer.state_dict(), ...)
    │
    ├─> 保存随机种子
    │       torch.save(rng_state, ...)
    │
    └─> 保存自定义注册对象
            for obj in registered_objects:
                obj.save_checkpoint()
```

### 2. Accelerator 内部依赖关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Accelerator                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  属性：                                                              │   │
│  │  ├─ state: AcceleratorState          # 完整状态                      │   │
│  │  ├─ gradient_state: GradientState    # 梯度累积状态                  │   │
│  │  ├─ device: torch.device             # 计算设备                      │   │
│  │  ├─ trackers: list[GeneralTracker]   # 实验追踪器列表                │   │
│  │  └─ project_configuration: ProjectConfiguration  # 项目配置        │   │
│  │                                                                     │   │
│  │  核心方法：                                                         │   │
│  │  ├─ prepare(...) -> tuple            # 准备模型/优化器/数据      │   │
│  │  ├─ backward(loss)                   # 反向传播                      │   │
│  │  ├─ gather_for_metrics(tensors)      # 收集分布式结果              │   │
│  │  ├─ save_state() / load_state()      # 检查点管理                  │   │
│  │  ├─ unwrap_model(model)              # 解包模型获取原始模型        │   │
│  │  └─ register_for_checkpointing(...) # 注册自定义检查点对象          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│           ┌────────────────────────┼────────────────────────┐               │
│           ▼                        ▼                        ▼               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────┐       │
│  │ AcceleratorState │  │ GradientState   │  │ ProjectConfiguration  │       │
│  │  (完整训练状态)   │  │ (梯度累积追踪)   │  │ (项目/日志目录配置)    │       │
│  └─────────────────┘  └─────────────────┘  └───────────────────────┘       │
│           │                                  │                             │
│           ▼                                  ▼                             │
│  ┌─────────────────┐              ┌───────────────────┐                    │
│  │   插件系统       │              │  实验追踪器系统   │                    │
│  │ DeepSpeedPlugin │              │   - TensorBoard   │                    │
│  │ FSDPPlugin      │              │   - Weights&Biases│                    │
│  │ ...             │              │   - MLflow        │                    │
│  └─────────────────┘              └───────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心代码片段

### 表意伪代码：prepare() 内部逻辑

```python
class Accelerator:
    def prepare(self, *args):
        """
        准备训练所需对象，返回分布式包装版本

        核心逻辑：根据 distributed_type 自动选择包装策略
        """
        result = []
        for obj in args:
            if isinstance(obj, nn.Module):
                prepared = self._prepare_model(obj)
            elif isinstance(obj, Optimizer):
                prepared = self._prepare_optimizer(obj)
            elif isinstance(obj, DataLoader):
                prepared = self._prepare_dataloader(obj)
            elif isinstance(obj, _LRScheduler):
                prepared = self._prepare_scheduler(obj)
            result.append(prepared)

        return result[0] if len(result) == 1 else tuple(result)

    def _prepare_model(self, model):
        """
        根据分布式类型包装模型
        """
        # 1. 设备放置
        if self.device_placement:
            model = model.to(self.device)

        # 2. 根据分布式类型包装
        if self.distributed_type == DistributedType.MULTI_GPU:
            # DDP 包装
            model = DistributedDataParallel(
                model,
                device_ids=[self.local_process_index],
                output_device=self.local_process_index
            )

        elif self.distributed_type == DistributedType.FSDP:
            # FSDP 包装
            model = FullyShardedDataParallel(
                model,
                auto_wrap_policy=self.fsdp_plugin.auto_wrap_policy
            )

        elif self.distributed_type == DistributedType.DEEPSPEED:
            # DeepSpeed 引擎
            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=self.deepspeed_plugin.optimizer,
                config=self.deepspeed_plugin.config
            )

        return model

    def _prepare_optimizer(self, optimizer):
        """
        包装优化器以支持梯度缩放和累积
        """
        return AcceleratedOptimizer(
            optimizer,
            device_placement=self.device_placement,
            scaler=self.scaler  # 用于混合精度
        )

    def _prepare_dataloader(self, dataloader):
        """
        包装数据加载器以支持分布式采样
        """
        return AcceleratedDataLoader(
            dataloader,
            device=self.device,
            sampler=DistributedSampler(dataloader.dataset)
                if self.distributed_type != DistributedType.NO else None
        )
```

### 表意伪代码：backward() 内部逻辑

```python
def backward(self, loss, **kwargs):
    """
    统一的反向传播接口

    自动处理：
    1. 混合精度梯度缩放
    2. DeepSpeed 特殊 backward
    3. 梯度同步时机
    """
    # 1. 检查是否需要跳过（梯度溢出时）
    if self.scaler is not None:
        # FP16: 使用 GradScaler
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward(**kwargs)
    else:
        # BF16/FP32: 直接 backward
        loss.backward(**kwargs)

    # 2. DeepSpeed 特殊处理
    if self.distributed_type == DistributedType.DEEPSPEED:
        self.deepspeed_engine.backward(loss)

    # 3. 更新梯度状态
    self.gradient_state._set_sync_gradients(True)
```

### 可运行代码：完整训练示例

```python
"""
Accelerator 核心 API 完整演示
运行：python demo_accelerator_api.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

# ============ 1. 创建模拟数据 ============
def create_dummy_data(batch_size=32, num_samples=1000):
    """创建模拟数据集"""
    X = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ============ 2. 定义模型 ============
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ============ 3. 训练函数 ============
def train():
    """完整的 Accelerate 训练流程"""

    # 3.1 初始化 Accelerator（仅需这一行）
    accelerator = Accelerator(
        mixed_precision="fp16",          # 启用混合精度
        gradient_accumulation_steps=2,   # 梯度累积
        log_with="tensorboard",          # 自动追踪
        project_dir="./logs"
    )

    # 3.2 打印环境信息
    accelerator.print(f"设备: {accelerator.device}")
    accelerator.print(f"混合精度: {accelerator.mixed_precision}")
    accelerator.print(f"进程数: {accelerator.num_processes}")
    accelerator.print(f"是否主进程: {accelerator.is_main_process}")

    # 3.3 创建模型、优化器、数据加载器（不需要 .to(device)）
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = create_dummy_data()

    # 3.4 准备阶段：自动处理分布式包装
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    accelerator.print("模型已准备完成，开始训练...")

    # 3.5 训练循环
    model.train()
    for epoch in range(3):
        total_loss = 0

        for step, (batch_x, batch_y) in enumerate(train_dataloader):
            # 注意：不需要 batch_x.to(device)，已自动处理

            # 前向传播
            outputs = model(batch_x)
            loss = nn.functional.mse_loss(outputs, batch_y)

            # 反向传播（替代 loss.backward()）
            accelerator.backward(loss)

            # 检查是否需要跳过（梯度溢出）
            if accelerator.optimizer_step_was_skipped:
                accelerator.print(f"步骤 {step}: 梯度溢出，跳过此步")
                continue

            # 优化器步进
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        # 只在主进程打印
        accelerator.print(f"Epoch {epoch}: avg_loss = {total_loss / len(train_dataloader):.4f}")

    # 3.6 保存状态
    if accelerator.is_main_process:
        accelerator.save_state("./checkpoint-epoch-3")
        accelerator.print("检查点已保存")

    # 3.7 解包模型并保存
    unwrapped_model = accelerator.unwrap_model(model)
    # 现在可以像普通 PyTorch 模型一样使用

    accelerator.print("训练完成")

if __name__ == "__main__":
    train()
```

### 可运行代码：检查点保存与恢复

```python
"""
检查点管理演示
"""
from accelerate import Accelerator
import torch.nn as nn

# ============ 定义需要保存状态的自定义对象 ============
class CustomTracker:
    """自定义追踪器示例"""
    def __init__(self):
        self.step = 0
        self.history = []

    def save_checkpoint(self, output_dir):
        """必须实现 save_checkpoint"""
        import json, os
        path = f"{output_dir}/custom_tracker.json"
        with open(path, 'w') as f:
            json.dump({'step': self.step, 'history': self.history}, f)

    def load_checkpoint(self, checkpoint_path):
        """必须实现 load_checkpoint"""
        import json
        path = f"{checkpoint_path}/custom_tracker.json"
        with open(path) as f:
            data = json.load(f)
            self.step = data['step']
            self.history = data['history']

# ============ 完整示例 ============
def checkpoint_demo():
    accelerator = Accelerator()

    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    tracker = CustomTracker()

    model, optimizer = accelerator.prepare(model, optimizer)

    # 注册自定义对象到检查点系统
    accelerator.register_for_checkpointing(tracker)

    # 训练...
    tracker.step += 1
    tracker.history.append(1.0)

    # 保存完整状态（包含模型、优化器、tracker）
    accelerator.save_state("checkpoint_dir")

    # 加载状态
    accelerator.load_state("checkpoint_dir")
    print(f"恢复的 step: {tracker.step}")
    print(f"恢复的历史: {tracker.history}")

if __name__ == "__main__":
    checkpoint_demo()
```

---

## 配置参数表

### Accelerator 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `device_placement` | bool | True | 是否自动将张量放到正确设备 |
| `mixed_precision` | str | None | "no"/"fp16"/"bf16"/"fp8" |
| `gradient_accumulation_steps` | int | 1 | 梯度累积步数 |
| `cpu` | bool | False | 强制使用 CPU |
| `dataloader_config` | DataLoaderConfiguration | None | 数据加载器配置 |
| `deepspeed_plugin` | DeepSpeedPlugin | None | DeepSpeed 配置 |
| `fsdp_plugin` | FullyShardedDataParallelPlugin | None | FSDP 配置 |
| `megatron_lm_plugin` | MegatronLMPlugin | None | Megatron-LM 配置 |
| `rng_types` | list | None | 随机数类型 ["generator"]/["torch"]/["cuda"] |
| `log_with` | str/list | None | 追踪器 "tensorboard"/"wandb"/["all"] |
| `project_dir` | str | None | 追踪日志保存路径 |
| `gradient_accumulation_plugin` | GradientAccumulationPlugin | None | 细粒度梯度累积配置 |
| `dynamo_plugin` | TorchDynamoPlugin | None | PyTorch 2.0 compile |
| `step_scheduler_with_optimizer` | bool | True | 是否在 optimizer.step 时更新 scheduler |

### Accelerator 常用属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `device` | torch.device | 当前进程的计算设备 |
| `distributed_type` | DistributedType | 分布式类型枚举 |
| `num_processes` | int | 总进程数 |
| `process_index` | int | 当前进程全局索引 |
| `local_process_index` | int | 当前节点内进程索引 |
| `is_main_process` | bool | 是否主进程 |
| `is_local_main_process` | bool | 是否本地主进程 |
| `mixed_precision` | str | 当前混合精度模式 |

---

## 常见问题排查

### Q1: prepare() 返回值的解包问题

```python
# 错误：Python 解包语法
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader  # 传入3个对象
)

# 正确：如果只传入1个对象，返回单个对象而非元组
model = accelerator.prepare(model)  # 正确

# 如果传入多个，使用解包
model, optimizer = accelerator.prepare(model, optimizer)

# 或者使用列表接收
prepared = accelerator.prepare(model, optimizer, dataloader)
model, optimizer, dataloader = prepared
```

### Q2: 访问原始模型参数

```python
# 包装后的模型（DDP/FSDP）不能直接使用
model = accelerator.prepare(model)

# 错误：访问原始属性
hidden_size = model.config.hidden_size  # 可能报错！

# 正确：先解包
unwrapped_model = accelerator.unwrap_model(model)
hidden_size = unwrapped_model.config.hidden_size

# 或者使用 overlay 方式
from accelerate.utils import DDPParameters
if accelerator.distributed_type == DistributedType.MULTI_GPU:
    # model.module 是原始模型（DDP）
    hidden_size = model.module.config.hidden_size
```

### Q3: gather_for_metrics 的使用

```python
# 场景：分布式评估时需要收集所有进程的预测结果
all_preds = []
all_labels = []

for batch in eval_dataloader:
    with torch.no_grad():
        outputs = model(**batch)

    preds = outputs.logits.argmax(dim=-1)
    labels = batch["labels"]

    # 收集所有进程的结果
    # 只在主进程返回完整数据，其他进程返回 None
    gathered_preds, gathered_labels = accelerator.gather_for_metrics((preds, labels))

    if accelerator.is_main_process:
        all_preds.extend(gathered_preds.cpu().numpy())
        all_labels.extend(gathered_labels.cpu().numpy())

# 只在主进程计算指标
if accelerator.is_main_process:
    accuracy = (all_preds == all_labels).mean()
    print(f"Accuracy: {accuracy}")
```

### Q4: 梯度累积与优化器同步

```python
accelerator = Accelerator(gradient_accumulation_steps=4)

for step, batch in enumerate(dataloader):
    with accelerator.accumulate(model):  # 梯度累积上下文
        loss = model(**batch)
        accelerator.backward(loss)

        # 注意：在 accumulate 块内不需要手动判断
        # 到第4步会自动同步梯度并更新
        optimizer.step()
        optimizer.zero_grad()

# 或者手动控制
for step, batch in enumerate(dataloader):
    loss = model(**batch)
    accelerator.backward(loss)

    # 手动判断是否需要同步
    if accelerator.sync_gradients:
        optimizer.step()
        optimizer.zero_grad()
```

### Q5: 保存模型的最佳实践

```python
# 方法1：使用 save_state（完整保存）
accelerator.save_state(f"checkpoint-{step}")
accelerator.load_state(f"checkpoint-{step}")

# 方法2：只保存模型权重（HF 格式推荐）
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    save_directory,
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,  # 重要！使用 accelerate.save
    state_dict=accelerator.get_state_dict(model)  # 获取原始状态字典
)

# 方法3：PyTorch 原生保存
accelerator.wait_for_everyone()  # 确保所有进程完成
if accelerator.is_main_process:
    torch.save(
        accelerator.get_state_dict(model),  # 解包后的状态字典
        "model.pt"
    )
```

---

## 与其他文档的交叉引用

| 内容 | 引用文档 |
|------|---------|
| `Accelerator` 设计理念和架构 | [00_整体架构与设计理念.md](1.infra/training/huggingface/accelerate/00_整体架构与设计理念.md) |
| `PartialState/AcceleratorState` 详解 | [02_状态管理层.md](02_状态管理层.md) |
| 分布式后端实现 | [03_分布式训练与Parallel.md](03_分布式训练与Parallel.md) |
| 大模型加载机制 | [04_大模型加载与Hook系统.md](04_大模型加载与Hook系统.md) |
| DeepSpeed/FSDP 特殊处理 | [05_DeepSpeed与FSDP集成.md](05_DeepSpeed与FSDP集成.md) |
| 混合精度细节 | [06_混合精度训练.md](06_混合精度训练.md) |
| 配置文件和启动命令 | [07_实战配置指南.md](07_实战配置指南.md) |
| 源码文件速查 | [README.md](1.infra/training/huggingface/accelerate/README.md) |
