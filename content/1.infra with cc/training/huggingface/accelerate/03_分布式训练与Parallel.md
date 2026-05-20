---
created: 2026-05-06
---

# 03_分布式训练与Parallel

> **【源码定位】** `/Users/danchen/Documents/1.RL_fw/huggingface/accelerate/src/accelerate/utils/operations.py`
>
> **【阅读建议】** 理解分布式通信原语对调试多卡训练问题很有帮助
>
> **【前置知识】** PyTorch DDP、NCCL/Gloo后端、all-reduce/all-gather操作

---

## 模块概述

Accelerate 的分布式训练支持建立在 PyTorch 分布式通信后端之上，通过统一的抽象层屏蔽了不同后端（NCCL/Gloo/XLA）的差异。核心设计目标是让同一套代码能在单卡、多卡、TPU、多节点等不同环境中无缝运行。

### 关键设计决策【重点】

| 设计决策 | 说明 | 技术实现 |
|---------|------|---------|
| **后端自动选择** | 根据硬件自动选择最优通信后端 | `nccl` (GPU), `gloo` (CPU), `xla` (TPU) |
| **统一操作接口** | 不同后端使用相同的 Python API | `gather()`, `reduce()`, `broadcast()` |
| **嵌套结构支持** | 支持对 dict/list/tuple 的递归处理 | `recursively_apply()` |
| **设备感知** | 自动处理设备间数据传输 | 检查 tensor 设备并自动转换 |
| **TPU 特殊处理** | TPU 有独特的 mark_step 机制 | `xm.mark_step()` 自动调用 |

---

## ASCII 架构图

### 1. 分布式训练整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Training Script                            │
│                                                                            │
│   for batch in dataloader:                                                  │
│       loss = model(batch)                                                   │
│       accelerator.backward(loss)  ◄── 统一的反向传播接口                     │
│       optimizer.step()                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Accelerator 门面层                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         prepare(model)                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │   │
│  │  │  DDP Wrapper │  │ FSDP Wrapper │  │ DeepSpeed Engine         │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     backward(loss)                                  │   │
│  │     自动选择：GradScaler / DeepSpeed / 原生 backward                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     gather(tensor)                                │   │
│  │     自动调用：_gpu_gather / _tpu_gather / _cpu_gather               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│      MULTI_GPU      │  │        TPU          │  │       MULTI_CPU     │
│   ┌───────────┐     │  │   ┌───────────┐     │  │   ┌───────────┐     │
│   │  NCCL     │     │  │   │    XLA    │     │  │   │   Gloo    │     │
│   │ Backend   │     │  │   │  Backend  │     │  │   │  Backend  │     │
│   └───────────┘     │  │   └───────────┘     │  │   └───────────┘     │
│                     │  │                     │  │                     │
│   GPU 0 ◄──► GPU 1  │  │   TPU Core 0        │  │   CPU 0 ◄──► CPU 1  │
│     ▲        ▲      │  │      │              │  │     ▲        ▲      │
│     └────┬───┘      │  │      ▼              │  │     └────┬───┘      │
│       NVLink        │  │   TPU Core 1        │  │    网络通信         │
│                     │  │      │              │  │                     │
└─────────────────────┘  │      ▼              │  └─────────────────────┘
                         │   TPU Core 2       │
                         │      │              │
                         └─────────────────────┘
```

### 2. 分布式通信操作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        分布式通信核心操作                                     │
└─────────────────────────────────────────────────────────────────────────────┘

【gather 操作 - 收集所有进程的 Tensor】

进程 0: Tensor([1, 2])         进程 1: Tensor([3, 4])         进程 2: Tensor([5, 6])
       │                                │                               │
       └────────────────┬───────────────┘                               │
                        ▼                                               │
                  ┌─────────────┐                                       │
                  │ All Gather  │◄─────────────────────────────────────┘
                  │  收集操作    │
                  └──────┬──────┘
                         ▼
                  进程 0: [Tensor([1, 2]), Tensor([3, 4]), Tensor([5, 6])]
                  进程 1: [Tensor([1, 2]), Tensor([3, 4]), Tensor([5, 6])]
                  进程 2: [Tensor([1, 2]), Tensor([3, 4]), Tensor([5, 6])]


【reduce 操作 - 跨进程规约（如求和/平均）】

进程 0: Tensor([1, 2])         进程 1: Tensor([3, 4])         进程 2: Tensor([5, 6])
       │                                │                               │
       └────────────────┬───────────────┘                               │
                        ▼                                               │
                  ┌─────────────┐                                       │
                  │ All Reduce  │◄─────────────────────────────────────┘
                  │   (SUM)     │
                  └──────┬──────┘
                         ▼
                  所有进程: Tensor([9, 12])  # (1+3+5, 2+4+6)


【broadcast 操作 - 广播单进程数据到所有进程】

进程 0: Tensor([1, 2])         进程 1: None                   进程 2: None
       │                                │                               │
       └────────────────────────────────┼───────────────────────────────┘
                                          ▼
                                    ┌────────────┐
                                    │  Broadcast │
                                    └─────┬──────┘
                                          ▼
                  进程 0: Tensor([1, 2])  进程 1: Tensor([1, 2])  进程 2: Tensor([1, 2])
```

### 3. 后端检测与初始化流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PartialState._prepare_backend() 自动检测流程                    │
└─────────────────────────────────────────────────────────────────────────────┘

开始
  │
  ▼
检查 ACCELERATE_USE_DEEPSPEED == "true"?
  │
  ├─是──► 返回 backend="deepspeed", type=DEEPSPEED
  │
  ▼
检查 ACCELERATE_USE_FSDP == "true"?
  │
  ├─是──► 返回 backend="nccl/gloo", type=FSDP
  │
  ▼
检查 ACCELERATE_USE_MEGATRON_LM == "true"?
  │
  ├─是──► 返回 backend="nccl", type=MEGATRON_LM
  │
  ▼
检查 TPU/XLA 可用?
  │
  ├─是──► 返回 backend="xla", type=XLA/TPU
  │
  ▼
检查 CUDA 可用?
  │
  ├─是──► 检查 WORLD_SIZE > 1?
  │           │
  │           ├─是──► 返回 backend="nccl", type=MULTI_GPU
  │           │
  │           └─否──► 返回 backend=None, type=NO
  │
  ▼
检查 WORLD_SIZE > 1?
  │
  ├─是──► 返回 backend="gloo", type=MULTI_CPU
  │
  └─否──► 返回 backend=None, type=NO (单进程 CPU)
```

---

## 核心代码片段

### 表意伪代码：operations.py 核心操作

```python
# ============================================================
# 递归处理嵌套数据结构
# ============================================================
def recursively_apply(func, data, *args, test_type=is_torch_tensor, **kwargs):
    """
    递归处理嵌套数据结构（list/tuple/dict）中的张量

    这使得 gather/reduce 操作可以处理返回值为 dict 的模型
    """
    if isinstance(data, (tuple, list)):
        # 保留原始类型（tuple/list）
        return honor_type(
            data,
            (recursively_apply(func, o, *args, **kwargs) for o in data)
        )

    elif isinstance(data, Mapping):  # dict 等映射类型
        # 递归处理每个值
        return type(data)({
            k: recursively_apply(func, v, *args, **kwargs)
            for k, v in data.items()
        })

    elif test_type(data):  # 是张量
        return func(data, *args, **kwargs)  # 应用操作

    else:  # 其他类型（int/str等）直接返回
        return data


# ============================================================
# gather 操作：跨进程收集张量
# ============================================================
def gather(tensor):
    """
    跨进程收集张量，返回所有进程的 tensor 列表
    """
    state = AcceleratorState()

    # 特殊情况：非分布式
    if state.distributed_type == DistributedType.NO:
        return [tensor]

    # 特殊情况：单进程（但可能有分布式环境）
    if state.num_processes == 1:
        return [tensor]

    # GPU 收集
    if state.distributed_type == DistributedType.MULTI_GPU:
        return _gpu_gather(tensor)

    # TPU 收集
    if state.distributed_type == DistributedType.TPU:
        return _tpu_gather(tensor)

    # CPU 收集
    if state.distributed_type == DistributedType.MULTI_CPU:
        return _cpu_gather(tensor)

    # DeepSpeed 收集
    if state.distributed_type == DistributedType.DEEPSPEED:
        return _deepspeed_gather(tensor)


def _gpu_gather(tensor):
    """使用 NCCL all-gather 收集 GPU 张量"""
    world_size = torch.distributed.get_world_size()

    # 创建输出列表
    output_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

    # NCCL all_gather
    torch.distributed.all_gather(output_tensors, tensor)

    return output_tensors


def _tpu_gather(tensor):
    """使用 XLA all-gather 收集 TPU 张量"""
    import torch_xla.core.xla_model as xm

    # XLA all-gather
    gathered = xm.all_gather(tensor, dim=0)

    # 转换为列表
    world_size = xm.xrt_world_size()
    return [gathered[i] for i in range(world_size)]


# ============================================================
# reduce 操作：跨进程规约张量
# ============================================================
def reduce(tensor, reduction="sum"):
    """
    跨进程规约张量

    reduction: "sum" | "mean"
    """
    state = AcceleratorState()

    if state.distributed_type == DistributedType.NO:
        return tensor

    if state.distributed_type == DistributedType.MULTI_GPU:
        # NCCL all_reduce
        torch.distributed.all_reduce(tensor, op=ReduceOp.SUM)

    elif state.distributed_type == DistributedType.TPU:
        import torch_xla.core.xla_model as xm
        xm.all_reduce(xm.REDUCE_SUM, tensor)

    # 如果是平均，需要除以进程数
    if reduction == "mean":
        tensor = tensor / state.num_processes

    return tensor


# ============================================================
# broadcast 操作：广播张量
# ============================================================
def broadcast(tensor, from_process=0):
    """
    从指定进程广播张量到所有进程
    """
    state = AcceleratorState()

    if state.distributed_type == DistributedType.NO:
        return tensor

    if state.distributed_type == DistributedType.MULTI_GPU:
        torch.distributed.broadcast(tensor, src=from_process)

    return tensor


# ============================================================
# 设备间数据传输工具
# ============================================================
def send_to_device(tensor, device):
    """
    将张量发送到设备，支持嵌套结构
    """
    def _send(t):
        # 处理特殊类型
        if hasattr(t, "to"):
            return t.to(device)
        return t

    return recursively_apply(_send, tensor)
```

### 表意伪代码：进程同步原语

```python
# ============================================================
# wait_for_everyone - 同步屏障
# ============================================================
def wait_for_everyone():
    """
    阻塞直到所有进程到达此点
    """
    state = PartialState()

    if state.distributed_type == DistributedType.NO:
        return  # 单进程无需同步

    if state.distributed_type == DistributedType.MULTI_GPU:
        torch.distributed.barrier()

    elif state.distributed_type == DistributedType.TPU:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("wait_for_everyone")

    elif state.distributed_type == DistributedType.MULTI_CPU:
        torch.distributed.barrier()


# ============================================================
# on_main_process - 主进程执行装饰器
# ============================================================
def on_main_process(function):
    """
    装饰器：只在主进程执行函数
    """
    def wrapper(*args, **kwargs):
        state = PartialState()
        if state.is_main_process:
            return function(*args, **kwargs)
        return None
    return wrapper


# ============================================================
# main_process_first - 主进程优先上下文
# ============================================================
@contextmanager
def main_process_first():
    """
    上下文管理器：主进程先执行代码块

    用途：避免多进程争抢资源（如下载数据）
    """
    state = PartialState()

    if state.is_main_process:
        yield  # 主进程先执行
        wait_for_everyone()  # 等待其他进程
    else:
        wait_for_everyone()  # 先等待
        yield  # 非主进程后执行


# ============================================================
# synchronize_rng - 同步随机种子
# ============================================================
def synchronize_rng(rng_type: list[str], generator=None):
    """
    同步所有进程的随机种子

    rng_type: 要同步的随机数类型 ["generator", "torch", "cuda"]
    """
    state = PartialState()

    # 主进程生成种子并广播
    if state.is_main_process:
        seed = torch.randint(0, 2**32, (1,)).item()
    else:
        seed = None

    # 广播种子
    seed_tensor = torch.tensor([seed] if seed else [0])
    broadcast(seed_tensor, from_process=0)
    seed = int(seed_tensor.item())

    # 设置随机种子
    if "torch" in rng_type:
        torch.manual_seed(seed)
    if "cuda" in rng_type and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if "generator" in rng_type and generator is not None:
        generator.manual_seed(seed)
```

### 可运行代码：验证分布式操作

```python
"""
演示 operations.py 中的分布式操作
运行：accelerate launch --num_processes 2 demo_operations.py
"""
import torch
from accelerate import Accelerator
from accelerate.utils import gather, reduce, broadcast

accelerator = Accelerator()
device = accelerator.device

print(f"\n{'='*60}")
print(f"进程 {accelerator.process_index}/{accelerator.num_processes} 开始测试")
print(f"{'='*60}")

# ============ 测试 1: gather ============
print("\n--- 测试 1: gather ---")

# 每个进程创建不同的张量
local_tensor = torch.tensor([accelerator.process_index * 2,
                              accelerator.process_index * 2 + 1], device=device)

print(f"进程 {accelerator.process_index}: local_tensor = {local_tensor}")

# 收集所有进程的 Tensor
gathered = gather(local_tensor)

print(f"进程 {accelerator.process_index}: gathered = {gathered}")

# ============ 测试 2: reduce ============
print("\n--- 测试 2: reduce (sum) ---")

local_sum = torch.tensor([1, 2], dtype=torch.float32, device=device)
print(f"进程 {accelerator.process_index}: 输入 = {local_sum}")

reduced = reduce(local_sum.clone(), reduction="sum")
print(f"进程 {accelerator.process_index}: reduce sum = {reduced}")

# ============ 测试 3: reduce (mean) ============
print("\n--- 测试 3: reduce (mean) ---")

local_mean = torch.tensor([10.0, 20.0], device=device)
print(f"进程 {accelerator.process_index}: 输入 = {local_mean}")

reduced_mean = reduce(local_mean.clone(), reduction="mean")
print(f"进程 {accelerator.process_index}: reduce mean = {reduced_mean}")

# ============ 测试 4: broadcast ============
print("\n--- 测试 4: broadcast ---")

# 只有主进程有值
if accelerator.is_main_process:
    tensor_to_broadcast = torch.tensor([999, 888], device=device)
else:
    tensor_to_broadcast = torch.tensor([0, 0], device=device)

print(f"进程 {accelerator.process_index}: before broadcast = {tensor_to_broadcast}")

broadcast(tensor_to_broadcast, from_process=0)

print(f"进程 {accelerator.process_index}: after broadcast = {tensor_to_broadcast}")

# ============ 测试 5: 嵌套结构 gather ============
print("\n--- 测试 5: 嵌套结构 gather ---")

# 模拟模型返回的 dict
nested_data = {
    "loss": torch.tensor(float(accelerator.process_index), device=device),
    "logits": torch.tensor([[accelerator.process_index]], device=device),
    "metadata": {
        "batch_size": 32,
        "accuracy": torch.tensor(0.9 + 0.01 * accelerator.process_index, device=device)
    }
}

print(f"进程 {accelerator.process_index}: 输入 = {nested_data}")

gathered_nested = gather(nested_data)

if accelerator.is_main_process:
    print(f"进程 {accelerator.process_index}: gathered 结果类型 = {type(gathered_nested)}")
    print(f"  loss: {gathered_nested['loss']}")
    print(f"  logits: {gathered_nested['logits']}")

accelerator.wait_for_everyone()
print(f"\n进程 {accelerator.process_index}: 测试完成")
```

### 可运行代码：设备管理和自动切换

```python
"""
演示设备管理和自动后端选择
运行：python demo_device_management.py
"""
import torch
import os

def detect_optimal_backend():
    """模拟 PartialState._prepare_backend 的检测逻辑"""

    print("="*60)
    print("分布式后端自动检测")
    print("="*60)

    # 检查已设置的标志
    use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"
    use_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"
    use_megatron = os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false") == "true"

    print(f"\n环境标志:")
    print(f"  ACCELERATE_USE_DEEPSPEED: {use_deepspeed}")
    print(f"  ACCELERATE_USE_FSDP: {use_fsdp}")
    print(f"  ACCELERATE_USE_MEGATRON_LM: {use_megatron}")

    print(f"\n硬件检查:")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 设备数: {torch.cuda.device_count()}")
        print(f"  当前设备: {torch.cuda.current_device()}")
        print(f"  设备名称: {torch.cuda.get_device_name(0)}")

    print(f"\n环境变量:")
    env_vars = ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    for var in env_vars:
        value = os.environ.get(var, "未设置")
        print(f"  {var}: {value}")

    # 检测逻辑
    print(f"\n检测结果:")

    if use_deepspeed:
        print("  分布式类型: DEEPSPEED")
        print("  Backend: deepspeed")
        print("  说明: DeepSpeed ZeRO 优化")
    elif use_fsdp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        print("  分布式类型: FSDP")
        print(f"  Backend: {backend}")
        print("  说明: PyTorch FullyShardedDataParallel")
    elif use_megatron:
        print("  分布式类型: MEGATRON_LM")
        print("  Backend: nccl")
        print("  说明: Megatron-LM 框架")
    elif os.environ.get("XRT_TPU_CONFIG"):
        print("  分布式类型: TPU/XLA")
        print("  Backend: xla")
        print("  说明: Google Cloud TPU")
    elif torch.cuda.is_available():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            print("  分布式类型: MULTI_GPU")
            print("  Backend: nccl")
            print(f"  说明: DDP 训练，{world_size} 个 GPU")
        else:
            print("  分布式类型: NO (单卡)")
            print("  Backend: None")
            print("  说明: 单机单卡训练")
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            print("  分布式类型: MULTI_CPU")
            print("  Backend: gloo")
            print(f"  说明: CPU 分布式训练，{world_size} 个进程")
        else:
            print("  分布式类型: NO (CPU)")
            print("  Backend: None")
            print("  说明: 单机 CPU 训练")


# 运行检测
detect_optimal_backend()

print("\n" + "="*60)
print("设备选择策略")
print("="*60)

def get_device_selection():
    """演示设备选择逻辑"""

    cpu = os.environ.get("ACCELERATE_USE_CPU", "false") == "true"

    if cpu or not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return torch.device(f"cuda:{local_rank}")

device = get_device_selection()
print(f"\n选择设备: {device}")
print(f"设备类型: {'GPU' if device.type == 'cuda' else 'CPU'}")

if device.type == "cuda":
    print(f"显存总量: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    print(f"计算能力: {torch.cuda.get_device_capability(device)}")
```

---

## 配置参数表

### 分布式环境变量

| 环境变量 | 说明 | 典型值 |
|---------|------|-------|
| `LOCAL_RANK` | 当前节点内进程索引 | 0, 1, 2, 3... |
| `RANK` | 全局进程索引 | 0, 1, 2... |
| `WORLD_SIZE` | 全局进程总数 | 4, 8, 16... |
| `MASTER_ADDR` | 主节点地址 | 192.168.1.1 |
| `MASTER_PORT` | 主节点端口 | 29500 |
| `LOCAL_WORLD_SIZE` | 当前节点进程数 | 4 |

### NCCL 特定环境变量

| 环境变量 | 说明 | 适用场景 |
|---------|------|---------|
| `NCCL_DEBUG` | 调试日志级别 | INFO/WARN |
| `NCCL_P2P_DISABLE` | 禁用 P2P 通信 | RTX 4000 系列 |
| `NCCL_IB_DISABLE` | 禁用 InfiniBand | 非 IB 网络 |
| `NCCL_SOCKET_IFNAME` | 指定网络接口 | eth0 |

### 后端选择决策表

| 环境 | Backend | DistributedType | 说明 |
|------|---------|-----------------|------|
| 单机单卡 GPU | None | NO | 非分布式 |
| 单机多卡 GPU | nccl | MULTI_GPU | DDP |
| 多机多卡 GPU | nccl | MULTI_GPU | 多节点 DDP |
| 单机 DeepSpeed | deepspeed | DEEPSPEED | ZeRO |
| 单机 FSDP | nccl | FSDP | 全分片 |
| TPU | xla | TPU | XLA 后端 |
| CPU 多进程 | gloo | MULTI_CPU | CPU 集群 |

---

## 常见问题排查

### Q1: NCCL 连接超时

```python
# 错误信息：NCCL timeout
RuntimeError: NCCL operation failed: timeout

# 解决方案：增加超时时间
import os
os.environ["NCCL_TIMEOUT"] = "3600"  # 秒

# 或使用 accelerate launch 时设置
# NCCL_TIMEOUT=3600 accelerate launch train.py
```

### Q2: RTX 40 系列 NCCL P2P 错误

```python
# 错误信息
RuntimeError: NCCL_P2P_DISABLE is not set

# 原因：RTX 4000 系列显卡在旧驱动上不支持 P2P

# 解决方案：设置环境变量
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 或使用 accelerate config 配置
```

### Q3: gather 操作内存不足

```python
# 问题：gather 会复制所有进程的数据，容易OOM

# 解决方案1：增量 gather
for batch in dataloader:
    outputs = model(batch)
    # 只 gather 必要的指标，而不是整个输出
    predictions = outputs.logits.argmax(-1)
    gathered_preds = accelerator.gather(predictions)

# 解决方案2：只在主进程 gather
all_preds = []
for batch in dataloader:
    outputs = model(batch)
    preds = outputs.logits.argmax(-1)
    preds = accelerator.gather_for_metrics(preds)  # 自动截断 padding
    if accelerator.is_main_process:
        all_preds.append(preds)
```

### Q4: TPU mark_step 频繁导致性能问题

```python
# 问题：TPU 上频繁调用 mark_step 会降低性能

# 原因：每次 mark_step 都会触发 XLA 图编译

# 解决方案：增加梯度累积步数
accelerator = Accelerator(
    gradient_accumulation_steps=8  # 减少 step 频率
)

# 或使用 accumulate 上下文
with accelerator.accumulate(model):
    loss = model(**batch)
    accelerator.backward(loss)
    # 只有在累积完成时才触发 mark_step
```

### Q5: 多节点通信失败

```python
# 检查清单：
# 1. 确保所有节点可以互相通信
# nc -zv <master_ip> <master_port>

# 2. 确保防火墙开放端口
# 通常需要开放 29500 和后续几个端口

# 3. 确保所有节点使用相同的 PyTorch/Accelerate 版本

# 4. 显式设置网络接口
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # 根据实际情况修改

# 5. 启动命令示例（每个节点）
# Node 0 (主节点):
# accelerate launch --num_machines 2 --machine_rank 0 \
#     --main_process_ip 192.168.1.1 --main_process_port 29500 train.py

# Node 1:
# accelerate launch --num_machines 2 --machine_rank 1 \
#     --main_process_ip 192.168.1.1 --main_process_port 29500 train.py
```

---

## 与其他文档的交叉引用

| 内容 | 引用文档 |
|------|---------|
| 状态管理层详解 | [02_状态管理层.md](02_状态管理层.md) |
| Accelerator API | [01_Accelerator核心API.md](01_Accelerator核心API.md) |
| DeepSpeed 分布式 | [05_DeepSpeed与FSDP集成.md](05_DeepSpeed与FSDP集成.md) |
| 启动配置 | [07_实战配置指南.md](07_实战配置指南.md) |
| 整体架构 | [00_整体架构与设计理念.md](1.infra%20with%20cc/training/huggingface/accelerate/00_整体架构与设计理念.md) |
| 源码文件速查 | [README.md](1.infra%20with%20cc/training/huggingface/accelerate/README.md) |
