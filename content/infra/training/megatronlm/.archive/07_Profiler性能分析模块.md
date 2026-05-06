# Megatron-LM Profiler 性能分析模块详解

> 训练流程辅助：性能分析与调优

---

## 1. 模块概述

Profiler 模块负责：
- 训练过程性能分析
- GPU 时间线追踪
- 内存使用分析
- 通信热点定位
- 性能瓶颈诊断

```
┌─────────────────────┐
│  启动配置            │
│  - profile          │
│  - profile_step_start│
│  - profile_step_end │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  工具选择            │
│  - PyTorch Profiler │
│  - NVIDIA Nsight    │
│  - NVTX Markers     │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  数据收集            │
│  - CUDA 时间        │
│  - 内存分配         │
│  - 通信事件         │
│  - Python 调用栈    │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  结果输出            │
│  - Chrome Trace     │
│  - TensorBoard      │
│  - Memory Snapshot  │
└─────────────────────┘
```

---

## 2. ProfilingConfig 配置类

**文件位置**: `megatron/training/config/common_config.py`

### 2.1 配置参数详解

```python
@dataclass(kw_only=True)
class ProfilingConfig:
    """性能分析配置"""

    # 是否启用 profiling
    use_nsys_profiler: bool = field(
        default=False,
        metadata={"argparse_meta": {"arg_names": ["--profile"], "dest": "profile"}}
    )
    """启用 nsys profiling。需要配合 nsys 命令行使用。

    示例命令:
    nsys profile -s none -t nvtx,cuda -o <output_file> --force-overwrite true \\
        --capture-range=cudaProfilerApi --capture-range-end=stop
    """

    # 分析时间窗口
    profile_step_start: int = 10
    """开始 profiling 的全局步数"""

    profile_step_end: int = 12
    """停止 profiling 的全局步数"""

    # PyTorch Profiler 选项
    use_pytorch_profiler: bool = False
    """使用 PyTorch 内置 profiler，可在 TensorBoard 中查看"""

    pytorch_profiler_collect_shapes: bool = False
    """收集张量形状信息"""

    pytorch_profiler_collect_callstack: bool = False
    """收集 Python 调用栈"""

    pytorch_profiler_collect_chakra: bool = False
    """收集 Chakra 执行追踪"""

    # Rank 选择
    profile_ranks: list[int] = field(default_factory=lambda: [])
    """指定要 profile 的全局 rank 列表（空列表表示所有 rank）"""

    # 内存分析
    record_memory_history: bool = False
    """在最后一个 rank 记录内存历史"""

    memory_snapshot_path: str = "snapshot.pickle"
    """内存快照输出路径"""

    # 其他选项
    record_shapes: bool = False
    """记录张量形状"""

    nvtx_ranges: bool = False
    """启用 NVTX 范围标注，用于 profiler 输出分类"""
```

### 2.2 命令行参数映射

| 命令行参数 | 配置字段 | 说明 |
|-----------|---------|------|
| `--profile` | `use_nsys_profiler` | 启用 profiling |
| `--profile-step-start` | `profile_step_start` | 开始步数 |
| `--profile-step-end` | `profile_step_end` | 结束步数 |
| `--use-pytorch-profiler` | `use_pytorch_profiler` | 使用 PyTorch profiler |
| `--pytorch-profiler-collect-shapes` | `pytorch_profiler_collect_shapes` | 收集形状 |
| `--pytorch-profiler-collect-callstack` | `pytorch_profiler_collect_callstack` | 收集调用栈 |
| `--profile-ranks` | `profile_ranks` | 指定 rank |
| `--record-memory-history` | `record_memory_history` | 记录内存历史 |

---

## 3. 两种 Profiler 模式

### 3.1 NVIDIA Nsight Systems (NSYS) Profiler

**特点**:
- 更底级的 CUDA 分析
- 支持 NVTX 标注
- 系统级性能视图
- 适合分析通信和 kernel 重叠

**使用方式**:

```bash
# 方式1: 使用 nsys 启动整个训练
nsys profile -s none -t nvtx,cuda -o profile_output \
    --force-overwrite true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    torchrun --nproc_per_node=8 train.py \
    --profile \
    --profile-step-start 10 \
    --profile-step-end 12

# 方式2: 多机训练
nsys profile -s none -t nvtx,cuda -o profile_%h_%p \
    --force-overwrite true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    torchrun --nnodes=4 --nproc_per_node=8 train.py \
    --profile \
    --profile-step-start 100 \
    --profile-step-end 102
```

**实现原理**:

```python
# megatron/training/training.py

# 开始 profiling
if iteration == args.profile_step_start:
    # 调用 CUDA Profiler API 开始
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
    # 启用 NVTX 范围追踪
    nsys_nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=True)
    nsys_nvtx_context.__enter__()

# 结束 profiling
if iteration == args.profile_step_end:
    # 退出 NVTX 上下文
    nsys_nvtx_context.__exit__(None, None, None)
```

### 3.2 PyTorch Profiler

**特点**:
- 集成在 PyTorch 中
- 输出 Chrome Trace 格式
- 支持 TensorBoard 可视化
- 支持 memory snapshot

**使用方式**:

```bash
torchrun --nproc_per_node=8 train.py \
    --profile \
    --use-pytorch-profiler \
    --profile-step-start 10 \
    --profile-step-end 15 \
    --pytorch-profiler-collect-shapes \
    --pytorch-profiler-collect-callstack
```

**实现原理**:

```python
# megatron/training/training.py

# 初始化 PyTorch Profiler
if args.profile and args.use_pytorch_profiler:
    # 可选: Chakra 执行追踪
    if args.pytorch_profiler_collect_chakra:
        et_dir = Path(f"{args.tensorboard_dir}/../chakra")
        et_dir.mkdir(parents=True, exist_ok=True)
        et = torch.profiler.ExecutionTraceObserver().register_callback(
            f"{et_dir}/rank-{torch.distributed.get_rank()}.json.gz"
        )

    # 设置 trace 处理函数
    def trace_handler(p):
        profile_dir = Path(f"{args.tensorboard_dir}/../torch_profile")
        profile_dir.mkdir(parents=True, exist_ok=True)
        p.export_chrome_trace(f"{profile_dir}/rank-{rank}.json.gz")

    # 创建 profiler
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start - 1, 0),  # 等待步数
            warmup=1 if args.profile_step_start > 0 else 0,  # 预热步数
            active=args.profile_step_end - args.profile_step_start,  # 活跃步数
            repeat=1,  # 重复次数
        ),
        on_trace_ready=trace_handler,
        record_shapes=args.pytorch_profiler_collect_shapes,
        with_stack=args.pytorch_profiler_collect_callstack,
        execution_trace_observer=et,
    )
    prof.start()

# 训练循环中
while iteration < args.train_iters:
    if args.profile and args.use_pytorch_profiler:
        prof.step()  # 推进 profiler 状态

    # ... 训练代码 ...

# 结束 profiling
if args.profile and iteration == args.profile_step_end:
    prof.stop()
    if prof.execution_trace_observer is not None:
        prof.execution_trace_observer.unregister_callback()
```

---

## 4. NVTX 标注系统

### 4.1 什么是 NVTX

NVTX (NVIDIA Tools Extension) 是 NVIDIA 提供的 API，用于在 profiler 输出中标注代码区域。

```
┌─────────────────────────────────────────────────────────┐
│  Timeline                                               │
├─────────────────────────────────────────────────────────┤
│  ████████████ forward_linear ████████████              │
│  ████████ attention ████████                            │
│  ████ qkv_proj ████  ████ out_proj ████                 │
│  ██████████ all_reduce ██████████                       │
└─────────────────────────────────────────────────────────┘
```

### 4.2 get_nvtx_range 实现

**文件位置**: `megatron/training/utils.py`

```python
def get_nvtx_range():
    """创建 NVTX 范围上下文管理器

    返回一个上下文管理器，用于标记代码块

    Example:
        nvtx_range = get_nvtx_range()
        with nvtx_range("forward_pass"):
            output = model(input)
    """
    try:
        from torch.cuda import nvtx

        @contextmanager
        def nvtx_range(msg, time=False):
            if time:
                timers = get_timers()
                timers(msg, log_level=0).start()
            try:
                nvtx.range_push(msg)  # 压入范围
                yield
            finally:
                nvtx.range_pop()  # 弹出范围
                if time:
                    timers(msg, log_level=0).stop()

        return nvtx_range
    except:
        # 降级为空操作
        @contextmanager
        def nvtx_range(msg, time=False):
            yield
        return nvtx_range
```

### 4.3 NVTX 使用示例

```python
from megatron.training.utils import get_nvtx_range

nvtx_range = get_nvtx_range()

def train_step(model, batch):
    with nvtx_range("train_step", time=True):
        # Forward
        with nvtx_range("forward"):
            output = model(batch)
            loss = compute_loss(output, batch)

        # Backward
        with nvtx_range("backward"):
            loss.backward()

        # Optimizer step
        with nvtx_range("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad()

        return loss
```

### 4.4 内置 NVTX 标注

Megatron 在关键位置已添加 NVTX 标注：

```python
# FSDP forward
with torch.autograd.profiler.record_function("CustomFSDP.forward"):
    output = self.module.forward(*inputs, **kwargs)

# RL 推理
with nvtx_range("offload-optimizer-state-and-grad-buffers-during-inference"):
    # Offload 优化器状态
    ...
```

---

## 5. 内存分析

### 5.1 记录内存历史

```python
# megatron/training/training.py

if args.record_memory_history and (is_last_rank() or torch.distributed.get_backend() == 'fake'):
    # 捕获内存快照
    snapshot = torch.cuda.memory._snapshot()
    from pickle import dump

    with open(args.memory_snapshot_path, 'wb') as f:
        dump(snapshot, f)
```

### 5.2 使用内存分析

```bash
# 启用内存历史记录
torchrun train.py \
    --profile \
    --record-memory-history \
    --memory-snapshot-path memory_snapshot.pickle

# 分析内存快照
python -c "
import pickle
with open('memory_snapshot.pickle', 'rb') as f:
    snapshot = pickle.load(f)
# 使用 torch.cuda.memory._dump_snapshot 可视化
"
```

### 5.3 TensorBoard 内存分析器

```bash
# 启用 PyTorch Profiler 的内存追踪
torchrun train.py \
    --profile \
    --use-pytorch-profiler \
    --profile-step-start 5 \
    --profile-step-end 10

# 在 TensorBoard 中查看
tensorboard --logdir runs/
# 打开 http://localhost:6006 -> Profile -> Memory
```

---

## 6. 多机 Profiling

### 6.1 指定 Rank 分析

```bash
# 只分析 rank 0 和 rank 4
torchrun --nnodes=2 --nproc_per_node=4 train.py \
    --profile \
    --use-pytorch-profiler \
    --profile-ranks 0 4 \
    --profile-step-start 100 \
    --profile-step-end 102
```

### 6.2 实现原理

```python
# megatron/training/training.py

# 检查当前 rank 是否需要 profile
should_profile = (
    args.profile and
    (len(args.profile_ranks) == 0 or
     torch.distributed.get_rank() in args.profile_ranks)
)

if should_profile:
    # 启动 profiler
    if args.use_pytorch_profiler:
        prof.start()
    else:
        # NSYS profiler
        pass
```

### 6.3 多机 NSYS 分析

```bash
# 每个 rank 生成独立文件
nsys profile -o profile_%h_%p \
    torchrun --nnodes=4 --nproc_per_node=8 train.py \
    --profile

# %h - 主机名
# %p - 进程 ID

# 合并分析 (使用 Nsight Systems GUI)
# File -> Open -> 选择所有 .nsys-rep 文件
```

---

## 7. PyTorch Profiler 调度详解

### 7.1 调度参数

```python
schedule = torch.profiler.schedule(
    wait=9,    # 前 9 步不记录
    warmup=1,  # 第 10 步预热（记录但性能开销小）
    active=2,  # 第 11-12 步完全记录
    repeat=1,  # 只执行一次
)
```

```
步数:    0  1  2  ...  8  9  10  11  12  13  ...
状态:   等待 等待 等待 ... 等待 等待 预热 记录 记录 结束
                              ↑    ↑    ↑
                         warmup active active
```

### 7.2 状态转换

```python
while iteration < args.train_iters:
    prof.step()  # 推进状态机

    # 状态: wait -> warmup -> active -> (repeat 或 finish)
```

---

## 8. 使用示例

### 8.1 快速性能分析

```bash
# 单机 8 卡，分析第 10-15 步
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --profile \
    --use-pytorch-profiler \
    --profile-step-start 10 \
    --profile-step-end 15 \
    ... 其他参数 ...

# 结果输出: runs/../torch_profile/rank-*.json.gz
```

### 8.2 详细 Kernel 分析

```bash
# 使用 NSYS 分析 kernel 和通信
nsys profile -s none -t nvtx,cuda,osrt,cudnn,cublas \
    -o detailed_profile \
    --force-overwrite true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    torchrun --nproc_per_node=8 pretrain_gpt.py \
    --profile \
    --profile-step-start 20 \
    --profile-step-end 22

# 打开分析
nsys-ui detailed_profile.nsys-rep
```

### 8.3 内存泄漏诊断

```bash
# 记录内存历史
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --profile \
    --use-pytorch-profiler \
    --record-memory-history \
    --memory-snapshot-path memory.pkl \
    --profile-step-start 1000

# 分析内存快照
python analyze_memory.py memory.pkl
```

### 8.4 通信热点定位

```bash
# 使用 NSYS 的 NCCL 追踪
nsys profile -s none -t nvtx,cuda,nccl \
    -o comm_profile \
    --force-overwrite true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    torchrun --nnodes=2 --nproc_per_node=8 pretrain_gpt.py \
    --profile \
    --profile-step-start 50
```

---

## 9. 结果可视化

### 9.1 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir runs/

# 访问
# http://localhost:6006
# -> Profile 页面
```

**TensorBoard Profile 视图**:
- Overview: 整体性能摘要
- Operator View: 算子级别时间分布
- Kernel View: CUDA kernel 时间
- Memory View: 内存使用曲线
- Trace View: 时间线详细视图

### 9.2 Chrome Trace

```python
# 输出 Chrome Trace 格式
p.export_chrome_trace("trace.json")
```

在 Chrome 浏览器打开:
```
chrome://tracing
# 然后加载 trace.json
```

### 9.3 Nsight Systems

```bash
# 打开 .nsys-rep 文件
nsys-ui profile.nsys-rep
```

---

## 10. 性能优化建议

### 10.1 Profiling 开销

| 模式 | 时间开销 | 内存开销 | 适用场景 |
|------|---------|---------|---------|
| 无 profile | 0% | 0% | 生产训练 |
| PyTorch Profile (基础) | ~5-10% | ~100MB | 常规分析 |
| PyTorch Profile (+shapes) | ~10-20% | ~500MB | 内存分析 |
| PyTorch Profile (+stack) | ~20-30% | ~1GB | 调用栈分析 |
| NSYS (cuda) | ~5% | ~100MB | kernel 分析 |
| NSYS (nvtx+cuda) | ~10% | ~200MB | 详细时间线 |

### 10.2 最佳实践

```python
# 1. 选择合适的分析窗口
# - 避免前几步（有初始化开销）
# - 选择稳定训练阶段
--profile-step-start 20
--profile-step-end 30

# 2. 只分析需要的 rank
# - 单机训练：分析 rank 0 即可
# - 多机训练：每节点分析 1-2 个 rank
--profile-ranks 0 4 8 12

# 3. 按需开启详细模式
# - 基础模式：先了解整体情况
# - 详细模式：针对性分析

# 4. 异步保存 trace
# - 避免阻塞训练
```

### 10.3 常见问题诊断

```python
# 问题1: GPU 利用率低
# 分析点:
# - 查看 Trace 中的 kernel 间隙
# - 检查 CPU-GPU 同步点
# - 查看 DataLoader 时间

# 问题2: 内存 OOM
# 分析点:
# - Memory View 查看峰值分配
# - 检查梯度累积
# - 查看临时张量

# 问题3: 通信瓶颈
# 分析点:
# - 查看 all_reduce 时间
# - 分析 compute-communication 重叠
# - 检查 NCCL kernel
```

---

## 11. 自定义 NVTX 标注

### 11.1 添加自定义标注

```python
from megatron.training.utils import get_nvtx_range
from contextlib import contextmanager

nvtx_range = get_nvtx_range()

class MyModel(torch.nn.Module):
    def forward(self, x):
        with nvtx_range("MyModel.forward"):
            # Embedding
            with nvtx_range("embedding"):
                x = self.embedding(x)

            # Transformer layers
            for i, layer in enumerate(self.layers):
                with nvtx_range(f"layer_{i}"):
                    x = layer(x)

            # Output
            with nvtx_range("output"):
                x = self.output(x)

            return x
```

### 11.2 条件性标注

```python
# 只在 profiling 时添加标注
def maybe_nvtx_range(msg, enabled=False):
    if enabled:
        return nvtx_range(msg)
    else:
        return contextmanager(lambda: iter([None]))()

# 使用
with maybe_nvtx_range("expensive_op", enabled=args.nvtx_ranges):
    result = expensive_operation()
```

---

## 12. 常见问题

### Q1: 多机 profiling 时文件冲突

**解决方案**:
```bash
# 使用 %h (hostname) 和 %p (PID) 区分
nsys profile -o profile_%h_%p.nsys-rep ...
```

### Q2: Profiling 步数选择

**建议**:
- **开始步数**: 跳过前 5-10 步（初始化开销）
- **持续步数**: 分析 3-10 步足够
- **特殊场景**: 如分析 checkpoint，选择包含保存的步数

### Q3: TensorBoard 无法打开 trace

**解决方案**:
```bash
# 确保文件完整
gunzip -t rank-0.json.gz

# 检查 TensorBoard 版本
pip install tensorboard-plugin-profile
```

### Q4: NSYS 无法捕获数据

**解决方案**:
```bash
# 确保使用 cudaProfilerApi 触发
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...

# 确保代码中有 --profile 参数
```

---

## 参考资料

- `megatron/training/config/common_config.py` - ProfilingConfig 定义
- `megatron/training/training.py` - Profiler 集成
- `megatron/training/utils.py` - get_nvtx_range 实现
- [PyTorch Profiler 文档](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVTX 文档](https://docs.nvidia.com/cuda/nvtx/index.html)