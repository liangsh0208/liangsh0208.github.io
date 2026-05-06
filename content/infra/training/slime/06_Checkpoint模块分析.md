# Slime Checkpoint 模块实现原理与性能优化

> **【源码定位】** `slime/backends/megatron_utils/checkpoint.py`, `slime/backends/megatron_utils/model.py`  
> **【阅读建议】** 需理解 Megatron 分布式 checkpoint、<bridge> 双解耦架构  
> **【前置知识】** PyTorch StateDict、HuggingFace Safetensors、分布式一致性

## 1. 模块概述

Checkpoint 模块是 Slime 训练框架的**持久化组件**，负责模型权重和训练状态的保存与加载。该模块支持 **Megatron 原生格式**和 **HuggingFace 格式**的无缝转换，实现了**分布式一致性保证**，并针对大规模模型提供了多项性能优化。

**关键设计决策【重点】**：
1. **格式自动识别**：自动检测 HF / Megatron / Distributed 格式，无缝切换
2. **在线权重更新**：训练后即时同步到 SGLang（非 checkpoint 恢复）
3. **分布式保存**：支持 DP=1 或 DP>1 场景的一致 checkpoint
4. **Bridge 双写**：同时保存 Megatron 分片和 HF 格式，便于部署

## 2. Checkpoint 格式支持

### 2.1 支持的格式

| 格式 | 特点 | 适用场景 |
|-----|------|---------|
| **Megatron 分布式** | 分片存储，每个 rank 独立文件 | 大规模训练，继续训练 |
| **HuggingFace** | 单一模型文件，易于部署 | 模型发布，推理部署 |
| **Bridge 模式** | Megatron ↔ HF 双向转换 | 灵活切换训练/推理 |

### 2.2 Checkpoint 格式判断

```python
def _is_megatron_checkpoint(path):
    """判断是否为 Megatron 格式的 checkpoint

    Megatron 格式特征：
    1. 包含 latest_checkpointed_iteration.txt 文件
    2. 目录名匹配 iter_\d{7} 格式
    """
    return (
        Path(path) / "latest_checkpointed_iteration.txt"
    ).is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )
```

### 2.3 目录结构

```
Megatron Checkpoint 目录结构：
checkpoint_path/
├── latest_checkpointed_iteration.txt  # 最新迭代号
├── iter_0000001/                      # 迭代目录
│   ├── mp_rank_00/                    # TP rank 0
│   │   └── model_optim_rng.pt         # 模型+优化器+随机状态
│   ├── mp_rank_01/                    # TP rank 1
│   │   └── model_optim_rng.pt
│   └── ...
├── iter_0000002/
│   └── ...
└── iter_0000003/
    └── ...

HuggingFace Checkpoint 目录结构：
hf_checkpoint_path/
├── config.json                        # 模型配置
├── model.safetensors                  # 模型权重（safetensors 格式）
├── model.safetensors.index.json       # 分片索引（大模型）
├── tokenizer.json                     # Tokenizer
├── tokenizer_config.json
└── special_tokens_map.json
```

## 3. Checkpoint 加载流程

### 3.1 加载入口

```python
def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    """统一加载入口，自动识别格式

    流程：
    1. 检测 checkpoint 格式
    2. 根据格式选择加载函数
    3. 加载模型权重和优化器状态
    """
    load_path = args.load

    if _is_megatron_checkpoint(load_path):
        # Megatron 格式
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        # HuggingFace 格式
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )
```

### 3.2 HuggingFace Checkpoint 加载

```python
def _load_checkpoint_hf(ddp_model, optimizer, args, load_path):
    """从 HuggingFace 格式加载模型

    流程：
    1. 使用 AutoBridge 创建 Megatron 模型桥接
    2. 加载 HF 权重并转换为 Megatron 格式
    3. 初始化优化器状态
    """
    from megatron.bridge import AutoBridge

    # 创建桥接器
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_checkpoint,
        trust_remote_code=True
    )

    # 加载权重到 Megatron 模型
    with patch_megatron_model(ddp_model):
        bridge.load_hf_weights(ddp_model)

    # 处理混合精度
    if (args.fp16 or args.bf16) and optimizer is not None:
        optimizer.reload_model_params()

    return iteration=0, num_floating_point_operations_so_far=0
```

### 3.3 Bridge 模式详解

```
┌─────────────────────────────────────────────────────────────────┐
│                    HuggingFace ↔ Megatron 桥接                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HuggingFace 格式                 Megatron 格式                  │
│  ┌─────────────────┐            ┌─────────────────┐            │
│  │ model.safetensors│            │ mp_rank_*/      │            │
│  │                 │   Bridge   │ model_optim_rng │            │
│  │ - 统一命名      │ ────────> │ - 分片存储       │            │
│  │ - 单一/分片文件 │            │ - TP/PP 分布    │            │
│  │ - FP16/BF16     │            │ - 包含优化器    │            │
│  └─────────────────┘            └─────────────────┘            │
│                                                                 │
│  权重映射规则：                                                  │
│  HF: model.layers.{i}.self_attn.q_proj.weight                   │
│  Megatron: encoder.layers.{i}.self_attention.query.weight       │
│                                                                 │
│  并行映射：                                                      │
│  HF (单卡) -> Megatron (TP 分片)：按列/行切分                    │
│  HF (单卡) -> Megatron (PP 分片)：按层切分                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Checkpoint 保存流程

### 4.1 保存入口

```python
def save(iteration, model, optimizer, opt_param_scheduler):
    """安全保存 Checkpoint

    流程：
    1. 禁用前向 pre-hook（避免干扰）
    2. 调用 Megatron save_checkpoint
    3. 重新启用 pre-hook
    """
    args = get_args()

    # 禁用 pre-hook（分布式优化器场景）
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)

    # 调用 Megatron 保存
    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
        train_data_iterator=None,
    )

    # 重新启用 pre-hook
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)
```

### 4.2 HuggingFace 格式保存

```python
def save_hf_model(args, rollout_id, model):
    """将 Megatron 模型保存为 HuggingFace 格式

    用途：
    - 模型发布
    - 推理部署
    - 与 HF 生态集成
    """
    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
    )

    try:
        from megatron.bridge import AutoBridge

        path = Path(args.save_hf.format(rollout_id=rollout_id))
        path.mkdir(parents=True, exist_ok=True)

        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        # 创建桥接器
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_checkpoint,
            trust_remote_code=True
        )

        # 保存为 HF 格式
        with patch_megatron_model(model):
            bridge.save_hf_pretrained(model, path=path)

        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")

    except Exception as e:
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")
```

## 5. 分布式 Checkpoint 一致性

### 5.1 Rank 协调机制

```
┌─────────────────────────────────────────────────────────────────┐
│                    分布式保存协调                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                       │
│  │ Rank 0      │ ──────> 创建目录，写入元数据           │
│  │ (主 Rank)   │         latest_checkpointed_iteration │
│  └─────────────┘                                       │
│        │                                               │
│        │ 同步屏障                                      │
│        ▼                                               │
│  ┌─────────────────────────────────────────────────┐  │
│  │ 所有 Rank 并行写入各自的分片                      │  │
│  │                                                 │  │
│  │ Rank 0: mp_rank_00/model_optim_rng.pt           │  │
│  │ Rank 1: mp_rank_01/model_optim_rng.pt           │  │
│  │ Rank 2: mp_rank_02/model_optim_rng.pt           │  │
│  │ ...                                             │  │
│  └─────────────────────────────────────────────────┘  │
│        │                                               │
│        │ 同步屏障                                      │
│        ▼                                               │
│  ┌─────────────┐                                       │
│  │ 完成保存    │                                       │
│  └─────────────┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 分片写入策略

```python
# Megatron 的分片写入策略
# 每个 TP rank 写入独立的文件，避免冲突
for tp_rank in range(tp_size):
    for pp_rank in range(pp_size):
        rank_dir = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        # 或简化为：mp_rank_{tp_rank:02d}
```

## 6. 性能优化实现

### 6.1 ShardedTensor 验证跳过

```python
# 跳过耗时的 shard 元数据验证
# 原因：大模型分片数多，验证耗时长
try:
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    # 原始 __post_init__ 会验证分片元数据
    # 这里替换为空实现，跳过验证
    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    # 同样跳过跨 rank 验证
    @classmethod
    def _init_from_local_shards_and_global_metadata(
        cls, local_shards, sharded_tensor_metadata, ...
    ):
        # 不做跨 rank 验证，直接构建 ShardedTensor
        ...

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass
```

### 6.2 ROCm 兼容性补丁

**文件**：`slime/utils/rocm_checkpoint_writer.py`

ROCm/HIP 平台上使用分布式 checkpoint 的异步保存时，`non_blocking=True` 会导致张量被放入 pinned memory。后续若 fork 子进程（Megatron 部分保存路径的实现方式），会触发段错误。

```python
class ROCmFileSystemWriterAsync(FileSystemWriterAsync):
    """
    FileSystemWriterAsync wrapper for ROCm compatibility.

    On ROCm/HIP, using non_blocking=True causes tensors to be stored in pinned memory,
    which triggers segmentation faults when forking subprocesses afterward.
    """

    @staticmethod
    def preload_tensors(*args, **kwargs):
        if torch.version.hip:
            # 强制 non_blocking=False，避免 pinned memory 导致的 fork 后段错误
            if "non_blocking" in kwargs:
                kwargs["non_blocking"] = False
            elif len(args) > 1 and isinstance(args[-1], bool):
                args = args[:-1] + (False,)
        return FileSystemWriterAsync.preload_tensors(*args, **kwargs)
```

**应用方式**：
```python
def initialize_model_and_optimizer(args, role="actor"):
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync
        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")
    ...
```

### 6.3 内存高效加载

```python
def initialize_model_and_optimizer(args, role="actor"):
    """内存高效的模型初始化"""

    # 创建模型
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role

    # 清理显存
    clear_memory()

    # 加载 checkpoint
    iteration, _ = load_checkpoint(
        model, optimizer, opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )

    # 再次清理显存
    clear_memory()

    return model, optimizer, opt_param_scheduler, iteration
```

## 7. 训练状态恢复

### 7.1 状态恢复内容

```
┌─────────────────────────────────────────────────────────────────┐
│                    Checkpoint 包含的状态                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  模型状态：                                                      │
│  ├── 模型参数 (model weights)                                   │
│  ├── 模型配置 (config)                                          │
│  └── 参数缓冲区 (buffers, 如 BatchNorm 统计量)                  │
│                                                                 │
│  优化器状态：                                                    │
│  ├── 一阶矩 (exp_avg)                                           │
│  ├── 二阶矩 (exp_avg_sq)                                        │
│  ├── 步数 (step)                                                │
│  └── 学习率调度状态                                              │
│                                                                 │
│  训练状态：                                                      │
│  ├── 当前迭代次数 (iteration)                                   │
│  ├── 随机数生成器状态 (rng state)                               │
│  └── 浮点运算累计 (num_floating_point_operations_so_far)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 迭代恢复

```python
def initialize_model_and_optimizer(args, role="actor"):
    """从 checkpoint 恢复训练迭代"""

    # 加载 checkpoint 获取迭代次数
    iteration, _ = load_checkpoint(...)

    # 恢复学习率调度器状态
    opt_param_scheduler.step(increment=iteration * args.global_batch_size)

    return model, optimizer, opt_param_scheduler, iteration
```

### 7.3 随机状态恢复

```python
# 训练随机状态的保存
checkpoint_state = {
    'model': model_state_dict,
    'optimizer': optimizer_state_dict,
    'rng_state': {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state(),
        'numpy': np.random.get_state(),
        'random': random.getstate(),
    },
}

# 训练随机状态的恢复
torch.set_rng_state(checkpoint['rng_state']['torch'])
torch.cuda.set_rng_state(checkpoint['rng_state']['cuda'])
np.random.set_state(checkpoint['rng_state']['numpy'])
random.setstate(checkpoint['rng_state']['random'])
```

## 8. 增量训练支持

### 8.1 部分加载

```python
# 仅加载模型权重，不加载优化器状态
iteration, _ = load_checkpoint(
    model,
    optimizer=None,           # 不加载优化器
    opt_param_scheduler=None,
    skip_load_to_model_and_opt=False,
)

# 重置优化器状态（用于 fine-tuning）
if args.reset_optimizer_states:
    for chained_optimizer in optimizer.chained_optimizers:
        for group in chained_optimizer.optimizer.param_groups:
            if "step" in group:
                group["step"] = 0
        for state in chained_optimizer.optimizer.state.values():
            if "step" in state:
                state["step"] = 0
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
```

### 8.2 参数冻结场景

```python
# 仅训练部分参数时，冻结参数从原始 checkpoint 加载
# 可训练参数从最新 checkpoint 加载

def load_partial_checkpoint(model, frozen_patterns, trainable_patterns):
    """部分加载 checkpoint

    策略：
    1. 冻结参数：从原始 HF checkpoint 加载
    2. 可训练参数：从训练 checkpoint 加载（如果存在）
    """
    # 加载原始权重
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint)
    bridge.load_hf_weights(model)

    # 加载训练权重（仅可训练部分）
    if args.load is not None:
        checkpoint = torch.load(args.load)
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in trainable_patterns):
                param.data.copy_(checkpoint['model'][name])
```

## 9. 性能优化注意事项

### 9.1 保存优化

| 优化策略 | 实现 | 效果 |
|---------|------|------|
| 异步写入 | `FileSystemWriterAsync` | 训练与保存并行 |
| 跳过验证 | 替换 `__post_init__` | 大模型加载加速 2-5x |
| 分片存储 | 每秩独立文件 | 避免合并操作 |
| 压缩存储 | FP16/BF16 | 减少存储空间 |

### 9.2 加载优化

| 优化策略 | 实现 | 效果 |
|---------|------|------|
| 内存高效加载 | 分步清理显存 | 避免峰值内存 |
| 惰性加载 | 按需加载参数 | 减少初始内存 |
| 分片并行加载 | 各秩独立读取 | 加速加载过程 |

### 9.3 配置建议

```bash
# Checkpoint 配置
--save /path/to/checkpoints    # 保存路径
--save-interval 100            # 保存间隔
--load /path/to/checkpoints    # 加载路径

# HuggingFace 格式保存
--save-hf /path/to/hf_models/rollout_{rollout_id}

# 优化器重置（fine-tuning）
--reset-optimizer-states

# 性能优化
--dist-ckpt-save-pre-mcore-014 true  # 兼容性优化
```

### 9.4 常见问题与解决方案

#### 问题1：Checkpoint 加载慢
```
原因：ShardedTensor 验证耗时
解决：代码已内置跳过验证逻辑
```

#### 问题2：保存时 OOM
```
原因：保存过程占用额外内存
解决：使用异步保存，减少 active 显存
```

#### 问题3：权重不匹配
```
原因：HF 与 Megatron 参数命名不一致
解决：检查 AutoBridge 映射规则
```

#### 问题4：大模型保存失败
```
原因：单文件过大
解决：使用分布式 checkpoint，启用分片存储
```

## 10. Checkpoint 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Checkpoint 完整流程                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  加载流程:                                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  load_checkpoint()                                     │    │
│  │       │                                                │    │
│  │       ├─> _is_megatron_checkpoint() ?                  │    │
│  │       │        │                                       │    │
│  │       │   Yes──┴──No                                   │    │
│  │       │      │      │                                  │    │
│  │       │      │      └─> _load_checkpoint_hf()          │    │
│  │       │      │              │                          │    │
│  │       │      │              ├─> AutoBridge.from_hf()   │    │
│  │       │      │              └─> bridge.load_hf_weights()│   │
│  │       │      │                                          │    │
│  │       │      └─> _load_checkpoint_megatron()           │    │
│  │       │              │                                  │    │
│  │       │              ├─> 各 rank 并行读取分片           │    │
│  │       │              └─> 加载模型、优化器、随机状态     │    │
│  │       │                                                │    │
│  │       └─> clear_memory()                               │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  保存流程:                                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  save()                                                │    │
│  │       │                                                │    │
│  │       ├─> disable_forward_pre_hook() (可选)            │    │
│  │       │                                                │    │
│  │       ├─> save_checkpoint()                            │    │
│  │       │       │                                        │    │
│  │       │       ├─> Rank 0 创建目录                      │    │
│  │       │       ├─> 所有 rank 并行写入分片               │    │
│  │       │       └─> Rank 0 写入迭代号文件                │    │
│  │       │                                                │    │
│  │       └─> enable_forward_pre_hook() (可选)             │    │
│  │                                                        │    │
│  │  save_hf_model() (可选)                                │    │
│  │       │                                                │    │
│  │       ├─> AutoBridge.from_hf()                         │    │
│  │       └─> bridge.save_hf_pretrained()                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 11. Checkpoint 配置实战速查

### 11.1 断点续训命令

```bash
# 从最近一次 checkpoint 恢复
python train.py \
    --load "/checkpoints/my_run/latest_checkpointed_iteration.txt" \
    --start-rollout-id 50  # 指定从第 50 轮开始（可选，自动推断）

# 从指定迭代恢复
python train.py \
    --load "/checkpoints/my_run/iter_0000100"

# 加载 HF 格式作为起点（非续训）
python train.py \
    --hf-checkpoint "/models/Qwen2.5-7B" \
    --start-rollout-id 0   # 明确从头开始
```

### 11.2 双格式保存配置

```bash
# 同时保存 Megatron 和 HF 格式（推荐）
python train.py \
    --save "/checkpoints/my_run" \
    --save-interval 10 \
    --hf-save "/checkpoints/my_run_hf" \
    --hf-save-interval 10

# 只保存 HF 格式（快速部署）
python train.py \
    --only-save-hf-model \
    --save "/checkpoints/my_run"
```

### 11.3 在线权重同步（非 checkpoint，训练过程中）

```bash
# 每轮训练后同步权重到 SGLang（默认行为）
--update-rollout-every-epochs 1
--megatron-to-hf-mode bridge  # 更高效

# 检查权重同步质量（调试时用）
--check-weight-update-equal
--offload-rollout  # Colocate 模式必须开启
```

### 11.4 Checkpoint 故障排查

| 问题 | 诊断 | 解决方案 |
|------|------|---------|
| `FileNotFoundError: latest_checkpointed_iteration.txt` | 路径错误 | 检查 `--load` 路径是否包含迭代目录 |
| `KeyError: 'optimizer'` | 只加载模型，没有优化器 | 使用 `bridge` 模式重新初始化优化器 |
| 各 rank checkpoint 不一致 | 分布式保存失败 | 检查所有 rank 的存储是否可写 |
| HF 转换后推理结果差异大 | Bridge 转换错误 | 使用 `--megatron-to-hf-mode bridge` |
| Checkpoint 文件过大 | 未压缩 | 使用 `--bf16` 存储 |

## 12. 相关章节导航

### 前置阅读
- **[00_整体架构概览](./00_整体架构概览.md)** - Checkpoint 与 Weight Sync 的关系
- **[02_模型架构分析](./02_模型架构分析.md)** - Bridge 机制原理

### 后续阅读
- **[07_性能优化实践](./07_性能优化实践.md)** - Checkpoint 性能优化

### 横向参考
- **[05_调度与资源管理](./05_调度与资源管理模块分析.md)** - 在线权重同步与 Rollout Manager

## 13. 参考资料

- **Megatron-LM Checkpointing**: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/checkpointing.py
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **Safetensors 格式**: https://huggingface.co/docs/safetensors/
- **Megatron Bridge**: https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/bridge
- **DeepSpeed Checkpoint**: https://www.deepspeed.ai/tutorials/model-checkpointing/