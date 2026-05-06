# Nanotron Checkpoint 与序列化

## 1. 顶部信息栏

```yaml
文档标题: Nanotron Checkpoint 与序列化
文件路径: 
  - src/nanotron/serialize/checkpoint.py (实际为 src/nanotron/serialize/main.py)
  - src/nanotron/serialize/weights.py
  - src/nanotron/serialize/optimizer.py
  - src/nanotron/serialize/metadata.py
  - src/nanotron/serialize/random.py
  - src/nanotron/serialize/utils.py
核心概念: CheckpointManager, safetensors, sharded weights, topology-agnostic loading, S3 support
创建日期: 2026-04-20
```

---

## 2. 模块概述

Nanotron 的 Checkpoint 系统提供了分布式场景下的完整状态保存与恢复能力，支持跨不同并行配置（TP/PP/DP 拓扑变化）加载检查点。

### 2.1 主要组件

| 组件 | 文件 | 功能描述 |
|------|------|----------|
| **save/load** | `serialize/main.py` | 主入口，协调各类状态的保存与加载 |
| **save_weights** | `serialize/weights.py` | 模型权重序列化（safetensors 格式） |
| **save_optimizer** | `serialize/optimizer.py` | 优化器状态序列化 |
| **TrainingMetadata** | `serialize/metadata.py` | 训练元数据结构定义 |
| **save_random_states** | `serialize/random.py` | 随机状态保存（可复现性） |
| **S3Mover** | `s3_checkpoints.py` | S3 存储集成 |

### 2.2 Checkpoint 结构

```
checkpoint/
├── checkpoint_metadata.json          # 检查点元数据
├── config.yaml                       # 训练配置副本
├── model/                            # 模型权重 (safetensors 格式)
│   ├── model_lm_head_weight.safetensors
│   ├── decoder_0_attention_qkv_proj_weight_pp-rank-0-of-2_tp-rank-0-of-2.safetensors
│   └── ...
├── optimizer/                        # 优化器状态
│   ├── optimizer_config.json
│   └── optimizer_pp-*_dp-*_tp-*.pt  # 各 rank 的优化器状态
├── lr_scheduler/                     # 学习率调度器状态
│   └── lr_scheduler_pp-*_tp-*.pt
└── random/                           # 随机状态
    └── tp-*-of-*_dp-*-of-*_pp-*-of-*.pt
```

### 2.3 核心特性

- **分片存储**: 支持 TP、PP、DP 维度的权重视分片
- **拓扑无关加载**: 可在不同并行配置下恢复检查点
- **格式版本控制**: 检查点版本管理，向后兼容
- **S3 集成**: 支持从 S3 加载和保存检查点
- **HuggingFace 互转**: 支持与 HF transformers 格式双向转换

---

## 3. 架构图

### 3.1 Checkpoint 保存流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Checkpoint Save Flow                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   save()  # src/nanotron/serialize/main.py                               │
│     │                                                                   │
│     ├───▶ save_config() ──────┐                                        │
│     │     - config.yaml         │                                        │
│     │                          ▼                                        │
│     ├───▶ save_weights() ─────┼──────┐                                 │
│     │     - Model state       │      │  ┌──────────────────────────┐   │
│     │     - safetensors       │      └──│ DP Rank 0 负责保存      │   │
│     │     - Sharded by TP/PP  │         │ 其他 rank 跳过          │   │
│     │                        ▼         └──────────────────────────┘   │
│     ├───▶ save_optimizer() ───┼──────┐                                 │
│     │     - Optim states      │      │  ┌──────────────────────────┐   │
│     │     - .pt format        │      └──│ Save: param_name_to_dp   │   │
│     │     - ZeRO sharding     │         │        _rank_offsets    │   │
│     │                        ▼         │ Save: orig_param_shapes  │   │
│     ├───▶ save_lr_scheduler()─┤        └──────────────────────────┘   │
│     │     - LambdaLR state    │                                        │
│     │     - .pt format        │                                        │
│     │                        ▼         ┌──────────────────────────┐   │
│     └───▶ save_meta() ────────┴───────│ 仅 Rank 0 保存元数据      │   │
│           - checkpoint_metadata.json    │ - version, tp, dp, pp    │   │
│           - TrainingMetadata            │ - consumed_samples      │   │
│                                         │ - last_train_step      │   │
│                                         └──────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ SANITY CHECKS (可选)                                           │    │
│  │ 1. 验证模型参数在所有 DP 间同步                                │    │
│  │ 2. 验证 tied 参数在相关 rank 间同步                            │    │
│  │ 3. 验证优化器状态在非 ZeRO 情况下同步                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Checkpoint 加载流程（拓扑无关）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Checkpoint Load Flow (Topology Agnostic)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  load() / resume training                                                │
│     │                                                                   │
│     ├───▶ parse_ckpt_path()                                            │
│     │     - 解析本地或 S3 路径                                         │
│     │     - 读取 latest.txt 获取最新步数                              │
│     │                                                                  │
│     ├───▶ load_meta()                                                  │
│     │     - 读取 checkpoint_metadata.json                              │
│     │     - 获取训练的 parallelism 配置 (ckp_tp, ckp_dp, ckp_pp)       │
│     │     - 版本兼容性检查                                             │
│     │                                                                  │
│     ├───▶ load_weights()                                               │
│     │     ┌─────────────────────────────────────────────────────────┐  │
│     │     │  For each parameter:                                    │  │
│     │     │                                                         │  │
│     │     │  IF ckp_tp_size == current_tp AND                      │  │
│     │     │     ckp_pp_size == current_pp:                         │  │
│     │     │     ├─▶ Direct load from safetensors                   │  │
│     │     │                                                         │  │
│     │     │  ELSE (topology mismatch):                             │  │
│     │     │     ├─▶ Find all related shards                        │  │
│     │     │     ├─▶ Merge TP/PP shards → unsharded tensor         │  │
│     │     │     ├─▶ Re-shard according to current topology        │  │
│     │     │     └─▶ Load into parameter                            │  │
│     │     │                                                         │  │
│     │     └─────────────────────────────────────────────────────────┘  │
│     │                                                                  │
│     ├───▶ load_optimizer()                                           │
│     │     ┌─────────────────────────────────────────────────────────┐  │
│     │     │  IF topology MATCH and NOT ZeRO-1:                     │  │
│     │     │     ├─▶ Direct load                                     │  │
│     │     │                                                         │  │
│     │     │  ELSE (topology MISMATCH or ZeRO-1):                   │  │
│     │     │     ├─▶ Load all shards into memory                     │  │
│     │     │     ├─▶ IF ZeRO-1: Merge DP shards first                │  │
│     │     │     ├─▶ Merge TP shards                                 │  │
│     │     │     ├─▶ Match param_name to optim state index           │  │
│     │     │     ├─▶ Reshard according to current topology           │  │
│     │     │     ├─▶ IF ZeRO-1: Shard across DP                      │  │
│     │     │     └─▶ Load into optimizer                             │  │
│     │     └─────────────────────────────────────────────────────────┘  │
│     │                                                                  │
│     ├───▶ load_lr_scheduler()                                        │
│     │     - 加载学习率调度器状态                                     │
│     │                                                                  │
│     └───▶ load_random_states()                                       │
│           - 恢复随机数生成器状态（保证可复现性）                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 分片权重存储结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Sharded Weight Storage Format                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  参数: model.decoder.0.attention.qkv_proj.weight                        │
│  配置: TP=2, PP=2                                                       │
│                                                                          │
│  Full Shape: [hidden_size * 3, hidden_size]                             │
│       │                                                                 │
│       ▼  PP 切分                                                        │
│  ┌──────────────────────────┬──────────────────────────┐               │
│  │     PP Rank 0 (layers)   │     PP Rank 1 (layers)   │               │
│  └──────────────────────────┴──────────────────────────┘               │
│              │                      │                                   │
│              ▼  TP 切分              ▼                                  │
│  ┌────────────────┐          ┌────────────────┐                        │
│  │   TP Rank 0    │          │   TP Rank 0    │                        │
│  │                │          │                │                        │
│  │ Sharded Shape  │          │ Sharded Shape  │                        │
│  │ [h*3/2, h]     │          │ [h*3/2, h]     │                        │
│  │                │          │                │                        │
│  │ local_slices   │          │ local_slices   │                        │
│  │ global_slices  │          │ global_slices  │                        │
│  └────────────────┘          └────────────────┘                        │
│         △                          △                                    │
│  ┌────────────────┐          ┌────────────────┐                        │
│  │   TP Rank 1    │          │   TP Rank 1    │                        │
│  │ Sharded Shape  │          │ Sharded Shape  │                        │
│  │ [h*3/2, h]     │          │ [h*3/2, h]     │                        │
│  └────────────────┘          └────────────────┘                        │
│                                                                          │
│  文件名格式:                                                            │
│  model_qkv_proj_weight                                                 │
│    _pp-rank-{pp_rank}-of-{pp_size}                                     │
│    _tp-rank-{tp_rank}-of-{tp_size}                                     │
│    .safetensors                                                        │
│                                                                          │
│  例如:                                                                 │
│  - model_qkv_proj_weight_pp-rank-0-of-2_tp-rank-0-of-2.safetensors    │
│  - model_qkv_proj_weight_pp-rank-0-of-2_tp-rank-1-of-2.safetensors    │
│  - model_qkv_proj_weight_pp-rank-1-of-2_tp-rank-0-of-2.safetensors    │
│  - model_qkv_proj_weight_pp-rank-1-of-2_tp-rank-1-of-2.safetensors    │
│                                                                          │
│  元数据 (safetensors metadata):                                       │
│  {                                                                      │
│    "version": "1.2",                                                  │
│    "local_global_slices_pairs": "[(0:1536, 0:1536), (0:768, 0:768)]", │
│    "unsharded_shape": "(3072, 768)"                                   │
│  }                                                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 检查点版本演进

```
Checkpoint Version History
│
├── v1.0 (legacy)
│   - 基础版本
│   - 无 TensorMetadata
│   - 从 checkpoint_metadata.json 推断
│
├── v1.1
│   - 引入 TensorMetadata
│   - local_global_slices_pairs 定义
│
├── v1.2 (current)
│   - 完整的分片信息
│   - DataStageMetadata 支持
│   - consumed_train_samples per stage
│   - S3 路径支持
│
└── Future
    - FP8 精度支持
    - 增量检查点
    - 异步保存优化
```

---

## 4. 核心实现与代码

### 4.1 保存主流程 (`save()`)

位置: `src/nanotron/serialize/main.py:49-204`

```python
def save(
    config: "Config",
    model: nn.Module,
    optimizer: optim.BaseOptimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    parallel_context: ParallelContext,
    training_metadata: TrainingMetadata,
    root_folder: Path,
    should_save_config: bool = True,
    should_save_model: bool = True,
    should_save_optimizer: bool = True,
    should_save_lr_scheduler: bool = True,
    sanity_checks: bool = True,
) -> None:
    """保存训练检查点
    
    保存内容包括：
    - 配置文件 (config.yaml)
    - 模型权重 (safetensors 格式，分片存储)
    - 优化器状态 (.pt 格式)
    - 学习率调度器状态 (.pt 格式)
    - 训练元数据 (checkpoint_metadata.json)
    """
    try:
        # 1. 保存配置
        if should_save_config:
            config.save_as_yaml(root_folder / "config.yaml", sanity_checks=sanity_checks)
    except Exception as e:
        raise e
        
    try:
        # 2. 保存模型权重
        if should_save_model:
            save_weights(model=model, parallel_context=parallel_context, root_folder=root_folder)
    except Exception as e:
        raise e
        
    try:
        # 3. 保存优化器
        if should_save_optimizer:
            save_optimizer(optimizer=optimizer, parallel_context=parallel_context, root_folder=root_folder)
    except Exception as e:
        raise e
        
    try:
        # 4. 保存学习率调度器
        if should_save_lr_scheduler:
            save_lr_scheduler(
                lr_scheduler=lr_scheduler,
                is_zero=config.optimizer.zero_stage,
                parallel_context=parallel_context,
                root_folder=root_folder,
            )
    except Exception as e:
        raise e

    # 5. 保存元数据
    save_meta(root_folder=root_folder, parallel_context=parallel_context, training_metadata=training_metadata)

    # 6. 可选的 Sanity Checks
    if sanity_checks:
        # 验证模型参数跨 DP 同步
        for name, param_or_buffer in sorted(model.state_dict().items(), key=lambda x: x[0]):
            assert_tensor_synced_across_pg(
                tensor=param_or_buffer,
                pg=parallel_context.dp_pg,
                msg=lambda err: f"{name} are not synced across DP {err}",
            )
        # 验证 tied 参数同步
        # 验证优化器状态同步
```

### 4.2 权重保存 (`save_weights`)

位置: `src/nanotron/serialize/weights.py:30-109`

```python
def save_weights(model: nn.Module, parallel_context: ParallelContext, root_folder: Path):
    """保存模型权重到 safetensors 格式
    
    特点：
    - 仅 DP Rank 0 保存（避免重复）
    - expert_rank=0 保存所有权重，其他仅保存 MLP 专家权重
    - 支持 tied weights，只有主副本保存
    """
    root_folder = root_folder / "model"
    
    # 仅 DP-0 保存权重（避免重复）
    if dist.get_rank(parallel_context.dp_pg) != 0:
        return
    
    # 构建模块到前缀的映射
    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    module_id_to_prefix[id(model)] = ""  # 根模型无前缀
    
    for name, param_or_buffer in tqdm(model.state_dict().items(), desc="Saving weights"):
        
        # 非 expert_rank=0 只保存 MoE 相关权重
        if dist.get_rank(parallel_context.ep_pg) != 0:
            if "experts" not in name:
                continue
        
        try:
            param = model.get_parameter(name)
        except AttributeError:
            param = None  # Buffer 类型
        
        if isinstance(param, NanotronParameter):
            metadata = {}
            
            if param.is_tied:
                # Tied 参数：只有主副本的第一个 rank 保存
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
                group_ranks = tied_info.global_ranks
                group = parallel_context.world_ranks_to_pg[group_ranks]
                
                if dist.get_rank(group) != 0:
                    continue  # 其他 rank 跳过
            else:
                base_name = name
            
            if param.is_sharded:
                # 分片参数：构建 TensorMetadata
                sharded_info: ShardedInfo = param.get_sharded_info()
                group = parallel_context.world_ranks_to_pg[sharded_info.global_ranks]
                exp_tp_pp_rank_and_size = get_exp_tp_pp_rank_and_size_from(...)
                is_expert_sharded = sharded_info.is_expert_sharded(parallel_context)
                
                metadata = TensorMetadata(
                    version=CHECKPOINT_VERSION,  # 例如 "1.2"
                    local_global_slices_pairs=sharded_info.local_global_slices_pairs,
                    unsharded_shape=sharded_info.unsharded_shape,
                ).to_str_dict()
            
            # 构建保存路径
            path = get_path(
                base_name,
                type=ObjectType.MODEL,
                exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
                is_expert_sharded=is_expert_sharded,
                prefix=root_folder,
            )
            path.parent.mkdir(exist_ok=True, parents=True)
            
            # 使用 safetensors 保存
            tensors = {"data": param_or_buffer}
            save_file(tensors=tensors, filename=path, metadata=metadata)
```

### 4.3 优化器状态保存

位置: `src/nanotron/serialize/optimizer.py:41-106`

```python
def save_optimizer(
    optimizer: optim.BaseOptimizer,
    parallel_context: ParallelContext,
    root_folder: Path,
):
    """保存优化器状态
    
    特点：
    - Zero-0: 只有 DP Rank 0 保存（完整状态）
    - Zero-1: 所有 DP Rank 保存各自分片
    """
    # Zero-0 只有 DP-0 保存；Zero-1 所有 rank 保存
    if (not optimizer.inherit_from(optim.ZeroDistributedOptimizer)) and \
       dist.get_rank(parallel_context.dp_pg) > 0:
        return
    
    root_folder = root_folder / "optimizer"
    root_folder.mkdir(exist_ok=True, parents=True)
    
    # Rank 0 保存优化器配置（用于拓扑变化的检查）
    if dist.get_rank(parallel_context.world_pg) == 0:
        config = {
            "type": str(optimizer.__class__.__name__),
            "parallelism": {
                "tp_size": str(parallel_context.tp_pg.size()),
                "dp_size": str(parallel_context.dp_pg.size()),
                "pp_size": str(parallel_context.pp_pg.size()),
                "expert_parallel_size": str(parallel_context.expert_parallel_size),
            },
            "configs": {},
        }
        
        if isinstance(optimizer, ZeroDistributedOptimizer):
            # ZeRO-1: 保存分片信息用于合并
            config["configs"]["param_name_to_dp_rank_offsets"] = convert_to_string(
                optimizer.param_name_to_dp_rank_offsets
            )
            # 保存原始参数形状（TP 分片参数被展平）
            config["configs"]["orig_param_shapes"] = convert_to_string(
                optimizer._orig_param_shapes
            )
        
        with open(root_folder / "optimizer_config.json", "w") as fo:
            json.dump(config, fo)
    
    # 保存优化器状态
    filename = optimizer_filename(
        parallel_context, 
        is_zero=optimizer.inherit_from(optim.ZeroDistributedOptimizer)
    )
    # 格式: optimizer_pp-{pp}-of-{pp}_dp-{dp}-of-{dp}_tp-{tp}-of-{tp}_exp-{ep}-of-{exp}.pt
    torch.save(optimizer.state_dict(), root_folder / filename)
```

### 4.4 拓扑无关的权重加载

位置: `src/nanotron/serialize/weights.py:190-320`

```python
def load_weights(
    model: nn.Module,
    parallel_context: ParallelContext,
    root_folder: Path,
    filtered_state_dict: Optional[Dict[str, Any]] = None,
):
    """从检查点加载权重，支持拓扑变化"""
    param_root_folder = root_folder / "model"
    filtered_state_dict = filtered_state_dict if filtered_state_dict is not None else model.state_dict()
    param_shard_metadata = {}  # 用于后续优化器加载
    
    for name, param_or_buffer in tqdm(filtered_state_dict.items(), desc="Loading weights"):
        param = model.get_parameter(name)
        
        if isinstance(param, NanotronParameter):
            # 处理 tied 参数名映射
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(...)
            else:
                base_name = name
            
            # 构建文件路径
            path = get_path(base_name, type=ObjectType.MODEL, ...)
            
            if path.exists():
                # 直接加载（拓扑匹配）
                with safe_open(path, framework="pt", device=str(param.device)) as fi:
                    param_or_buffer[:] = fi.get_tensor("data")
            else:
                # 拓扑不匹配：需要合并分片
                if not param.is_sharded:
                    raise ValueError(f"'{name}' is not sharded but file not found")
                
                # 查找所有相关分片
                suffix = base_name.rsplit(".", 1)[-1]
                shards_path = list(path.parent.glob(f"model_{suffix}*.safetensors"))
                
                # 合并分片并重新分片
                param_shard_metadata[name] = {}
                load_sharded_param_latest(
                    param_or_buffer=param_or_buffer,
                    sharded_info=sharded_info,
                    shards_path=shards_path,
                    param_shard_metadata=param_shard_metadata[name],
                )
    
    return param_shard_metadata  # 供优化器加载使用


def load_sharded_param_latest(
    param_or_buffer: torch.Tensor,
    sharded_info: ShardedInfo,
    shards_path: List[Path],
    param_shard_metadata: Optional[Dict] = None,
):
    """加载分片参数（支持拓扑变化）"""
    shards_and_slices_maps = []
    
    # 读取所有分片和元数据
    for shard_path in shards_path:
        with safe_open(shard_path, framework="pt", ...) as fi:
            param_metadata = TensorMetadata.from_str_dict(fi.metadata())
            shards_and_slices_maps.append(
                (fi.get_tensor("data"), param_metadata.local_global_slices_pairs)
            )
            
            if param_shard_metadata is not None:
                pp_rank, tp_rank = extract_tp_pp_rank_from_shard_path(shard_path)
                param_shard_metadata[(pp_rank, tp_rank)] = param_metadata
    
    # 分配完整张量
    unsharded_tensor = torch.empty(checkpoint_unsharded_shape, device=param_or_buffer.device)
    
    # 合并分片到完整张量，然后切分到当前拓扑
    merge_and_shard_tp_tensors(
        buffer=param_or_buffer,
        unsharded_buffer=unsharded_tensor,
        shards_and_slices_maps=shards_and_slices_maps,
        shard_metadata=sharded_info,
    )
```

### 4.5 拓扑无关的优化器加载

位置: `src/nanotron/serialize/optimizer.py:149-357`

```python
def load_optimizer(
    optimizer: optim.BaseOptimizer,
    parallel_context: ParallelContext,
    root_folder: Path,
    map_location: Optional[str] = None,
    param_shard_metadata: Dict = None,  # 从权重加载获取的分片信息
    model: nn.Module = None,
):
    """加载优化器状态，支持拓扑变化和 ZeRO-1"""
    root_folder = root_folder / "optimizer"
    
    # 读取检查点的并行配置
    with open(root_folder / "optimizer_config.json", "r") as file:
        ckp_optimizer_config = json.load(file)
    
    ckp_tp_size = int(ckp_optimizer_config["parallelism"]["tp_size"])
    ckp_pp_size = int(ckp_optimizer_config["parallelism"]["pp_size"])
    ckp_dp_size = int(ckp_optimizer_config["parallelism"]["dp_size"])
    
    # 检查拓扑是否变化
    if ckp_tp_size != parallel_context.tp_pg.size() or ckp_pp_size != parallel_context.pp_pg.size():
        # 需要拓扑无关的加载
        assert param_shard_metadata is not None, "需要分片元数据"
        
        ckp_optim_type = ckp_optimizer_config["type"]
        
        if ckp_optim_type == "ZeroDistributedOptimizer":
            # ZeRO-1: 先合并 DP 分片
            shard_paths = list(root_folder.glob(
                f"optimizer_pp-*-of-{ckp_pp_size}_dp-*-of-{ckp_dp_size}_tp-*-of-{ckp_tp_size}*.pt"
            ))
            ckp_sharded_optim_states = merge_dp_shard_in_zero1_optimizer(
                model, ckp_optimizer_config, shard_paths, parallel_context, map_location
            )
        else:
            # Zero-0: 直接加载所有 TP/PP 分片
            shard_paths = list(root_folder.glob(f"optimizer_pp-*-of-{ckp_pp_size}_tp-*-of-{ckp_tp_size}.pt"))
            ckp_sharded_optim_states = {}
            for shard_path in shard_paths:
                pp_rank, tp_rank = extract_parallel_ranks_from_shard_path(shard_path, is_zero1=False)
                ckp_sharded_optim_states[(pp_rank, tp_rank)] = torch.load(shard_path, map_location=map_location)
        
        # 构建新的优化器状态字典
        new_optim_state_dict = optimizer.state_dict()
        new_optim_state_dict["state"] = defaultdict(dict)
        
        # 获取优化器状态键（exp_avg, exp_avg_sq 等）
        OPTIMIZER_STATE_NAMES = sorted(
            ckp_sharded_optim_states[(0, 0)]["state"][0].keys() - ["step"]
        )
        
        # 遍历当前模型的所有参数
        for param_index, param_name in enumerate(model.state_dict().keys()):
            param = model.get_parameter(param_name)
            base_name = param.get_tied_info().name if param.is_tied else param_name
            
            if param.is_sharded:
                # 分片参数：合并旧检查点的分片，重新切分到当前拓扑
                new_shard_metadata = param.get_sharded_info()
                
                for state_key in OPTIMIZER_STATE_NAMES:  # exp_avg, exp_avg_sq
                    buffer = torch.zeros_like(param, device=map_location, dtype=OPTIMIZER_STATE_DTYPE)
                    unsharded_buffer = torch.empty(new_unshared_shape, device=map_location, dtype=OPTIMIZER_STATE_DTYPE)
                    
                    # 合并所有检查点分片
                    for (pp_rank, tp_rank), ckp_optim_state in ckp_sharded_optim_states.items():
                        old_optim_state_index = find_optim_index_from_param_name(
                            base_name, ckp_sharded_optim_states, is_zero1=False, pp_rank=pp_rank
                        )
                        if old_optim_state_index is None:
                            continue
                        
                        ckp_shard_data = ckp_optim_state["state"][old_optim_state_index][state_key]
                        ckp_shard_metadata = get_checkpoint_state_metadata(param_name, pp_rank, tp_rank)
                        
                        # 如果是 ZeRO-1，数据是展平的，需要 reshape
                        if ckp_optim_type == "ZeroDistributedOptimizer":
                            orig_shape = ckp_optimizer_config["configs"]["orig_param_shapes"][param_name]
                            ckp_shard_data = ckp_shard_data.view(orig_shape)
                        
                        # 合并到未分片缓冲区
                        # ... (merge logic)
                    
                    # 重新分片到当前拓扑
                    # ... (reshard logic)
                    
            else:
                # 非分片参数（如 LayerNorm）：直接复制
                # ... (direct copy logic)
            
            # 复制 step 值
            if step is not None:
                new_optim_state_dict["state"][param_index]["step"] = step
        
        state_dict = new_optim_state_dict
    else:
        # 拓扑匹配：直接加载
        state_dict = torch.load(
            root_folder / optimizer_filename(parallel_context, is_zero=...),
            map_location=map_location,
        )
    
    # 如果是 ZeRO-1，需要跨 DP 重新分片
    if isinstance(optimizer, ZeroDistributedOptimizer):
        if ckp_tp_size != parallel_context.tp_pg.size() or ckp_dp_size != parallel_context.dp_pg.size():
            current_dp_rank = dist.get_rank(parallel_context.dp_pg)
            for param_index in state_dict["state"]:
                param_name = [name for idx, name in state_dict["names"].items() if idx == param_index][0]
                for state_name in OPTIMIZER_STATE_NAMES:
                    # 切分到当前 DP rank 的偏移范围
                    sliced_tensor = get_sliced_tensor(
                        param=state_dict["state"][param_index][state_name],
                        start_offset=optimizer.param_name_to_dp_rank_offsets[param_name][current_dp_rank][0],
                        end_offset=optimizer.param_name_to_dp_rank_offsets[param_name][current_dp_rank][1],
                    )
                    state_dict["state"][param_index][state_name] = sliced_tensor
    
    optimizer.load_state_dict(state_dict, map_location=map_location)
```

### 4.6 检查点路径解析（支持 S3）

位置: `src/nanotron/serialize/main.py:207-284`

```python
def parse_ckpt_path(config: Config, parallel_context: ParallelContext) -> Optional[Path]:
    """解析检查点路径，支持本地和 S3
    
    支持两种格式：
    1. 检查点目录（包含 model_config.json）
    2. 检查点根目录（包含 latest.txt，指向最新 step）
    """
    load_from_candidate = config.checkpoints.resume_checkpoint_path
    
    if load_from_candidate is not None:
        if check_path_is_local(load_from_candidate):
            # 本地路径
            latest_meta_path = config.checkpoints.resume_checkpoint_path / "latest.txt"
            if latest_meta_path.exists():
                # latest.txt 存在，从中读取最新 step
                with fs_open(latest_meta_path, mode="r") as fi:
                    load_from_candidate = int(fi.read())
                checkpoint_path = config.checkpoints.resume_checkpoint_path / str(load_from_candidate)
            elif (config.checkpoints.resume_checkpoint_path / "model_config.json").exists():
                # 直接是检查点目录
                checkpoint_path = config.checkpoints.resume_checkpoint_path
            else:
                return None  # 未找到检查点
        else:
            # S3 路径
            latest_meta_path = config.checkpoints.resume_checkpoint_path / "latest.txt"
            if latest_meta_path.exists():
                with fs_open(latest_meta_path, mode="r") as fi:
                    latest_iteration = int(fi.read())
                s3_path = config.checkpoints.resume_checkpoint_path / str(latest_iteration)
                checkpoint_path = config.checkpoints.checkpoints_path / str(latest_iteration)
            else:
                s3_path = config.checkpoints.resume_checkpoint_path
                checkpoint_path = config.checkpoints.checkpoints_path / load_from_candidate.name
            
            # 从 S3 下载检查点
            s3_mover = S3Mover(
                local_path=os.path.join(checkpoint_path),
                s3_path=os.path.join(s3_path),
                s5cmd_numworkers=config.s3_upload.s5cmd_numworkers,
                s5cmd_concurrency=config.s3_upload.s5cmd_concurrency,
                s5cmd_path=config.s3_upload.s5cmd_path,
                dummy=bool(int(os.environ.get("LOCAL_RANK", None)) != 0),
            )
            s3_mover.distributed_wait_for_completion(parallel_context.world_pg)
            s3_mover.start_downloading()
            s3_mover.distributed_wait_for_completion(parallel_context.world_pg)
        
        return checkpoint_path
```

---

## 5. 配置参数表

### 5.1 CheckpointsArgs 配置

```yaml
# 检查点配置
checkpoints:
  checkpoints_path: /path/to/checkpoints              # 保存路径
  checkpoint_interval: 1000                           # 保存间隔（steps）
  save_initial_state: false                           # 是否保存初始状态（step 0）
  save_final_state: true                              # 是否保存最终状态
  resume_checkpoint_path: /path/to/checkpoint         # 恢复路径（可选）
  load_lr_scheduler: true                             # 是否加载学习率调度器
  load_optimizer: true                                # 是否加载优化器
  checkpoints_path_is_shared_file_system: false       # 是否为共享文件系统
```

### 5.2 S3UploadArgs 配置

```yaml
# S3 上传配置（可选）
s3_upload:
  upload_s3_path: s3://bucket/checkpoints             # S3 目标路径
  remove_after_upload: false                          # 上传后是否删除本地文件
  s5cmd_numworkers: 16                                # s5cmd 工作线程数
  s5cmd_concurrency: 10                               # s5cmd 并发数
  s5cmd_path: /usr/local/bin/s5cmd                    # s5cmd 路径
```

### 5.3 TrainingMetadata 结构

```python
@dataclass
class TrainingMetadata:
    consumed_train_samples: int                        # 全局已消费的样本数
    last_train_step: int                              # 最后训练步数
    last_stage_idx: Optional[int] = None              # 最后阶段的索引
    data_stages: Optional[List[DataStageMetadata]] = None    # 各阶段的元数据

@dataclass
class DataStageMetadata:
    name: str                                         # 阶段名称
    start_training_step: int                         # 阶段起始步
    consumed_train_samples: int                      # 该阶段已消费样本数
    consumed_tokens_per_dataset_folder: Dict[str, int]  # 各数据集的 token 数
```

### 5.4 CheckpointMetadata 结构

```json
{
  "version": "1.2",
  "tp": 2,                      // 保存时的 TP 大小
  "dp": 4,                      // 保存时的 DP 大小
  "metas": {
    "consumed_train_samples": 1000000,
    "last_train_step": 1000,
    "last_stage_idx": 0,
    "data_stages": [
      {
        "name": "stage_1",
        "start_training_step": 1,
        "consumed_train_samples": 1000000,
        "consumed_tokens_per_dataset_folder": {
          "/data/train": 1000000000
        }
      }
    ]
  },
  "custom_metas": null
}
```

### 5.5 文件命名规范

| 类型 | 命名格式 | 示例 |
|------|----------|------|
| 模型权重 | `model_{name}_pp-rank-{p}-of-{P}_tp-rank-{t}-of-{T}.safetensors` | `model_qkv_proj_weight_pp-rank-0-of-2_tp-rank-0-of-4.safetensors` |
| 优化器 (ZeRO-0) | `optimizer_pp-{p}-of-{P}_tp-{t}-of-{T}_exp-{e}-of-{E}.pt` | `optimizer_pp-0-of-2_tp-0-of-4_exp-0-of-1.pt` |
| 优化器 (ZeRO-1) | `optimizer_pp-{p}-of-{P}_dp-{d}-of-{D}_tp-{t}-of-{T}.pt` | `optimizer_pp-0-of-2_dp-0-of-4_tp-0-of-1.pt` |
| 学习率调度器 (ZeRO-0) | `lr_scheduler_pp-{p}-of-{P}_tp-{t}-of-{T}.pt` | `lr_scheduler_pp-0-of-2_tp-0-of-4.pt` |
| 学习率调度器 (ZeRO-1) | `lr_scheduler_pp-{p}-of-{P}_dp-{d}-of-{D}_tp-{t}-of-{T}.pt` | `lr_scheduler_pp-0-of-2_dp-0-of-4_tp-0-of-1.pt` |
| 随机状态 | `tp-{t}-of-{T}_dp-{d}-of-{D}_pp-{p}-of-{P}.pt` | `tp-0-of-4_dp-0-of-2_pp-0-of-2.pt` |

---

## 6. 常见问题 (FAQ)

### 6.1 检查点保存问题

**Q1: 如何设置合适的 checkpoint_interval？**

A: 建议：
- **开发/调试**: 100-500 steps（频繁保存以防失败）
- **短训练 (<10K steps)**: 1000 steps
- **长训练 (100K+ steps)**: 2000-5000 steps
- **考虑因素**: 
  - 保存时间（与模型大小、存储速度相关）
  - 训练失败的概率和可接受的重新训练成本
  - 存储空间限制

**Q2: 检查点太大，存储空间不足？**

A: 解决方案：
- 启用 S3 上传并设置 `remove_after_upload: true`
- 限制保留的检查点数量（手动清理或使用脚本）
- 只保存模型权重（跳过优化器）：`should_save_optimizer: false`（不推荐用于恢复训练）

**Q3: 保存时遇到 IO 错误？**

A: 检查：
- 磁盘空间是否充足
- 文件系统权限是否正确
- 网络存储（如 NFS）是否稳定
- 如果是 S3：检查 `s5cmd` 配置和网络连接

### 6.2 检查点加载问题

**Q4: 能否在不同并行配置下加载检查点？**

A: **可以！** Nanotron 支持拓扑无关的加载：
- 模型权重：自动合并和重新分片
- 优化器状态：支持 ZeRO-0 和 ZeRO-1 的跨配置加载
- 需要 `param_shard_metadata` 来正确映射参数

示例：从 TP=4,PP=2 加载到 TP=2,PP=4：
```yaml
checkpoints:
  resume_checkpoint_path: /path/to/old_checkpoint  # TP=4,PP=2

parallelism:
  tp: 2  # 新配置
  pp: 4  # 新配置
```

**Q5: 加载后学习率调度器状态丢失？**

A: 确保 `load_lr_scheduler: true`，且 LambdaLR 的 `lr_lambdas` 数量与优化器参数组匹配。

**Q6: 从检查点恢复后随机性不一致？**

A: 确保：
- `load_random_states` 被调用
- 所有 worker 加载各自 rank 的随机状态文件
- 检查 `DataStageMetadata` 以恢复正确的数据消费位置

### 6.3 HuggingFace 转换问题

**Q7: 如何转换为 HuggingFace 格式？**

```bash
# Llama 模型转换示例
torchrun --nproc_per_node=1 \
  examples/llama/convert_nanotron_to_hf.py \
  --checkpoint_path=/path/to/nanotron/ckpt \
  --save_path=/path/to/hf/model \
  --tokenizer_name=meta-llama/Llama-2-7b-chat-hf \
  --config_cls=LlamaConfig
```

**Q8: QKV 权重格式不同？**

A: Nanotron 使用合并的 QKV 权重（`qkv_proj`），HuggingFace 使用分离的 Q、K、V。转换时：
- Nanotron: `[hidden_size * 3, hidden_size]`
- HF: 分开 `q_proj`, `k_proj`, `v_proj`，各 `[hidden_size, hidden_size]`

转换脚本中的 `_handle_attention_block` 函数处理此映射。

**Q9: Flash Attention 的旋转位置编码格式不同？**

A: 处理策略：
```python
# Nanotron (GPT-J style - 交错)
# [d_0, d_2, d_4, ..., d_1, d_3, d_5, ...]

# HuggingFace (GPT-NeoX style - 半半)
# [d_0, d_1, d_2, ..., d_{n/2}, ..., d_n]

# 转换时重排
if interleave_qkv:
    # 从 Nanotron 交错格式拆分到 HF 半半格式
    pass
```

### 6.4 拓扑变化加载

**Q10: 改变 TP size 后加载失败？**

A: 常见原因：
- 参数名称不匹配（检查 `module_id_to_prefix`）
- `param_shard_metadata` 未正确传递
- 检查点版本太旧（<1.2），缺少 `TensorMetadata`

**Q11: 优化器状态加载后不一致？**

A: 确保：
1. 模型权重先加载（生成正确的分片元数据）
2. `param_shard_metadata` 传递给 `load_optimizer()`
3. 使用拓扑无关加载逻辑（自动触发当 TP/PP 大小不匹配）

### 6.5 S3 集成问题

**Q12: S3 下载慢或失败？**

A: 优化建议：
- 增加 `s5cmd_numworkers` 和 `s5cmd_concurrency`
- 使用 `s5cmd` 替代 `aws s3`，性能更好
- 检查 region 配置，使用最近的 S3 endpoint
- 启用 S3 的 Transfer Acceleration

**Q13: 本地和 S3 检查点如何同步？**

A: 配置建议：
```yaml
checkpoints:
  checkpoints_path: /local/checkpoints            # 本地保存路径
  resume_checkpoint_path: s3://bucket/checkpoints # S3 加载路径

s3_upload:
  upload_s3_path: s3://bucket/checkpoints         # S3 上传目标
```

---

## 7. 参考资料

### 7.1 核心文件位置

```
src/nanotron/serialize/
├── __init__.py                           # 模块导出
├── main.py                              # save(), parse_ckpt_path()
├── weights.py                           # save_weights(), load_weights()
├── optimizer.py                         # save_optimizer(), load_optimizer()
├── metadata.py                          # TrainingMetadata, CheckpointMetadata, TensorMetadata
├── random.py                            # save_random_states(), load_random_states()
└── utils.py                             # merge_and_shard_tp_tensors(), get_path(), ObjectType

src/nanotron/
├── s3_checkpoints.py                    # S3Mover 类
└── sanity_checks.py                     # 加载/保存的验证函数

examples/
├── llama/convert_nanotron_to_hf.py      # Llama HF 转换
├── llama/convert_hf_to_nanotron.py      # Llama 反向转换
├── mamba/convert_nanotron_to_hf.py      # Mamba HF 转换
└── config_resume_training.py            # 恢复训练配置示例
```

### 7.2 相关库和格式

1. **safetensors**
   - 文档: https://huggingface.co/docs/safetensors/
   - 优点：安全的 PyTorch 替代格式，支持元数据，加载速度快

2. **s5cmd**
   - GitHub: https://github.com/ Peak/s5cmd
   - 功能：高性能 S3 命令行工具，用于 checkpoint S3 上传/下载

3. **dacite**
   - GitHub: https://github.com/konradhalas/dacite
   - 功能：将字典转换为 Python dataclass，用于配置解析

### 7.3 与其他框架对比

| 特性 | Nanotron | DeepSpeed | Megatron-LM | FSDP |
|------|----------|-----------|-------------|------|
| safetensors 格式 | 原生支持 | 否 (torch) | 否 (torch) | 否 (torch) |
| 拓扑无关加载 | 原生支持 | 有限支持 | 有限支持 | 有限支持 |
| S3 集成 | 内置 | 需要额外配置 | 需要额外配置 | 需要额外配置 |
| 检查点版本管理 | 内置 | 无 | 无 | 无 |
| DataStage 支持 | 内置 | 无 | 无 | 无 |
| HF 转换脚本 | 提供 | 需要额外工具 | 需要额外工具 | 需要额外工具 |
| 存储格式 | 分片 (TP/PP/DP) | ZeRO 分片 | 分片 | FSDP 分片 |
| 精度支持 | FP16/BF32/FP32 | FP16/FP32/Int8 | BF16/FP16/FP32 | BF16/FP32 |

### 7.4 设计决策说明

1. **为什么使用 safetensors？**
   - 安全性：防止 pickle 反序列化攻击
   - 效率：支持延迟加载和元数据读取
   - 跨语言：可被 Python 之外的生态系统使用

2. **为什么分片存储？**
   - 分布式训练：每个 rank 只负责自己的分片
   - 并行 I/O：多 rank 同时写入
   - 灵活加载：可选择性加载特定分片

3. **为什么需要 TensorMetadata？**
   - 拓扑变化：记录分片方式以支持重新组合
   - 版本控制：检查点格式演进
   - 调试：可视化参数分片方案

4. **为什么优化器单独存储？**
   - 模型部署：通常不需要优化器状态
   - 存储效率：避免重复存储 tied 参数的状态
   - ZeRO 优化：与分片策略配合

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-04-20 | 初始版本，基于 nanotron-main 代码 |
