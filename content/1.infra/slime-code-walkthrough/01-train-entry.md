---
title: "Slime 代码走读 — 01. 训练入口与 Ray 调度层"
date: 2026-06-15T12:00:00+08:00
draft: false
---

> 本文档从 `train.py` 入口出发，逐层解析 Ray Placement Group 分配、RolloutManager 创建、Training Model 初始化 的完整调用链。

---

## 一、入口文件：`train.py`

```python
# slime/train.py
import ray
from slime.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking

def train(args):
    configure_logger()
    pgs = create_placement_groups(args)           # ① 分配 GPU
    init_tracking(args)

    # ② 先创建 RolloutManager（里面需要 SGLang engines）
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # ③ 再创建 Actor 和 Critic 模型
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # ④ 初始权重同步：Megatron 权重 → SGLang
    if not args.critic_train_only:
        actor_model.update_weights()

    # ⑤ 主训练循环
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval and rollout_id == 0:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # 保存 + eval + weight sync ...

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)
```

**核心动作分解**：

| 动作 | 代码 | 说明 |
|------|------|------|
| GPU 分配 | `create_placement_groups()` | Ray Placement Group，按角色绑定 GPU bundle |
| Rollout 初始化 | `create_rollout_manager()` | 启动 SGLang engines、router |
| 训练初始化 | `create_training_models()` | SPMD 启动 Megatron 训练进程 |
| 权重同步 | `actor_model.update_weights()` | Megatron 权重 → SGLang（首次） |
| 生成 | `rollout_manager.generate()` | 用 SGLang 生成 response |
| 训练 | `actor_model.async_train()` | Megatron 训练（fwd+bwd+opt.step） |
| 评测 | `rollout_manager.eval()` | AIME2024 / 自定义评测 |

---

## 二、Placement Group 分配：`slime/ray/placement_group.py`

### 2.1 `create_placement_groups()`

```python
def create_placement_groups(args):
    pg = PlacementGroup([{"GPU": args.num_gpus_per_node}] * args.num_nodes)
    
    # reordered_bundle_indices + reordered_gpu_ids: PG 内 GPU 重排序
    # 原因：多节点时节点间通信带宽不同，需要优化 bundle 排序
    
    # 创建 RolloutManager 的 PG
    # 创建 Train 的 PG（actor + critic）
    return {
        "rollout": (pg, reordered_bundle_indices, reordered_gpu_ids),
        "actor_train": (pg, reordered_bundle_indices, reordered_gpu_ids),
        # critic_train optional
    }
```

**关键理解**：

- **一个 Ray Placement Group** 横跨多个节点的 GPU，扮演 `Bundle`（`{"GPU": 8}` 表示一节点 8 GPU）
- `reordered_bundle_indices` 和 `reordered_gpu_ids` 是**人工重排序后的 GPU 映射**——因为实际集群中节点间 NVLink/IB 拓扑不是线性的，通过重排序让通信密集的角色相邻
- 当 `--colocate` 为 True 时，rollout 和 train 共用同一个 Placement Group；否则分开

### 2.2 `create_rollout_manager()`

```python
def create_rollout_manager(args, pg):
    RolloutManager = ray.remote(RolloutManagerClass)
    rollout_manager = RolloutManager.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg[0],
            placement_group_bundle_index=0,  # 在第一个 bundle 上
        )
    ).remote(args, pg)
    
    # 计算每轮 rollout 需要循环多少 epoch（global dataset 模式下）
    if args.rollout_global_dataset:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
    else:
        num_rollout_per_epoch = args.num_rollout
    
    return rollout_manager, num_rollout_per_epoch
```

**RolloutManager** 是一个 Ray @ray.remote 类，它的 `__init__` 中：
1. 调用 `init_http_client()` 初始化共享 HTTP 客户端
2. 调用 `start_rollout_servers(args, pg)` 创建所有 SGLang engines + router
3. 如果启用故障恢复，启动 `RolloutHealthMonitor`

### 2.3 `create_training_models()`

```python
def create_training_models(args, pgs, rollout_manager):
    # Actor
    actor_model = RayTrainGroup(
        actor_train_pgs,
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        backend="megatron",
        role="actor",
        with_ref=args.use_kl_loss,
    )
    
    # Critic（optional，用于 PPO）
    if args.use_critic:
        critic_model = RayTrainGroup(..., role="critic")
    
    # 设置 rollout_manager 引用，让训练进程知道怎么访问引擎
    actor_model.set_rollout_manager(rollout_manager)
    
    return actor_model, critic_model
```

---

## 三、`RayTrainGroup`：`slime/ray/actor_group.py`

```python
class RayTrainGroup:
    """管理一组同角色的 TrainRayActor（SPMD 模式）"""
    def __init__(self, pgs, num_nodes, num_gpus_per_node, backend, role, ...):
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(...),
            ).remote(world_size, rank, master_addr, master_port)
            self._actor_handlers.append(actor)

    def async_train(self, rollout_id, rollout_data_ref):
        return [actor.train.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]

    def update_weights(self):
        return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
```

### SPMD 启动逻辑

| 步骤 | 说明 |
|------|------|
| `__init__` | 为每个 rank 创建一个 `TrainRayActor.remote(...)` |
| `set_rollout_manager()` | rank=0 的 actor 和 RolloutManager 建立反向引用 |
| `init()` | 调用 `init_process_group(backend="nccl")`，完成 PyTorch 分布式初始化 |
| `init(args, role, ...)` | 调用 Megatron 初始化（`init()` + `initialize_model_and_optimizer()`） |

**关键**：每个 `TrainRayActor` 是一个独立的 Python 进程，内部有自己的 `torch.distributed` 进程组。它们的 `MASTER_ADDR`/`MASTER_PORT` 是 actor_group 分配的，形成一个独立的 Megatron 并行世界。

---

## 四、`TrainRayActor`：`slime/ray/train_actor.py`

```python
class TrainRayActor(abc.ABC):
    def __init__(self, world_size, rank, master_addr, master_port):
        # 设置环境变量
        os.environ["MASTER_ADDR"] = master_addr or self._get_current_node_ip_and_free_port()
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        torch.cuda.set_device(f"cuda:{local_rank}")
        dist.init_process_group(backend, timeout=timedelta(minutes=args.distributed_timeout_minutes))
        init_gloo_group()  # GLOO 用于 DP gather_object / Python 对象通信

        # NUMA affinity（仅限 NVIDIA）
        pynvml.nvmlDeviceSetCpuAffinity(handle)
```

**MegatronTrainRayActor**（继承自 `TrainRayActor`）在 `init()` 中进一步调用：

```python
# slime/backends/megatron_utils/actor.py
class MegatronTrainRayActor(TrainRayActor):
    def init(self, args, role, with_ref, with_opd_teacher):
        monkey_patch_torch_dist()  # 替换 NCCL 库，支持进程组可重载
        super().init(...)
        init(args)  # Megatron initialize_distributed() + init()
        
        # 读取 HF config（每个 rank 串行读，避免竞争）
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(...)
                self.tokenizer = AutoTokenizer.from_pretrained(...)
            dist.barrier(group=get_gloo_group())
        
        # 初始化模型 + optimizer + scheduler + 加载 checkpoint
        self.model, self.optimizer, self.opt_param_scheduler, start_rollout_id = \
            initialize_model_and_optimizer(args, role)
        
        # 初始化权重备份系统（用于 ref/old_actor/teacher 切换）
        self.weights_backuper = TensorBackuper.create(...)
        
        # 初始化权重同步器
        update_weight_cls = UpdateWeightFromTensor if args.colocate else UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(...)
```

---

## 五、调用链总结

```
train.py
  ├── parse_args()              [slime/utils/arguments.py: ~1777 行的巨型参数系统]
  ├── create_placement_groups() [slime/ray/placement_group.py]
  │     └── PlacementGroup
  ├── create_rollout_manager()  [slime/ray/placement_group.py]
  │     └── RolloutManager.__init__()
  │           └── start_rollout_servers()
  │                 ├── _resolve_sglang_config()
  │                 ├── _start_router()
  │                 ├── _make_group() → ServerGroup
  │                 └── group.start_engines() → SGLangEngine Ray Actor
  ├── create_training_models()  [slime/ray/placement_group.py]
  │     └── RayTrainGroup.__init__()
  │           └── for rank: TrainRayActor.remote() → init() → Megatron 初始化
  ├── rollout_manager.generate()  [slime/ray/rollout.py:RolloutManager]
  │     └── _get_rollout_data()
  │           └── generate_rollout() [slime/rollout/sglang_rollout.py]
  └── actor_model.async_train()   [slime/ray/actor_group.py]
        └── for actor: actor.train.remote()
              └── MegatronTrainRayActor.train()
                    └── train_actor() / train_critic()
```

---

## 六、关键设计洞察

1. **SPMD 模式**：Ray 创建 N 个 `MegatronTrainRayActor`，每个有自己的 `RANK`，但不是 Ray 的 Actor 间通信做并行——它们内部用 NCCL。Ray 只负责生命周期管理和跨角色通信。

2. **Colocate vs 分离**：
   - Colocate = rollout 和 train 同 PG，`colocate=True` 时 `needs_offload=True`，需要显存切换
   - 分离 = rollout 和 train 不同 PG，GPU 不重叠，weight sync 走网络/PCIe

3. **为什么先 init rollout 再 init train**：因为 `rollout_manager` 被传给 `actor_model.set_rollout_manager()`，训练进程需要知道 SGLang 引擎地址来做 weight update

4. **`rollout_data_ref` 是 `ray.put()` 的对象**：`RolloutManager.generate()` 返回的数据被 `ray.put()` 放入共享内存，训练 Actor 通过 `process_rollout_data()` 从 Ray Object Store 取出来
