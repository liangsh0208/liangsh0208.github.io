---
created: 2026-05-06
---

# Checkpoint Engine 模块

**路径**：`verl/checkpoint_engine/`

Checkpoint Engine 负责在训练过程中高效地保存模型状态（模型权重 + 优化器状态）以及将更新后的模型权重同步给 Rollout Worker（vLLM/SGLang）。

---

## 1. 设计目标

传统的 `torch.save()` 在大模型场景下效率低下：
- 需要在 rank 0 聚合全量权重，造成内存瓶颈
- 磁盘 I/O 是训练的主要阻塞点

verl 的 Checkpoint Engine 解决了两个核心问题：
1. **分布式保存**：每个 rank 保存自己的分片，并行写入，无需聚合
2. **高速权重同步**：训练完成后快速将权重传给 Rollout Worker

---

## 2. 抽象接口

**文件**：`verl/checkpoint_engine/base.py`

```python
class CheckpointEngine(ABC):
    """检查点引擎抽象基类"""

    @abstractmethod
    def save_checkpoint(
        self,
        model,
        optimizer,
        lr_scheduler,
        path: str,
        global_step: int,
    ):
        """保存训练状态到指定路径"""

    @abstractmethod
    def load_checkpoint(
        self,
        model,
        optimizer,
        lr_scheduler,
        path: str,
    ) -> int:
        """从路径加载训练状态，返回恢复的 global_step"""

    @abstractmethod
    def update_weights(
        self,
        model,
        rollout_engine,
        global_step: int,
    ):
        """将模型权重同步给 Rollout Worker（每步训练后调用）"""
```

**注册机制**：

```python
class CheckpointEngineRegistry:
    _registry: dict[str, type[CheckpointEngine]] = {}

    @staticmethod
    def register(backend: str):
        def wrapper(cls):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls
        return wrapper

    @staticmethod
    def get(backend: str) -> type[CheckpointEngine]:
        return CheckpointEngineRegistry._registry[backend]
```

---

## 3. `NCCLCheckpointEngine`

**文件**：`verl/checkpoint_engine/nccl_checkpoint_engine.py`

基于 NCCL 的权重同步引擎，是最常用的默认实现（适用于 NVIDIA GPU 集群）。

### 3.1 权重同步流程

```
训练完成（Actor FSDP 分片权重）
        │
        ▼
step 1: FSDP all-gather 收集完整权重（每个 rank 都得到完整模型）
        │
        ▼
step 2: 将权重传输给同节点的 vLLM 引擎
        │  ├─ 同节点：CUDA IPC（零拷贝，最快）
        │  └─ 跨节点：NCCL broadcast
        ▼
vLLM 引擎调用 update_weights() 加载新权重
        │
        ▼
释放临时聚合的权重，FSDP 恢复分片状态
```

### 3.2 关键实现

```python
@register("nccl")
class NCCLCheckpointEngine(CheckpointEngine):

    def update_weights(self, model: FSDP, rollout_engine, global_step: int):
        """将 FSDP 训练后的权重同步给 vLLM"""

        # 1. FSDP unshard：将分片参数合并为完整参数
        with FSDP.summon_full_params(model, writeback=False):
            # 2. 将完整权重打包
            state_dict = {}
            for name, param in model.named_parameters():
                state_dict[name] = param.data

        # 3. 通知 vLLM 更新权重
        rollout_engine.update_weights(state_dict, global_step)

    def save_checkpoint(self, model, optimizer, lr_scheduler, path, global_step):
        """分布式保存，每个 rank 保存自己的 FSDP 分片"""
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,  # 每个 rank 保存自己的分片
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            state_dict = model.state_dict()

        # rank 0 保存优化器状态和超参数
        if dist.get_rank() == 0:
            torch.save({
                "global_step": global_step,
                "lr_scheduler": lr_scheduler.state_dict(),
            }, f"{path}/meta.pt")

        # 所有 rank 并行保存分片
        torch.save(state_dict, f"{path}/model_rank{dist.get_rank()}.pt")
```

---

## 4. `NIXLCheckpointEngine`

**文件**：`verl/checkpoint_engine/nixl_checkpoint_engine.py`

NIXL（NVIDIA Interconnect eXtensions Library）是 NVIDIA 的高性能传输库，支持：
- GPU Direct RDMA：绕过 CPU，直接在 GPU 之间传输数据
- NVLink / InfiniBand：利用高速互连网络
- 异步传输：传输与计算 overlap

```python
@register("nixl")
class NIXLCheckpointEngine(CheckpointEngine):
    """基于 NIXL 的高速权重传输"""

    def update_weights(self, model, rollout_engine, global_step):
        # 注册 GPU 内存到 NIXL
        nixl_handle = nixl.register_memory(model.parameters())

        # 发起异步传输（GPU Direct RDMA）
        transfer_future = nixl.transfer_async(
            src=nixl_handle,
            dst=rollout_engine.nixl_handle,
        )

        # 继续计算（overlap 传输与下一 batch 的 rollout 启动）
        # ...

        # 等待传输完成
        nixl.wait(transfer_future)
```

---

## 5. `MooncakeCheckpointEngine`

**文件**：`verl/checkpoint_engine/mooncake_checkpoint_engine.py`

Mooncake 是字节跳动开源的分布式 KV 存储系统，适合超大模型的分布式保存：

```python
@register("mooncake")
class MooncakeCheckpointEngine(CheckpointEngine):
    """基于 Mooncake 分布式存储的检查点引擎"""

    def save_checkpoint(self, model, optimizer, path, global_step):
        # 每个 rank 将自己的分片直接写入 Mooncake
        client = MooncakeClient(self.server_url)
        shard = get_local_shard(model)
        client.put(
            key=f"ckpt/{global_step}/rank{dist.get_rank()}",
            value=shard,
        )

    def load_checkpoint(self, model, path):
        client = MooncakeClient(self.server_url)
        shard = client.get(
            key=f"ckpt/latest/rank{dist.get_rank()}"
        )
        load_shard_to_model(model, shard)
```

---

## 6. `HCCLCheckpointEngine`

**文件**：`verl/checkpoint_engine/hccl_checkpoint_engine.py`

华为昇腾 NPU 环境下使用 HCCL（Huawei Collective Communication Library）代替 NCCL，接口与 `NCCLCheckpointEngine` 完全相同。

---

## 7. `KimiCheckpointEngine`

**文件**：`verl/checkpoint_engine/kimi_checkpoint_engine.py`

字节跳动内部存储系统集成，适配大规模集群的检查点需求。

---

## 8. CheckpointManager

**文件**：`verl/utils/checkpoint/checkpoint_manager.py`

`CheckpointManager` 是 `RayPPOTrainer` 使用的高层封装，协调保存检查点和更新权重：

```python
class CheckpointManager:
    """协调保存、加载、权重同步的统一接口"""

    def update_weights(self, global_step: int):
        """每步训练后调用：将新权重同步给 Rollout Worker"""
        self.checkpoint_engine.update_weights(
            model=self.actor_engine.model,
            rollout_engine=self.rollout_worker,
            global_step=global_step,
        )

    def save_checkpoint(self, global_step: int, path: str = None):
        """保存训练状态"""
        if path is None:
            path = f"{self.save_dir}/step_{global_step}"
        self.checkpoint_engine.save_checkpoint(
            model=self.actor_engine.model,
            optimizer=self.actor_engine.optimizer,
            path=path,
            global_step=global_step,
        )

    def sleep_replicas(self):
        """让 Rollout Worker 的副本进入睡眠（释放显存给训练用）"""
        for replica in self.rollout_replicas:
            replica.sleep()

    def find_latest_checkpoint(self):
        """自动找到最新的检查点路径"""
        return find_latest_ckpt_path(self.save_dir)
```

---

## 9. ESI 检查点（Elastic Server Instance）

verl 支持弹性云实例（ESI）场景，在实例即将被回收前自动保存检查点：

```python
def should_save_ckpt_esi(max_steps_duration, redundant_time) -> bool:
    """检查 ESI 实例是否即将到期，决定是否强制保存检查点"""
    remaining_time = get_esi_remaining_time()
    # 如果剩余时间不足以再跑一步 + 冗余时间，立刻保存
    return remaining_time < max_steps_duration + redundant_time
```

---

## 10. 检查点保存策略

```yaml
trainer:
  save_freq: 100          # 每 100 步保存一次
  esi_redundant_time: 60  # ESI 实例预留 60 秒冗余

# 触发保存的条件（满足任一即保存）：
# 1. global_step % save_freq == 0
# 2. is_last_step（最后一步）
# 3. ESI 实例即将到期
```
