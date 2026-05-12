# Single Controller 模块

**路径**：`verl/single_controller/`

single_controller 是 verl 的**分布式通信抽象层**，实现了"单控制器多 Worker"的计算模型：Driver 进程持有所有控制流，Worker 只暴露远程可调用的计算方法，Driver 通过 RPC 调用 Worker。

---

## 1. 设计哲学

### 传统框架 vs verl

| 维度 | 传统 RL 框架（如 OpenRLHF） | verl |
|------|--------------------------|------|
| 控制流 | 分散在每个 Worker | 集中在 Driver |
| 算法修改 | 需要修改多个 Worker | 只修改 Driver（`ray_trainer.py`）|
| 数据路由 | Worker 间点对点通信 | Driver 汇总后再分发 |
| 扩展性 | Worker 数量变化需改代码 | ResourcePool 自动扩展 |

核心优势：把**算法逻辑**（如 compute_advantage、apply_kl_penalty）留在 Driver，**计算密集型操作**（前向传播、反向传播、推理生成）交给 Worker，实现了清晰的分工。

---

## 2. 目录结构

```
single_controller/
├── base/
│   ├── worker.py        # Worker 基类
│   ├── worker_group.py  # WorkerGroup、ResourcePool 基类
│   └── decorator.py     # @register 装饰器、Dispatch/Execute 枚举
└── ray/
    └── base.py          # RayWorkerGroup（Ray 实现）
```

---

## 3. `ResourcePool`

**文件**：`verl/single_controller/base/worker_group.py`

ResourcePool 描述了一个 Ray 集群中的资源分配方案，记录每个节点上的进程数：

```python
class ResourcePool:
    def __init__(
        self,
        process_on_nodes: list[int],  # 每个节点上的进程数，如 [8, 8]（2 节点各 8 进程）
        max_colocate_count: int = 10, # 同一 GPU 上最多可共存的 Worker 数
        n_gpus_per_node: int = 8,
    ):
        ...

    @property
    def world_size(self) -> int:
        return sum(self._store)  # 总进程数（= 总 GPU 数）

    def local_world_size_list(self) -> list[int]:
        """返回每个进程的 local_world_size，展开后长度 == world_size"""

    def local_rank_list(self) -> list[int]:
        """返回每个进程的 local_rank，展开后长度 == world_size"""
```

**ResourcePoolManager**：管理多个 ResourcePool，支持不同 Role 使用不同 Pool：

```python
# 示例：Actor 和 Critic 使用同一个 Pool（共享 GPU）
manager = ResourcePoolManager(
    resource_pool_spec={
        "actor_rollout_ref": ResourcePool([8, 8]),  # 16 个 GPU
        "critic": ResourcePool([4]),                 # 4 个 GPU
    },
    mapping={
        Role.ActorRollout: "actor_rollout_ref",
        Role.Critic: "critic",
    }
)
```

---

## 4. Dispatch 和 Execute 模式

**文件**：`verl/single_controller/base/decorator.py`

这两个枚举控制 Driver 如何将数据分发给 Worker 以及如何聚合结果。

### 4.1 Dispatch 模式（数据如何分发到各 Rank）

| 模式 | 语义 |
|------|------|
| `RANK_ZERO` | 只发给 rank 0，其余不接收 |
| `ONE_TO_ALL` | 把相同数据广播给所有 rank |
| `ALL_TO_ALL` | 将 batch 切分后，每个 rank 拿到一份（数据并行）|
| `DP_COMPUTE` | 数据并行计算，自动按 DP 维度切分 |
| `DP_COMPUTE_PROTO` | 同上，但数据封装为 `DataProto`，支持自动 padding |
| `DP_COMPUTE_PROTO_WITH_FUNC` | 同上，附带函数参数 |
| `DP_COMPUTE_METRIC` | 数据并行计算，聚合时 reduce metrics |
| `DIRECT_ROLLOUT_METHOD` | vLLM ExternalRayDistributedExecutor 专用 |

### 4.2 Execute 模式（哪些 Rank 执行）

| 模式 | 语义 |
|------|------|
| `ALL` | 所有 rank 都执行 |
| `RANK_ZERO` | 只有 rank 0 执行 |

### 4.3 `@register` 装饰器

Worker 方法通过 `@register(dispatch_mode, execute_mode)` 标记，WorkerGroup 自动生成对应的远程调用代理：

```python
from verl.single_controller.base.decorator import register, Dispatch, Execute

class MyWorker(Worker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, execute_mode=Execute.ALL)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """这个方法会被自动分发到所有 Worker，输入按 DP 维度切分"""
        ...

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO)
    def save_checkpoint(self, path: str):
        """只在 rank 0 执行"""
        ...
```

---

## 5. `WorkerGroup` 基类

**文件**：`verl/single_controller/base/worker_group.py`

WorkerGroup 是 Driver 持有的 Worker 代理，所有方法调用都会路由到实际的分布式 Worker：

```python
class WorkerGroup:
    """对外暴露与 Worker 相同的接口，但内部通过 RPC 调用远端 Worker"""

    def _bind_methods(self, worker_cls):
        """扫描 worker_cls 上所有被 @register 标记的方法，
        为 WorkerGroup 生成对应的代理方法"""
        for method_name in dir(worker_cls):
            if hasattr(method, MAGIC_ATTR):  # 有 @register 标记
                dispatch_fn = get_predefined_dispatch_fn(dispatch_mode)
                execute_fn  = get_predefined_execute_fn(execute_mode)
                # 生成代理方法：自动处理数据切分、RPC 调用、结果聚合
```

**`_split_args_kwargs_data_proto`**：将 `DataProto` 按 `chunks` 切分，均匀分配给各 Worker。

---

## 6. `RayWorkerGroup`

**文件**：`verl/single_controller/ray/base.py`

Ray 后端的 WorkerGroup 实现，将 Worker 方法的调用转换为 `ray.remote()` 异步调用：

```python
class RayWorkerGroup(WorkerGroup):
    def __init__(
        self,
        resource_pool: ResourcePool,
        ray_cls_with_init: RayClassWithInitArgs,  # 包含初始化参数的 Ray actor 类
        device_name: str = "cuda",
    ):
        # 为每个 GPU 创建一个 Ray actor（Worker 实例）
        self._workers: list[ray.actor.ActorHandle] = [
            ray_cls.remote(...) for _ in range(world_size)
        ]

    def execute_all_async(self, method_name, *args, **kwargs):
        """向所有 Worker 发起异步 RPC，返回 future 列表"""
        futures = [w.method_name.remote(*args) for w in self._workers]
        return futures

    def execute_all(self, method_name, *args, **kwargs):
        """同步版本，等待所有 Worker 完成并返回结果"""
        return ray.get(self.execute_all_async(...))
```

### 共置 Worker（Colocated Worker）

当多个 Role（如 ActorRollout + RefPolicy）共享同一组 GPU 时，使用 `create_colocated_worker_cls` 将它们合并为一个 Ray actor：

```python
def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    将多个 Role 的 Worker 类封装成一个 ColocatedWorker 类。
    每次调用 ColocatedWorker.method() 时，会路由到对应 Role 的内部 Worker 上。
    """
    class ColocatedWorker:
        def __init__(self):
            for role, cls_with_args in class_dict.items():
                setattr(self, role, cls_with_args.cls(*cls_with_args.args))
```

---

## 7. `RayClassWithInitArgs`

包装类，延迟实例化 Ray actor（等 Worker 被分配到 GPU 节点才创建）：

```python
class RayClassWithInitArgs:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def remote(self, *extra_args, **extra_kwargs):
        """真正创建 Ray remote actor"""
        return self.cls.remote(*self.args, *extra_args, **self.kwargs, **extra_kwargs)
```

---

## 8. 数据流示意

```
Driver（RayPPOTrainer）
    │
    │  actor_rollout_wg.generate_sequences(batch)
    │         │
    │         ▼  RPC（Ray remote call）
    │   ┌─────────────────────────────────────────┐
    │   │  RayWorkerGroup._dispatch(batch)         │
    │   │    → 将 batch 按 DP 切分为 N 份           │
    │   │    → 并发 ray.remote 调用 N 个 Worker     │
    │   │    → ray.get() 等待所有结果               │
    │   │    → 拼合结果 DataProto                  │
    │   └─────────────────────────────────────────┘
    │         │
    │         ▼
    │   批量结果 DataProto 返回给 Driver
```

---

## 9. 异步调用支持

WorkerGroup 也支持异步调用模式，Driver 发出 RPC 后继续执行其他逻辑，在需要结果时再同步：

```python
# 发起异步调用
future = wg.generate_sequences_async(batch)

# 在等待的间隙做其他工作
do_something_else()

# 获取结果（阻塞）
result = ray.get(future)
```

这是 `experimental/fully_async_policy` 的基础，实现 rollout 和 update 的流水线并行。
