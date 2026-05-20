---
created: 2026-05-06
---

# Slime Infrastructure 分析文档

> 分析路径: `/Users/danchen/Documents/RL_fw/slime/slime`
> 分析日期: 2026-03-14
> 分析目标: 基础设施架构、调用逻辑、组件关系

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构](#2-目录结构)
3. [核心组件详解](#3-核心组件详解)
4. [完整调用流程](#4-完整调用流程)
5. [数据流分析](#5-数据流分析)
6. [分布式架构](#6-分布式架构)
7. [关键代码路径](#7-关键代码路径)
8. [配置与参数](#8-配置与参数)

---

## 1. 项目概述

**Slime** 是一个大规模**强化学习训练框架**，专门用于大语言模型的 **RLHF (Reinforcement Learning from Human Feedback)** 训练。

### 1.1 核心依赖

| 依赖库 | 用途 |
|--------|------|
| **Megatron-LM** | 分布式模型训练后端 |
| **SGLang** | 高效模型推理/生成引擎 |
| **Ray** | 分布式Actor协调框架 |
| **PyTorch** | 深度学习框架 |

### 1.2 支持的算法

| 算法 | 文件位置 | 描述 |
|------|----------|------|
| PPO | `loss.py`, `ppo_utils.py` | Proximal Policy Optimization |
| GRPO | `loss.py:448-452` | Group Relative Policy Optimization |
| GSPO | `loss.py`, `ppo_utils.py:95-121` | Group Sequence Policy Optimization |
| REINFORCE++ | `ppo_utils.py:211-278` | 改进的REINFORCE算法 |
| REINFORCE++-baseline | `ppo_utils.py:281-308` | 带基线的REINFORCE++ |

---

## 2. 目录结构

```
slime/
├── backends/                    # 后端实现
│   ├── megatron_utils/          # Megatron训练后端
│   │   ├── actor.py             # [核心] 训练Actor实现
│   │   ├── model.py             # [核心] 模型设置、训练循环
│   │   ├── loss.py              # [核心] PPO/GRPO等损失函数
│   │   ├── data.py              # 数据迭代器
│   │   ├── checkpoint.py        # 检查点保存/加载
│   │   ├── initialize.py        # Megatron初始化
│   │   ├── cp_utils.py          # Context Parallel工具
│   │   ├── megatron_to_hf/      # Megatron→HuggingFace转换
│   │   │   ├── llama.py         # LLaMA模型转换
│   │   │   ├── qwen2.py         # Qwen2模型转换
│   │   │   ├── qwen3moe.py      # Qwen3 MoE转换
│   │   │   └── ...              # 其他模型
│   │   ├── update_weight/       # 权重更新机制
│   │   │   ├── update_weight_from_tensor.py     # 张量直接更新
│   │   │   └── update_weight_from_distributed.py # 分布式更新
│   │   └── kernels/             # CUDA内核
│   │       ├── fp8_kernel.py    # FP8量化
│   │       └── int4_qat/        # INT4量化感知训练
│   └── sglang_utils/            # SGLang推理后端
│       ├── sglang_engine.py     # [核心] SGLang引擎封装
│       ├── sglang_config.py     # 引擎配置
│       └── arguments.py         # SGLang参数解析
│
├── ray/                         # Ray分布式组件
│   ├── ray_actor.py             # Ray Actor基类
│   ├── rollout.py               # [核心] 推理服务器管理
│   ├── train_actor.py           # 训练Actor基类
│   ├── actor_group.py           # Actor组管理
│   └── placement_group.py       # GPU放置组
│
├── rollout/                     # Rollout生成模块
│   ├── base_types.py            # 基础数据类型
│   ├── data_source.py           # 数据源管理
│   ├── sglang_rollout.py        # [核心] SGLang推理生成
│   ├── sft_rollout.py           # SFT训练rollout
│   ├── on_policy_distillation.py # 在线蒸馏
│   ├── rm_hub/                  # 奖励模型
│   │   ├── math_utils.py        # 数学验证
│   │   ├── gpqa.py              # GPQA评估
│   │   └── f1.py                # F1评估
│   ├── filter_hub/              # 样本过滤器
│   │   └── dynamic_sampling_filters.py
│   └── generate_hub/            # 生成器
│       └── benchmarkers.py      # 基准测试
│
├── router/                      # 请求路由
│   ├── router.py                # 自定义路由器
│   └── middleware_hub/          # 路由中间件
│       ├── radix_tree.py        # Radix树实现
│       └── radix_tree_middleware.py # 前缀缓存中间件
│
└── utils/                       # 工具函数
    ├── ppo_utils.py             # [核心] PPO算法实现
    ├── types.py                 # 数据类型定义
    ├── arguments.py             # 参数配置
    ├── data.py                  # 数据处理
    ├── memory_utils.py          # 内存管理
    ├── metric_utils.py          # 指标计算
    ├── distributed_utils.py     # 分布式工具
    └── timer.py                 # 计时器
```

---

## 3. 核心组件详解

### 3.1 MegatronTrainRayActor (训练节点)

**文件**: `backends/megatron_utils/actor.py`

```python
class MegatronTrainRayActor(TrainRayActor):
    """Megatron训练Ray Actor，负责模型训练"""

    # 初始化流程
    def init(self, args, role, with_ref, with_opd_teacher) -> int:
        # 1. 分布式初始化
        # 2. 加载HuggingFace配置和tokenizer
        # 3. 初始化模型、优化器、学习率调度器
        # 4. 创建权重备份器(支持多模型切换)
        # 5. 创建权重更新器

    # 训练入口
    def train(self, rollout_id, rollout_data_ref) -> None:
        # 分发到 train_actor() 或 train_critic()

    # Actor训练
    def train_actor(self, rollout_id, rollout_data) -> None:
        # 1. 获取数据迭代器
        # 2. (可选) 填充路由重放数据
        # 3. 计算参考模型log probs
        # 4. 计算当前模型log probs
        # 5. 计算advantages和returns
        # 6. 执行训练
        # 7. 备份权重

    # 权重同步到推理引擎
    def update_weights(self) -> None:
        # 将训练后的权重同步到SGLang引擎
```

### 3.2 SGLangEngine (推理节点)

**文件**: `backends/sglang_utils/sglang_engine.py`

```python
class SGLangEngine(RayActor):
    """SGLang推理引擎封装"""

    def init(self, dist_init_addr, port, nccl_port, ...):
        # 1. 计算服务参数
        # 2. 启动SGLang HTTP服务进程
        # 3. 注册到路由器

    # 权重更新API
    def update_weights_from_tensor(self, serialized_named_tensors, ...):
        # 从张量数据更新权重

    def update_weights_from_distributed(self, names, dtypes, shapes, ...):
        # 从分布式源更新权重

    # 内存管理
    def release_memory_occupation(self):
        # 释放GPU内存(训练时)

    def resume_memory_occupation(self, tags):
        # 恢复GPU内存(推理时)
```

### 3.3 RolloutServer (推理服务器集群)

**文件**: `ray/rollout.py`

```python
@dataclasses.dataclass
class ServerGroup:
    """一组同构的SGLang引擎"""
    engines: list[SGLangEngine]
    tp_size: int                   # 张量并行大小
    nodes_per_engine: int          # 每引擎节点数
    pg: PlacementGroup             # Ray放置组

class RolloutServer:
    """推理服务器管理器"""

    def __init__(self, args):
        # 创建多个ServerGroup
        # 配置负载均衡

    async def generate(self, samples) -> list[Sample]:
        # 分发生成请求

    def update_weights(self, weight_updater):
        # 批量更新所有引擎权重
```

### 3.4 SlimeRouter (请求路由)

**文件**: `router/router.py`

```python
class SlimeRouter:
    """自定义HTTP路由器"""

    def __init__(self, args):
        self.worker_request_counts: dict[str, int]  # 负载计数
        self.dead_workers: set[str]                 # 故障节点

    async def proxy(self, request, path):
        # 1. 选择最少负载的worker
        # 2. 转发请求
        # 3. 返回响应

    def _use_url(self):
        # 基于请求数的负载均衡算法
        return min(self.worker_request_counts,
                   key=self.worker_request_counts.get)
```

---

## 4. 完整调用流程

### 4.1 系统初始化流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         系统启动流程                                 │
└─────────────────────────────────────────────────────────────────────┘

1. Ray集群初始化
   └─ ray.init()

2. 创建PlacementGroup (GPU资源分配)
   └─ placement_group.py: create_colocate_pg()
       ├─ Actor组GPU槽位
       ├─ Critic组GPU槽位
       └─ Rollout引擎GPU槽位

3. 启动训练Actor
   └─ ray/actor_group.py: ActorGroup
       └─ train_actor.py: TrainRayActor.init()
           ├─ dist.init_process_group()  # NCCL通信组
           ├─ init_gloo_group()          # Gloo通信组
           └─ MegatronTrainRayActor.init()
               ├─ init(args)             # Megatron初始化
               ├─ initialize_model_and_optimizer()
               │   ├─ get_model()        # 构建模型
               │   ├─ get_megatron_optimizer()  # 创建优化器
               │   └─ load_checkpoint()  # 加载检查点
               └─ TensorBackuper.create() # 权重备份器

4. 启动推理引擎
   └─ ray/rollout.py: RolloutServer.__init__()
       └─ sglang_engine.py: SGLangEngine.init()
           ├─ launch_server_process()   # 启动SGLang进程
           ├─ _wait_server_healthy()    # 等待就绪
           └─ 注册到Router

5. (可选) 启动Critic Actor
   └─ 同训练Actor流程

6. 连接Actor和Critic
   └─ connect_actor_critic()
       └─ init_process_group() # 创建Actor-Critic通信组
```

### 4.2 单轮Rollout训练流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Rollout训练主循环                                 │
│                                                                 [Main] │
└─────────────────────────────────────────────────────────────────────┘

for rollout_id in range(start_rollout_id, num_rollout):
    │
    ├──[阶段1: 生成 samples]───────────────────────────────────────────┤
    │
    │   主进程调用:
    │   rollout_server.generate(rollout_id, data_source)
    │   │
    │   └─► rollout/sglang_rollout.py: generate_rollout()
    │       │
    │       ├─ data_source.get_samples(num_samples)
    │       │   └─ 获取prompt数据
    │       │
    │       ├─ generate_and_rm_group() [异步批量生成]
    │       │   │
    │       │   ├─► generate(args, sample, sampling_params)
    │       │   │   │
    │       │   │   ├─ POST /generate → SGLang引擎
    │       │   │   ├─ 返回: response, log_probs, routed_experts
    │       │   │   └─ 更新sample状态
    │       │   │
    │       │   └─► async_rm(args, sample)
    │       │       └─ 计算奖励 (数学验证/代码执行/etc.)
    │       │
    │       └─ 返回: list[list[Sample]]
    │
    ├──[阶段2: 训练]───────────────────────────────────────────────────┤
    │
    │   主进程调用:
    │   actor_group.train(rollout_id, rollout_data_ref)
    │   │
    │   └─► actor.py: MegatronTrainRayActor.train()
    │       │
    │       ├─ _get_rollout_data()
    │       │   └─ 处理数据，移动到GPU
    │       │
    │       └─ train_actor(rollout_id, rollout_data)
    │           │
    │           ├─► data.py: get_data_iterator()
    │           │   └─ 创建数据迭代器
    │           │
    │           ├─► [可选] 切换到ref模型计算log_probs
    │           │   └─ compute_log_prob(..., store_prefix="ref_")
    │           │
    │           ├─► [可选] 切换到teacher模型(OPD)
    │           │   └─ compute_log_prob(..., store_prefix="teacher_")
    │           │
    │           ├─► 切换到actor模型计算log_probs
    │           │   └─ compute_log_prob(..., store_prefix="")
    │           │
    │           ├─► loss.py: compute_advantages_and_returns()
    │           │   │
    │           │   ├─ 计算KL散度
    │           │   ├─ 根据算法选择:
    │           │   │   ├─ PPO: get_advantages_and_returns_batch()
    │           │   │   ├─ GRPO: get_grpo_returns()
    │           │   │   └─ REINFORCE++: get_reinforce_plus_plus_returns()
    │           │   └─ 可选: 归一化advantages
    │           │
    │           └─► model.py: train()
    │               │
    │               └─ for step_id in num_steps_per_rollout:
    │                   └─ train_one_step()
    │                       │
    │                       ├─ get_batch()           # 获取批次
    │                       ├─ forward_backward_func() # 前向传播
    │                       ├─ loss_function()       # 计算损失
    │                       │   │
    │                       │   └─► loss.py: policy_loss_function()
    │                       │       ├─ 计算 log_probs, entropy
    │                       │       ├─ compute_policy_loss() # PPO clip
    │                       │       ├─ 计算 entropy_loss
    │                       │       └─ 计算 kl_loss (可选)
    │                       │
    │                       ├─ optimizer.step()      # 更新参数
    │                       └─ opt_param_scheduler.step() # 更新学习率
    │
    ├──[阶段3: 权重同步]───────────────────────────────────────────────┤
    │
    │   主进程调用:
    │   actor_group.update_weights()
    │   │
    │   └─► actor.py: MegatronTrainRayActor.update_weights()
    │       │
    │       ├─ 获取可更新的引擎列表
    │       │   └─ rollout_manager.get_updatable_engines_and_lock()
    │       │
    │       └─► weight_updater.update_weights()
    │           │
    │           ├─ [Colocate模式]
    │           │   └─► update_weight_from_tensor.py
    │           │       └─ engine.update_weights_from_tensor()
    │           │
    │           └─ [分布式模式]
    │               └─► update_weight_from_distributed.py
    │                   ├─ init_weights_update_group()
    │                   └─ update_weights_from_distributed()
    │
    └──[阶段4: 检查点保存]─────────────────────────────────────────────┤
        │
        └─ if (rollout_id + 1) % save_interval == 0:
            └─► actor.py: save_model(rollout_id)
                └─► model.py: save()
                    └─ save_checkpoint()
```

### 4.3 模型权重更新流程详解

```
┌─────────────────────────────────────────────────────────────────────┐
│                    权重同步机制                                     │
└─────────────────────────────────────────────────────────────────────┘

训练进程 (Megatron)                      推理进程 (SGLang)
        │                                       │
        │  1. 权重备份到CPU/CUDA                 │
        ├─► weights_backuper.backup("actor")    │
        │                                       │
        │  2. 建立通信组                         │
        ├──────────────────────────────────────►│ init_weights_update_group()
        │   (NCCL ProcessGroup)                 │
        │                                       │
        │  3. 发送权重元数据                     │
        ├──────────────────────────────────────►│ update_weights_from_distributed()
        │   names, dtypes, shapes               │
        │                                       │
        │  4. 权重直接传输 (GPU-to-GPU)          │
        ├══════════════════════════════════════►│
        │   NCCL AllGather/Broadcast            │
        │                                       │
        │                                       ├─► 应用新权重
        │                                       │
        │  5. 确认更新                           │
        │◄──────────────────────────────────────┤ weight_version++
        │                                       │
        │  6. 重启引擎(可选)                     │
        │                                       ├─► flush_cache()
        │                                       │
```

---

## 5. 数据流分析

### 5.1 Sample数据结构

**文件**: `utils/types.py`

```python
@dataclass
class Sample:
    """单条样本数据结构"""

    # === 标识信息 ===
    group_index: int | None       # 组索引 (同一prompt的多响应)
    index: int | None             # 全局唯一索引

    # === 输入数据 ===
    prompt: str | list[dict]      # 输入提示 (文本或多模态消息)
    tokens: list[int]             # Token IDs (prompt + response)
    multimodal_inputs: dict       # 多模态输入 (图像/视频/音频)
    multimodal_train_inputs: dict # 处理后的多模态数据

    # === 输出数据 ===
    response: str                 # 生成的响应文本
    response_length: int          # 响应token长度

    # === 标签和奖励 ===
    label: str | None             # 标准答案
    reward: float | dict          # 奖励值 (可能为字典如{"format": 1.0, "accuracy": 0.5})

    # === 训练相关 ===
    loss_mask: list[int]          # 损失掩码 (0/1)
    rollout_log_probs: list[float] # 推理时log概率
    rollout_routed_experts: list  # MoE路由信息

    # === 状态 ===
    status: Status                # PENDING/COMPLETED/TRUNCATED/ABORTED
    weight_versions: list[str]    # 权重版本追踪

    # === 扩展 ===
    metadata: dict                # 用户自定义元数据
    train_metadata: dict          # 训练时使用的元数据
```

### 5.2 数据流转过程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         数据流转图                                   │
└─────────────────────────────────────────────────────────────────────┘

[磁盘数据集]
     │
     ▼
┌──────────────────┐
│ Dataset          │  utils/data.py
│ - 加载prompt     │
│ - tokenize       │
│ - 处理多模态     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ DataSource       │  rollout/data_source.py
│ - get_samples()  │
│ - add_samples()  │  (支持缓冲区回放)
│ - save/load      │
└────────┬─────────┘
         │
         │ list[list[Sample]]
         ▼
┌──────────────────┐
│ SGLang生成       │  rollout/sglang_rollout.py
│ - 批量推理       │
│ - 计算reward     │
│ - 更新状态       │
└────────┬─────────┘
         │
         │ RolloutFnTrainOutput(samples, metrics)
         ▼
┌──────────────────┐
│ 数据处理         │  utils/data.py: process_rollout_data()
│ - 分片到DP rank  │
│ - 转换为tensor   │
└────────┬─────────┘
         │
         │ RolloutBatch (dict)
         ▼
┌──────────────────┐
│ DataIterator     │  backends/megatron_utils/data.py
│ - 批次迭代       │
│ - 动态batch      │
│ - 序列填充       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 训练循环         │  backends/megatron_utils/model.py
│ - 前向传播       │
│ - 损失计算       │
│ - 反向传播       │
└──────────────────┘
```

### 5.3 RolloutBatch内部结构

```python
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]

# 典型内容:
{
    # === Token数据 ===
    "tokens": list[torch.Tensor],           # 每个样本的token ids
    "loss_masks": list[torch.Tensor],       # 每个样本的损失掩码

    # === 长度信息 ===
    "total_lengths": list[int],             # prompt + response长度
    "response_lengths": list[int],          # response长度

    # === 概率和价值 ===
    "log_probs": list[torch.Tensor],        # 当前策略log概率
    "ref_log_probs": list[torch.Tensor],    # 参考策略log概率
    "rollout_log_probs": list[torch.Tensor],# 推理时log概率
    "values": list[torch.Tensor],           # 价值函数预测 (Critic)

    # === RL数据 ===
    "rewards": list[float],                 # 奖励值
    "advantages": list[torch.Tensor],       # 优势函数
    "returns": list[torch.Tensor],          # 回报

    # === 多模态 ===
    "multimodal_train_inputs": list[dict],  # 多模态训练数据

    # === MoE路由 ===
    "rollout_routed_experts": list[torch.Tensor], # 专家路由信息
}
```

---

## 6. 分布式架构

### 6.1 集群拓扑

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Slime 集群架构                                  │
└─────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────┐
                            │  Ray Head   │
                            │  (主控节点)  │
                            └──────┬──────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   Worker 0    │          │   Worker 1    │          │   Worker N    │
│               │          │               │          │               │
│ ┌───────────┐ │          │ ┌───────────┐ │          │ ┌───────────┐ │
│ │Train Actor│ │          │ │Train Actor│ │          │ │Train Actor│ │
│ │ (Rank 0)  │ │          │ │ (Rank 1)  │          │ │ (Rank N)  │ │
│ │           │ │          │ │           │ │          │ │           │ │
│ │ Megatron  │ │          │ │ Megatron  │ │          │ │ Megatron  │ │
│ │ - TP=2    │ │          │ │ - TP=2    │ │          │ │ - TP=2    │ │
│ │ - PP=1    │ │          │ │ - PP=1    │ │          │ │ - PP=1    │ │
│ │ - DP=4    │ │          │ │ - DP=4    │ │          │ │ - DP=4    │ │
│ └───────────┘ │          └───────────┘ │          │ └───────────┘ │
│               │          │               │          │               │
│ ┌───────────┐ │          │ ┌───────────┐ │          │ ┌───────────┐ │
│ │SGLang引擎 │ │          │ │SGLang引擎 │ │          │ │SGLang引擎 │ │
│ │ (推理)    │ │          │ │ (推理)    │ │          │ │ (推理)    │ │
│ │ - TP=8    │ │          │ │ - TP=8    │ │          │ │ - TP=8    │ │
│ └───────────┘ │          └───────────┘ │          │ └───────────┘ │
│               │          │               │          │               │
│    GPU 0-7    │          │    GPU 0-7    │          │    GPU 0-7    │
└───────────────┘          └───────────────┘          └───────────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                            ┌──────▼──────┐
                            │   Router    │
                            │ (负载均衡)   │
                            └─────────────┘
```

### 6.2 并行策略矩阵

| 策略 | 训练 (Megatron) | 推理 (SGLang) | 说明 |
|------|-----------------|---------------|------|
| **TP** (Tensor Parallel) | 支持 | 支持 | 模型分片到多GPU |
| **PP** (Pipeline Parallel) | 支持 | 支持 | 层间流水线 |
| **DP** (Data Parallel) | 支持 | 支持 | 数据并行 |
| **CP** (Context Parallel) | 支持 | - | 超长序列支持 |
| **EP** (Expert Parallel) | 支持(MoE) | 支持(MoE) | 专家并行 |

### 6.3 通信组关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                         通信组层次结构                               │
└─────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   WORLD (NCCL)      │  所有GPU
                    │   size = world_size │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ TP Group        │  │ PP Group        │  │ DP Group        │
│ (Tensor交互)    │  │ (流水线阶段)    │  │ (梯度同步)      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ CP Group        │  │ EP Group        │  │ Gloo Group      │
│ (上下文并行)    │  │ (专家并行)      │  │ (CPU通信)       │
└─────────────────┘  └─────────────────┘  └─────────────────┘

特殊通信组:
┌─────────────────────────────────────┐
│ Actor-Critic Group (NCCL)           │
│ - 用于同步Value和Advantage          │
│ - world_size = 2                    │
│ - rank 0: Actor, rank 1: Critic    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Weight Update Group (NCCL)          │
│ - 用于训练-推理间权重同步           │
│ - 动态创建/销毁                     │
└─────────────────────────────────────┘
```

### 6.4 内存共存模式 (Colocate)

当 `--colocate` 启用时，训练和推理共享GPU:

```
┌───────────────────────────────────────────────────────────────┐
│                    单GPU内存布局 (Colocate)                    │
└───────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      GPU Memory                              │
├─────────────────────┬───────────────────────────────────────┤
│                     │                                       │
│   训练阶段          │   推理阶段                            │
│   (Active)          │   (Offloaded/Released)                │
│                     │                                       │
│   ├─ Model Weights  │   ├─ KV Cache (释放)                  │
│   ├─ Optimizer      │   ├─ Weights Backup (CPU)             │
│   │  States         │   │                                   │
│   ├─ Activations    │   │                                   │
│   └─ Gradients      │   │                                   │
│                     │                                       │
│   ────────────────  │   ─────────────────────────────────   │
│                     │                                       │
│   推理阶段          │   训练阶段                            │
│   (Active)          │   (Offloaded/Released)                │
│                     │                                       │
│   ├─ Model Weights  │   ├─ Optimizer States (CPU)           │
│   ├─ KV Cache       │   ├─ Activations (释放)               │
│   └─ Activation     │   └─ Gradients (释放)                 │
│                     │                                       │
└─────────────────────┴───────────────────────────────────────┘

时序:
[Rollout生成] → [训练: wake_up()] → [训练] → [训练: sleep()] → [权重同步] → [Rollout生成] → ...
```

---

## 7. 关键代码路径

### 7.1 PPO损失计算路径

```
loss_function()                          # loss.py:943
    │
    ├─ get_sum_of_sample_mean()          # 归约函数
    │
    └─ policy_loss_function()            # loss.py:613
        │
        ├─ get_log_probs_and_entropy()   # loss.py:225
        │   ├─ get_responses()           # 提取response部分
        │   └─ calculate_log_probs_and_entropy() # ppo_utils.py:649
        │
        ├─ compute_policy_loss()         # ppo_utils.py:124
        │   └─ PPO Clipping:
        │       ratio = exp(-ppo_kl)
        │       pg_loss = max(-ratio*adv, -clip(ratio)*adv)
        │
        ├─ entropy_loss = -entropy_coef * entropy
        │
        └─ kl_loss = kl_coef * kl        # 可选
```

### 7.2 GAE计算路径

```
compute_advantages_and_returns()         # loss.py:400
    │
    ├─ compute_approx_kl()               # ppo_utils.py:12
    │   └─ KL估算 (k1/k2/k3/low_var_kl)
    │
    └─ 根据advantage_estimator选择:
        │
        ├─ PPO:
        │   get_advantages_and_returns_batch()  # ppo_utils.py:374
        │   └─ chunked_gae()              # 分块并行GAE
        │
        ├─ GRPO:
        │   get_grpo_returns()            # ppo_utils.py:201
        │   └─ reward广播到每个token
        │
        └─ REINFORCE++:
            get_reinforce_plus_plus_returns() # ppo_utils.py:211
            └─ 折扣累积回报
```

### 7.3 检查点保存路径

```
save_model()                             # actor.py:508
    │
    ├─ save()                            # model.py:687
    │   ├─ disable_forward_pre_hook()
    │   ├─ save_checkpoint()             # checkpoint.py
    │   │   └─ 保存: model, optimizer, scheduler
    │   └─ enable_forward_pre_hook()
    │
    └─ save_hf_model()                   # model.py:715
        └─ 转换Megatron格式到HuggingFace格式
```

---

## 8. 配置与参数

### 8.1 核心参数分类

```yaml
# === 集群配置 ===
--actor-num-nodes: 1              # Actor节点数
--actor-num-gpus-per-node: 8      # 每节点GPU数
--critic-num-nodes: 1             # Critic节点数
--critic-num-gpus-per-node: 8

# === Rollout配置 ===
--rollout-num-gpus-per-engine: 8  # 每引擎GPU数
--rollout-batch-size: 256         # Rollout批次大小
--n-samples-per-prompt: 8         # 每个prompt生成样本数
--rollout-max-response-len: 2048  # 最大响应长度

# === SGLang配置 ===
--sglang-tp-size: 8               # 张量并行大小
--sglang-dp-size: 1               # 数据并行大小
--sglang-server-concurrency: 256  # 并发请求数

# === 并行策略 ===
--tensor-model-parallel-size: 2   # TP大小
--pipeline-model-parallel-size: 1 # PP大小
--context-parallel-size: 1        # CP大小

# === PPO超参数 ===
--advantage-estimator: ppo        # 算法选择
--eps-clip: 0.2                   # PPO clip范围
--eps-clip-high: 0.2
--gamma: 1.0                      # 折扣因子
--lambd: 0.95                     # GAE lambda
--kl-coef: 0.01                   # KL系数
--entropy-coef: 0.01              # 熵系数

# === 学习率 ===
--lr: 1e-6                        # 主学习率
--min-lr: 1e-7                    # 最小学习率
--lr-warmup-iters: 100            # 预热迭代数
--lr-decay-style: cosine          # 衰减策略

# === 内存优化 ===
--offload-train: true             # 训练时offload
--offload-rollout: true           # Rollout时offload
--colocate: false                 # 是否共存

# === 检查点 ===
--save-interval: 10               # 保存间隔
--save: /path/to/save             # 保存路径
--load: /path/to/load             # 加载路径
```

### 8.2 关键环境变量

```bash
# CUDA设备可见性
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 分布式通信
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
WORLD_SIZE=8
RANK=0
LOCAL_RANK=0

# Slime特定
ENABLE_ROUTING_REPLAY=1           # 路由重放
ROUTING_REPLAY_STAGE=replay       # 重放阶段
```

---

## 附录: 主要文件功能速查表

| 文件路径 | 主要功能 | 关键类/函数 |
|----------|----------|-------------|
| `backends/megatron_utils/actor.py` | 训练Actor实现 | `MegatronTrainRayActor` |
| `backends/megatron_utils/model.py` | 模型训练循环 | `train()`, `forward_only()` |
| `backends/megatron_utils/loss.py` | 损失计算 | `policy_loss_function()`, `compute_advantages_and_returns()` |
| `backends/sglang_utils/sglang_engine.py` | SGLang引擎封装 | `SGLangEngine` |
| `ray/rollout.py` | 推理服务器管理 | `ServerGroup`, `RolloutServer` |
| `ray/train_actor.py` | 训练Actor基类 | `TrainRayActor` |
| `rollout/sglang_rollout.py` | Rollout生成 | `generate_rollout()`, `generate()` |
| `rollout/data_source.py` | 数据源管理 | `RolloutDataSource` |
| `router/router.py` | 请求路由 | `SlimeRouter` |
| `utils/ppo_utils.py` | PPO算法实现 | `compute_policy_loss()`, `get_advantages_and_returns_batch()` |
| `utils/types.py` | 数据类型定义 | `Sample`, `RolloutBatch` |
| `utils/arguments.py` | 参数解析 | `get_slime_extra_args_provider()` |