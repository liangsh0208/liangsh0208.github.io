---
title: "Slime 代码走读 — 00. 仓库架构宏观介绍"
date: 2026-06-15T12:00:00+08:00
draft: false
---

> 本文档是对 slime 项目的系统性代码走读笔记。slime 是一个基于 **Megatron-LM + SGLang + Ray** 构建的大规模 RL（GRPO/PPO）训练框架。

---

## 一、项目定位

slime 定位为一个**生产级的大规模 LLM RL 训练框架**，核心特征：

- **训练后端**：Megatron-LM（支持 TP/PP/CP/DP/EP 多级并行）
- **推理引擎**：SGLang（支持 CUDA Graph、Prefix Cache、Speculative Decoding）
- **调度层**：Ray（Placement Group + Actor 模型，跨节点分布式调度）
- **算法支持**：GRPO、PPO、Reinforce++、GSPO、OPD、SFT
- **部署模式**：支持 Colocate（训练推理同 GPU 分时复用）和分离部署

---

## 二、目录结构总览

```
slime/
├── train.py                          # 训练入口（主循环）
├── train_async.py                    # 异步训练入口变体
├── submit.py / submit_*.py           # a6o/平台提交脚本
├── shell/                            # 训练 Shell 脚本（调 train.py）
│   └── run-qwen3-32B.sh             # 4 节点 32 GPU 训练脚本示例
├── scripts/                          # 模型配置的模块化参数
│   └── models/qwen3-32B.sh
├── slime/
│   ├── ray/                          # Ray 分布式调度层
│   │   ├── placement_group.py       # Placement Group 分配、Actor 初始化
│   │   ├── rollout.py               # RolloutServer / RolloutManager
│   │   ├── actor_group.py           # RayTrainGroup（训练 Actor 集合）
│   │   ├── train_actor.py           # TrainRayActor 抽象基类
│   │   └── utils.py                 # Ray 辅助工具
│   ├── backends/
│   │   ├── megatron_utils/          # Megatron 训练后端（核心）
│   │   │   ├── actor.py             # MegatronTrainRayActor（训练生命周期）
│   │   │   ├── model.py             # train_one_step / forward_only / save
│   │   │   ├── loss.py              # 所有 loss 计算 + advantage 估计
│   │   │   ├── data.py              # get_batch / DataIterator / 日志聚合
│   │   │   ├── model_provider.py    # GPTModel 封装 + freeze 逻辑
│   │   │   ├── checkpoint.py        # 加载 / 保存 checkpoint
│   │   │   ├── cp_utils.py          # Context Parallel 工具（slice/allgather）
│   │   │   ├── initialize.py        # torch.distributed + Megatron 初始化
│   │   │   ├── arguments.py         # 参数校验 + 默认设置
│   │   │   └── update_weight/      # 权重同步到 SGLang 的实现
│   │   └── sglang_utils/            # SGLang 配置和引擎封装
│   │       ├── sglang_engine.py     # SGLangEngine Ray Actor
│   │       └── sglang_config.py     # SglangConfig / ModelConfig / ServerGroupConfig
│   ├── rollout/                     # Rollout 生成逻辑
│   │   ├── sglang_rollout.py        # 异步 generate + reward 计算（核心）
│   │   ├── data_source.py           # 数据集加载（prompt 数据源）
│   │   ├── rm_hub/                  # Reward Model 集合（DeepScaler、Math等）
│   │   │   ├── deepscaler.py
│   │   │   └── math_utils.py
│   │   └── filter_hub/              # 动态采样过滤器
│   ├── utils/                       # 通用工具库（非常庞大）
│   │   ├── arguments.py             # 全局参数解析（1777 行）
│   │   ├── ppo_utils.py             # KL / GAE / Policy Loss / Entropy
│   │   ├── flops_utils.py           # TFLOPS 计算
│   │   ├── timer.py                 # 性能计时器
│   │   ├── data.py                  # Dataset / 数据预处理
│   │   ├── types.py                 # RolloutBatch / Sample 类型定义
│   │   └── ...
│   └── router/                      # slime 自定义 router（替代 sglang router）
├── configs/                          # 预设配置 YAML
├── datasets/                         # 数据集处理脚本
├── examples/                         # 使用示例
└── docs/                             # 文档
```

---

## 三、运行调用链（鸟瞰图）

```
┌────────────────────────────────────────────────────────────────┐
│                        启动阶段                                    │
│  shell/run-qwen3-32B.sh → train.py                             │
│    → parse_args() → create_placement_groups()                  │
│    → create_rollout_manager()  [Ray Actor: RolloutManager]       │
│    → create_training_models()  [Ray Actor: MegatronTrainRayActor]│
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                      主训练循环 (Rollout × N)                     │
│                                                                  │
│  for rollout_id in range(num_rollout):                           │
│                                                                  │
│    ┌──────────────────────────────────────────────────────────┐  │
│    │ ① GENERATE PHASE (SGLang)                                 │  │
│    │   rollout_manager.generate.remote()                      │  │
│    │   → sglang_rollout.generate_rollout_async()               │  │
│    │   → generate_and_rm_group() [asyncio 并发]                │  │
│    │   → generate() [HTTP → SGLang router → engine]           │  │
│    │   → async_rm() [DeepScaler reward]                       │  │
│    │   → 返回 RolloutFnTrainOutput (samples)                  │  │
│    └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│    ┌──────────────────────────────────────────────────────────┐  │
│    │ ② DATA PREP + LOG_PROB (Megatron)                         │  │
│    │   _convert_samples_to_train_data()                       │  │
│    │   → _split_train_data_by_dp() [按 DP rank 分发]           │  │
│    │   → MegatronTrainRayActor.train()                        │  │
│    │   → _get_rollout_data() [CPU → GPU]                       │  │
│    │   → _switch_model("ref") → forward_only() [ref log_probs]│  │
│    │   → _switch_model("actor") → forward_only() [log_probs]  │  │
│    │   → compute_advantages_and_returns() [GRPO/PPO/...]      │  │
│    └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│    ┌──────────────────────────────────────────────────────────┐  │
│    │ ③ TRAIN PHASE (Megatron)                                   │  │
│    │   train() → train_one_step() x num_steps_per_rollout     │  │
│    │   → forward_backward_func() [Megatron pipeline]          │  │
│    │   → loss_function() → policy_loss_function()             │  │
│    │   → optimizer.step() → opt_param_scheduler.step()      │  │
│    │   → log_rollout_data() + log_perf_data() + 训练日志      │  │
│    └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│    ┌──────────────────────────────────────────────────────────┐  │
│    │ ④ SYNC + EVAL (Ray)                                        │  │
│    │   actor_model.update_weights()                           │  │
│    │   → UpdateWeightFromDistributed.update_weights()         │  │
│    │   → Megatron 权重广播 → SGLang engines                  │  │
│    │   → [可选] eval.remote() → AIME2024 评测                  │  │
│    └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## 四、核心设计模式

### 4.1 Ray Actor 分层

| Layer | 类型 | 角色 |
|-------|------|------|
| `RolloutManager` | `@ray.remote` | 全局协调者，管理所有 rollout server |
| `SGLangEngine` | `@ray.remote` | 单个 SGLang 推理实例 |
| `MegatronTrainRayActor` | `@ray.remote` | 单个训练进程（一个 Megatron rank） |
| `RayTrainGroup` | 本地对象 | 管理一组同角色的 TrainRayActor |

### 4.2 训练与推理的 GPU 关系

```
Colocate 模式（同 GPU 分时复用）:
  GPU 0-7: SGLang engine (generate) ↔ Megatron actor (train)
           [offload → train → update_weights → onload]

分离部署模式:
  GPU 0-15: SGLang engines (generate)
  GPU 16-31: Megatron actors (train)
           [update_weights 跨 GPU 传输]
```

### 4.3 多阶段流水线

```
Generate (Async)      Data Prep             Train (Sync)            Sync
   |                       |                     |                   |
   |─── GPU 0-7 ─────────|── CPU + GPU ─────────|─── GPU 16-31 ─────|── NVLink/PCIe

时间线:
  generate_0 ──► prep_0 ──► train_0 ──► sync_0 ──► generate_1 ──► ...

关键: generate_1 理论上可以和 train_0 并行（async），但 slime 当前实现中
      rollout_n 的 generate 必须等 rollout_{n-1} 的 train 完成后才启动
      （因为 GPU 在 colocate 模式下互斥）
```

---

## 五、关键概念速查

| 概念 | 含义 | 代码体现 |
|------|------|---------|
| **Rollout** | 一轮完整的生成+训练 | `rollout_id` 递增 |
| **Sample** | 一条 (prompt, response, reward) | `slime/utils/types.py:Sample` |
| **RolloutBatch** | 一个 microbatch 的训练数据 | `dict[str, list[Tensor]]` |
| **DataIterator** | microbatch 迭代器 | `slime/backends/megatron_utils/data.py` |
| **ServerGroup** | 一组同构 SGLang engines | `slime/ray/rollout.py` |
| **RolloutServer** | 一个模型 + 一个 router + N 个 groups | `slime/ray/rollout.py` |
| **EngineGroup** | 逻辑概念，等价 ServerGroup | colocate 决策的单位 |
| **Weight Sync** | Megatron 权重 → SGLang | `update_weight/` 目录 |
| **Offload** | SGLang 显存释放给 Megatron | `release_memory_occupation` |
| **Colocate** | 训练推理同 GPU | `--colocate` 参数 |
| **Dynamic Batch** | 按 token 数切 microbatch | `--use-dynamic-batch-size` |
| **Context Parallel (CP)** | 序列维度的模型并行 | `cp_utils.py` |
| **Advantage Estimator** | GRPO / PPO / REINFORCE++ | `loss.py:compute_advantages_and_returns` |
| **KL Loss** | 策略与参考模型的 KL 散度 | `ppo_utils.py:compute_approx_kl` |
| **TIS / OPSM** | 离策略 / 序列掩码修正 | `loss.py:vanilla_tis_function` |

---

## 六、文档索引

本系列代码走读分为以下专题：

| # | 文档 | 覆盖范围 |
|---|------|---------|
| 00 | `00-overview.md` | 仓库架构宏观介绍（本文） |
| 01 | `01-train-entry.md` | train.py → PlacementGroup → Ray Actor 初始化 |
| 02 | `02-rollout-system.md` | RolloutManager → RolloutServer → ServerGroup → SGLangEngine |
| 03 | `03-training-backend.md` | MegatronTrainRayActor → train_actor → train_one_step |
| 04 | `04-loss-and-advantage.md` | Loss 计算、Advantage 估计、Policy Loss |
| 05 | `05-data-pipeline.md` | DataSource → Sample → DataIterator → get_batch |
| 06 | `06-parallelism-and-context.md` | TP/PP/CP/DP/EP 并行策略 + 权重同步 |
| 07 | `07-optimization-and-utils.md` | FLOPS 计算、Timer、Arguments 解析、性能优化 |
