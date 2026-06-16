---
created: 2026-06-09
paper: https://arxiv.org/abs/2409.19256
code: https://github.com/volcengine/verl
authors: Guangming Sheng et al. (ByteDance)
tags:
  - RLHF
  - Framework
  - System
  - LLM
---

# HybridFlow: A Flexible and Efficient RLHF Framework

## 一句话总结

HybridFlow（veRL）通过**单控制器+多控制器的混合编程范式**，结合**零冗余的3D-HybridEngine权重重分片机制**，实现了比现有系统最高 **20.57 倍**吞吐提升的统一RLHF框架，支持PPO、ReMax、Safe-RLHF等多种算法灵活表达。

![](HybridFlow_fig1_dataflow.png)

> **Figure 1**: PPO、ReMax、Safe-RLHF 三种 RLHF 算法的 dataflow 图。每个节点代表一个分布式 LLM 训练或生成程序，每条边代表多对多的数据广播关系。RLHF 将传统 RL 的简单 dataflow 大幅复杂化。

---

## 1. 研究背景与动机

### 1.1 问题定义

RLHF（Reinforcement Learning from Human Feedback）是大语言模型对齐的核心后训练技术。传统 RL 可以被建模为一个 dataflow：节点表示神经网络的计算，边表示网络间的数据依赖。

但 RLHF 将每个节点扩展为一个**分布式 LLM 训练或生成程序**，每条边扩展为**多对多的数据广播**，使得整个 dataflow 的复杂性急剧上升。

### 1.2 现有方法的三大瓶颈

| 瓶颈类型 | 具体表现 |
|---------|---------|
| **灵活性不足** | 现有系统（OpenRLHF、DeepSpeed-Chat）采用纯多控制器范式，用户必须在代码中混写集体通信、模型计算和点对点数据传输，缺乏模块化封装，导致不同 RLHF 算法需要 case-by-case 实现 |
| **算法支持有限** | OpenRLHF 仅支持 PPO；NeMo-Aligner 支持 PPO 但扩展性差 |
| **执行效率低下** | DeepSpeed-Chat：ZeRO 训练 + TP 生成，模型重分片开销大；OpenRLHF：两份 Actor 权重导致冗余内存和频繁同步；NeMo-Aligner：训练和生成使用相同并行策略，生成吞吐量极低（瓶颈占比 81.2%） |

**关键观察**：Actor 模型的训练阶段是**compute-bound**（计算密集），而生成阶段是**memory-bandwidth-bound**（访存密集），两者需要截然不同的并行策略。

---

## 2. 系统架构详解

### 2.1 核心设计：混合编程范式

![](HybridFlow_fig2_programming_models.png)

> **Figure 2**: (a) 多控制器范式：每个节点内部有自己的控制器，适合分布式计算但不适合灵活表达 dataflow；(b) HybridFlow 的混合范式：**单控制器**协调节点间通信，**多控制器**管理节点内分布式计算。

HybridFlow 的关键创新是**将单控制器和多控制器范式以混合方式结合**：
- **单控制器**（Ray）：负责节点间的数据依赖调度和通信编排（inter-node）
- **多控制器**（PyTorch DDP/FSDP）：负责节点内的分布式计算（intra-node）

这种分离使得用户可以用高层 API 表达算法逻辑，底层自动处理分布式细节。

### 2.2 分层 APIs

| 层级 | API 类型 | 功能 |
|------|---------|------|
| **Intra-node** | `3DParallelWorker` 基类 | 封装模型分布式计算（ActorWorker, CriticWorker 等） |
| **Inter-node** | `@register` 传输协议 | 8 种协议：`3D_PROTO`, `DP_PROTO`, `ONE_TO_ALL`, 等 |

**PPO 实现仅需 8 行代码**：

```python
sequences = actor.generate_sequences(prompts)
values = critic.compute_values(sequences)
ref_logprobs = ref_policy.compute_logprobs(sequences)
rewards = reward_model.compute_rewards(sequences)
actor.update_actor(advantages)
critic.update_critic(returns, values)
```

从 PPO 扩展到 Safe-RLHF 只需额外 5 行（加入 cost model 和 pretrain loss）。

### 2.3 3D-HybridEngine：零冗余权重重分片

![](HybridFlow_fig5_3d_engine.png)

> **Figure 5**: 3D-HybridEngine 的工作流程。每一轮迭代包括：(1) 在 micro DP group 内收集更新后的 Actor 参数用于生成；(2) 加载 prompt 到各模型副本；(3) All-gather 生成结果；(4) 重新分片参数用于训练；(5) 计算 loss 并更新权重。

#### 并行组表示

- **训练阶段**并行配置：$p \times t \times d$（PP size, TP size, DP size）
- **生成阶段**并行配置：$p_g \times t_g \times d_g \times d$，其中 $d_g = pt / (p_g t_g)$

#### 零冗余重分片原理

传统方法的冗余来源：训练权重和生成权重在设备上重叠存储。HybridFlow 通过**按固定间隔选择 rank**的方式重组 TP/PP group，使得训练与生成阶段的模型权重完全复用同一套物理存储，实现**零内存冗余**。

#### 通信与内存对比

| 系统 | 通信量 | 峰值内存 | 冗余 |
|------|-------|---------|------|
| DeepSpeed-Chat | $\frac{(tpd-1)}{tpd} \cdot M$ | $M$ | $\frac{1}{tpd} \cdot M$ |
| HybridFlow-V | $\frac{(tp-1)}{tp} \cdot M$ | $M$ | $\frac{1}{tp} \cdot M$ |
| **HybridFlow** | $\frac{(tp - t_g p_g)}{t_g p_g \cdot tp} \cdot M$ | $\frac{1}{t_g p_g} \cdot M$ | **0** |

其中 $M$ 为模型参数量。HybridFlow 的通信量与冗余严格小于等于所有基线。

### 2.4 自动设备映射（Auto-Mapping）

**输入**：Dataflow 图 $D$、LLM 集合 $L=[l_1, ..., l_k]$、Workload $W$、总 GPU 数 $N$、单 GPU 显存容量 $Q$

**算法流程**：
1. 枚举所有可能的模型共置方案（Bell partition，4 个模型时为 15 种）
2. 确定每个共置模型的最小 GPU 数 $A_{min}$
3. 从 $A_{min}$ 开始枚举所有可行设备分配
4. 对每个模型调用 `auto_parallel()` 搜索最优并行策略
5. 用 `d_cost()` 估计各方案 latency，取最小值

**复杂度**：$O\left(\frac{(N-1)!}{(k-1)!(N-k)!}\right)$，但通过缓存和剪枝，实际运行时间最多半小时，且呈线性增长。

**关键洞察**：小集群上将所有模型共置在同一套设备上性能最好；大集群上将模型分离共置更优。

---

## 3. 实验结果

### 3.1 实验设置

- **硬件**：16 台机器 × 8 A100-80GB = **128 GPUs**，NVLink 600GB/s，网络互联 200Gbps
- **模型**：7B、13B、70B Actor + 7B、13B、70B Critic
- **基线**：DeepSpeed-Chat、OpenRLHF、NeMo-Aligner
- **算法**：PPO、ReMax、Safe-RLHF

### 3.2 端到端吞吐提升

| 对比基线 | 平均加速 | 最大加速 |
|---------|---------|---------|
| DeepSpeed-Chat | **3.67 倍** | 最高 7.84 倍 |
| OpenRLHF | **3.25 倍** | 最高 5.93 倍 |
| NeMo-Aligner | **12.52 倍** | 最高 **20.57 倍** |

- 70B 模型平均加速：**9.64 倍**
- 8 GPU 最小加速：**2.09 倍**
- 128 GPU 上 7B 模型：PPO 1.68 倍、ReMax 1.53 倍、Safe-RLHF 1.71 倍

### 3.3 扩展性与开销分析

- **强扩展效率**：66.8%
- **阶段切换开销降低**：平均 55.2%（11.7s），最高 89.1%（70B 模型减少 78.2s）
- **生成延迟降低**（最优 $t_g$ 配置）：7B 降低 60.3%，13B 降低 36.4%

### 3.4 模型放置策略

对于 13B Actor + 70B Critic 的不对等工作负载：
- 小集群（≤64 GPU）：**共置（colocate）**优于分离放置 44.8%
- 大集群（>64 GPU）：分离放置更优（固定 batch size 下）

---

## 4. 局限性与未来方向

1. **显存仍是瓶颈**：70B 模型 + 大 batch 场景下，单 GPU 80GB 仍可能不足
2. **自动映射的搜索开销**：虽然通过缓存优化到半小时内，但对于超大规模集群（>1024 GPU）可能仍需进一步优化
3. **通信优化空间**：跨节点 All-gather 在高延迟网络下仍是瓶颈
4. **多模态扩展**：当前仅支持文本 RLHF，多模态（图像、视频）场景待探索

---

## 5. 个人思考

### 5.1 设计优雅之处

HybridFlow 的**混合控制器范式**是一个极具洞察力的设计决策。它没有盲目追求某种"最优"架构，而是认识到 RLHF dataflow 的两个层面（节点间通信 vs 节点内计算）有本质不同的需求，分别用最合适的工具解决（Ray 做调度、PyTorch 做计算）。这种"分层解耦"的思路与计算机系统中经典的 OS 内核 vs 用户态设计有异曲同工之妙。

**3D-HybridEngine 的零冗余重分片**是另一个亮点。它不是简单地在训练前后做全量参数同步，而是通过精巧的 group 重组让同一套物理权重在两个阶段被复用。这提示我们：在分布式系统中，"物理布局"与"逻辑视图"的解耦往往能带来巨大的优化空间。

### 5.2 与 StreamRL 的对比

HybridFlow 采用的是**共置架构（colocation）**——将训练和生成放在同一套 GPU 上，通过时间复用调度；StreamRL 则采用**训推分离架构（disaggregation）**。两者的取舍反映了不同假设下的最优解：HybridFlow 假设资源受限且需要低延迟通信，因此共置减少跨节点开销；StreamRL 假设大规模场景下资源弹性更重要。HybridFlow 在小集群（≤64 GPU）上性能更好，而 StreamRL 在大规模异构/跨数据中心场景下更具优势。两者并非替代关系，而是互补——HybridFlow 解决"如何在有限资源内高效复用"，StreamRL 解决"如何在弹性资源下最大化吞吐"。

### 5.3 启发

- **API 设计至关重要**：HybridFlow 能让 PPO 在 8 行内实现，体现了好的系统抽象能把复杂性"压到下面"
- **自动并行搜索**正在从训练扩展到 RLHF 全流程，未来可能实现端到端的自动分布式策略搜索

---

## 6. 关键引用

```bibtex
@inproceedings{sheng2025hybridflow,
  title={HybridFlow: A Flexible and Efficient RLHF Framework},
  author={Sheng, Guangming and Zhang, Chi and Ye, Zilingfeng and Wu, Xibin and Zhang, Wang and Zhang, Ru and Peng, Yanghua and Lin, Haibin and Wu, Chuan},
  booktitle={ECCS},
  year={2025},
  url={https://arxiv.org/abs/2409.19256}
}
```
