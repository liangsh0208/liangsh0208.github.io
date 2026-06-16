---
created: 2026-06-09
paper: https://arxiv.org/abs/2504.15930
authors: Yinmin Zhong et al.
tags:
  - RL
  - System
  - Disaggregation
  - LLM
---

# StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation

## 一句话总结

StreamRL 从**训推分离（disaggregation）**的第一性原理出发，通过**流式生成（streaming generation）**消除流水线气泡、以**偏度感知调度（skewness-aware dispatching）**消除长尾气泡，并支持**弹性扩缩容**，在 LLM 强化学习场景中实现了最高 **2.66 倍**吞吐提升和 **1.33 倍**成本优化。

![](StreamRL_fig1_arch_comparison.png)

> **Figure 1**: 两种 RL 框架架构对比。(a) **Colocated（共置架构）**：生成与训练共享同一套资源，通过时间复用调度；(b) **Disaggregated（训推分离架构）**：SGS（流式生成服务）与 Trainer 部署在物理隔离的资源上，支持异构硬件和跨数据中心部署。

---

## 1. 研究背景与动机

### 1.1 问题定义

LLM 强化学习包含两个阶段：
1. **生成（Generation）**：模型在线生成样本
2. **训练（Training）**：利用生成的样本计算 reward 并更新模型权重

传统观点认为**共置架构**（两个阶段共享资源）优于**分离架构**（专用资源分配给每个阶段），因为共置避免了跨节点通信开销。

### 1.2 共置架构的三大问题

| 问题类型 | 根本原因 | 影响 |
|---------|---------|------|
| **资源耦合** | 两个阶段被迫使用同一套设备和并行策略 | 无法独立优化各自的需求 |
| **负载不均衡** | 随着序列长度增加，生成时间增长远快于训练时间（KV cache 膨胀迫使 batch size 减小） | 导致大量 GPU 空闲时间 |
| **异构支持缺失** | 无法将访存带宽型硬件（如 H20）用于生成，将计算型硬件（如 H800）用于训练 | 成本效率低下 |

#### 生成 vs 训练的本质差异

| 维度 | 生成（Generation） | 训练（Training） |
|------|-------------------|-------------------|
| **瓶颈类型** | Memory-bandwidth-bound（访存密集） | Compute-bound（计算密集） |
| **计算特征** | 每步仅计算新 token，需加载全部参数 | 前向+反向同时计算所有 token |
| **可扩展性** | 快速达到吞吐平台期（仅 TP 提升带宽，但受限于 intra-node NVLink） | 从 DP/PP/TP 多维扩展中获益更多 |
| **硬件偏好** | 高 HBM 带宽（H20: 4TB/s） | 高算力（H800） |

#### 性能敏感度差异（Figure 2）

![](StreamRL_fig2_sensitivity.png)

> **Figure 2**: 生成和训练对于资源和序列长度的敏感度差异。生成时间的增长随资源增加迅速饱和，而训练时间持续下降；随着序列长度增加，生成时间的增长显著大于训练时间。

---

## 2. 系统架构详解

### 2.1 训推分离架构设计

StreamRL 将生成与训练抽象为两个独立服务：

- **SGS（Stream Generation Service）**：负责 LLM 推理/生成
  - 使用自研推理引擎，支持 continuous batching 和 prefix sharing
  - 提供两个核心 API：`update(weights)` 和 `generate(prompts)`
- **Trainer**：负责模型训练更新
  - 实现 3D 并行（DP/TP/PP）
  - 支持动态 CPU offloading

两个服务可部署在**物理隔离的资源上**，甚至通过点对点链路连接不同的数据中心。

### 2.2 流水线气泡与流式生成

**核心问题**：传统同步 RL（如 PPO）的生成和训练是严格交替的——必须等所有样本生成完成后才能开始训练，导致 trainer 在生成期间空闲，SGS 在训练期间空闲。

**StreamRL 的流式机制**：

一旦 SGS 开始生成并产生第一个完成的样本，立即以**流式（stream）方式**返回给 Trainer，Trainer 无需等待整批样本生成完成即可开始处理。这打破了传统的阶段边界。

| 模式 | 生成-训练关系 | 气泡消除程度 |
|------|-------------|------------|
| **同步基线** | 严格交替，等待整批完成 | 气泡最大 |
| **StreamRL-Sync** | 流式返回样本，但 trainer 按 batch 训练 | 部分重叠 |
| **StreamRL-Async** | 完全异步：trainer 用实时到达的样本流持续训练 | **完全消除** |

### 2.3 偏度气泡与偏度感知调度

#### 问题：长尾输出长度分布

LLM 生成样本的输出长度服从长尾分布（Figure 7）：大多数样本较短，但少数样本极长。在 batch 生成中，**必须等待最长样本完成后整个 batch 才能释放**，导致短样本严重等待。

#### 样本延迟模型

单样本延迟：
$$\text{Sample Latency} = \text{PTL}(\text{BS}) \times L$$

其中 PTL(BS) = per-token latency（与 batch size 单调递增），$L$ = 输出长度。

实例生成延迟（包含多个样本）：
$$\text{Latency} = \text{PTL}(\text{BS}) \times L_{\text{avg}} \times \lceil M / \text{BS} \rceil$$

#### Skewness-aware Dispatching 算法

核心洞察：**对于输出长度更大的样本，应减小其 batch size**（因为 PTL 单调递增，较大 BS 的 PTL 更高，长样本在高 BS 下浪费严重）。

算法步骤：
1. 用轻量级 **Ranker 模型**预测每个 prompt 的输出长度，按降序排序
2. 将最长 $\alpha\%$（实践中 $\alpha = 20$）标记为**长尾样本**，其余为**常规样本**
3. 长尾样本使用 $P_{90}(D)$ 作为长度估计，常规样本使用 $P_{50}(D)$
4. 枚举所有 $(N_l, N_r)$ 分配方案（$N_l$ = 长尾实例数，$N_r$ = 常规实例数，$N_l + N_r = N$）
5. 用公式 (2) 计算每种方案的总延迟，取最小值
6. 实例内部采用 **Longest-Processing-Time-First (LPT)** 调度

Ranker 模型召回率：

| Base Model | Tail 20% | Tail 10% | Tail 5% |
|-----------|----------|---------|---------|
| Qwen2.5-7B | **0.87** | 0.82 | 0.76 |
| Qwen2.5-3B | 0.85 | 0.79 | 0.72 |
| Qwen2.5-1.5B | 0.81 | 0.75 | 0.68 |

### 2.4 弹性扩缩容

StreamRL 动态监控两个阶段间的执行时间差 $\delta$。

- 当生成阶段慢于训练阶段时，SGS 估计**增加一个 DP 单元**后能减少的生成时间 $\delta'$
- 当 $\delta \geq \delta'$ 时，触发扩容，为 SGS 增加一个 DP 单元
- 由于 SGS 实例天然解耦，**扩容不中断训练**，开销仅包含新实例初始化

这种弹性设计使 StreamRL 能适应负载变化，始终保持生成与训练的平衡。

---

## 3. 实验结果

### 3.1 实验设置

- **模型规模**：7B ~ 72B
- **数据集规模**：数 K ~ 数 10K 样本
- **基线**：veRL（共置架构代表）、ColocationRL
- **对比模式**：StreamRL-Sync（流式同步）、StreamRL-Async（完全异步）

### 3.2 端到端吞吐对比

| 对比 | 加速范围 |
|------|---------|
| StreamRL-Sync vs. veRL | **1.12 倍 ~ 2.12 倍** |
| StreamRL-Sync vs. ColocationRL | **1.06 倍 ~ 1.41 倍** |
| StreamRL-Async vs. ColocationRL | **1.30 倍 ~ 2.66 倍** |

### 3.3 消融实验（72B 模型，20K 数据集）

| 技术组合 | 归一化吞吐 | 提升 |
|---------|-----------|------|
| Colocation Baseline | 1.00 | — |
| + skewness-aware scheduling | 1.08 | **+8%** |
| + disaggregation + streaming | 1.23 | **+15%**（累计） |
| + asynchronous | 1.48 | **+25%**（累计） |

关键结论：
- **偏度感知调度**独立贡献 8%
- **训推分离 + 流式生成**独立贡献约 15%
- **异步训练**进一步大幅提升 25%
- 三者叠加可达 **1.48 倍**（注意这里的 1.48 是相对于同步基线，与 Async vs ColocationRL 的 2.66 倍并不矛盾，因为后者还包含了异构硬件优势）

### 3.4 资源均衡优化

将生成和训练阶段的资源分配调均衡后，可获得 **1.25 倍** 额外加速。

### 3.5 跨数据中心与异构部署

- **配置**：H20（4TB/s HBM，成本 1.00/机）用于生成 + H800（成本 2.85/机）用于训练
- **成本归一化吞吐**：比同构部署高 **1.23 倍 ~ 1.31 倍**
- 证明分离架构在真实异构/跨 DC 场景下具有显著成本优势

### 3.6 同步 vs 异步训练质量

![](StreamRL_fig7_throughput.png)

> **Figure 8**: 端到端吞吐对比。StreamRL-Async 在所有配置下均显著超越基线。

实验中同步和异步 PPO 的 reward 曲线基本一致，说明异步训练没有牺牲收敛质量。

---

## 4. 局限性与未来方向

1. **网络带宽假设**：跨数据中心的分离架构依赖足够的网络带宽，低带宽场景下通信可能成为新瓶颈
2. **Ranker 模型开销**：虽然轻量，但高频调用仍有计算成本，且对未见过的 prompt 分布可能泛化不足
3. **异步训练的理论保证**：实验验证了异步 PPO 的收敛性，但缺乏严格的理论证明
4. **与在线服务的结合**：当前 SGS 仅面向 RL 训练场景，与生产推理服务的资源共享/调度有待探索
5. **多轮对话 RLHF**：当前针对单轮生成优化，多轮对话中的 KV cache 管理更复杂

---

## 5. 个人思考

### 5.1 从第一性原理出发

StreamRL 的最大亮点在于它**挑战了行业传统认知**。长期以来，大家默认共置架构优于分离架构，但 StreamRL 通过深入分析生成与训练的本质差异（访存密集 vs 计算密集），证明了在真实大规模部署中分离架构反而更优。这提醒我们：**系统设计的"默认选项"往往基于过时的假设（如通信开销 > 一切），当硬件能力和工作负载特征发生变化时，需要重新审视这些假设**。

### 5.2 三类气泡的优雅解法

StreamRL 针对三种不同类型的"气泡"给出了针对性解法：

| 气泡类型 | 根源 | StreamRL 解法 |
|---------|------|---------------|
| **Pipeline bubbles** | 阶段间严格依赖 | 流式生成 + 异步训练 |
| **Skewness bubbles** | 长尾输出长度分布 | 偏度感知调度 + LPT |
| **Resource bubbles** | 静态资源分配 | 弹性扩缩容 |

这种"识别问题 -> 分类 -> 针对性解决"的方法论，值得在系统设计中借鉴。

### 5.3 与 HybridFlow 的对比与融合可能

| 维度 | HybridFlow | StreamRL |
|------|-----------|----------|
| **架构哲学** | 共置 + 时间复用 | 分离 + 异步流式 |
| **优化目标** | 单节点内的零冗余 + 灵活 API | 全网资源利用率 + 弹性伸缩 |
| **适合场景** | 中小规模集群（≤64 GPU） | 大规模/异构/跨 DC |
| **核心创新** | 3D-HybridEngine 零冗余重分片 | 流式生成 + 偏度调度 |
| **算法灵活性** | 支持多种 RLHF 算法（PPO/ReMax/Safe-RLHF） | 聚焦 PPO，强调吞吐 |

我认为两者的**融合**是一个很有前景的方向：
- **HybridFlow 的 API 层**可以被 StreamRL 的分离架构复用，解决 StreamRL 当前算法支持有限的问题
- **StreamRL 的流式异步机制**可以集成到 HybridFlow 的 3D-HybridEngine 中，进一步消除阶段切换等待
- 在 **中等规模集群**（64~256 GPU）上，可以自适应选择共置或分离：小 batch 时共置减少通信，大 batch 时分离提升并行度

### 5.4 对工业界的影响

StreamRL 的**异构硬件 + 跨数据中心**支持对工业界有深远意义。云厂商可以动态调配不同型号的 GPU 分别用于生成和训练，最大化性价比。弹性扩缩容能力也意味着 RL 训练可以更灵活地接入空闲推理资源，提升整体集群利用率。

---

## 6. 关键引用

```bibtex
@article{zhong2025streamrl,
  title={StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation},
  author={Zhong, Yinmin and Zhang, Zili and Song, Xiaoniu and Hu, Hanpeng and Jin, Chao and Wu, Bingyang and Chen, Nuo and Chen, Yukun and Zhou, Yu and Wan, Changyi and Zhou, Hongyu and Jiang, Yimin and Zhu, Yibo and Jiang, Daxin},
  journal={arXiv preprint arXiv:2504.15930},
  year={2025}
}
```
