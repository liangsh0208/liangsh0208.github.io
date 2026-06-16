---
created: 2026-06-09
published: 2025-10-15
paper: https://www.alphaxiv.org/abs/2510.13786
code: https://github.com/periodic-labs/ScaleRL
authors: Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S. Dhillon, David Brandfonbrener, Rishabh Agarwal (Meta, UT Austin, UCL, UC Berkeley, Harvard University, Periodic Labs)
tags:
  - LLM
  - Reinforcement-Learning
  - Scaling-Law
  - RLHF
  - Post-Training
  - Compute-Efficiency
---

# The Art of Scaling Reinforcement Learning Compute for LLMs

## 一句话总结

本文通过超过 **40 万 GPU 小时** 的大规模系统性实验，首次为 LLM 的 RL 训练建立了**可预测的 Sigmoid 扩缩律框架**，系统消融了损失函数、聚合方式、精度、off-policy 算法等关键设计选择，提出了 **ScaleRL** 最佳实践配方，并成功将小规模（8k GPU 小时）的 Sigmoid 曲线外推到 **10 万 GPU 小时**的单一 RL 运行，使 RL 训练的扩缩可预测性逼近预训练水平。

| 100k GPU 小时预测扩缩 | AIME-24 下游验证 |
|:---:|:---:|
| ![](ScaleRL_fig1a_100k.png) | ![](ScaleRL_fig1b_100k_aime24.png) |

> **Figure 1a & 1b**: 左图为 8B dense 模型在 100,000 GPU 小时下的 Sigmoid 扩缩曲线拟合与 extrapolation，训练曲线（蓝点）与预测曲线（橙色虚线）高度吻合；右图为 AIME-24 下游评估，同样呈现稳定可预测的扩缩行为。两条曲线均从早期低算力点（~1.5k GPU 小时后）外推，验证后延至 100k GPU 小时，展示了 Sigmoid 框架强大的预测能力。

---

## 1. 研究背景与动机

### 1.1 问题定义

强化学习（RL）已成为训练大语言模型（LLM）的核心环节——从 RLHF 到 DeepSeek-R1-Zero 的纯 RL 推理训练，RL 算力消耗迅速攀升。例如，**DeepSeek-R1-Zero 的 RL 训练消耗了约 10 万 H800 GPU 小时**，占其预训练算力的 3.75%（Guo et al., 2025）。然而，与预训练领域已有成熟的 Power Law / Scaling Law 预测框架不同，**RL 训练一直缺乏系统性的、可预测的扩缩律方法论**。

具体而言，RL 领域存在以下核心问题：
- 没有一个统一的框架来评价算法改进对扩缩行为的影响
- 不知道哪些设计选择会影响**渐近性能（asymptotic performance）**，哪些只影响**计算效率（compute efficiency）**
- 无法从小规模实验可靠地外推大规模结果

### 1.2 本文核心贡献

本文的核心贡献可以概括为三点：

1. **科学框架**：提出用 **Sigmoid 函数** 拟合 `pass rate — log(compute)` 曲线，首次为 RL 训练建立可预测的扩缩律。相比预训练中常用的 Power Law，Sigmoid 更适合有上界（如准确率 ≤ 1）的 RL 评估指标。

2. **大规模消融研究**：消耗 **400,000+ GPU 小时**，在 8B dense 模型上对常见设计选择进行系统消融（loss function、aggregation、normalization、off-policy algorithm、precision、filtering、curriculum 等），清晰区分了影响**渐近上限 A** vs **计算效率 B** 的因素。

3. **最佳实践配方 ScaleRL**：综合消融结论，提出 ScaleRL 配方，并在 10 万 GPU 小时的单一 RL 运行中验证了其可预测性和有效性。

---

## 2. 预liminaries 与实验设置

### 2.1 RL 训练范式

本文考虑标准 LLM RL 设定：从数据分布 $D$ 中采样 prompt $x$，对每個 prompt 生成 $G$ 条响应 $\{y_i\}_{i=1}^G$，通过可验证奖励信号（如数学题目答案是否正确）计算奖励 $r_i$，并更新策略模型 $\pi_\theta$。

**Generator-Trainer 分离架构**：
- **Generator**（生成端）：使用优化推理核（ vLLM / TensorRT-LLM）高吞吐生成 rollout，运行旧参数 $\theta_{\text{old}}$
- **Trainer**（训练端）：运行 FSDP 训练后端，更新参数 $\theta$
- 两者异步运行形成 **off-policy** 数据流

### 2.2 训练配置

| 配置项 | 默认值 |
|--------|--------|
| 基础模型 | 8B dense / Llama-4 Scout 17B×16 MoE |
| 序列长度 | 16,384 tokens（thinking 12,288 + solution 2,048 + prompt 2,048） |
| 扩展长度 | 32,768 tokens |
| 数据集 | Polaris-53K（可验证数学问题） |
| Batch size | 768 = 48 prompts × 16 generations |
| 大 batch | 2,048 prompts |
| 验证集 | 从 Polaris-53K  Hold out 1,000 prompts |
| 评估频率 | 每 100 训练步 |
| 每次评估生成数 | 16 |
| GPU | Nvidia GB200 |

**长度控制（Interruption）**：为防止生成长度过长，当 thinking 部分超过 12,288 tokens 时，强制插入结束语 `"Okay, time is up. Let me stop thinking and formulate a final answer now. \|end_thinking\|>` 截断思考过程。

### 2.3 基础 RL 算法

本文以 **GRPO（Group Relative Policy Optimization）** 为起点，做了以下修改：
- **移除 KL 散度正则项**（类似 DeepSeek-R1-Zero 的做法）
- **采用 DAPO 非对称裁剪（asymmetric clipping）**

#### 重要性采样比（Importance Sampling Ratio）

$$\rho_{i,t}(\theta) := \frac{\pi^{\theta}_{\text{train}}(y_{i,t} \mid x, y_{i,<t})}{\pi^{\theta_{\text{old}}}_{\text{gen}}(y_{i,t} \mid x, y_{i,<t})} = \frac{\pi^{\theta}_{\text{train}}(y_{i,t})}{\pi^{\theta_{\text{old}}}_{\text{gen}}(y_{i,t})}$$

| 符号 | 含义 |
|------|------|
| $\rho_{i,t}(\theta)$ | 第 $i$ 条响应在第 $t$ 个 token 的重要性采样比 |
| $\pi^{\theta}_{\text{train}}$ | 训练策略（当前参数 $\theta$） |
| $\pi^{\theta_{\text{old}}}_{\text{gen}}$ | 生成策略（旧参数，可能在异步 off-policy 场景下滞后多步） |
| $y_{i,t}$ | 第 $i$ 条响应的第 $t$ 个 token |
| $y_{i,<t}$ | 第 $i$ 条响应中第 $t$ 个 token 之前的所有 token |

> **直觉**：$\rho_{i,t}$ 衡量了当前训练策略生成该 token 的概率与生成策略生成该 token 的概率之比。如果比值过大，说明训练策略与生成策略偏差太大，需要裁剪（clip）来稳定训练。

#### 非对称裁剪

$$\mathrm{clip}_{\mathrm{asym}}(\rho, \epsilon^-, \epsilon^+) := \mathrm{clip}(\rho,\, 1-\epsilon^-, 1+\epsilon^+)$$

| 符号 | 含义 |
|------|------|
| $\epsilon^-$ | 下界裁剪阈值（允许策略降低概率的幅度） |
| $\epsilon^+$ | 上界裁剪阈值（允许策略提高概率的幅度） |
| $\mathrm{clip}(x, a, b)$ | 将 $x$ 裁剪到 $[a, b]$ 区间 |

> **直觉**：DAPO 允许**非对称**的裁剪范围——通常 $\epsilon^+ > \epsilon^-$，这意味着 RL 可以更激进地**提升**好 token 的概率，但温和地**降低**坏 token 的概率。这与传统 PPO 的对称裁剪 $\mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon)$ 形成对比。

#### 优势函数计算（Advantage Estimation）

$$\hat{A}_i = r_i - \mathrm{mean}(\{r_j\}_{j=1}^G)$$

$$\hat{A}_i^G = \frac{\hat{A}_i}{\mathrm{std}(\{r_j\}_{j=1}^G) + \epsilon}$$

| 符号 | 含义 |
|------|------|
| $r_i$ | 第 $i$ 条响应的奖励（通常为 0 或 1，即 pass/fail） |
| $G$ | 每個 prompt 生成的响应数量（group size） |
| $\mathrm{mean}(\{r_j\})$ | 同组响应的奖励均值 |
| $\mathrm{std}(\{r_j\})$ | 同组响应的奖励标准差 |
| $\epsilon$ | 数值稳定性常数，防止除以零 |
| $\hat{A}_i^G$ | 组归一化后的优势函数（GRPO 标准做法） |

> **直觉**：GRPO 的优势估计采用**组内相对对比**——在同组 $G$ 条响应中，比平均表现好的响应对应正优势，比平均差的对应负优势。除以标准差是为了消除不同 prompt 难度带来的尺度差异。注意：由于奖励是二元（0/1）的，当同组所有响应全对或全错时，$\mathrm{std}=0$，优势无法计算——这就是后文 "Zero-Variance Filtering" 要解决的问题。

#### 替代目标函数（Surrogate Objective，Sample-Level Aggregation）

$$\mathcal{J}(\theta) = \mathbb{E}_{\begin{subarray}{l}x \sim D,\\ \{y_i\}_{i=1}^G \sim \pi_{\text{gen}}^{\theta_{\text{old}}}(\cdot \mid x)\end{subarray}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( \rho_{i,t}(\theta) \hat{A}_i^G,\; \mathrm{clip}_{\text{asym}}(\rho_{i,t}(\theta), \epsilon^-, \epsilon^+) \hat{A}_i^G \right) \right]$$

| 符号 | 含义 |
|------|------|
| $\mathcal{J}(\theta)$ | 待最大化的替代目标（期望收益） |
| $\mathbb{E}$ | 对 prompt 分布 $D$ 和生成策略的期望 |
| $G$ | 每個 prompt 生成的响应数 |
| $|y_i|$ | 第 $i$ 条响应的 token 长度 |
| $\min(\cdot, \cdot)$ | 取未裁剪目标和裁剪目标中的较小值（PPO-clip） |
| $\hat{A}_i^G$ | 组归一化优势（与 token 位置 $t$ 无关，整条响应共享） |

> **直觉**：这个目标函数是 PPO-clip 的 GRPO 变体。
> 1. **内循环** $\sum_t$：对每条响应的所有 token 求和，按 token 长度 $|y_i|$ 平均——这是 **sample-level**（或 token-level）归一化
> 2. **外循环** $\frac{1}{G}\sum_i$：对同组 $G$ 条响应求平均
> 3. **PPO-clip**：$\min$ 操作确保当重要性采样比 $\rho$ 偏离 1 太远时，策略不会单方向过度更新（防止破坏性大梯度步）

---

## 3. Sigmoid 扩缩律框架

### 3.1 核心公式

本文的核心创新之一是提出用 **Sigmoid 函数** 建模 RL 的 `pass rate — log(compute)` 关系：

$$\underbrace{R_C - R_0}_{\text{Reward Gain}} = \underbrace{(A - R_0)}_{\text{Asymptotic Reward Gain}} \times \underbrace{\frac{1}{1 + (C_{\text{mid}} / C)^B}}_{\text{Compute Efficiency}}$$

展开形式：

$$\boxed{R_C = R_0 + (A - R_0) \times \frac{1}{1 + (C_{\text{mid}} / C)^B}}$$

| 参数 | 符号 | 含义 | 作用 |
|------|------|------|------|
| $R_C$ | 当前 pass rate | 在计算量 $C$ 时的通过率 | 输出变量 |
| $R_0$ | 初始 pass rate | 训练开始前的基线通过率 | 通常为 SFT 后的初始性能 |
| $A$ | Asymptotic pass rate | 渐近通过率（$0 \leq A \leq 1$） | **最终能达到的上限**，决定"天花板" |
| $B$ | Scaling exponent | 扩缩指数（$B > 0$） | **计算效率**，$B$ 越大曲线越陡峭（收敛越快） |
| $C_{\text{mid}}$ | Midpoint compute | 曲线中点对应的计算量 | 达到 $(A + R_0)/2$ 所需的计算量 |
| $C$ | Compute | 当前已消耗的计算量（GPU 小时） | 输入变量 |

> **直觉**：可以将这个公式理解为一个**有上限的增长曲线**。$R_C$ 从 $R_0$ 出发，随着 $C$ 增大趋近于 $A$，增长速度由 $B$ 和 $C_{\text{mid}}$ 共同决定。
>
> - 当 $C \ll C_{\text{mid}}$：$(C_{\text{mid}}/C)^B \gg 1$，分母很大，$R_C \approx R_0$（早期增长缓慢）
> - 当 $C = C_{\text{mid}}$：$(C_{\text{mid}}/C)^B = 1$，$R_C = R_0 + (A-R_0)/2$（恰好达到中点）
> - 当 $C \gg C_{\text{mid}}$：$(C_{\text{mid}}/C)^B \approx 0$，$R_C \approx A$（趋近渐近上限）

### 3.2 为什么选 Sigmoid 而非 Power Law？

预训练领域常用 **Power Law**：$L(N) = aN^{-b}$（损失随模型参数量幂律下降）。但 Power Law 的关键假设是**无界**的——损失可以无限降低。

RL 训练中评估指标通常是 **pass rate（通过率）**，天然有上界 1（100%）。因此：
- **Power Law** 在中长期会高估性能（预测值 > 1）
- **Sigmoid** 自然收敛到 $(0, 1)$ 区间内的渐近值 $A$

此外，RL 训练的早期低算力阶段（前 1,500 GPU 小时）存在较大噪声和不稳定性，**不适合用于拟合**。本文在所有扩缩拟合中均**排除早期低算力阶段**（~1.5k GPU 小时之前的点），从稳定区域开始拟合。

![](ScaleRL_fig3_interpreting_fit.png)

> **Figure 3**: Sigmoid 曲线参数的可视化解释。左图展示 $A$（渐近上限）、$B$（扩缩效率/曲线陡峭度）、$C_{\text{mid}}$（中点位置）三个参数如何影响曲线形态。该图清晰地说明：**$A$ 决定能走多远，$B$ 和 $C_{\text{mid}}$ 决定走多快**。

### 3.3 对数空间线性化

为了方便拟合和外推，本文将 Sigmoid 公式变形为对数线性形式：

$$\mathcal{F}(R_C) = C_{\text{mid}}^B \times \left[ \frac{A - R_0}{R_C - R_0} - 1 \right]^{-1} = C^B$$

取对数后：

$$\log \mathcal{F}(R_C) = B \cdot \log C$$

> **直觉**：变形后斜率直接就是 $B$。在 $\log C$ 对 $\log \mathcal{F}(R_C)$ 的坐标系中，数据应该呈一条直线，斜率为 $B$。这使得不仅便于最小二乘拟合，还**直观展示了扩缩效率**——越大的 $B$ 意味着越陡的斜率（越快的收敛速度）。本文多个实验采用了这种可视化形式。

---

## 4. 大规模消融实验：发现 design choice 的影响

本节总结 §3 的系统性消融，将 design choices 分为三类：
- **影响渐近性能 $A$**：改变最终能达到的上限
- **影响计算效率 $B$**：改变收敛速度，但不改变上限
- **两者兼有影响**

### 4.1 Off-Policy 算法架构

![](ScaleRL_fig4a_infra.png)

> **Figure 4a**: 两种异步 off-policy RL 架构对比。上图为 **PPO-off-policy**：Trainer 完成参数更新后推送到 Generator，Generator 拉取最新模型生成 rollout。下图为 **PipelineRL**：Generator 和 Trainer 独立运行，通过流水线（pipeline）缓冲旧数据，允许更大的 off-policyness。

**关键发现**：PipelineRL 与 PPO-off-policy 达到**相似的渐近性能 $A$**，但 PipelineRL **显著提升了计算效率 $B$**（收敛更快）。这说明 off-policy 架构本身主要影响效率，不会降低最终上限。

![](ScaleRL_fig4b_pipelinerl_offpolicy.png)

> **Figure 4b**: PipelineRL 在不同最大 off-policyness 步数 $k$ 下的对比。$k$ 表示 Generator 产生的 rollout 最多滞后 Trainer 多少步。实验证明 **$k=8$ 为最优值**——更大的 $k$ 不会继续提升效率，反而可能导致不稳定。

### 4.2 损失函数对比：DAPO vs GSPO vs CISPO

本文对比了三种主流损失函数设计：

#### DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）

$$\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}\left[ \frac{1}{T} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \min\left( \rho_{i,t}(\theta) \hat{A}_i,\; \mathrm{clip}_{\text{asym}}(\rho_{i,t}(\theta), \epsilon^-, \epsilon^+) \hat{A}_i \right) \right]$$

其中 $T = \sum_{i=1}^G |y_i|$ 为所有响应的 token 总数。

| 特性 | 说明 |
|------|------|
| 聚合方式 | Token-level（按总 token 数 $T$ 归一化） |
| 裁剪 | 非对称裁剪 $\mathrm{clip}_{\text{asym}}$ |
| 优势 | $\hat{A}_i$（未做组标准差归一化） |

#### GSPO（Group-relative policy optimization via importance Sampling with a POisson）

GSPO 采用**序列级（sequence-level）**重要性采样比：

$$\rho_i(\theta) = \frac{\pi_{\text{train}}(y_i \mid x, \theta)}{\pi_{\text{gen}}(y_i \mid x, \theta_{\text{old}})} = \prod_{t=1}^{|y_i|} \rho_{i,t}(\theta)$$

目标函数使用序列级 $\rho_i$ 而非 token 级 $\rho_{i,t}$。

#### CISPO（Clipped Importance Sampling Policy Optimization）

$$\mathcal{J}_{\text{CISPO}}(\theta) = \mathbb{E}_{\begin{subarray}{c}x \sim D,\\ \{y_i\}_{i=1}^G \sim \pi_{\text{gen}}(\cdot \mid x, \theta_{\text{old}})\end{subarray}} \left[ \frac{1}{T} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \texttt{sg}\left(\min(\rho_{i,t}, \epsilon_{\max})\right) \cdot \hat{A}_i \cdot \log\left(\pi_{\text{train}}(y_{i,t} \mid x, y_{i,<t}, \theta)\right) \right]$$

| 符号 | 含义 |
|------|------|
| $\texttt{sg}(\cdot)$ | stop gradient（不计算梯度），确保裁剪只在优势权重中生效 |
| $\min(\rho_{i,t}, \epsilon_{\max})$ | 将重要性采样比硬上限截断到 $\epsilon_{\max}$ |
| $\hat{A}_i$ | 优势函数（prompt-level 或 batch-level 归一化） |
| $\log \pi_{\text{train}}(\cdot)$ | 标准策略梯度（REINFORCE）的对数概率 |

> **CISPO 直觉**：
> 1. 用 **truncated importance sampling**（截断后冻结梯度）替代 PPO-clip 的 $\min$ 操作
> 2. 基础形式是 **REINFORCE + IS 加权**，而不是 PPO 的 clip  surrogate
> 3. $\texttt{sg}(\min(\rho, \epsilon_{\max}))$ 意味着：用截断后的 $\rho$ 作为优势权重，但不让裁剪操作本身影响策略梯度方向——这降低了训练的敏感性

![](ScaleRL_fig5a_loss_type.png)

> **Figure 5a**: DAPO、GSPO、CISPO 三种损失函数的扩缩曲线对比。**CISPO 和 GSPO 达到更高的渐近性能 $A$**，而 CISPO 在训练后期呈现出更持久的近线性增长趋势，最终略优于 GSPO。DAPO 收敛更快但上限更低。

**关键结论**：
- **CISPO > GSPO > DAPO**（按渐近性能 $A$）
- CISPO 对超参数 $\epsilon_{\max}$ 的选择**显著更鲁棒**（insensitive）
- 损失函数选择**直接影响渐近性能 $A$**（不仅仅是效率）

### 4.3 FP32 精度：常被忽视的巨大收益

![](ScaleRL_fig5b_fp32.png)

> **Figure 5b**: LM Head（最后一层输出 logits 的层）使用 FP32 vs BF16 精度的扩缩曲线对比。FP32 将渐近性能 $A$ 从 **0.52 提升到 0.61**，是绝对提升最显著的单项设计选择之一。

| 条件 | 渐近上限 $A$ |
|------|-------------|
| BF16（默认） | 0.52 |
| FP32（仅 LM Head） | 0.61 |
| **绝对提升** | **+0.09 (+17.3%)** |

> **直觉**：RL 训练涉及大量重要性采样比的计算（概率比值的对数差），BF16 的数值精度可能导致概率估计出现系统性偏差（尤其是小概率 token），这种偏差在 RL 的 multi-step 更新中被放大。仅将**最后一层 LM Head** 改为 FP32 就能解决绝大多数数值问题，同时计算开销极小。

**注意**：这个发现在 **Scout MoE（17B×16）** 上同样成立，说明其普适性。

### 4.4 损失聚合方式

| 聚合方式 | 公式 | 结论 |
|----------|------|------|
| Sample-level | $\frac{1}{G} \sum_i \frac{1}{|y_i|} \sum_t \dots$（GRPO 默认） | 基准 |
| Prompt-level | $\frac{1}{T} \sum_i \sum_t \dots$，$T=\sum_i |y_i|$（DAPO 默认） | **渐近性能最高**，采用 |
| Token-level | $\frac{1}{\sum_g |y_g|} \sum_i \sum_t \dots$ | 与 prompt-level 接近 |

> **直觉**：Prompt-level 聚合将所有 token 平等对待（不按样本数平均），对长响应更公平，避免了 sample-level 中短响应对 loss 的过度稀释。

### 4.5 优势归一化方式

| 方式 | 做法 | 结论 |
|------|------|------|
| Prompt-level（GRPO） | 仅在同 prompt 的 $G$ 条响应内归一化 | 基准 |
| Batch-level | 在整个 batch 的所有响应上计算 mean/std | **理论上更合理，实际略优**，采用 |
| No normalization（Dr. GRPO） | 不除以 std | 性能接近 |

三者性能差异不大，本文选择 **batch-level** 因其理论基础更坚实。

### 4.6 Zero-Variance Filtering（零方差过滤）

问题：在 GRPO 风格的优势计算中，
$$\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_j\})}{\mathrm{std}(\{r_j\}) + \epsilon}$$

当同 prompt 的所有 $G$ 条响应全部正确（$r_j=1$）或全部错误（$r_j=0$）时，$\mathrm{std}=0$，优势无法计算。默认做法是将这些 prompt 保留但优势=0，或依赖 $\epsilon$ 兜底。

**"Effective batch" 做法**：在计算 loss 时**排除**所有响应奖励全同的 prompt。

![](ScaleRL_fig6a_0var_filtering.png)

> **Figure 6a**: Zero-variance filtering 的扩缩曲线。使用 effective batch（排除零方差 prompt）显著提升了**渐近性能 $A$**。这些 prompt 提供的梯度信号本就很差（所有响应都一样好/坏），保留它们反而稀释了有效梯度。

### 4.7 Adaptive Prompt Filtering（No-Positive-Resampling）

问题：如果某个 prompt 历史 pass rate 已经很高（≥ 0.9），继续训练它贡献的梯度信号很小（几乎所有响应都很接近最优）。

**No-Positive-Resampling** 策略：在后续 epoch 中，过滤掉 pass rate ≥ 0.9 的 prompt。

![](ScaleRL_fig6b_noposresample.png)

> **Figure 6b**: No-Positive-Resampling 的扩缩曲线。过滤高 pass rate prompt 后，**渐近性能 $A$ 提升**。这说明 curriculum / 动态数据筛选不仅影响效率，也会影响最终上限——将计算资源集中在"仍有学习空间"的 prompt 上。

---

## 5. ScaleRL：最佳实践配方

综合 §3 所有消融结论，本文提出 **ScaleRL** 配方：

| 组件 | ScaleRL 选择 | 理由 |
|------|-------------|------|
| **异步架构** | PipelineRL-8 | 最优 off-policyness 效率 |
| **长度控制** | Interruption-based（12,288 thinking budget） | 防止生成长度爆炸 |
| **精度** | FP32 for LM Head logits | $A$ 从 0.52→0.61 |
| **损失聚合** | Prompt-level | 最高渐近性能 |
| **优势归一化** | Batch-level | 理论合理且略优 |
| **损失函数** | CISPO | 最高 $A$ + 超参数鲁棒 |
| **零方差过滤** | 启用（effective batch）| 提升 $A$ |
| **动态过滤** | No-Positive-Resampling（pass_rate < 0.9）| 提升 $A$ |

### 5.1 ScaleRL 完整目标函数

$$\mathcal{J}_{\text{ScaleRL}}(\theta) = \mathbb{E}_{x, \{y_i\}} \left[ \frac{1}{\sum_{g=1}^G |y_g|} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \texttt{sg}\left(\min(\rho_{i,t}, \epsilon_{\max})\right) \cdot \hat{A}_i^{\text{norm}} \cdot \log \pi_{\text{train}}^{\theta}(y_{i,t}) \right]$$

**约束条件**（effective batch）：
- $0 < \mathrm{mean}(\{r_j\}) < 1$（排除零方差 prompt）
- $\text{pass\_rate}(x) < 0.9$（No-Positive-Resampling）

其中：
$$\rho_{i,t} = \frac{\pi_{\text{train}}^{\theta}(y_{i,t})}{\pi_{\text{gen}}^{\theta_{\text{old}}}(y_{i,t})}, \quad \hat{A}_i^{\text{norm}} = \frac{\hat{A}_i}{\hat{A}_{\text{std}}}$$

### 5.2 Leave-One-Out（LOO）消融验证

LOO 实验设计：在 16,000 GPU 小时的训练中，将前 8,000 GPU 小时的曲线用于 **Sigmoid 拟合 + 外推**，预测后 8,000 GPU 小时的性能，与实际对比。

技巧：将 Sigmoid 公式重排为 $\mathcal{F}(R_C) = C_{\text{mid}}^B \left[ \frac{A-R_0}{R_C-R_0} - 1 \right]^{-1} = C^B$，取对数后：

$$\log \mathcal{F}(R_C) = B \cdot \log C$$

在 $\log C$ 对 $\log \mathcal{F}(R_C)$ 的坐标中，斜率直接就是 $B$，呈一条直线。

![](ScaleRL_fig7_LOO.png)

> **Figure 7**: Leave-One-Out 消融实验结果。每個子图展现移除一个 ScaleRL 组件后的扩缩曲线。左侧在原始坐标（pass rate vs. log compute），右侧在对数线性化坐标（$\log \mathcal{F}(R_C)$ vs. $\log C$，斜率即 $B$）。关键观察：
> - 大多数 LOO 变体达到**相似的渐近上限 $A$**
> - 差异主要体现在**计算效率 $B$**
> - ScaleRL 综合所有最优组件后效率最高（CISPO 的 $B=2.01$ vs DAPO 的 $B=1.77$）
> - FP32、CISPO、No-Positive-Resampling 是对效率和上限都有正向贡献的关键组件

**结论**：LOO 实验确认每个组件都对 ScaleRL 的最终性能有实质性贡献，且 Sigmoid 外推在 2 倍计算量范围内高度可靠。

### 5.3 误差范围分析

![](ScaleRL_fig2_prevalent_methods.png)

> **Figure 2**: ScaleRL 与当前主流 RL 方法的扩缩曲线对比，包括 DeepSeek-R1、Qwen-2.5、Magistral、MiniMax 等代表性方法。ScaleRL 在相同计算量下达到更高性能，展现出比 prevalent methods 更优的扩缩特性。

---

## 6. 多轴扩缩：验证框架的普适性

### 6.1 模型规模（MoE）

在 **Llama-4 Scout 17B×16 MoE** 上验证 ScaleRL：
- 计算量：50,000 GPU 小时
- 步骤数：7,100 步
- 仅需 8B dense 模型 **1/6** 的 RL 计算量即可达到更高性能
- FP32 收益在 MoE 上同样显著

### 6.2 生成长度（Context Budget）

| 长度设置 | 观察 |
|----------|------|
| 14K tokens | 早期效率较高（$B$ 较大），但上限较低 |
| 32K tokens | 早期效率较低（更多无效思考），但最终上限 $A$ 更高 |

> **结论**：长上下文 RL 是**提升天花板**的有效手段（ceiling-raising knob），但代价是前期效率降低。

### 6.3 全局 Batch Size

- 大 batch（2,048 prompts）vs 小 batch：
  - 小 batch **早期看似更好**（$B$ 大）
  - 但随着计算量增长，**大 batch 反超并达到更高上限**

> **结论**：评估 RL 算法时，如果只在小计算量范围比较，可能得出误导性结论（小 batch 占优）；Sigmoid 框架能揭示长期趋势。

### 6.4 每 Prompt 生成数（Generations per Prompt）

在固定总 batch tokens 的条件下，测试 $G \in \{8, 16, 24, 32\}$：
- 拟合的 Sigmoid 曲线**几乎不变**
- 说明在总计算量固定的前提下，$G$ 的配置主要是一个二阶效应

---

## 7. 核心公式速查

| 公式 | 名称 | 关键符号 |
|------|------|----------|
| $R_C = R_0 + (A - R_0) \times \frac{1}{1 + (C_{\text{mid}}/C)^B}$ | **Sigmoid Scaling Law** | $A$: 上限, $B$: 效率, $C_{\text{mid}}$: 中点 |
| $\rho_{i,t} = \pi_{\text{train}} / \pi_{\text{gen}}$ | Importance Sampling Ratio | $\rho_{i,t}$: token 级概率比 |
| $\mathrm{clip}_{\text{asym}}(\rho, \epsilon^-, \epsilon^+)$ | DAPO 非对称裁剪 | $\epsilon^- <$ 下界, $\epsilon^+$ > 上界 |
| $\hat{A}_i = (r_i - \mathrm{mean}(\{r_j\})) / (\mathrm{std}(\{r_j\}) + \epsilon)$ | GRPO 优势 | 组内相对标准化 |
| $\mathcal{J}_{\text{DAPO}} = \mathbb{E}[\frac{1}{T}\sum_i\sum_t \min(\rho_{i,t}\hat{A}_i, \mathrm{clip}_{\text{asym}}(\rho)\hat{A}_i)]$ | DAPO 目标 | Token 级聚合 + 非对称裁剪 |
| $\mathcal{J}_{\text{CISPO}} = \mathbb{E}[\frac{1}{T}\sum_i\sum_t \texttt{sg}(\min(\rho_{i,t}, \epsilon_{\max})) \hat{A}_i \log \pi_{\text{train}}]$ | **CISPO 目标** | Truncated IS + REINFORCE |
| $\mathcal{J}_{\text{ScaleRL}} = \text{CISPO} + \text{Prompt-level} + \text{Batch-norm} + \text{Filters}$ | **ScaleRL 目标** | 综合最优配方 |

---

## 8. 局限性与未来方向

1. **单一领域**：目前实验集中在可验证数学推理（binary reward），在开放式生成（非二元奖励，如 RLHF 中的偏好模型奖励）上的适用性有待验证。
2. **固定模型架构**：Sigmoid 框架假设模型容量固定，未联合建模 "模型规模 + RL 计算" 的双轴扩展。
3. **数据规模**：Polaris-53K 数据集相对较小，更大规模数据上 Sigmoid 参数可能不同。
4. **领域外推**：在代码、科学推理等其他可验证领域是否遵循同样规律需要更多验证。
5. **理论解释**：为什么 CISPO 的渐近上限优于 DAPO？背后的理论机制尚不完全清楚。

---

## 9. 个人思考

### 9.1 方法论层面的启发

这篇论文最深刻的贡献不是某一个具体技巧，而是**为 RL 训练引入了类似预训练 Scaling Law 的系统性思维框架**。长期以来，RL 领域的改进往往是在固定计算量下比较最终性能，这种"点比较"容易得出误导性结论：

- 某些 trick 看似提升了最终 pass rate，实则是**用计算效率换上限**（或反之）
- 在小规模实验中表现好的方法，大规模可能完全失效
- Sigmoid 参数 $A$ 和 $B$ 的解耦提供了一个**正交分析维度**

这个框架的价值在于，今后评价一个 RL 改进时，可以问三个问题：
1. 它提升了 **$A$** 吗？（最终能走多远）
2. 它提升了 **$B$** 吗？（走多快）
3. 它改变了 **$C_{\text{mid}}$** 吗？（中间的转折点在哪）

### 9.2 工程实践层面的启发

几个反直觉但至关重要的发现：

1. **FP32 的重要性被严重低估**：在 BF16 大行其道的今天，仅将 LM Head 改为 FP32 就能将最终上限提升 17%（0.52 → 0.61），这种"免费午餐"级的收益值得所有 RL 训练框架开发者重新审视数值精度选择。

2. **Loss 函数选择比想象中更关键**：CISPO、GSPO、DAPO 三种看似相近的方法，在渐近性能上有实质性差异。这意味着 RL 算法的理论形式仍然有优化空间，REINFORCE-based 的方法可能更适合高算力 regime。

3. **Off-policy 的效率空间巨大**：PipelineRL-8 能在不损失上限的前提下大幅提升效率，这在当前算力越来越贵的背景下极具工程价值。

4. **数据筛选（Curriculum）不只是效率优化**：No-Positive-Resampling 改变了 $A$，说明动态调整训练数据分布可以影响**最终能达到的理论上限**——这暗示了"教什么"和"怎么教"同样重要。

### 9.3 对当前研究方向的关联

对于正在从事大模型 post-training / RL scaling 的研究者，ScaleRL 提供了三个可直接落地的 action items：
- **立即检查 LM Head 的精度设置**（0.5 天工作量，潜在收益巨大）
- **引入 Sigmoid 拟合到日常实验流程**（从此可以评估方法的 long-horizon 收益）
- **认真考虑 CISPO 替代 PPO/DAPO 作为主 RL loss**（尤其在高算力场景）

---

## 10. 关键引用

```bibtex
@article{khatri2025scalerl,
  title={The Art of Scaling Reinforcement Learning Compute for LLMs},
  author={Khatri, Devvrit and Madaan, Lovish and Tiwari, Rishabh and Bansal, Rachit and Duvvuri, Sai Surya and Zaheer, Manzil and Dhillon, Inderjit S. and Brandfonbrener, David and Agarwal, Rishabh},
  journal={arXiv preprint arXiv:2510.13786},
  year={2025}
}
```

---

## 附录：实验图表总览

| Figure | 文件名 | 内容 |
|--------|--------|------|
| Fig 1a | `ScaleRL_fig1a_100k.png` | 100k GPU 小时 Sigmoid 外推（in-distribution validation） |
| Fig 1b | `ScaleRL_fig1b_100k_aime24.png` | AIME-24 下游验证外推 |
| Fig 2 | `ScaleRL_fig2_prevalent_methods.png` | 与 DeepSeek/Qwen/Magistral/MiniMax 等方法对比 |
| Fig 3 | `ScaleRL_fig3_interpreting_fit.png` | Sigmoid 参数 $A, B, C_{\text{mid}}$ 可视化解释 |
| Fig 4a | `ScaleRL_fig4a_infra.png` | PipelineRL vs PPO-off-policy 架构对比 |
| Fig 4b | `ScaleRL_fig4b_pipelinerl_offpolicy.png` | PipelineRL 不同 $k$ 值消融 |
| Fig 5a | `ScaleRL_fig5a_loss_type.png` | DAPO vs GSPO vs CISPO 损失函数对比 |
| Fig 5b | `ScaleRL_fig5b_fp32.png` | FP32 vs BF16 精度对比 |
| Fig 6a | `ScaleRL_fig6a_0var_filtering.png` | Zero-variance filtering 消融 |
| Fig 6b | `ScaleRL_fig6b_noposresample.png` | No-Positive-Resampling 消融 |
| Fig 7 | `ScaleRL_fig7_LOO.png` | Leave-One-Out 综合消融 |
| Fig 9a | `ScaleRL_fig9a_large_scale_gen_len.png` | 生成长度扩缩（14K vs 32K） |
| Fig 10a | `ScaleRL_fig10a_large_scale_bsz.png` | Batch size 扩缩 |
| Fig 11 | `ScaleRL_fig11_large_scale_tasks.png` | Math + Code 联合训练扩缩 |
