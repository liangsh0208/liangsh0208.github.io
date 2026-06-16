---
created: 2026-06-09
paper: https://arxiv.org/abs/2402.03300
code: https://github.com/deepseek-ai/DeepSeek-Math
authors: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo et al. (DeepSeek-AI, Tsinghua, PKU)
tags:
  - RL
  - GRPO
  - Math
  - LLM
  - Reasoning
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

## 一句话总结
DeepSeekMath 7B 通过对 DeepSeek-Coder-Base 进行 120B 数学 token 的继续预训练，并结合去除 Critic 模型的 Group Relative Policy Optimization (GRPO) 算法，在 MATH 竞赛级基准上达到了 51.7%（接近 GPT-4/Gemini-Ultra），同时将 RL 显存占用降至 PPO 的 1/4 以下。

---

## 1. 研究背景与动机

### 1.1 问题定义
数学推理因其复杂性和结构化特性，对语言模型构成重大挑战。闭源顶尖模型（GPT-4、Gemini-Ultra）不公开可用，而开源模型存在显著性能差距。

### 1.2 现有方法的不足
1. **PPO 的内存瓶颈**：PPO 需要一个与策略模型同等大小的 Value Model（Critic），在大模型场景下带来巨额显存开销。
2. **PPO 的值估计不可靠**：获取精确且可扩展的值估计（尤其面对长响应和复杂任务）极为困难。
3. **RLHF 的 Reward Model 范式**：传统偏好评分的 Reward Model 难以直接用于数学推理这类需要明确正确/错误判断的任务。

![](GRPO_fig1_math_benchmark.png)
> **Figure 1**: MATH 基准 Top-1 准确率对比。DeepSeekMath-RL 7B 在没有外部工具包和投票技术的情况下，接近 GPT-4 和 Gemini-Ultra 的水平。

### 1.3 核心策略
- **数据侧**：从 Common Crawl 中通过精心设计的数据选择管道提取 120B 高质量数学相关 token。
- **训练侧**：基于代码预训练模型（DeepSeek-Coder-Base-v1.5）继续进行数学预训练，发现 "code training improves mathematical reasoning" 关键结论。
- **RL 侧**：提出 GRPO，放弃 Value Model，通过组内相对优势估计降低显存占用并提升推理能力。

---

## 2. 方法详解

### 2.1 数据构建：迭代式高质量数学语料提取

![](GRPO_fig2_data_pipeline.png)
> **Figure 2**: 迭代数据采集 pipeline。通过 fastText 分类器迭代筛选、人工审核标注，从 Common Crawl 中提取高质数学网页。

### 2.2 Preliminaries: PPO

PPO 的目标函数：

$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{\text{old}}}(O|q)} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left( \frac{\pi_{\theta}(o_t|q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o_{<t})} \hat{A}_t, \text{clip}\left(\frac{\pi_{\theta}(o_t|q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o_{<t})}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_t \right) \right]
$$

其中每一步的奖励包含 KL 惩罚：

$$
r_t = r_{\phi}(q, o_{\leq t}) - \beta \log \frac{\pi_{\theta}(o_t|q, o_{<t})}{\pi_{\text{ref}}(o_t|q, o_{<t})}
$$

| 符号 | 含义 |
|------|------|
| $\pi_{\theta}$ | 当前策略模型 |
| $q$ | 查询（数学问题） |
| $o$ | 策略生成的输出（解题过程） |
| $\hat{A}_t$ | 第 $t$ 步的优势估计（由 Value Model 计算） |
| $\varepsilon$ | 裁剪参数（通常 0.2） |
| $\beta$ | KL 散度惩罚系数 |
| $\pi_{\text{ref}}$ | 参考策略（通常是 SFT 模型） |

PPO 的核心问题在于需要一个额外的 Value Model 来估计 $\hat{A}_t$，且该模型通常与 Policy Model 规模相当，造成巨大显存负担。

### 2.3 GRPO: Group Relative Policy Optimization

GRPO 是本文的核心贡献之一，其设计思想是**完全去掉 Critic/Value Model**，改为对每个 query 采样一组输出，在组内做相对优势估计。

#### 2.3.1 GRPO 目标函数

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left( \frac{\pi_{\theta}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left( \frac{\pi_{\theta}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{KL}[\pi_{\theta} \| \pi_{\text{ref}}] \right) \right]
$$

**逐项解释**：

| 部分 | 含义 |
|------|------|
| $q \sim P(Q)$ | 从问题分布中采样 query |
| $\{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(O|q)$ | 对同一 query 从旧策略采样 $G$ 个输出（组内采样） |
| $\frac{1}{G} \sum_{i=1}^{G}$ | 对组内各输出取平均 |
| $\frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$ | 对第 $i$ 个输出的所有 token 取平均 |
| $w_{i,t} = \frac{\pi_{\theta}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$ | **token 级重要性比率** |
| $\min(\cdot, \text{clip}(\cdot))$ | PPO 标准裁剪机制，防止策略更新幅度过大 |
| $\hat{A}_{i,t} = \hat{A}_i$ | 同一输出内所有 token 共享相同优势估计 |
| $\beta \mathbb{D}_{KL}[\pi_{\theta} \| \pi_{\text{ref}}]$ | KL 散度正则项 |

#### 2.3.2 组相对优势估计（Outcome Supervision）

$$
\hat{A}_{i,t} = \hat{r}_i = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^{G})}{\text{std}(\{r_i\}_{i=1}^{G})}
$$

**直觉**：不再用 Value Model 估计每个 token 的优势，而是对同一 query 的 $G$ 个输出做组内标准化。如果某个回答的结果奖励高于组内平均，则组内所有 token 都获得正激励；反之则负激励。

| 符号 | 含义 |
|------|------|
| $G$ | 组大小（通常取 64） |
| $r_i$ | 第 $i$ 个输出 $o_i$ 的最终奖励（verifier 判定正确/错误） |
| $\text{mean}(\{r_i\})$ | 组内所有输出奖励的均值 |
| $\text{std}(\{r_i\})$ | 组内奖励的标准差 |

#### 2.3.3 KL 散度无偏估计

论文采用以下无偏估计量替代传统的 $\log$ 形式 KL 散度：

$$
\mathbb{D}_{KL}[\pi_{\theta} \| \pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta}(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta}(o_{i,t}|q, o_{i,<t})} - 1
$$

#### 2.3.4 过程监督变体（Process Supervision）

对每个推理步骤（由换行符等方式切分）提供奖励，优势计算为后续步骤的归一化奖励之和：

$$
\hat{A}_{i,t} = \sum_{\text{index}(j) \geq t} \hat{r}_i^{\text{index}(j)}, \quad \hat{r}_i^{\text{index}(j)} = \frac{r_i^{\text{index}(j)} - \text{mean}(R)}{\text{std}(R)}
$$

#### 2.3.5 GRPO vs PPO 架构对比

![16](GRPO_fig3_ppo_vs_grpo.png)
> **Figure 3**: PPO 与 GRPO 架构对比。PPO 需要额外的 Value Model（与 Policy Model 同等规模）来估计优势；GRPO 完全去掉 Value Model，改为对每个 query 采样多个输出，在组内做相对优势估计。

**GRPO vs PPO 的关键差异**：

| 特性 | PPO | GRPO |
|------|-----|------|
| Value Model (Critic) | 必需 | **不需要** |
| 优势估计来源 | Value Model | 组内相对归一化 |
| 显存占用 | ~2x Policy Model | ~1x Policy Model |
| 奖励粒度假设 | Token-level / 轨迹级 | 输出级 / 步骤级 |
| 采样复杂度 | 单输出 | 每组 $G$ 个输出 |

#### 2.3.6 迭代式 GRPO 算法

```
Algorithm 1: Iterative Group Relative Policy Optimization

输入：初始策略模型 π_θ_init；奖励模型 r_φ；任务提示 D；超参数 ε, β, μ

1. π_θ ← π_θ_init
2. for iteration = 1, ..., I do
3.   π_ref ← π_θ                          # 更新参考策略
4.   for step = 1, ..., M do
5.     从 D 中采样批次 D_b
6.     π_θ_old ← π_θ                      # 更新旧策略
7.     对每个问题 q ∈ D_b：
        采样 G 个输出 {o_i}_{i=1}^G ~ π_θ_old(·|q)
8.     通过 r_φ 计算奖励 {r_i}_{i=1}^G
9.     计算组相对优势估计 Â_{i,t}
10.    for GRPO iteration = 1, ..., μ do
11.      最大化 GRPO 目标更新 π_θ
12.   通过 10% 历史数据 replay 持续训练更新 r_φ
```

---

## 3. 实验设置与核心结果

### 3.1 训练配置

| 模型 | 基础模型 | 训练数据 | 训练 tokens | 学习率 | Batch 大小 |
|------|---------|---------|------------|--------|-----------|
| DeepSeekMath-Base 7B | DeepSeek-Coder-Base-v1.5 7B | 56% DeepSeekMath Corpus + 4% AlgebraicStack + 10% arXiv + 20% Github + 10% NL | 500B | 4.2e-4 | 10M |
| DeepSeekMath-Instruct 7B | DeepSeekMath-Base | 776K SFT 数据 | 500 steps | 5e-5 | 256 |
| DeepSeekMath-RL 7B | DeepSeekMath-Instruct | 144K GSM8K+MATH CoT 问题 | - | 1e-6 | 1024 |

GRPO 设置：KL 系数 $\beta = 0.04$，每组采样 $G = 64$ 个输出，最大长度 1024。

### 3.2 主实验结果（Chain-of-Thought）

| 模型 | 大小 | GSM8K | MATH | MGSM-zh | CMATH |
|------|------|-------|------|---------|-------|
| **闭源基线** |
| GPT-4 | - | 92.0% | 52.9% | - | 86.0% |
| Gemini Ultra | - | 94.4% | 53.2% | - | - |
| **开源基线** |
| DeepSeekMath-Instruct | 7B | 82.9% | 46.8% | 73.2% | 84.6% |
| **DeepSeekMath-RL** | **7B** | **88.2%** | **51.7%** | **79.6%** | **88.8%** |

- DeepSeekMath-RL 7B 达到 MATH 51.7%， Self-consistency over 64 samples 达到 **60.9%**。
- DeepSeekMath-RL 7B **击败了所有 7B-70B 的开源模型**。
- 验证 "code training improves models' ability to solve mathematical problems both with and without tool use"。

### 3.3 Tool-Integrated 结果

| 模型 | GSM8K | MATH |
|------|-------|------|
| GPT-4 Code Interpreter | 97.0% | 69.7% |
| DeepSeekMath-Instruct 7B | 83.7% | 57.4% |
| **DeepSeekMath-RL 7B** | **86.7%** | **58.8%** |

---

## 4. 消融实验关键结论

![](GRPO_fig4_training_methods.png)
> **Figure 4**: 不同训练方法性能对比（Online RFT、GRPO、过程监督、结果监督）。

### 4.1 代码预训练对数学推理的提升

| 训练设置 | GSM8K | MATH | CMATH | HumanEval |
|---------|-------|------|-------|-----------|
| 无持续训练 | 2.9% | 3.0% | 12.3% | 12.2% |
| 通用→数学 | 19.1% | 14.4% | 37.2% | 12.8% |
| **代码→数学（两阶段）** | **21.9%** | **15.3%** | **39.7%** | **12.2%** |
| 代码&数学混合 | 17.6% | 12.1% | 36.3% | **29.3%** |

**结论**：先代码后数学的两阶段预训练效果最优；代码与数学混合训练虽然在纯数学上稍有下降，但 HumanEval 编程能力大幅领先。

### 4.2 arXiv 数据无效

对 GSM8K、MATH 等基准，arXiv 训练 "no notable improvements or even deterioration"。

### 4.3 RL 训练对比

| 对比维度 | 关键发现 |
|---------|---------|
| Online vs Offline RFT | **Online RFT 显著优于 RFT** |
| GRPO vs Online RFT | GRPO 通过差异化梯度系数胜出 |
| 过程监督 vs 结果监督 | **GRPO+PS > GRPO+OS** |
| 迭代 RL | 显著提升性能，尤其第一轮 |

![](GRPO_fig5_iterative_rl.png)
> **Figure 5**: 迭代 RL 性能曲线。第 1 轮迭代 RL 提升最显著，后续轮次渐进提升。

### 4.4 Pass@K vs Maj@K 分析

![](GRPO_fig6_pass_maj.png)
> **Figure 6**: Pass@K 与 Maj@K 对比。"RL enhances Maj@K but not Pass@K"——RL 提升的是输出分布的鲁棒性（多数投票效果更好），而非单样本的基础能力。

---

## 5. 局限性与未来方向

1. **Tool-Integrated 仍有差距**：虽然 CoT 接近 GPT-4，但在使用代码解释器后仍落后 GPT-4 Code Interpreter。
2. **过程监督的标注成本**：PRM（Process Reward Model）的细粒度标注成本高昂。
3. **迭代 RL 成本**：每轮迭代需要重新采样数据集和训练，计算资源消耗大。
4. **长推理链的稳定性**：面对超长推理链的任务，GRPO 的奖励稀疏性问题可能加剧（这也是后来被 GSPO 解决的问题）。

---

## 6. 个人思考

### 6.1 方法的优雅之处
- **"去掉 Critic" 的减法美学**：GRPO 并非堆叠模块，而是通过 Group-relative 估计消除了对 Value Model 的依赖，这是一个典型的"做减法"思路。
- **组内相对归一化极其巧妙**：不需要外部参考，利用同一 query 下不同输出的相对优劣来估计优势。这种自举（self-bootstrapping）思想在 RL 中非常优雅。

### 6.2 局限与深层问题
- **Token-level 重要性比率的隐患**：GRPO 虽然去掉了 Value Model，但在目标函数中保留了 token-level 的重要性比率 $\pi_\theta / \pi_{\theta_{\text{old}}}$。每个采样 token 仅靠单样本估计分布偏移，引入了高方差噪声，在长序列和 MoE 场景下会积累放大，甚至导致模型崩溃。这一问题被后续的 GSPO（2025 Qwen 团队）识别并系统性解决。
- **Reward 设计的挑战**：论文聚焦于 Outcome/Process Supervision 的验证器设计，但实际应用中 verifier 的可靠性（尤其是 Process-level）仍是瓶颈。

### 6.3 实践启示
- **代码→数学的两阶段预训练** 是一个关键洞察，说明跨领域能力的迁移是可行的。
- GRPO 作为 PPO 的轻量替代方案，其思想已在 DeepSeek-V3、R1 等后续模型中被发扬光大，成为 LLM Post-Training RL 的标准范式之一。

---

## 7. 论文演进关系：GRPO → GSPO

| 维度 | GRPO (2024, DeepSeek) | GSPO (2025, Qwen) |
|------|----------------------|-------------------|
| **核心问题** | Value Model 显存开销大 | GRPO token-level 重要性比率引入高方差，长序列/ MoE 场景崩溃 |
| **优化粒度** | Token-level (每个 token 各有重要性比率) | **Sequence-level** (整条 response 统一重要性比率) |
| **重要性比率定义** | $w_{i,t} = \frac{\pi_\theta(y_{i,t})}{\pi_{\theta_\text{old}}(y_{i,t})}$ | $s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}$ |
| **裁剪对象** | Token-by-token | Response-by-response |
| **对 MoE 的兼容性** | 需要 Routing Replay 策略稳定训练 | **根本解决 expert-activation 不稳定性**，无需额外策略 |
| **显存/效率** | 相比 PPO 大幅降低 | 进一步简化，sequence-level 容忍推理-训练引擎精度差异 |
| **算法关系** | 基础范式 | **对 GRPO 的系统性改进** — 从"什么可以被去掉"进化为"什么优化粒度是正确的" |

详见 [GSPO 笔记](GSPO--Group-Sequence-Policy-Optimization.md)。

---

## 8. 关键引用

```bibtex
@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, YK and Wu, Y and others},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```
