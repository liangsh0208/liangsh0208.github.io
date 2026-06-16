---
created: 2026-06-09
paper: https://arxiv.org/abs/2507.18071
authors: Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba Inc.)
tags:
  - RL
  - GSPO
  - GRPO
  - LLM
  - PolicyOptimization
  - MoE
  - Qwen
---

# Group Sequence Policy Optimization

## 一句话总结
GSPO 从理论上识别并修正了 GRPO 中 token 级重要性比率的根本缺陷，提出基于序列似然的 sequence-level clipping, rewarding 和 optimization 策略，在训练稳定性、效率和性能上全面超越 GRPO，并从根本上解决了 MoE 模型 RL 训练中的 expert-activation 不稳定性问题。

---

## 1. 研究背景与动机

### 1.1 问题定义
大语言模型 RL 训练（尤其是长推理链和 MoE 模型上）的不稳定性是制约 RL Scaling 的关键瓶颈。GRPO 作为当前最先进的无 Value Model RL 算法，在大模型长序列训练场景下会遭遇灾难性且不可逆的模型崩溃（catastrophic and irreversible model collapse）。

### 1.2 核心洞察：GRPO 的重要性采样误用

论文指出 GRPO 的不稳定性**根源于重要性采样权重在算法设计中的根本性误用与失效**，而非超参数调优不足。

**关键论证**：

1. **重要性采样的第一条原则**：从行为分布 $\pi_{\text{beh}}$ 中采样以估计目标分布 $\pi_{\text{tar}}$ 的期望时，重要性权重 $\frac{\pi_{\text{tar}}(z)}{\pi_{\text{beh}}(z)}$ 的有效性依赖于**从行为分布中采集足够多的样本**（$N \gg 1$）进行平均，才能有效校正分布偏移。

   $$
   \mathbb{E}_{z \sim \pi_{\text{tar}}}[f(z)] = \mathbb{E}_{z \sim \pi_{\text{beh}}}\left[ \frac{\pi_{\text{tar}}(z)}{\pi_{\text{beh}}(z)} f(z) \right]
   $$

2. **GRPO 的 token-level 悖论**：GRPO 在每个 token 位置 $t$ 上应用重要性权重 $w_{i,t} = \frac{\pi_{\theta}(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x, y_{i,<t})}$。但这个权重是基于**每个 next-token 分布的单一样本**（即仅采样了一个 $y_{i,t}$）构造的，完全无法满足重要性采样"多样本平均"的前提条件。

3. **后果**：token-level 重要性权重不仅无法起到分布校正的作用，反而向训练梯度中注入**高方差噪声**。该噪声随着序列长度增加而累积，并被裁剪机制进一步放大，最终导致模型崩溃——且崩溃后即使回退 checkpoint、调参、切换 query 也无法恢复。

4. **核心原则**：**"the unit of optimization objective should match the unit of reward"**（优化目标的粒度应当与奖励的粒度匹配）。由于奖励授予的是整个序列，off-policy 校正应当在序列级别进行，而非 token 级别。

---

## 2. 方法详解

### 2.1 GSPO 核心目标函数

GSPO 的核心创新是将优化从 token 级别提升到**序列级别**，同时引入长度归一化以统一不同长度 response 的数值范围。

#### 2.1.1 Sequence-level 优化目标

$$
\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( s_i(\theta) \widehat{A}_i, \text{clip}\left(s_i(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_i \right) \right]
$$

| 符号 | 含义 |
|------|------|
| $x$ | query / prompt |
| $y_i$ | 第 $i$ 个采样响应 |
| $G$ | 组大小（同一 query 采样 $G$ 个 response） |
| $s_i(\theta)$ | **序列级重要性比率**（含长度归一化） |
| $\widehat{A}_i$ | 第 $i$ 个响应的组相对优势估计 |
| $\varepsilon$ | 裁剪区间（GSPO 中约为 3e-4 ~ 4e-4） |

#### 2.1.2 组相对优势估计

$$
\widehat{A}_i = \frac{r(x, y_i) - \text{mean}\left(\{r(x, y_i)\}_{i=1}^{G}\right)}{\text{std}\left(\{r(x, y_i)\}_{i=1}^{G}\right)}
$$

与 GRPO 相同：在同一组内对查询 $x$ 的 $G$ 个响应做标准化，奖励高的响应获得正优势，反之负优势。

#### 2.1.3 序列级重要性比率（含长度归一化）

$$
s_i(\theta) = \left( \frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)} \right)^{\frac{1}{|y_i|}} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_{\theta}(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x, y_{i,<t})} \right)
$$

**逐项解释**：

| 部分 | 含义 |
|------|------|
| $\pi_{\theta}(y_i|x)$ | 当前策略下整个响应的似然（所有 token 条件概率的乘积） |
| $\pi_{\theta_{\text{old}}}(y_i|x)$ | 旧策略下整个响应的似然 |
| $\left(\cdot\right)^{\frac{1}{|y_i|}}$ | **长度归一化**：将序列级比率转化为 per-token 平均值，避免不同长度响应的数值范围差异 |
| $\sum_{t=1}^{|y_i|} \log \frac{\pi_{\theta}}{\pi_{\theta_{\text{old}}}}$ | 所有 token 的 log-重要性比率之和 |

**为什么需要长度归一化**：
- 如果不做长度归一化，$\frac{\pi_{\theta}(y_i)}{\pi_{\theta_{\text{old}}}(y_i)}$ 是 $|y_i|$ 个小于 1 或大于 1 的数的乘积，序列越长，数值越极端。
- 归一化后 $s_i(\theta)$ 被控制在一个统一的数值范围内，不同长度的 response 可以用同一组裁剪参数。

### 2.2 梯度分析：GSPO vs GRPO（核心对比）

#### GSPO 的梯度推导（省略 clip）

$$
\nabla_{\theta} \mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}_{x, \{y_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \underbrace{s_i(\theta) \widehat{A}_i}_{\text{序列级权重}} \cdot \underbrace{\frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \nabla_{\theta} \log \pi_{\theta}(y_{i,t}|x, y_{i,<t})}_{\text{各 token 梯度}} \right]
$$

#### GRPO 的梯度推导

$$
\nabla_{\theta} \mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \widehat{A}_i \cdot \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \underbrace{\frac{\pi_{\theta}(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x, y_{i,<t})}}_{\text{token 级不均匀权重}} \nabla_{\theta} \log \pi_{\theta}(y_{i,t}|x, y_{i,<t}) \right]
$$

**核心差异对比**：

| 维度 | GRPO | GSPO |
|------|------|------|
| **梯度权重方式** | 各 token 按自身的 token-level 重要性比率分别加权 | **同一条 response 内所有 token 共享同一个序列级权重 $s_i(\theta)$** |
| **token 权重范围** | $\frac{\pi_\theta}{\pi_{\theta_\text{old}}} \in (0, 1+\varepsilon]$（正优势时）或 $[1-\varepsilon, +\infty)$（负优势时） | 所有 token 权重相同，数值稳定 |
| **不稳定性来源** | token 间不均匀权重的累积，导致梯度不可预测 | 消除了不均匀权重，梯度估计更可靠 |
| **与奖励粒度匹配** | token 级优化 vs 序列级奖励（不匹配） | **序列级优化 vs 序列级奖励（匹配）** |

### 2.3 GSPO-token 变体（Token-level 的兼容方案）

为了兼容需要 token-wise 优势调整的场景（如多轮 RL），论文提出了 GSPO-token 变体。

$$
\mathcal{J}_{\text{GSPO-token}}(\theta) = \mathbb{E}_{x, \{y_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( s_{i,t}(\theta) \widehat{A}_{i,t}, \text{clip}\left(s_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon\right) \widehat{A}_{i,t} \right) \right]
$$

其中 token 级重要性比率定义为：

$$
s_{i,t}(\theta) = \text{sg}\left[s_i(\theta)\right] \cdot \frac{\pi_{\theta}(y_{i,t}|x, y_{i,<t})}{\text{sg}\left[\pi_{\theta}(y_{i,t}|x, y_{i,<t})\right]}
$$

**逐项解释**：

| 符号 | 含义 |
|------|------|
| $\text{sg}[\cdot]$ | stop gradient（detach，只取数值，不参与反向传播） |
| $\text{sg}[s_i(\theta)]$ | 序列级比率的数值（detach 后），主导梯度缩放 |
| $\frac{\pi_\theta}{\text{sg}[\pi_\theta]}$ | token 级条件概率与其 detach 版本的比值（数值为 1，梯度仅来自 $\pi_\theta$ 分子） |

**关键性质**：由于 $\frac{\pi_\theta}{\text{sg}[\pi_\theta]}$ 数值恒为 1，$s_{i,t}(\theta)$ 的数值等于 $s_i(\theta)$。当所有 token 的 advantage 设为同一值时，GSPO-token 与 GSPO 在目标函数、裁剪条件和理论上完全相同。**GSPO-token 仅在需要 per-token 不同 advantage 时提供额外灵活性**。

---

## 3. 实验结果

### 3.1 实验设置

| 项目 | 配置 |
|------|------|
| 基础模型 | Qwen3-30B-A3B-Base 的 cold-start 模型 |
| 评估基准 | AIME'24（32 采样平均 Pass@1）、LiveCodeBench（202410-202502，8 采样平均 Pass@1）、CodeForces（Elo Rating） |
| Mini-batch | 每批 rollout 数据划分为 4 个 mini-batches |
| GSPO 裁剪范围 | 左 3e-4，右 4e-4 |
| GRPO 裁剪范围 | 左 0.2，右 0.27（精心调参后的公平对比） |
| MoE 特殊设置 | GRPO 需要 Routing Replay 策略才能正常收敛 MoE 训练；GSPO 不需要 |

### 3.2 主实验结果

 className="float-left"

![](GSPO_fig1_training_curves.png)
> **Figure 1**: GSPO 与 GRPO 的训练曲线对比。GSPO 训练全程稳定，通过增加训练计算量、定期更新 query set、延长生成长度可持续提升性能。在同等训练计算量和 query 消耗下，GSPO 的训练准确率和基准测试性能均优于 GRPO。

**核心结论**：
1. GSPO 训练全程稳定收敛。
2. 在相同训练计算量和已消耗 query 下，GSPO 达到更好的训练准确率和基准测试性能。
3. GSPO 具备通过增加计算量、扩展生成长度等方式**持续性能提升**的能力。
4. GSPO 已成功应用于 Qwen3 模型的 RL 训练中，证明了其在超大规模 RL 训练中的有效性。

### 3.3 消融实验：Clipping Fractions

![](GSPO_fig2_clipping_fractions.png)
> **Figure 2**: GSPO 与 GRPO 的被裁剪 token 比例。两者之间存在**两个数量级**的差异——GSPO 裁剪了远远更多的 token，但训练效率反而更高。

**反直觉发现**：
- GSPO 的序列级裁剪会将**整个 response** 排除在梯度估计之外，导致被裁剪 token 比例远高于 GRPO。
- 然而，裁剪更多 token 的 GSPO 反而训练效率更高。
- 这进一步说明 **GRPO 的 token-level 梯度估计本质上就是有噪声且低效的**，即便保留更多 token 供训练，这些 token 提供的信号质量也远不如 GSPO 序列级筛选后的信号。
- **结论**：数量的多少不如质量的好坏；序列级信号比 token 级噪声更有价值。

### 3.4 MoE 训练的根本性改善

#### 背景：MoE 的 expert-activation 不稳定性

在 MoE 模型中，每次 RL 梯度更新后，同一 rollout 样本在新旧策略下的 activated experts 可能发生显著变化。对于 48 层的 Qwen3-30B-A3B-Base 模型，**每次 RL 梯度更新后约有 10% 的 activated experts 不同**。

这使得 GRPO 的 token-level 重要性比率 $\frac{\pi_\theta(y_{i,t})}{\pi_{\theta_\text{old}}(y_{i,t})}$ 剧烈波动，进一步失效，阻碍 RL 正常收敛。

#### 先前方案：Routing Replay

缓存旧策略的 activated experts 并在计算 token-level 比率时 replay 这些路由模式。虽然有效，但：
- 增加额外的内存和通信开销
- 可能限制 MoE 模型的实际容量

#### GSPO 的优势

![](GSPO_fig3_routing_replay.png)
> **Figure 3**: Routing Replay 在 GRPO MoE 训练收敛中的关键作用。GSPO 摆脱了对 Routing Replay 的依赖，仅关注序列似然 $\pi_\theta(y_i|x)$，对单个 token 似然不敏感。

**关键洞察**：由于 MoE 模型始终维持其语言建模能力（整体序列似然不会剧烈波动），GSPO 的序列级重要性比率不受单个 token 路由变化的影响。

**GSPO 带来的改变**：
1. **根本解决** expert-activation volatility，无需 Routing Replay
2. 简化训练流程并稳定训练
3. 释放 MoE 模型的完整容量

### 3.5 RL 基础设施简化

由于训练引擎（如 Megatron）和推理引擎（如 SGLang、vLLM）之间存在精度差异，实践中通常需要用训练引擎重新计算 old policy 下的 token 级似然。

GSPO 仅使用序列级似然，**对精度差异的容忍度远高于 token 级似然**。因此：
- 可以直接使用推理引擎返回的序列级似然进行优化
- 避免训练引擎的重新计算
- 在 partial rollout、多轮 RL、训练-推理解耦框架中特别有益

---

## 4. 局限性与未来方向

1. **裁剪范围仍需调参**：GSPO 的裁剪范围（~3e-4）与 GRPO（~0.2）存在数量级差异，这反映了序列级比率与 token 级比率数值范围的本质不同。理论上是否存在自适应裁剪策略？
2. **长度归一化的假设**：长度归一化对所有 token 一视同仁，但在某些任务中不同位置的 token 重要性可能不同。
3. **Token-level 优势的兼容性**：GSPO-token 变体提供了向后兼容，但在复杂场景（如多轮对话、分层奖励）中的实际表现待进一步验证。
4. **无代码仓库**：论文未提供公开的代码实现。

---

## 5. 个人思考

### 5.1 理论之美
GSPO 的核心洞察不是"增加一个新技巧"，而是**识别并修正了 GRPO 在重要性采样基本原理上的根本性误用**。这种"回归第一性原理"的思考方式极具启发性：

- 重要性采样的有效性前提（多样本平均）被违反
- 优化目标粒度与奖励粒度不匹配
- 修复方式是回到序列级，并在序列级重新做 clip

### 5.2 "裁剪更多反而更好"的深层含义
这是一个非常反直觉但有深刻洞察的现象。GRPO 保留了更多 token 看似"信息丰富"，但这些 token 的梯度权重因 token-level 重要性噪声而被严重污染；GSPO 虽然裁剪掉更多 token，但保留的信号**信噪比**远更高。这再次验证了"质量 > 数量"。

### 5.3 对 MoE 的特殊意义
MoE 模型的 expert-activation 不稳定性是 RL Scaling 的一个具体且现实的阻碍。GSPO 通过序列级抽象天然克服了这一问题，展示了**正确的算法设计可以消除复杂 workaround 的必要性**，这比任何 patch 方案都更有价值。

### 5.4 与 GRPO 的演进关系
如果将 GRPO 视为"通过去掉 Critic 简化 PPO" 的第一步，则 GSPO 是**"修正优化粒度以匹配奖励粒度"** 的第二步。两者共同构成了 LLM Post-Training RL 的核心算法演进路径：

```
PPO (需要 Value Model) 
  → GRPO (去掉 Value Model，保留 token-level 目标)
    → GSPO (修正为 sequence-level 目标，根本解决稳定性)
```

### 5.5 对工程实践的启示
GSPO 的 sequence-level 设计使得 inference engine 可以直接返回序列似然用于训练，这对当前训练-推理分离的 RL 基础设施（如 vLLM/SGLang + Megatron 的组合）是一大利好。可以预见，GSPO 可能成为未来大规模 RL training system 的默认优化目标。

---

## 6. 与前序论文 GRPO 的关系

详见 [GRPO 笔记](GRPO--DeepSeekMath.md)。

| 比较维度 | GRPO (DeepSeek, 2024) | GSPO (Qwen, 2025) |
|---------|-----------------------|-------------------|
| **核心贡献** | 去掉 Critic/Value Model，用组相对优势替代 | 修正优化粒度，从 token 级提升到序列级 |
| **问题意识** | PPO 的显存和值估计问题 | GRPO token-level 比率的方差问题 |
| **理论根基** | 无 Value Model 的经验设计 | 重要性采样的第一性原理 |
| **重要性比率** | token 级：$w_{i,t} = \frac{\pi_\theta(y_{i,t})}{\pi_{\theta_\text{old}}(y_{i,t})}$ | 序列级 + 长度归一化：$s_i(\theta) = \left(\frac{\pi_\theta(y_i)}{\pi_{\theta_\text{old}}(y_i)}\right)^{1/|y_i|}$ |
| **裁剪对象** | 每个 token | 每条 response |
| **裁剪比例** | ~10%（GRPO, $\varepsilon \approx 0.2$） | ~90%+（GSPO, $\varepsilon \approx 3\times 10^{-4}$） |
| **梯度信号** | token 级不均匀加权 | 序列级均匀加权（信噪比更高） |
| **MoE 训练** | 需 Routing Replay 稳定 | **无需额外策略** |
| **工程简化** | 相比 PPO 省显存 | 可直接使用推理引擎序列似然 |
| **适用模型** | DeepSeekMath, DeepSeek-V3 | Qwen3 系列 |

**演进视角**：GSPO 不是对 GRPO 的替代，而是**在正确识别 GRPO 理论基础缺陷后的系统性修正**。两者一脉相承：都使用组采样 + 相对优势估计 + KL 正则，不同之处在于优化的基本单位。从这个角度看，GSPO 是 GRPO 的 natural successor。

---

## 7. 关键引用

```bibtex
@article{zheng2025gspo,
  title={Group Sequence Policy Optimization},
  author={Zheng, Chujie and Liu, Shixuan and Li, Mingze and Chen, Xiong-Hui and Yu, Bowen and Gao, Chang and Dang, Kai and Liu, Yuqiong and Men, Rui and Yang, An and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2507.18071},
  year={2025}
}
```
