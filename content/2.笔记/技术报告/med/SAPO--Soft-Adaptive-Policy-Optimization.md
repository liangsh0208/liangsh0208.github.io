---
created: 2026-06-15
published: 2025-12-01
paper: https://arxiv.org/abs/2511.20347v2
authors: Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba Inc)
tags:
  - LLM-RL
  - Policy-Optimization
  - Group-Based-RL
  - GRPO
  - GSPO
  - 推理能力
---

# Soft Adaptive Policy Optimization (SAPO)

## 一句话总结

SAPO 用 **sigmoid 软门控** 替代 RL 训练中 GRPO/GSPO 的硬裁剪（hard clipping），通过 **非对称温度**（negative token 衰减更快）实现 token 级自适应与序列级一致性的统一，在 MoE 和 Dense 模型上均获得更稳定的训练动态和更高的 Pass@1 性能。

> **声明**：本论文是纯方法论文，重点在算法设计与统一分析框架，对数据管线（SFT 数据、奖励数据构造）几乎未披露，训练超参数披露也有限。缺失信息见「附录：论文未披露的关键信息」。

| SAPO vs. 基线 | 核心差异 |
|----------------|---------|
| vs GRPO | 以光滑 $\mathrm{sech}^2$ 门控替代硬裁剪，避免 all-or-nothing 梯度截断 |
| vs GSPO | 在序列级平滑门控基础上保留 token 级自适应性，异常 token 不拖累整条序列 |

---

## 1. 研究背景与动机

### 1.1 问题定义

LLM 的推理能力主要依赖强化学习（RL）提升。当前最实用的范式是**组内相对策略优化（group-based policy optimization）**：对每个 query 采样 $G$ 个 response，计算组内奖励标准化后的优势值 $\widehat{A}_i$，用重要性比率 $r_{i,t}$ 加权策略梯度更新。

### 1.2 现有方法的不足

- **GRPO** [DeepSeekMath]：token 级硬裁剪，对 $r_{i,t}$ 超出 $[1-\varepsilon, 1+\varepsilon]$ 区间直接截断梯度。问题：过紧则浪费样本，过松则引入高方差 off-policy 梯度。
- **GSPO** [Group Sequence Policy Optimization]：序列级硬裁剪，用几何平均序列比率 $s_i(\theta)$ 替代每 token 的比率，长度归一化降低方差。问题：当序列中存在少数严重 off-policy token 时，整条序列被硬裁剪抑制，丢失了大量近 on-policy token 的信息信号。
- **MoE 模型加剧方差**：Mixture-of-Experts 的结构使得 token 级比率差异被路由异质性进一步放大，hard clipping 的 brittleness 更突出。

### 1.3 本文核心思路

SAPO 提出**平滑、温度控制的 sigmoid 门控**，兼具：
1. **序列级一致性**：序列内 token log-ratio 低方差时，token 门控的平均值集中为序列级 soft gate；
2. **token 级自适应性**：方差大/异常 token 存在时，soft gate 只衰减异常 token，保留正常 token 梯度，避免 GSPO 的序列级"一刀切"。

---

## 2. 数据准备与策展

> **说明**：本文作为纯方法论文，对数据管线几乎未作披露，以下为论文信息空白说明。

### 2.1 数据来源

- 数学推理 RL 实验使用 AIME25、HMMT25、BeyondAIME 等公开基准作为 validation。
- Qwen3-VL 的大规模训练涉及多任务混合（数学、代码、逻辑推理），但各任务的数据量、采样比例、来源均未披露。

### 2.2 缺失信息

- SFT/冷启动阶段的训练数据来源与规模；
- RL 阶段的 rollout 数据构造方式（规则奖励 vs 模型奖励）；
- 是否使用了拒绝采样（Rejection Sampling）、Best-of-N 等数据增强；
- 数据去重/过滤方法。

---

## 3. 训练管线详解

### 3.1 基础模型

- 控制实验：从 **Qwen3-30B-A3B-Base** 冷启动微调；
- 大规模验证：Qwen3-VL 系列，覆盖 MoE 和 Dense 架构；
- 论文未披露 base model 的具体参数系列细节（如 dense 基座是否来自 Qwen3-4B）。

### 3.2 RL 阶段

#### 3.2.1 目标函数

SAPO 的整体目标函数：

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{y_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)}\left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} f_{i,t}(r_{i,t}(\theta)) \widehat{A}_{i,t} \right]$$

其中：

| 符号 | 含义 |
|------|------|
| $G$ | 每 query 采样 response 数（group size） |
| $r_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t} \mid q, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid q, y_{i,<t})}$ | token 级重要性比率 |
| $\widehat{A}_{i,t} = \widehat{A}_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$ | 组内标准化优势（per-response 共享） |
| $f_{i,t}(x) = \sigma(\tau_{i,t}(x-1)) \cdot \frac{4}{\tau_{i,t}}$ | 软门控函数（sigmoid 形状） |
| $\sigma(x) = 1/(1+e^{-x})$ | sigmoid 函数 |
| $\tau_{i,t} = \begin{cases} \tau_{\text{pos}} & \text{if } \widehat{A}_{i,t} > 0 \\ \tau_{\text{neg}} & \text{otherwise} \end{cases}$ | **非对称温度**（关键设计） |

#### 3.2.2 梯度形式

对目标求导后，梯度为加权的 log-policy gradient：

$$\nabla_{\theta} \mathcal{J}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} w_{i,t}(\theta) \cdot r_{i,t}(\theta) \cdot \nabla_{\theta} \log \pi_{\theta}(y_{i,t} \mid q, y_{i,<t}) \cdot \widehat{A}_{i,t} \right]$$

其中梯度权重 $w_{i,t}(\theta)$ 为：

$$w_{i,t}(\theta) = 4 \, p_{i,t}(\theta) \, (1 - p_{i,t}(\theta)), \quad p_{i,t}(\theta) = \sigma\!\left(\tau_{i,t}\,(r_{i,t}(\theta)-1)\right)$$

**关键性质**：
- 当 $r_{i,t}(\theta) = 1$（完全 on-policy）时，$w_{i,t}(\theta) = 1$，梯度与未裁剪目标完全一致；
- $w_{i,t}(\theta)$ 随 $r_{i,t}$ 偏离 1 而**平滑衰减**，近似指数衰减而非硬截断。

#### 3.2.3 为什么 $\tau_{\text{neg}} > \tau_{\text{pos}}$？

论文给出了**梯度传播分析**（对 softmax logit 的导数），证明：

$$\frac{\partial \log \pi_{\theta}(y_{i,t} \mid \cdot) \cdot \widehat{A}_{i,t}}{\partial z_v} = \begin{cases} (1 - \pi_{\theta}(y_{i,t} \mid \cdot)) \cdot \widehat{A}_{i,t} & v = y_{i,t} \text{ (sampled)} \\ -\pi_{\theta}(v \mid \cdot) \cdot \widehat{A}_{i,t} & \text{otherwise} \end{cases}$$

直觉：**负优势 token 的梯度会提升大量未采样 token 的 logit**，因为 action space 极大（$|\mathcal{V}| \sim 10^5$），而 desirable actions 极少。这种"多目标 logit 提升"天然比正优势 token 更不稳定。因此，给负 token 设置更大的温度 $\tau_{\text{neg}} > \tau_{\text{pos}}$，使负 token 的梯度权重衰减更快。

实验验证（Figure 5）：$\tau_{\text{neg}} = 1.05$（正 1.0）最稳定；$\tau_{\text{neg}} = 0.95$ 导致显著不稳定；对称 $\tau_{\text{neg}} = \tau_{\text{pos}} = 1.0$ 居中。

#### 3.2.4 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| $\tau_{\text{pos}}$ | 1.0 | 正优势 token 温度 |
| $\tau_{\text{neg}}$ | 1.05 | 负优势 token 温度 |
| 每 batch rollout 数据分成 mini-batch | 4（控制实验）/ 2（大规模） | 梯度更新 |

---

## 4. 方法

### 4.1 核心思想与架构

SAPO 的本质是一个**连续信任区域（continuous trust region）**设计：

- 近 on-policy 区域：梯度完全保留，鼓励有效更新和探索；
- 偏离区域：梯度平滑衰减，维持学习信号的同时降低优化噪声；
- 极端偏离区域：梯度趋于 0，但不会像硬裁剪那样突然归零。

### 4.2 算法详解（含公式 + 符号解释）

#### 4.2.1 门控函数视角的统一框架

论文提出用**统一门控函数框架**理解 GRPO、GSPO 和 SAPO：

$$\mathcal{J}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} f_{i,t}(r_{i,t}(\theta)) \cdot \widehat{A}_{i,t} \right]$$

三种算法对应不同的 $f_{i,t}$：

| 算法 | 门控 $f_{i,t}$ | 导数 $f'_{i,t}$ | 特点 |
|------|---------------|------------------|------|
| **SAPO** | $\frac{4}{\tau_i} \sigma(\tau_i(r_{i,t}-1))$ | $\mathrm{sech}^2\!\left(\frac{\tau_i}{2}(r_{i,t}-1)\right)$ | 光滑软门控 |
| **GRPO** | $\min(r_{i,t}, 1+\varepsilon)$ 或 $\max(r_{i,t}, 1-\varepsilon)$ | 二元门控：区间内=1，区间外=0 | 硬裁剪 |
| **GSPO** | 同上但序列比率 $s_{i,t}$ 替代 $r_{i,t}$；token-invariant | 同上，但作用于整条序列 | 序列级硬裁剪 |

#### 4.2.2 SAPO → GSPO 的归约（在 A1+A2 假设下）

设：
- (A1) 小步近 on-policy：$r_{i,t}(\theta) \approx 1$，故 $\log r_{i,t} \approx r_{i,t} - 1$；
- (A2) 序列内低方差：$\mathrm{Var}_i(\theta) = \frac{1}{|y_i|}\sum_t (z_{i,t} - \mu_i)^2$ 很小。

则 SAPO 的梯度权重 $f'^{\text{SAPO}}_{i,t}(r_{i,t}(\theta)) = \mathrm{sech}^2\left(\frac{\tau_i}{2}(r_{i,t}-1)\right)$ 对整条序列取平均后，由 Taylor 展开得：

$$\underbrace{\frac{1}{|y_i|}\sum_t g_{\tau_i}(z_{i,t})}_{\text{average token gate}} \approx g_{\tau_i}(\mu_i) + \underbrace{\frac{\tau_i^2}{4} \mathrm{Var}_i(\theta)}_{\text{bounded error}}$$

即：在 (A1+A2) 下，SAPO 退化为 **GSPO-like 序列级更新**，但门控是光滑的 $\mathrm{sech}^2$ 而非分段常数的硬裁剪。**当 A2 被违反（序列内存在异常 token）时，SAPO 不摧毁整条序列，只衰减异常 token**——这是相比 GSPO 的本质优势。

#### 4.2.3 SAPO → GRPO 的对比

GRPO 的梯度权重为硬门控（式 24）：

$$f'^{\text{GRPO}}_{i,t} = \begin{cases} 1 & \text{if on-policy} \\ 0 & \text{if off-policy} \end{cases}$$

SAPO 用 $\mathrm{sech}^2\left(\frac{\tau_i}{2}(r_{i,t}-1)\right)$ 替代二元门控，避免了紧邻裁剪边界的 token 从全梯度突降至零梯度的数值不稳定性。

### 4.3 关键设计选择

#### Why sigmoid × $4/\tau$？

- 因子 $\frac{4}{\tau_i}$ 的选取使得在 $r_{i,t} = 1$ 处的梯度精确等于未裁剪目标 $r_{i,t} \widehat{A}_{i,t}$ 的梯度，保证 on-policy 行为不因门控而改变。

#### Why $\tau_{\text{neg}} > \tau_{\text{pos}}$？

- 正更新只抑制单个采样 token 并降低其他 token logit，影响范围可控；
- 负更新（提升大量未采样 token logit）在 off-policy 时容易引入噪声。更大温度 → sigmoid 更陡峭 → 负 token 门控衰减更快 → 更稳定。

---

## 5. 实验结果

### 5.1 实验设置

- **Base Model**：Qwen3-30B-A3B-Base（控制实验）；Qwen3-VL-30B-A3B（大规模验证）
- **任务**：数学推理（AIME25、HMMT25、BeyondAIME）、代码（LiveCodeBench v6）、逻辑推理（ZebraLogic）、多模态推理（MathVision）
- **推理配置**：AIME25 Pass@1 with 16 samples；LiveCodeBench Pass@1 with 8 samples
- **基线**：GSPO、GRPO-R2（GRPO + routing replay）
- **SAPO 超参**：$\tau_{\text{pos}} = 1.0$，$\tau_{\text{neg}} = 1.05$

### 5.2 主实验结果

#### 5.2.1 控制实验：Qwen3-30B-A3B 冷启动数学推理

![](SAPO_fig8_training_reward.png)

> **Figure 4**: Qwen3-30B-A3B-Base 冷启动模型上不同 RL 算法的训练奖励与验证性能。SAPO 在所有基准上保持持续稳定提升，GSPO 和 GRPO-R2 均出现早期训练崩溃。

**核心结论**：
- SAPO **稳定的训练动态**：奖励训练曲线持续上升，无明显震荡或崩溃；
- SAPO **不依赖 routing replay**：GRPO-R2 虽然优于纯 GRPO，但仍需 routing replay 辅助稳定；SAPO 天然不需要此工程手段；
- **最终 Pass@1 更高**：在 AIME25 / HMMT25 / BeyondAIME 上均优于基线。

#### 5.2.2 温度消融实验

![](SAPO_fig9_temperature_ablation.png)

> **Figure 5**: SAPO 在三种温度配置下的训练奖励与验证性能。$\tau_{\text{neg}}=1.05 > \tau_{\text{pos}}=1.0$ 最稳定；$\tau_{\text{neg}}=0.95 < \tau_{\text{pos}}=1.0$ 导致显著不稳定（红色曲线）；对称 $\tau_{\text{neg}}=\tau_{\text{pos}}=1.0$ 居中。

#### 5.2.3 大规模验证：Qwen3-VL 训练

![](SAPO_fig10_qwen3vl.png)

> **Figure 6**: Qwen3-VL-30B-A3B 上 SAPO、GSPO、GRPO-R2 的训练奖励与验证性能。SAPO 在所有四个基准（AIME25、LiveCodeBench、ZebraLogic、MathVision）上均获得持续提升并超越基线。

**跨任务一致性**：SAPO 在数学（AIME）、代码（LiveCodeBench）、逻辑（ZebraLogic）、多模态推理（MathVision）上均稳定且优于基线，证明方法不局限于单一任务类型。

### 5.3 SAPO 假设的实证验证

论文验证了 (A1) 近 on-policy 和 (A2) 低序列内方差假设在真实训练中是否成立：

| 假设 | MoE (Qwen3-30B-A3B) | Dense (Qwen3-4B) | 结论 |
|------|----------------------|-------------------|------|
| (A1) $r_{i,t}$ 集中在 1 | 是（直方图峰尖在 1） | 是 | 小步假设成立 |
| (A2) $\mathrm{Var}_i(\theta)$ 低 | 大多 $<0.02$，分布略宽 | 更集中 | MoE 假设稍显宽松，但趋于成立 |
| 预测误差 $D_i(\theta)$ 与方差的关系 | 散点线性相关 | 线性相关 | Taylor 近似误差可控 |

---

## 6. 局限性与讨论

### 6.1 论文自我分析的局限

- SAPO 最终仍可能表现出训练不稳定的迹象，只是比 GRPO/GSPO 出现得更晚、程度更轻；
- 未对**极长序列**（如 >32K reasoning chain）的场景做专门验证；
- MoE 模型虽然假设成立，但其方差分布比 Dense 更宽，soft gate 的近似误差可能更大。

### 6.2 SAPO 方法的深层局限

- **超参数不敏感但非无感**：$\tau_{\text{pos}} = 1.0, \tau_{\text{neg}} = 1.05$ 的调参空间有限，但不同任务/模型规模是否需要不同温度？论文未做系统扫描。
- **Reward hacking 风险未涉及**：SAPO 解决的是优化稳定性，而非奖励信号本身的可操纵性（如规则奖励可被策略利用）。
- **Replace KL penalty？** SAPO 不包含显式 KL divergence penalty，soft gate 本身替代了 KL 约束的作用，但当策略偏离极大时是否足够？论文未讨论。

### 6.3 未来方向

- 探索 task-specific 的动态温度调度；
- 将 SAPO 的 soft gate 思想扩展到其他 group-based RL 变体（如 REINFORCE++、RLOO 等）；
- 结合外部 Verifier 对抗 reward hacking，同时保持 SAPO 的优化稳定性。

---

## 7. 个人思考

### 7.1 方法的优雅之处

- **门控统一框架**是本文最大的理论贡献。将 GRPO、GSPO、SAPO 统一为 $f_{i,t}(r)$ 的选择，不仅澄清了各方法的差异，也为未来设计新门控提供了系统方法论。
- **为什么选 sigmoid + sech²？** 因为 sigmoid 的导数 $\sigma(1-\sigma)$ 天然对称且在 0 处取最大值，其形状恰好对应"on-policy 全梯度 → off-policy 平滑衰减"的直觉。
- **非对称温度设计**的洞见来自对 softmax logit 梯度传播的分析，而不是简单的工程试错，这体现了方法论的深度。

### 7.2 与相关工作的关联

- [[GRPO]] → SAPO 的整体目标继承了 group-relative + importance ratio 框架；
- [[GSPO]] → SAPO 在 A1+A2 假设下可归约为 GSPO，但保留了 token 级自适应；
- [[DeepSeek-R1]] → 使用了 GRPO，SAPO 可视为对其优化稳定性的升级；
- [[Baichuan-M4]] → 引用了 SAPO 的前身 DCPO 和 GSPO 作为 RL 稳定训练的技术来源，形成工业界方法层叠引用的闭环。

### 7.3 最值得关注的数字

| 数字 | 含义 | 为什么值得关注 |
|------|------|----------------|
| **1 / 6** 的推理压缩（M4 论文引用 SAPO） | SAPO 的核心应用 | 证明 SAPO 已在工业级系统中被验证 |
| $\tau_{\text{neg}}=1.05$ vs $0.95$ | 温度消融 | 约 5% 的相对温度差异即可导致训练崩溃 vs 稳定的定性差别 |
| $10^5$ sequences / $10^9$ tokens | 假设验证数据量 | 用强统计力量化验证了 (A1)(A2) 的现实性 |

### 7.4 可复现的洞察

- **GRPO 的硬裁剪是 RL 训练不稳定的一个被低估的根源**：不仅是"KL 约束太弱"或"reward 信号差"，优化器本身的 all-or-nothing 梯度截断就在制造脆性。
- **正/负优势的不对称性是 LLM RL 的结构性特征**，因为负更新在巨大词表上扩散。任何不考虑这种不对称性的 group-based RL 方法都存在稳定性隐患。

---

## 8. 关键引用

```bibtex
@article{gao2025sapo,
  title={Soft Adaptive Policy Optimization},
  author={Gao, Chang and Zheng, Chujie and Chen, Xiong-Hui and Dang, Kai and Liu, Shixuan and Yu, Bowen and Yang, An and Bai, Shuai and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2511.20347},
  year={2025},
  url={https://arxiv.org/abs/2511.20347v2}
}
```

---

## 附录：论文未披露的关键信息

| 参数类别 | 缺失信息 | 说明 |
|---------|---------|------|
| SFT 阶段 | 冷启动微调数据规模与来源 | 论文仅提及"cold-start model"，未说明 SFT 数据 |
| RL 阶段 | 训练总步数/学习率/scheduler | 未披露 |
| RL 阶段 | Group size $G$ 的具体值 | 未披露 |
| RL 阶段 | 每个 batch 的 query 数量 | 未披露 |
| RL 阶段 | KL divergence penalty 是否使用及系数 | SAPO 未使用显式 KL？未明确说明 |
| RL 阶段 | 奖励函数构造 | 数学任务使用规则奖励？模型奖励？未说明 |
| RL 阶段 | 是否使用 rejection sampling / best-of-N 预热 | 未说明 |
| 数据管线 | 各任务的 sampling ratio | 仅提到"fixed sampling ratio"，未给出具体值 |
| 数据管线 | 代码/逻辑推理任务的数据来源 | 未说明 |
| 大规模实验 | Qwen3-VL 系列的具体规模（30B-A3B 外的其他尺寸） | 提到"different model sizes"但未逐一列出 |
| 大规模实验 | 训练总计算量（GPU hours） | 未披露 |
| 超参数 | 系统性的超参扫描（$\tau$ 取值范围） | 仅做了三点的消融（0.95, 1.0, 1.05），是否最优未知 |
| 基线 | GRPO 的裁剪范围 $\varepsilon$ 具体值 | 未披露 |
| 基线 | GSPO 的 $\varepsilon$ 具体值 | 未披露 |
| 实验 | Pass@1 的精确数字（而非仅曲线对比） | 论文未给出具体数值表格，仅曲线图 |
| 复现 | 代码未开源 | 截至论文发表无公开代码实现 |
