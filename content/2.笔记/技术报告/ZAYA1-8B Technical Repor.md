地址; https://www.alphaxiv.org/abs/2605.05365?chatId=019e1a0f-d0f8-76ee-9737-ebe347e1e3e4


<alphaxiv-thinking-title title="Reading full paper details" />

<alphaxiv-thinking-title title="Composing detailed blog post" />

---

# ZAYA1-8B 深度解析：用 0.7B 活跃参数挑战前沿大模型

> **作者：** Zyphra 团队 | **arXiv：** 2605.05365 | **发布：** 2026年5月

---

## 导言：一个大胆的命题

Zyphra 的这篇技术报告提出了一个挑衅性的问题：如果将架构创新、推理数据、可验证强化学习和测试时计算方法**协同设计**，一个不足 1B 活跃参数的模型能走到哪一步？

ZAYA1-8B 给出的答案令人惊讶——在 AIME'25 达到 91.9%，在 HMMT'25 达到 89.6%，逼近甚至超越了参数量大数十倍的前沿模型。

---

## 第一章：模型架构（Section II）

ZAYA1-8B 基于 Zyphra 的 **MoE++ 架构**，在标准 Transformer MoE 基础上引入了三项关键变化。

### 1.1 压缩卷积注意力（CCA）

CCA 是对标准 Multi-Head Attention 的一次激进重构。其核心思路是：**在压缩的隐空间中做序列混合**，而不是在完整的高维 embedding 空间中操作。

具体来说，CCA 使用一个轻量卷积下投影器（convolutional downprojector）压缩序列表征，然后在低维空间中完成注意力计算，再投影回原维度。这带来三重好处：

- 训练和 prefill 阶段的 FLOP 显著减少
- KV-cache 尺寸压缩（相对于完整 MHA 压缩 8×，query 压缩 2×）
- 长上下文训练的通信开销降低

> "CCA also improves training speed relative to GQA and MLA and reduces prefill FLOPs while maintaining comparable KV-cache compression rates."

[CCA Overview](https://alphaxiv.org/abs/2605.05365?page=3)

正是 CCA 对 prefill FLOPs 的大幅削减，使得 ZAYA1-8B 能够在有限算力下完成 131K 上下文的长上下文训练，这对后续 RL 阶段至关重要。

- 这是论文示意图，但是没有给太多

![394](Pasted%20image%2020260512113509.png)


### 1.2 ZAYA1 路由器

标准 MoE 使用线性层做路由（token → expert 得分向量），ZAYA1 路由器用**多层 MLP 替代线性路由**，并引入**指数深度平均（EDA）**机制跨层融合路由表征。

完整路由流程如下：

**步骤 1**：下投影到路由隐空间（维度 $R=256$）：

$$r^l = W^{\text{down}} x^l$$

**步骤 2**：EDA 跨层融合（$\gamma$ 为可学习系数）：

$$r^l = r^l + \gamma r^{l-1}$$

**步骤 3**：三层 MLP + RMSNorm 生成路由得分：

$$s^l = \text{softmax}(\text{MLP}(\text{RMSnorm}(r^l)))$$

**步骤 4**：带偏置的 top-1 专家选择：

$$e_{\text{idx}} = \text{argmax}_e (s^l_e + b^l_e)$$

路由偏置 $b^l$ 通过**类 PID 控制器**动态更新，误差信号为当前 batch 路由分布与均匀分布之差：

$$\nabla b^l_e = p_{l,e} - \frac{1}{E}$$

这套机制的效果非常显著：路由收敛更快，均衡失效更少，专家专业化程度更高。与线性路由器相比，ZAYA1 路由器生成的 per-token 路由概率熵更低，说明路由决策更加"自信"。

实验还发现一个重要洞见：**路由器是边际参数的最佳投向**——少量路由器参数控制着远多于自身规模的专家参数，路由质量的提升带来的模型质量收益远超直接扩大专家规模。

> "A small number of router parameters controls a much larger number of expert parameters, and better routing decisions significantly reduce balancing instability and improve model quality."

[Router Efficiency](https://alphaxiv.org/abs/2605.05365?page=1)

### 1.3 残差缩放

第三项变化是**可学习残差缩放**，对残差流和每层输出同时施加仿射变换：

$$\text{Res-scale}(x) = \alpha x + \beta$$
$$x^{l+1} = \text{Res-scale}_{\text{res}}(x^l) + \text{Res-scale}_{\text{out}}(\text{Layer}(\text{RMSnorm}(x^l)))$$

其中 $\alpha \in \mathbb{R}^D$ 和 $\beta$ 均为可学习参数，初始化为 $\alpha=1, \beta=0$（即等价于标准残差连接）。该设计仅增加 $4 \times L \times D$ 个参数（与 LayerNorm 同量级），却能有效控制深层网络的残差范数增长，作用类似 Qwen 的注意力门控机制但参数开销更低。

### 1.4 核心架构参数

| 参数 | 配置 |
|---|---|
| 活跃参数 / 总参数 | 0.76B / 8.4B |
| Transformer 层数 | 40 |
| 隐藏维度 | 2048 |
| 每层专家数 | 16 |
| 路由策略 | Top-1，无 residual expert |
| Expert FFN 宽度 | 4096 pre-activation / 2048 post-activation |
| 路由隐维度 | 256 |
| 注意力变体 | CCGQA（CCA preconditioner） |
| KV 压缩比 | 8× 相对于 MHA |
| Position Embedding | 每头 50% 通道用 RoPE |
| Tokenizer | Gemma3（262,272 词表） |

![](Pasted%20image%2020260512110225.png)

---

## 第二章：预训练与中训练（Section III）

### 2.1 训练阶段概览

ZAYA1-8B 经历了四个主要训练阶段，全部在 AMD MI300X + Pensando Pollara 400 网络上完成：

| 阶段 | 上下文长度 | RoPE Base | Token 预算 | 主要数据 |
|---|---|---|---|---|
| 基础预训练 Phase 1 | 4K | 10K | 8T | 广泛网络爬取、代码、数学、多语言 |
| 基础预训练 Phase 2 | 4K | 10K | 4T | 加大代码/数学/推理/指令格式比例 |
| 32K 中训练 | 32K | 1M | 1.2T | 长 CoT 推理、代码、数学 |
| SFT | 131K | 5M | 660B | 对话、推理、代码、TTC traces |

[Training Phases](https://alphaxiv.org/abs/2605.05365?page=6)

### 2.2 推理感知预训练

ZAYA1-8B 的一个核心设计理念是：**推理数据应从预训练阶段起就贯穿始终**，而非仅在后训练阶段引入。

数据配比如下：

| 类别 | 32K 中训练 | 131K SFT |
|---|---|---|
| 长 CoT 推理轨迹 | **86.1%** | **75.0%** |
| 网络/合成/多语言 | 5.7% | 9.8% |
| 原生长上下文数据 | 0.8% | 6.4% |
| 代码 | 3.0% | 5.0% |
| 数学/STEM | 3.0% | 2.6% |
| 短指令/few-shot | 1.4% | 1.2% |

中训练阶段高达 86.1% 的数据是长 CoT 推理轨迹，这一比例在业界极为罕见。

### 2.3 保答案截断（AP-Trimming）

在 4K 上下文的早期预训练中，强教师模型生成的推理轨迹往往超过 10K token（尾部甚至超过 30K），如何处理这些过长样本是个难题。

**三种朴素做法的问题：**
1. **完全丢弃**：损失宝贵的推理信号
2. **简单截断**：通常保留推理前缀但丢失最终答案，训练模型"推理永远不收敛"
3. **从中间截断**：破坏推理链的因果一致性

AP-Trimming 的做法是**从推理轨迹尾部截断，同时保留最终答案**。具体流程：

```
给定目标上下文预算 C：
1. 若样本完整长度 ≤ C → 保留全部
2. 若超出 → 截去最后一个推理块（<think>...</think>）的尾部，
   保留推理起始部分 + 完整答案
3. 多轮对话：额外删除早期轮次的推理块（保留其答案）
4. 若仅答案部分就超出 C → 丢弃该样本
```

这一设计的理论依据是：推理链**前半部分**通常包含问题分解、计划制定和多方案探索，价值更高；**尾部**通常是对已选方案的收敛计算，价值相对较低。因此从尾部截断能最大程度保留有价值的规划信号。

同时，AP-Trimming 随训练阶段的上下文长度变化而重新应用（4K → 32K → 131K），在更长的阶段保留更完整的推理轨迹。

---

## 第三章：后训练——四阶段 RL 级联（Section IV）

### 3.1 总体架构

后训练采用 **SFT + 四阶段 RL 级联**结构。关键设计原则是：**可验证推理优先，行为调优最后**。

```
SFT (660B tokens, 131K context)
    ↓
Stage 1: 推理热身 RL (232步, 数学/谜题/TTC)
    ↓
Stage 2: RLVE-Gym 课程 RL (400步, 400个自适应环境)
    ↓
Stage 3: 数学+代码+TTC RL (384+464步)
    ↓
Stage 4: 行为 RL (384步, 对话/指令跟随)
```

### 3.2 共享算法核（Algorithmic Spine）

所有 RL 阶段共用同一套算法架构，仅数据和奖励信号不同：

#### PipelineRL（异步训练）
Rollout 生成与梯度更新在**完全隔离的 GPU 池**上异步运行，rollout worker 与 trainer worker 比例为 2–5:1。Trainer 每 2 步同步一次权重，稳态下 rollout 策略最多落后 trainer 策略 2 步。

#### DPPO Binary-TV（信任域约束）
**完全抛弃** PPO 的 per-token ratio clipping，改用 Binary Total-Variation 信任域掩码：

> 对每个 token，若策略散度估计超过阈值 $\delta$，则将该 token 从梯度中掩码（mask out）；其余 token 按标准策略梯度更新。

生产配置 $\delta = 0.1$。

#### Dr-GRPO SMTSN（损失聚合）
采用 Sequence-Mean over Token-Sum-Normalized（SMTSN）聚合：

- Token-level 损失在每个 rollout **内部求和**（而非平均）
- 再在 rollout 之间**求平均**

这避免了标准 GRPO 中隐含的"长度归一化"偏置（标准 per-token 平均会让梯度偏向更短的回复）。

#### MaxRL 优势估计
对于每个 prompt，采样 $G$ 个 rollout，reward $r_i \in \{0, 1\}$，优势计算为：

$$\hat{A}_i = \frac{r_i - \bar{r}}{\bar{r}}$$

其中 $\bar{r}$ 是组内平均奖励。相对于标准 GRPO（除以奖励标准差），MaxRL 对难题给出更强梯度信号，因为难题的 $\bar{r}$ 小，优势估计的分母更小。

#### 无 KL 正则
整个级联**完全不使用 KL 正则**（既不作为 reward 惩罚，也不作为 loss 项），信任域完全由 DPPO Binary-TV 负责。

论文专门分析了为何 KL-in-reward 在 PipelineRL 场景下是危险的——异步训练中，一个长 rollout 的不同 token 可能由不同版本的策略生成，导致有符号 log-ratio 估计量在序列级累积产生**长度依赖偏置**，最终造成 rollout 长度不受控地增长。[KL Bias](https://alphaxiv.org/abs/2605.05365?page=14)

### 3.3 各阶段详情

#### Stage 1：推理热身（232步）

| 数据类别 | 占比 |
|---|---|
| Reasoning-gym 和核心谜题 | 54.4% |
| 竞赛级数学 | 31.2% |
| Enigmata 谜题 | 14.4% |

刻意偏向难题：样本先验通过率 ≤ 0.75，中位 rollout 长度约 17.6K tokens，p90 约 30K tokens。TTC 格式的 prompt 也在此阶段引入，让模型从 RL 开始就接触聚合式推理。

#### Stage 2：RLVE-Gym 自适应课程（400步）

从 400 个可验证环境中学习，核心挑战是如何为每个环境找到"让模型以约 0.5 通过率解题"的合适难度。

解决方案是 **Thompson Sampling + IRT（项目反应理论）**：

将每个环境的通过率建模为逻辑曲线：

$$p_{\text{success}} = \sigma\left(-\frac{d - \mu}{s}\right) = \frac{1}{1 + e^{(d-\mu)/s}}$$

维护参数池 $\Theta = \{(\mu_m, s_m)\}$，按后验权重采样参数，计算目标 $p_{\text{target}} = 0.5$ 处的难度，运行 rollout 后更新后验权重：

$$w_j \propto w_j \cdot \text{Binomial}(k; G, p_{\text{success},j})$$

在线难度调度器随训练进展动态上调/下调每个环境的难度，确保每个环境始终处于"可解但未饱和"的信息密度最大区间。

#### Stage 3：数学+代码+TTC RL（384+464步）

这是能力构建的核心阶段，分两个 phase：

| 类别 | Phase 1 (18,656行) | Phase 2 (12,092行) |
|---|---|---|
| 标准代码 | 20.3% | 31.4% |
| 代码辅助/代码 TTC | 33.4% | 26.8% |
| 标准数学 | 33.8% | 22.5% |
| 数学 TTC/RSA/PaCoRe | 12.5% | 19.4% |

从竞赛编程参考题目构造了三类**辅助合成代码环境**：

1. **CodeI/O 预测**：给定代码和输入预测输出，或给定代码和输出反推输入——强化执行追踪和逆向程序推理
2. **CodeARC 重构**：给定问题描述和 I/O 示例合成代码——强化从稀疏行为证据合成代码
3. **Falsification（反驳）**：给定规范和实现，找到使其违规的输入——强化对抗性测试构造

三类任务均为**二值可验证**，与数学/谜题 prompt 共用同一 RL 目标。

#### Stage 4：行为 RL（384步）

最后阶段聚焦对话风格、指令跟随、偏好优化。此阶段切换回**标准 GRPO**（用奖励标准差归一化），不再使用 MaxRL 或长度奖励。两个 IF 子阶段引入**二值指令跟随检查器门控**：若 completion 不满足显式指令约束，则无论 reward model 评分多高，reward 一律归零。

### 3.4 工程稳定性关键技术

#### 路由器回放（Router Replay）

MoE Top-1 路由的离散性在 RL 训练中会产生一个微妙的梯度毒化问题：vLLM 推理引擎和训练器即使在 BF16 设置下，也会因微小数值差异对边界 token 做出不同的专家选择。若 token 在推理时走了专家 A，训练时走了专家 B，则梯度是针对专家 B 的输出计算的，但 rollout 是专家 A 生成的——梯度被污染。

Router Replay 的解法是：vLLM 在 decode 时将每层每 token 的专家选择索引写入共享内存缓冲区，随 batch 发送给 trainer，trainer 强制复用这些路由决策，保证 $e_{\text{train}} \equiv e_{\text{inference}}$。

[Router Replay](https://alphaxiv.org/abs/2605.05365?page=11)

实现细节：索引写入与 decode 工作重叠，不影响推理速度；传输时与 token IDs 打包在一起，无额外传输步骤。

#### 精度控制（BF16 + 选择性 FP32）

默认使用 BF16 权重和激活，**仅以下操作升至 FP32**，且推理引擎和训练器**两侧都需一致**：

- 损失/输出：fused cross-entropy + LM head matmul
- 注意力/归一化：CCA cache state、QK-norm、QK-mean、RMSNorm
- 路由/残差：router softmax 和残差流加法

配置完成后，vLLM 与 trainer 的 per-token log-probability 差异极小：KL 散度 = $1.3 \times 10^{-4}$，Pearson r > 0.9996。[Precision Match](https://alphaxiv.org/abs/2605.05365?page=12)

#### 无动量 Muon 优化器

所有 RL 阶段使用 Muon 优化器，但**动量设为零**（$\mu = 0$）：

$$\Delta W_t = -\eta_t \mathcal{M}(g_t)$$

其中 $\mathcal{M}(\cdot)$ 是 Muon 的 Newton-Schulz 正交化步骤。

动量为零意味着每次参数更新**只依赖当前 rollout batch**，不携带历史梯度信息。这对异步 RL 的非平稳信号结构更为合适——相邻 batch 的 prompt、轨迹、reward 和策略快照可能完全不同，历史动量的积累可能引入过时信息。

词嵌入和 LM head 仍用 AdamW。

#### 训练稳定性监控

- **LZ77 流式可压缩性检测**：实时监测每个 rollout 各区块的压缩率，若某区块压缩率 $r_c < 0.05$（高度可压缩 = 重复/复制），将该 rollout 的 task reward 归零，即使最终答案正确也不给正向学习信号
- **稀有 token 比例监控**：追踪 token ID 落在 tokenizer 词表高序区间的比例，作为 gibberish/退化文本的早期预警指标

---

## 第四章：实验结果（Section V）

### 4.1 同级别模型对比

| 模型 | 活跃参数 | AIME'26 | HMMT'26 | LCB-v6 | GPQA-D | IFEval |
|---|---|---|---|---|---|---|
| **ZAYA1-8B** | **0.7B** | **89.1** | **71.6** | **64.8** | **71.0** | **85.6** |
| Qwen3-4B-Thinking-2507 | 4B | 79.0 | 53.6 | 54.9 | 66.1 | 86.8 |
| Qwen3.5-4B | 4B | 84.5 | 63.6 | 55.8 | 76.2 | 89.8 |
| Gemma-4-E4B-it | 4B | 50.3 | 32.1 | 54.2 | 57.4 | 88.5 |

ZAYA1-8B 在数学推理上以 0.7B 活跃参数大幅领先所有 4B 同类模型，但在 EQBench、Creative Writing、BFCL-v4、$\tau^2$ 等风格/智能体任务上相对较弱，这与其专注于可验证推理的训练取向一致。

### 4.2 跨规模对比

<!-- Bar Chart: AIME'26 成绩 vs. 活跃参数（跨规模对比） -->

ZAYA1-8B（0.7B 活跃参数）的 AIME'26 成绩与 Nemotron-3-Nano（3B 活跃，30B 总参数）和 Qwen3-Next-80B（3B 活跃，80B 总参数）几乎持平，这是本文最有说服力的结果之一。

### 4.3 后训练的增益量化

| 基准 | SFT checkpoint | 最终 ZAYA1-8B | 增益 |
|---|---|---|---|
| AIME'26 | 68.3 | 89.1 | **+20.8** |
| HMMT'26 | 39.2 | 71.6 | **+32.4** |
| LCB-v6 | 54.8 | 64.8 | **+10.0** |
| GPQA-Diamond | 59.3 | 71.0 | **+11.7** |
| IFEval | 66.6 | 85.6 | **+19.0** |

RL 级联带来的增益是全方位的，且数学推理的增益幅度最大（HMMT +32.4 点）。

---

## 第五章：Markovian RSA——测试时计算方法（Section VI）

这是本文方法论贡献的核心，也是 ZAYA1-8B 最独特的能力来源。

### 5.1 方法动机

两条先行工作的启发：

- **RSA（Recursive Self-Aggregation）**：维护候选推理链种群，反复聚合精炼，让小模型逼近大模型
- **Markovian Thinker**：将 RL 推理环境重构为固定块大小、有界状态转移的 Markov 过程，将长上下文推理解耦于上下文长度

Markovian RSA 将两者结合：**RSA 的递归候选聚合** + **Markovian Thinker 的有界工作空间原理**。

### 5.2 算法详解

**输入**：问题 $q$，基础策略 $\pi$，配置 $(N, C, T, \beta, \tau)$

**Round 0**：并行生成 $N$ 个独立候选推理轨迹，每个长度上限 $\beta$，提取最后 $\tau$ token 作为"尾部（tail）"

**Round $t \geq 1$**（重复 $T$ 次）：
- 对每个新候选：从当前种群随机采样 $C$ 个尾部
- 拼接成聚合 prompt（包含原问题 $q$ 和 $C$ 个候选尾部）
- 策略生成新的推理轨迹（长度上限 $\beta$），提取尾部加入种群

**最终答案**：从最后一轮所有候选的输出中提取

**聚合 prompt 的预填充长度严格有界**：

$$L_{\text{prefill}} \leq |q| + C\tau + O(1)$$

**关键性质**：$L_{\text{prefill}}$ 与 $\beta$ 完全无关——可以无限增大单候选推理深度 $\beta$，而不增加聚合阶段的上下文成本。

### 5.3 与其他 TTC 方法的统一视角

| 方法 | Batch Size | 上下文状态 | 对应特殊情况 |
|---|---|---|---|
| 单次长 CoT | 1，位置随推理增长 | 完整历史 | — |
| 并行采样 | N，单轮 | 无聚合 | $T=0$ |
| Delethink 延续 | N，分块轮次 | 1个有界尾部 | $C=1$ |
| 全链 RSA | N，聚合轮次 | $C$ 条完整链，最长 $C\beta$ | $\tau=\beta$ |
| **Markovian RSA** | **N，聚合轮次** | **$C$ 个尾部，最长 $C\tau$** | **通用情况** |

### 5.4 训练时集成

仅仅在推理时用 Markovian RSA 而不训练它是次优的。ZAYA1-8B 在三个阶段引入聚合训练数据：

**SFT 阶段**：从开源数据集中（如 OpenMathReasoning 每题 8 个 rollout），构造 round-0 → round-1 聚合样本：采样 $C$ 个 rollout 的尾部 → 构建聚合 prompt → 教师模型生成聚合 rollout 作为 SFT 目标。优点：离线构造，不需要新的教师推理；不依赖验证器。

**RL 阶段**（Stage 3）：两种变体：
- **专家聚合**：聚合 prompt 来自教师模型 rollout，policy 生成聚合结果并获取可验证奖励
- **自聚合**：聚合 prompt 来自当前 SFT checkpoint 或前一个 RL checkpoint 的 rollout

无需特殊的多轮 RL 机制——聚合结构编码在 prompt 构造中，梯度更新与普通 RL 完全相同。

### 5.5 Inference 扩展实验结果

<!-- Line Chart: Markovian RSA 配置扫描：AIME'25 vs. 生成 token 数 -->

配置扫描的关键发现：

- 在 $T=2, N=16, C=4$ 下，增大每候选推理预算 $\beta$（8K → 16K → 40K）持续提升精度
- **增大尾部 $\tau$（从 4K 到 8K，$\beta=40K$ 固定）不提升精度**（91.9% vs 90.8%），说明在这些基准上，4K token 已足够捕获足够的推理状态进行有效聚合
- 跨候选聚合（$C=4$ vs $C=1$）带来 AIME+4.4点、HMMT+5.8点的增益

最优配置 $(\beta=40K, \tau=4K, T=2, N=16, C=4)$：AIME'25 **91.9%**，HMMT'25 **89.6%**，总生成 decode token 约 740K/题。

推荐部署配置 $(\beta=16K, \tau=4K)$：约 440K token/题，仍达 88.8%/87.1%。

---

## 第六章：讨论与展望（Section VII）

### 主要结论

> "ZAYA1-8B is designed to maximize reasoning performance per active parameter, with a particular focus on reasoning-intensive mathematics and coding."

[Conclusion](https://alphaxiv.org/abs/2605.05365?page=22)

论文指出，以下五个设计的**协同**是取得这些结果的关键：
1. CCA 架构降低计算成本，使长上下文训练可行
2. MoE++ 路由器最大化稀疏激活的参数效率
3. AP-Trimming 使推理数据在预训练阶段就可用
4. 四阶段 RL 级联系统性地积累可验证推理能力
5. Markovian RSA 在推理时可预测地扩展计算投入

### 已知局限

- **Agentic 任务较弱**：无专门的多轮工具使用 RL，BFCL-v4 和 $\tau^2$ 分数低于许多专注智能体训练的模型
- **风格/创意任务**：EQBench 和 Creative Writing 指标不突出，因为训练优先级在可验证推理
- **AMD 验证范围**：仅在 ZAYA1-8B 规模验证了 AMD 全栈，更大模型和更广并行配置留待未来工作
- **Markovian RSA vs 全链 RSA 的受控对比**：留待未来工作

---

## 总结

ZAYA1-8B 是一篇系统工程与算法创新并重的工作。它最重要的贡献不是某一个单点技巧，而是展示了一条**从架构到数据到训练到推理的全链路协同设计路线**：当每个组件都为推理场景定制优化时，0.7B 活跃参数已经可以在竞赛数学领域与 37B 活跃参数的前沿模型正面竞争。Markovian RSA 则进一步揭示了一个值得深入探索的方向：**用可预测的计算预算换取更高精度，而不是用更大的模型**。





# 论文中提及的数据集整理

基于《ZAYA1-8B 技术报告》及相关参考来源，我为您汇总了模型训练中使用的主要数据集、其对应的 Hugging Face (HF) 地址以及各阶段所采用的训练算法。

### ZAYA1-8B 训练数据集与算法汇总表

|训练阶段|数据集名称|Hugging Face 地址 / 来源|训练算法 / 方法|
|:--|:--|:--|:--|
|**有监督微调 (SFT)**|**OpenMathReasoning**|[nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning)|**交叉熵损失 (CE Loss)**；使用最佳拟合递减分箱包装 (Bin Packing)。|
||**rStar-Coder**|[microsoft/rStar-Coder](https://huggingface.co/datasets/microsoft/rStar-Coder)|同上|
||**Markovian RSA 聚合示例**|内部构建（通过专家模型轨迹重新洗牌生成）|同上；引入测试时计算 (TTC) 聚合训练。|
|**推理 RL 热身**|**Reasoning Gym**|[PNYX/reasoning_gym_lmeh](https://huggingface.co/datasets/PNYX/reasoning_gym_lmeh)|**异步 PipelineRL**；**DPPO Binary-TV** 信任区域掩码；**Dr-GRPO** 序列级损失聚合；**MaxRL** 优势估计；**无动量 Muon 优化器**。|
||**竞赛数学 / Enigmata 谜题**|内部构建 (Zyphra)|同上；使用可验证任务奖励。|
|**RLVE 课程训练**|**RLVE-Gym**|[hamishivi/rlve (集合)](https://huggingface.co/collections/hamishivi/rlve)|在上述 RL 算法基础上，增加 **Thompson/IRT 自适应难度调度**。|
|**数学与代码 RL**|**CodeI/O (预测任务)**|[hkust-nlp/CodeIO-PyEdu-Reasoning](https://huggingface.co/datasets/hkust-nlp/CodeIO-PyEdu-Reasoning)|**PipelineRL + MaxRL 架构**；侧重于逻辑流规划和执行追踪。|
||**CodeARC (代码重构)**|[Anjiang-Wei/CodeARC (GitHub)](https://github.com/Anjiang-Wei/CodeARC)|同上；侧重于从行为证据中合成代码。|
||**Falsification (伪造检测)**|内部环境|同上；侧重于对抗性测试构建。|
||**PaCoRe / RSA 聚合提示**|基于推理轨迹构建|同上；包含 **专家聚合** 与 **自我聚合** 训练。|
|**行为 RL (对齐)**|**HelpSteer2 / HelpSteer3**|[nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) /(https://huggingface.co/datasets/nvidia/HelpSteer3)|**标准 GRPO**；使用奖励模型 (RM) 分数和 **奖励标准差归一化**；**不使用**长度奖励。|
||**IFBench (指令遵循)**|[allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test)|同上；奖励由 **二元指令遵循检查器 (IF Checker)** 门控。|

### 核心训练算法解析

1. **SFT 阶段算法**：采用标准的**下一步 Token 预测 (Next-token Prediction)**，配合交叉熵损失。特别之处在于对长推理轨迹使用了**答案保留裁剪 (Answer-Preserving Trimming)** 方案，以在固定上下文中保留最终答案。
2. **强化学习 (RL) 骨干架构**：
    - **PipelineRL**：将 Rollout 生成与梯度更新异步化，在不同的 GPU 池上运行。
    - **DPPO Binary-TV**：放弃了 PPO 的比例裁剪，改为对策略散度超过阈值的 Token 进行梯度掩码，且**奖励中不包含 KL 正则化**。
    - **Dr-GRPO**：对标准 GRPO 进行了改进，采用**序列均值/Token 总和归一化 (SMTSN)**，以避免由于隐式长度归一化导致的奖励偏好偏差。
    - **MaxRL**：使用样本组的平均值而非标准差进行优势归一化，在困难提示词上产生更强的梯度信号。
    - **无动量 Muon (Momentum-free Muon)**：针对 RL 训练的非平稳性，每一轮更新仅依赖当前批次，不保留第一动量缓存，提高了稳定性并减少了内存占用。
3. **行为 RL 阶段算法**：此阶段回归到更接近标准对齐的方法，使用 **RLAIF (AI 反馈强化学习)** 和标准的奖励模型评分机制。