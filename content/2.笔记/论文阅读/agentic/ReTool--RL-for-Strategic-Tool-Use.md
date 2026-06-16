---
created: 2026-06-09
paper: https://arxiv.org/abs/2504.11536
authors: Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, Wanjun Zhong (ByteDance Seed)
tags:
  - AgenticRL
  - ToolUse
  - RL
  - CodeInterpreter
---

# ReTool: Reinforcement Learning for Strategic Tool Use in LLMs

## 一句话总结

ReTool 是一个将代码解释器（CI）集成到 RL 训练闭环中的框架，通过 outcome-based reward 让模型自主学习"何时调用、如何调用"工具，在 AIME 上仅用 400 步训练即达到 67% 准确率，并实现响应长度缩短 40% 的推理效率提升。

![](ReTool_fig1_main_results.png)

> **Figure 1**: ReTool 与纯文本 RL 基线在 Qwen2.5-32B-Instruct 上的 AIME 2024/2025 得分对比，ReTool 以更少训练步数获得显著更高的准确率。

---

## 1. 研究背景与动机

### 1.1 问题定义

现有推理模型（如 OpenAI o1、DeepSeek R1）在纯文本推理上表现出色，但在需要精确数值计算、符号操作或几何推理的任务中仍存在明显短板：

- **文本推理的局限**：长文本思维链（CoT）依赖于内部语言模式，面对复杂计算时容易产生累积误差
- **代码解释器的优势**：提供形式化、可执行的接口，支持枚举、验证和精确计算，显著降低推理歧义

核心问题是：**如何让 LLM 在推理过程中策略性地调用代码解释器，而不是盲目模仿固定的工具使用模式？**

### 1.2 现有方法的不足

| 方法 | 局限 |
|------|------|
| Prompting（如 PoT） | 仅通过提示词引导，无法自适应决策 |
| SFT（如 TIR） | 局限于模仿特定数据分布，无法泛化到未见模式 |
| 纯文本 RL（如 R1-Zero） | 缺乏工具调用能力，复杂计算仍是瓶颈 |
| 并发工作 ToRL | 仅在 1.5B/7B 小模型上验证，性能未达最优 |

现有 SFT 方法的主要缺陷在于：模型只能学习"见过的"工具使用模式，无法自主探索"何时调用"和"如何纠正"等策略性行为。RL 提供了一条更优路径——通过 outcome feedback 激励模型探索灵活的工具调用策略。

---

## 2. 方法

### 2.1 核心思想

ReTool 采用**两阶段训练框架**：

1. **Cold-start SFT**：通过高质量数据让模型学会"如何调用 CI"的基础能力
2. **Tool-enhanced RL**：通过 outcome-based reward 让模型自主探索最优工具使用策略

关键创新点在于：将代码解释器的实时执行反馈纳入 RL 的 rollout 过程，使模型在多轮交互中动态调整推理策略。

![](ReTool_fig2_training_process.png)

> **Figure 2**: (a) 传统文本 RL 训练流程；(b) ReTool 的 RL 训练流程，在 rollout 中动态交错自然语言推理与代码执行反馈。

### 2.2 工具调用策略

#### 2.2.1 Rollout 中的交错代码执行

与传统生成固定文本不同，ReTool 的 rollout 是一个**动态交互过程**：

1. 策略模型生成文本推理 $t_1$
2. 检测 `<code></code>` 终止触发器，暂停生成
3. 解析代码片段 $c_1$ 并发送至沙箱执行
4. 沙箱返回输出 $f_1$（成功结果或错误信息），封装在 `<interpreter></interpreter>` 标签中
5. 模型接收反馈后继续生成，直到给出最终答案 $o$ 或产生新代码

最终产生**混合推理轨迹**：

$$
[t_1 \oplus c_1 \oplus f_1 \oplus \dots \oplus o]
$$

关键设计：**同时返回成功执行结果和错误信息**。这种动态反馈机制使模型能够迭代地探索、修正和优化其推理与工具使用策略。

#### 2.2.2 训练优化技巧

| 技巧 | 作用 |
|------|------|
| **Interpreter Feedback Mask** | 将 `<interpreter>` 标签内的 token 从 loss 计算中屏蔽，避免外部 token 干扰训练稳定性 |
| **KV-Cache Reuse** | 在代码执行前缓存 KV-Cache，执行后仅追加新 token 的 KV-Cache，大幅降低内存开销 |
| **异步沙箱** | Worker Pool 模式实现并行环境交互，避免慢速线程阻塞训练流程 |

### 2.3 奖励设计

ReTool 采用极简的**基于规则的准确性奖励**：

$$
R(a, \hat{a}) = \begin{cases}
1, & \text{if } \texttt{is\_equivalent}(a, \hat{a}) \\
-1, & \text{otherwise}
\end{cases}
$$

**逐项解释**：

| 符号 | 含义 |
|------|------|
| $a$ | 问题的标准答案（ground-truth） |
| $\hat{a}$ | 模型的预测答案 |
| $\texttt{is\_equivalent}(\cdot, \cdot)$ | 等价性判断函数（支持数值等价、表达式等价等） |
| 奖励值 $\in \{1, -1\}$ | 正确得 +1，错误得 -1 |

**设计意图**：刻意简化奖励结构，仅依赖 outcome feedback，不引入代码可执行性奖励等额外信号。这样做的目的是：
- 缓解 reward hacking（避免模型为追求中间奖励而偏离最终目标）
- 鼓励更多样化的问题解决行为
- 让模型自主发现"何时调用代码"的策略，而非被显式规则约束

### 2.4 训练流程

#### 2.4.1 Cold-start 数据构建

1. **收集**：从 Open-Thoughts 等开源数据集收集初始推理数据 $\mathcal{D}_{\text{init}}$
2. **过滤**：结合人工专家筛选 + DeepSeek-R1 评估，去除无效数据
3. **转换**：用结构化 prompt 模板将可受益于代码执行的手动计算步骤替换为对应代码片段及执行结果
4. **双阶段验证**：
   - 格式验证：确保语法一致性，便于 RL 阶段检测代码调用触发器
   - 答案验证：剔除最终输出与正确答案不符的样本
5. **输出**：得到代码增强的长格式推理数据集 $\mathcal{D}_{\text{CI}}$

#### 2.4.2 RL 训练算法

基于 PPO（Proximal Policy Optimization），目标函数为：

$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \, o_{\leq t} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)} \left[ \min\left( \frac{\pi_\theta(o_t \mid q, o_{<t}; \mathcal{CI})}{\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t}; \mathcal{CI})} \hat{A}_t, \; \text{clip}\left( \frac{\pi_\theta(o_t \mid q, o_{<t}; \mathcal{CI})}{\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t}; \mathcal{CI})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_t \right) \right]
$$

**逐项解释**：

| 符号 | 含义 |
|------|------|
| $\theta$ | 当前策略网络的参数 |
| $\theta_{\text{old}}$ | 旧策略网络的参数（用于计算重要性采样比率） |
| $q$ | 输入问题（query） |
| $a$ | 标准答案 |
| $o_t$ | 第 $t$ 步生成的 token |
| $o_{<t}$ | 第 $t$ 步之前已生成的所有 token |
| $\pi_\theta(o_t \mid q, o_{<t}; \mathcal{CI})$ | 在代码解释器 $\mathcal{CI}$ 辅助下的条件生成概率 |
| $\hat{A}_t$ | 第 $t$ 步的优势函数估计值（advantage） |
| $\varepsilon$ | PPO 的裁剪系数（通常取 0.1~0.2） |
| $\text{clip}(\cdot, 1-\varepsilon, 1+\varepsilon)$ | 限制重要性采样比率，防止策略更新幅度过大 |
| $\min(\cdot, \cdot)$ | 取裁剪前后目标值的较小者，构成 pessimistic bound |

核心改动：策略 rollout 不再只是文本生成，而是**与代码解释器 $\mathcal{CI}$ 的交互过程**。

#### 2.4.3 超参数配置

| 参数 | 设置 |
|------|------|
| 框架 | VeRL |
| RL 方法 | PPO |
| Cold-start 轮数 | 2 epochs |
| 优化器 | AdamW |
| 学习率 | $1 \times 10^{-6}$ |
| 最大序列长度 | 16384 tokens |
| Mini-batch size | 512 |
| KL 系数 | 0.0 |
| 主模型 | Qwen2.5-32B-Instruct |

---

## 3. 实验结果

### 3.1 实验设置

- **评估基准**：AIME2024、AIME2025（MATH 奥林匹克级别）
- **评估方式**：每个数据集重复评估 32 次，取平均准确率估计 pass@1
- **推理超参数**：temperature=1.0，top-p=0.7
- **基线模型**：Qwen2.5-Math-72B-Instruct、Sky-T1、DeepSeek-R1-Zero-Qwen-32B、QwQ-32B-Preview、s1-32B、OpenAI o1-preview

### 3.2 主实验结果

| 模型 | AIME2024 | AIME2025 | 备注 |
|------|----------|----------|------|
| Qwen2.5-Math-72B-Instruct | 30.0 | - | 基线大模型 |
| Qwen2.5-Math-72B-Instruct-TIR | 40.0 | - | 工具集成推理 |
| Sky-T1 | 43.3 | - | 开源 o1 风格模型 |
| OpenAI o1-preview | 44.6 | 37.9 | 商业闭源 |
| DeepSeek-R1-Zero-Qwen-32B | 47.0 | - | 纯文本 RL |
| QwQ-32B-Preview | 50.0 | 33.5 | 强化学习推理 |
| s1-32B | 56.7 | - | Test-time scaling |
| **ReTool (Qwen2.5-32B-Instruct)** | **67.0** | **49.3** | **400 steps** |
| **ReTool (DeepSeek-R1-Distill-Qwen-32B)** | **72.5** | **54.3** | **扩展设置** |

**消融实验（基于 Qwen2.5-32B-Instruct）**：

| 变体 | AIME2024 | AIME2025 |
|------|----------|----------|
| w/o Training（Base Model） | 26.7 | - |
| w/o CI（Text-based RL） | 40.0 | 36.7 |
| w/o RL（仅 Cold-start） | 40.9 | 34.5 |
| **ReTool（完整方法）** | **67.0** | **49.3** |

关键发现：
1. **效率优势**：ReTool 仅用 400 步即达到 67%，而文本 RL 基线需 1080 步才达到 40%
2. **性能飞跃**：ReTool-32B（72.5%）超越 OpenAI o1-preview（44.6%）27.9 个百分点
3. **数据质量验证**：仅 Cold-start 即达 40.9%，接近文本 RL 基线，说明数据构建有效

### 3.3 消融实验

消融实验表明两个组件缺一不可：

- **去掉 CI（纯文本 RL）**：性能上限明显受限于模型自身计算能力，在复杂计算题上容易出错
- **去掉 RL（仅 Cold-start）**：模型只能模仿训练数据中的工具使用模式，无法自适应优化调用策略
- **完整 ReTool**：RL 驱动的自主探索使模型能够发现更高效的工具调用时机和方式

---

## 4. 认知分析

### 4.1 行为演化（Behavior Evolution）

![](ReTool_fig3_behavior_evolution.png)

> **Figure 3**: RL 训练过程中 CI 相关行为的演化趋势。(a) 响应长度；(b) 代码使用率；(c) 代码行数；(d) 正确代码数量；(e) 代码通过率；(f) 代码调用时机。

**关键观察**：

1. **响应长度（图 a）**：先急剧下降后温和上升，最终比训练前缩短约 40%（10k tokens → 6k tokens）。初期下降是因为代码替代了冗长文本计算，后期上升则源于更复杂多样的代码行为涌现。

2. **代码使用率（图 b）**：持续上升，最终覆盖近 98% 的问题。说明模型逐步掌握了策略性使用代码的能力。

3. **代码行数（图 c）**：持续上升，最终达到训练前的近 5 倍。表明模型学会了编写更复杂的代码策略。

4. **正确代码数量（图 d）**：从 1k 上升至 5k，反映出代码工具利用能力的显著增强。

5. **代码通过率（图 e）**：正确回答中的代码通过率始终接近 100%；错误回答中的通过率呈下降趋势。这说明代码可执行性直接影响推理过程和最终结果。

6. **代码调用时机（图 f）**：调用时机逐渐提前。模型学会了在推理早期就调用工具，而非等到推理末尾才求助代码。

### 4.2 "Aha Moment"——代码自纠正

![](ReTool_fig4_aha_moment.png)

> **Figure 4**: 代码自纠正的"顿悟时刻"案例。模型最初生成包含未定义函数 `greedy()` 的代码，在收到解释器错误反馈后，自主反思并生成修正版本。

这是论文中最引人注目的发现之一。模型展现出**涌现的自纠正能力**：

- 初始代码因调用未定义的 `greedy()` 函数而执行失败
- 收到解释器错误反馈后，模型生成反思："Oops, the functions need to be defined in the same scope. Let's correct that."
- 随后生成包含完整函数定义的修正代码，成功执行

这种能力**未经显式训练**，完全由 RL 的 outcome feedback 驱动涌现，标志着模型自主掌握了元认知层面的代码调试能力。

### 4.3 代码用途分析

![](ReTool_fig5_code_purpose.png)

> **Figure 5**: RL 训练前后代码用途的词云对比。计算（calculation）和验证（verification）是主要用途，RL 后代码用途更加多样化。

- 训练前：以计算和验证为主
- 训练后：代码用途更加多样，反映出模型发展出了**自适应工具选择**的元认知能力

### 4.4 CI 推理 vs. 文本推理

![](ReTool_fig6_ci_vs_text.png)

> **Figure 6**: 同个问题下 CI 驱动推理（左）与纯文本推理（右）的对比。CI 推理用简洁代码替代了冗长易错的手动计算，将模型注意力聚焦于高层推理策略。

---

## 5. 局限性与未来方向

### 5.1 局限性

1. **任务范围有限**：仅在数学问题求解（AIME）上验证，未覆盖更广泛的任务类型（如多跳推理、知识检索等）
2. **模型规模**：最大验证到 32B 参数，更大规模（72B/400B+）上的表现尚待验证
3. **冷启动依赖**：数据构建依赖现有高质量数据集（Open-Thoughts）+ DeepSeek-R1 过滤，数据获取成本不低
4. **工具类型单一**：仅使用沙箱式代码解释器，未探索其他工具类型（如搜索引擎、计算器、API 调用等）
5. **奖励稀疏**：仅依赖最终结果奖励，中间步骤缺乏细粒度反馈，可能限制学习效率

### 5.2 未来方向

1. **扩展到更多工具类型**：将 RL 驱动的策略性工具使用推广到搜索、数据库查询、API 调用等场景
2. **细粒度奖励设计**：探索过程奖励模型（PRM）辅助的稠密奖励，加速收敛
3. **更大规模验证**：验证 ReTool 在 72B/400B+ 模型上的扩展性
4. **多任务泛化**：在代码生成、科学推理、日常问题求解等非数学任务上验证通用性
5. **与并发工作 ToRL 的比较与融合**：ToRL 在更小模型上的探索与 ReTool 的规模化经验可以互补

---

## 6. 个人思考

### 6.1 核心启发

ReTool 最深刻的洞见在于：**工具使用不应该是被硬编码的规则，而应该是在 outcome feedback 下自主涌现的策略**。这与 Agentic RL 的核心哲学高度一致——环境交互产生的反馈信号是智能行为的根本驱动力。

几个特别值得关注的点：

1. **效率悖论**：通常 RL 会让模型生成越来越长的思维链（如 R1 的"过度思考"），但 ReTool 却实现了响应长度缩短 40%。这说明**引入外部工具反而能让推理更聚焦、更高效**——模型不再需要把计算资源浪费在冗长的手动推导上。

2. **自纠正的涌现**：Figure 4 展示的案例证明，当环境反馈（解释器错误信息）足够丰富时，RL 可以驱动模型发展出超出训练分布的能力。这验证了一个重要假设：元认知能力可以通过交互反馈自发涌现，无需专门标注。

3. **极简奖励的力量**：仅用 ±1 的准确性奖励就实现了如此显著的效果，说明在工具集成场景中，环境本身提供了足够的信息信号（成功/失败、执行输出等），过度设计奖励函数反而可能限制行为多样性。

### 6.2 方法优雅性

ReTool 的方法设计非常优雅——它没有引入复杂的架构改动或奖励函数，而是通过**将代码解释器视为环境的一部分**并纳入 RL rollout，让模型在与环境的交互中自主发现策略。这种"简单但有效"的设计哲学与 DeepSeek-R1 的 GRPO 有异曲同工之妙。

### 6.3 与本项目的关联

本论文与当前 Agentic RL 研究方向高度相关：
- 工具调用策略的学习与 `nsight system` 中的 tool-use design 直接相关
- 代码解释器作为环境交互接口的设计，可为训练基础设施的 sandbox 设计提供借鉴
- outcome-based reward 的极简设计思路，与强化学习中的 reward shaping 研究相呼应

### 6.4 疑问与待验证

1. **泛化性**：代码解释器在数学以外的领域（如多模态推理、代码生成）是否同样有效？
2. **Reward hacking**：虽然作者声称极简奖励缓解了 hacking，但如果模型发现通过特定代码模式总能获得 reward 时是否会出现新的问题？
3. **与 PRM 的结合**：如果引入过程奖励模型，能否进一步加速工具使用策略的学习？
4. **多工具协调**：当存在多个可选工具时，模型如何学习最优的工具组合策略？

---

## 7. 关键引用

```bibtex
@article{feng2025retool,
  title={ReTool: Reinforcement Learning for Strategic Tool Use in LLMs},
  author={Feng, Jiazhan and Huang, Shijue and Qu, Xingwei and Zhang, Ge and Qin, Yujia and Zhong, Baoquan and Jiang, Chengquan and Chi, Jinxin and Zhong, Wanjun},
  journal={arXiv preprint arXiv:2504.11536},
  year={2025},
  url={https://arxiv.org/abs/2504.11536}
}
```

**相关文献**：
- Li et al., 2025. ToRL: Scaling Tool-integrated RL. 与 ReTool 同期的工具集成 RL 工作，在 1.5B/7B 规模上探索。
- Chen et al., 2025. TIR: Tool-Integrated Reasoning. 基于 SFT 的工具使用方法，缺乏自适应能力。
- DeepSeek-AI et al., 2025. DeepSeek-R1. 纯文本 RL 推理的代表性工作。
