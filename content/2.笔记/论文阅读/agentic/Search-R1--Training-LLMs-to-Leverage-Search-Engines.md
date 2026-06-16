---
created: "2026-06-09"
paper: "https://arxiv.org/abs/2503.09516"
authors: "Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan O. Arik, Dong Wang, Hamed Zamani, Jiawei Han"
tags:
  - AgenticRL
  - ToolUse
  - Search
  - RL
  - MultiHopQA
  - RAG
---

# Search-R1: Training LLMs to Reason and Leverage Search Engines with RL

## 一句话总结
Search-R1 首次将强化学习（RL）训练框架扩展到让 LLM 在推理过程中**自主调用搜索引擎**进行多轮实时检索，通过**检索 token 掩码机制**稳定训练，在 7 个 QA 数据集上平均提升 20%–41%。

![](SearchR1_fig1_framework.png)
> **Figure 1**: Search-R1 整体训练框架。上半部分为 PPO w. Search Engine 的交互流程（含 Value Model、Reward Model、Reference LLM）；下半部分为 GRPO w. Search Engine 的组采样流程。虚线两侧展示了两种 RL 算法与搜索引擎环境的集成方式。

---

## 1. 研究背景与动机

### 1.1 问题定义

外部知识的高效获取和时效性信息检索对 LLM 的推理与生成至关重要。现有方案主要分为两类：

1. **RAG（Retrieval-Augmented Generation）**：检索一次后生成，无法处理多步推理
2. **Search-as-a-tool**：提示工程驱动 LLM 调用搜索引擎，但交互策略是手动设计的，非最优

### 1.2 现有方法的不足

作者指出三个核心挑战：

| 挑战 | 问题描述 |
|------|----------|
| **RL 框架与稳定性** | 搜索引擎返回的文本（检索 token）不应计算 loss，否则会导致训练不稳定 |
| **多轮交错推理与检索** | 需要 LLM 在推理过程中自主决定何时搜索、搜索什么、如何整合检索结果 |
| **奖励设计** | 过程奖励（Process Reward）需要人工设计且容易引入偏差；需要简单的可扩展方案 |

现有工作（如 IRCoT、Search-o1）依赖人工模板或 SFT 微调搜索行为，无法让模型自主探索最优交互策略。

---

## 2. 方法

### 2.1 核心思想

Search-R1 将**搜索引擎作为 RL 环境的一部分**，训练 LLM 自主生成搜索查询并整合检索结果。核心设计包括：

1. **多轮检索交互**：LLM 通过特殊 token（`<search>`、`</search>`、`<information>`、`</information>`、`<answer>`、`</answer>`）与环境交互
2. **检索 Token 掩码**：屏蔽搜索引擎返回的 token 的 loss，确保训练稳定
3. **结果导向奖励（Outcome-based Reward）**：仅根据最终答案正确性给予奖励，无需过程奖励

**整体优化目标**（式 1）：

```
max_π E_{x~D, y~π(·|x; R)} [r_φ(x, y)] - β · D_KL[π_θ(y|x; R) || π_ref(y|x; R)]
```

| 符号 | 含义 |
|------|------|
| `π_θ` | 待训练的策略模型（Policy LLM） |
| `x` | 输入问题，从数据集 D 中采样 |
| `y` | 完整生成序列（含推理、搜索调用、检索结果、最终答案） |
| `R` | 搜索引擎环境 |
| `r_φ(x, y)` | 奖励函数（基于最终答案与标准答案的匹配度） |
| `β` | KL 散度系数，控制与参考模型的偏离程度 |
| `π_ref` | 参考模型（通常是 SFT 初始化模型，冻结参数） |
| `D_KL` | KL 散度，防止策略模型偏离参考模型太远 |

### 2.2 检索 Token 掩码机制

这是 Search-R1 最关键的技术贡献。在多轮检索交互中，完整序列 `y` 包含两类 token：

- **LLM 生成 token**：推理过程、搜索查询、最终答案
- **检索 token**：搜索引擎返回的 passage 内容

如果对所有 token 计算 loss，模型会被迫去学习预测搜索引擎返回的外部文本，这是不合理且会导致训练崩溃的。因此引入**二元掩码函数** `I(y_t)`：

```
I(y_t) = 1  if y_t 是 LLM 生成的 token
I(y_t) = 0  if y_t 是搜索引擎检索返回的 token
```

**PPO 目标函数在 Search-R1 中的形式**（式 2）：

```
J_PPO(θ) = E[ 1/Σ_t I(y_t) · Σ_{t:I(y_t)=1} min( (π_θ/π_old) · A_t,  clip(π_θ/π_old, 1-ε, 1+ε) · A_t ) ]
```

| 符号 | 含义 |
|------|------|
| `Σ_t I(y_t)` | LLM 生成 token 的总数（归一化因子） |
| `Σ_{t:I(y_t)=1}` | **仅对 LLM 生成的 token 求和**，检索 token 被 mask 掉 |
| `π_θ / π_old` | 新策略与旧策略的概率比（importance ratio） |
| `A_t` | 优势函数（Advantage），由 GAE 估计 |
| `ε` | clip 系数（通常 0.2），防止策略更新过大 |
| `clip(...)` | 裁剪操作，限制 ratio 在 `[1-ε, 1+ε]` 范围内，保证训练稳定性 |

**GRPO 目标函数在 Search-R1 中的形式**（式 3）：

```
J_GRPO(θ) = E[ 1/G Σ_{i=1}^G 1/Σ_t I(y_{i,t}) · Σ_{t:I(y_{i,t})=1} 
              min( (π_θ/π_old) · Â_{i,t}, clip(...) · Â_{i,t} ) - β · D_KL[π_θ || π_ref] ]
```

| 符号 | 含义 |
|------|------|
| `G` | 组大小（Group Size），对每个问题采样 G 个输出 |
| `y_{i,t}` | 第 i 组输出的第 t 个 token |
| `Â_{i,t}` | 组内归一化的优势函数（减去组内均值，除以组内标准差） |
| `β · D_KL[π_θ || π_ref]` | KL 惩罚项，鼓励探究但不过度偏离参考策略 |

**GRPO 相比 PPO 的关键区别**：不需要单独的 Value Model，而是通过组内采样估计 baseline，降低内存开销；但 Search-R1 实验发现 PPO 训练更稳定。

### 2.3 奖励设计

Search-R1 采用极简的**基于结果的规则奖励**（式 4）：

```
r_φ(x, y) = EM(a_pred, a_gold)
```

| 符号 | 含义 |
|------|------|
| `a_pred` | 模型预测的最终答案（从 `<answer>` `</answer>` 标签中提取） |
| `a_gold` | 标准答案 |
| `EM` | Exact Match，完全匹配得 1，否则得 0 |

**设计选择背后的考量**：
- **不使用格式奖励**：避免模型为了迎合格式而牺牲内容质量
- **不使用过程奖励（Process Reward）**：减少人工设计成本，降低reward hacking风险
- **不使用神经奖励模型**：消除奖励模型训练开销和不稳定性

这种极简奖励设计是 Search-R1 可扩展性的关键——它让 RL 训练完全依赖于环境反馈，无需额外监督信号。

### 2.4 训练流程

**生成模板**（Algorithm 1 的 rollout 流程）：

1. LLM 接收系统指令：在 `<thinking>` `</thinking>` 中进行推理，在 `<search>` `</search>` 中发出搜索查询，在 `<answer>` `</answer>` 中给出最终答案
2. LLM 自回归生成，直到遇到以下三种情况之一：
   - `</search>` → 将查询提交给搜索引擎，检索结果包裹在 `<information>` `</information>` 中追加到上下文，继续生成
   - `</answer>` → 终止生成，提取答案
   - `<eos>` → 终止生成
3. 一个完整的 rollout 是 `q → o_1 → r_1 → o_2 → r_2 → ... → a`，其中 `o_i` 是 LLM 输出，`r_i` 是检索结果

**完整训练循环**：
1. 从训练集（NQ + HotpotQA merge）采样问题 batch
2. 每个问题生成 G 个 rollout（GRPO）或 1 个 rollout（PPO）
3. 调用搜索引擎获取检索结果，拼接完整序列
4. 应用检索 token 掩码，计算 PPO/GRPO loss
5. 更新 Policy LLM，Reference LLM 保持冻结

![](SearchR1_fig2_multi-turn.png)
> **Figure 2**: 多轮检索交互示意图。展示了 LLM 如何在推理过程中自主插入 `<search>` 查询，搜索引擎返回 `<information>` 结果，LLM 根据新信息继续推理，最终给出 `<answer>`。

---

## 3. 实验结果

### 3.1 实验设置

**数据集**：7 个 QA 基准

| 类型 | 数据集 | 说明 |
|------|--------|------|
| 单跳 QA | NQ (Natural Questions) | 谷歌真实搜索问题 |
| | TriviaQA | 冷知识问答 |
| | PopQA | 流行实体问答 |
| 多跳 QA | HotpotQA | 桥接式多跳推理 |
| | 2WikiMultiHopQA | 维基百科多跳 |
| | Musique | 复杂多跳 |
| | Bamboogle | 人工构建多跳 |

**模型**：Qwen-2.5-3B / 7B（Base 和 Instruct 两个版本）

**检索器**：E5 retriever，索引 2018 Wikipedia dump，每次检索 top-3 passages

**训练数据**：NQ + HotpotQA 合并训练（每个数据集内部划分 train/dev/test）

**基线方法**：

| 方法 | 说明 |
|------|------|
| Direct Inference | 无检索直接生成 |
| CoT | Chain-of-Thought 提示 |
| IRCoT | 交错检索与生成的 SOTA 基线 |
| Search-o1 | 基于模板的搜索调用 |
| RAG | 标准检索增强生成 |
| SFT | 监督微调搜索行为 |
| R1 (w/o search) | 纯 RL 推理训练，无搜索 |
| Rejection Sampling w/ search | 带搜索的拒绝采样 |

### 3.2 主实验结果

**Table 2：Qwen2.5-7B 和 Qwen2.5-3B 上的性能对比**

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle | Avg. |
|--------|-----|----------|-------|----------|-------|---------|-----------|------|
| **Qwen2.5-7B** |
| RAG | 0.349 | 0.585 | 0.392 | 0.299 | 0.235 | 0.058 | 0.208 | 0.304 |
| R1-base | 0.297 | 0.539 | 0.202 | 0.242 | 0.273 | 0.083 | 0.296 | 0.276 |
| **Search-R1-base** | **0.480** | **0.638** | **0.457** | **0.433** | **0.382** | **0.196** | **0.432** | **0.431** |
| Search-R1-instruct | 0.393 | 0.610 | 0.397 | 0.370 | 0.414 | 0.146 | 0.368 | 0.385 |
| **Qwen2.5-3B** |
| RAG | 0.348 | 0.544 | 0.387 | 0.255 | 0.226 | 0.047 | 0.080 | 0.270 |
| **Search-R1-base** | **0.406** | **0.587** | **0.435** | **0.284** | **0.273** | 0.049 | 0.088 | 0.303 |
| Search-R1-instruct | 0.341 | 0.545 | 0.378 | **0.324** | **0.319** | **0.103** | **0.264** | **0.325** |

**关键发现**：
- **Search-R1-base (7B)** 平均 EM = 0.431，相比 RAG（0.304）提升 **41.7%**
- **Search-R1-instruct (3B)** 平均 EM = 0.325，相比 RAG（0.270）提升 **20.4%**
- **多跳 QA 提升更显著**：HotpotQA、2WikiMultiHopQA 上 Search-R1 几乎是 RAG 的 1.5–2 倍
- **Base 模型经 RL 训练后反超 Instruct**：在 7B 上，Search-R1-base（0.431）> Search-R1-instruct（0.385），说明 RL 搜索训练可以替代甚至超越 SFT 得到的指令遵循能力

### 3.3 消融实验

#### 3.3.1 PPO vs. GRPO

![](SearchR1_fig2a_ppo_vs_grpo.png)
> **Figure 2a**: PPO（蓝色）与 GRPO（橙色）的训练曲线对比（Qwen2.5-3B-it）。GRPO 初期收敛更快，但 PPO 最终稳定性更好。两者最终 reward 水平接近。

| 对比维度 | PPO | GRPO |
|----------|-----|------|
| 收敛速度 | 较慢 | 更快 |
| 训练稳定性 | **更稳定**，曲线平滑 | 后期出现震荡 |
| 内存开销 | 需要 Value Model | 不需要 |
| 最终性能 | **略优** | 接近 |

**结论**：PPO 在搜索场景下表现出更好的稳定性，是首选算法。

#### 3.3.2 Base vs. Instruct 模型

![](SearchR1_fig2b_base_vs_instruct.png)
> **Figure 2b**: Base 模型（蓝色）与 Instruct 模型（橙色）的训练曲线对比。Instruct 启动 reward 更高（已有指令遵循能力），但 Base 模型经过 RL 训练后最终追平甚至超越。

- **Instruct 模型**：初始 reward 更高，因为已具备基础推理和格式遵循能力
- **Base 模型**：从零开始学习，但最终性能与 Instruct 相当甚至更好（尤其在 7B 上）
- **启示**：RL 搜索训练可以替代 SFT 预训练，直接赋予 Base 模型工具使用能力

#### 3.3.3 检索 Token 掩码（核心消融）

| 方法 | Avg. EM (7B) |
|------|-------------|
| w. mask | **0.431** |
| w.o. mask | 0.343 |

**提升幅度**：+25.6%（0.431 vs 0.343）

![](SearchR1_fig3a_masking_3b.png)
> **Figure 3a**: 检索 Token 掩码对训练稳定性的影响（Qwen2.5-3B-base）。蓝色（w. mask）稳定收敛到较高 reward；橙色（w.o. mask）在约 250 step 后 reward 急剧下降，训练崩溃。

**机制分析**：
- **不掩码**：模型被迫学习预测检索 passage 中的 token，相当于在无关外部文本上浪费梯度
- **掩码**：模型只关注自身生成内容的质量，梯度信号更纯净
- **重要性**：该消融验证了检索 token 掩码是稳定训练的必要条件，而非可选优化

#### 3.3.4 Response Length 与有效搜索次数

![](SearchR1_fig2c_response_length.png)
> **Figure 2c**: 训练过程中平均 response length 的变化趋势。呈现“先降后升再稳定”的三阶段模式：初期模型减少无效搜索，中期增加推理深度，后期稳定在最优长度。

![](SearchR1_fig2d_valid_search.png)
> **Figure 2d**: 训练过程中有效搜索次数的变化。随着训练进行，模型学会执行更多有效搜索查询，表明 RL 确实在优化搜索策略而非随机尝试。

#### 3.3.5 其他分析

- **检索 passage 数量**：top-3 是 sweet spot，过多或过少都会降低性能
- **组大小（Group Size）**：GRPO 中 G=8 性能较好，G 过小则 baseline 估计不准，G 过大则计算开销高
- **模型泛化**：在 Llama-3.2 系列上也验证了方法有效性（Figure 3 后续图表）

---

## 4. 局限性与未来方向

### 4.1 局限性

1. **静态检索器**：使用固定的 E5 + 2018 Wikipedia，未探索联合训练检索器（如 RL 同时优化生成器和检索器）
2. **奖励稀疏性**：仅使用最终结果奖励，对长轨迹的信用分配（credit assignment）可能不够精细
3. **计算开销**：每步 rollout 都需要调用搜索引擎，训练效率低于纯文本 RL
4. **评估范围**：仅在 QA 任务上验证，未扩展到代码、数学等更复杂的工具使用场景
5. **检索质量上限**：搜索引擎返回的 passage 质量直接决定了性能天花板

### 4.2 未来方向

1. **端到端检索器优化**：将检索器纳入 RL 训练闭环（如 REINFORCE 更新检索器参数）
2. **过程奖励探索**：在保持简单性的前提下，设计轻量级过程奖励辅助长轨迹训练
3. **多工具场景**：扩展到计算器、代码解释器、数据库查询等更多工具的组合使用
4. **在线部署优化**：搜索引擎调用延迟高，需要研究异步检索、缓存策略等工程优化
5. **更大规模模型验证**：在 14B、32B 甚至更大模型上验证 scaling 规律

---

## 5. 个人思考

### 5.1 方法层面的启发

**检索 Token 掩码是最优雅的洞察**。从形式上看，这是一个极其简单的实现（一个 binary mask），但从 RL 理论角度看，它解决了一个根本性问题：环境返回的观测（检索 passage）不应该成为策略梯度的一部分。这让我联想到在具身智能（Embodied AI）中，传感器观测也不需要被策略网络“预测”——Search-R1 把这个直觉做了严谨的消融验证（Table 4 中 +25.6% 的提升），从实验上证明了其必要性。

**极简奖励设计的 scaling 优势**。在 RLHF 和 DPO 的研究中，reward modeling 是一个巨大的瓶颈——模型越大，reward model 越难训练，且容易出现 reward hacking。Search-R1 的结果奖励方案把这个复杂度降到了零，代价是信用分配可能更困难。但实验表明，只要环境交互足够丰富（多轮搜索产生足够长的轨迹），稀疏奖励仍然能够驱动有效学习。这为 RL + Tool Use 的规模化提供了一条更可行的路径。

### 5.2 与当前研究方向的关联

**与 AgenticRL 的关联**：Search-R1 本质上是把“工具调用”从 prompt engineering 降级为 RL 优化问题。这与近期 OpenAI 的 DeepResearch、Anthropic 的 Computer Use 等方向一致——LLM 不再是被动的文本生成器，而是主动的环境交互者。不同之处在于 Search-R1 把这件事做成了一个可复现的训练框架，而不是一个闭源的系统提示。

**与 R1 / DeepSeek-R1 的关系**：DeepSeek-R1 证明了纯 RL 可以让 LLM 自主涌现长推理能力（如自我验证、反思）。Search-R1 进一步证明了 RL 也可以让 LLM 自主涌现**工具使用能力**（何时搜索、搜索什么）。两者结合可以预见：未来的推理模型不仅会“想得更久”，还会“查得更准”。

### 5.3 潜在的改进空间

1. **课程学习（Curriculum Learning）**：当前训练直接在 NQ + HotpotQA 混合数据上进行。如果按检索轮数或问题难度设计课程，可能加速收敛。
2. **检索结果摘要**：当前直接把完整 passage 拼接到上下文中，导致序列长度剧增。如果让模型生成检索摘要或只提取关键句，可能提升效率。
3. **多智能体视角**：可以把 LLM（生成查询）和搜索引擎（返回结果）看作一个双智能体系统，用博弈论框架分析其均衡策略。

### 5.4 对实际应用的启示

Search-R1 的训练范式特别适合以下场景：
- **企业内部知识库问答**：企业文档是动态更新的，静态 RAG 无法应对，而 Search-R1 可以让模型学会如何查询不同的内部系统
- **实时信息检索**：股票价格、天气、新闻等时效性强的查询
- **专业领域深度研究**：如医学文献检索、法律案例检索，需要多轮 refinement

核心门槛在于**搜索引擎的延迟和成本**——如果每次 rollout 需要 3–5 次搜索调用，训练成本会显著高于纯文本 RL。这可能是该方法在实际落地时需要解决的首要工程问题。

---

## 6. 关键引用

```bibtex
@article{jin2025searchr1,
  title={Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan O. and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025},
  url={https://arxiv.org/abs/2503.09516}
}
```

**相关论文推荐**：
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- Rejection Sampling for LLM Search-o1
- IRCoT: Interleaving Retrieval with Chain-of-Thought
- WebGPT: Browser-assisted question-answering with human feedback
- RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks