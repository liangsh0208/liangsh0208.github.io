---
created: 2026-06-09
paper: https://arxiv.org/abs/2511.19933
authors: Vaishali Vinay (Microsoft Security Research)
tags:
  - LLM系统
  - 可靠性
  - 故障模式
  - 系统工程
  - 评估方法论
---

# Failure Modes in LLM Systems: A System-Level Taxonomy for Reliable AI Applications

## 一句话总结

本文从系统工程视角出发，系统性地提出了 LLM 应用中的 **15 种隐藏故障模式分类体系**，涵盖推理层、输入/上下文层和系统/运营层三个维度，揭示了现有 benchmark 在评估稳定性、可复现性和系统鲁棒性方面的根本性缺失，并给出了可靠 LLM 系统的顶层设计原则。

> **[Figure 1: 多智能体 LLM 的失败率]** — 此图来自 PDF 第 1 页。展示了 MetaGPT、ChatDev、HyperAgent、AppWorks、AG2 五种多智能体框架在不同场景下的成功率（蓝色）与失败率（黄色）对比。MetaGPT 的失败率高达 66%，AG2 的失败率为 15.2%，说明多智能体系统的故障率显著高于单一模型。

---

## 1. 研究背景与动机

### 1.1 问题定义

<font color="#ff0000">LLM 正被快速集成到决策支持工具、自动化工作流和 AI 软件系统中。然而，这些系统在生产环境中的行为仍然 poorly understood，其故障模式与传统机器学习模型 fundamentally different。</font>

**核心问题**：**现有 benchmark 只测量知识或推理能力，但对稳定性、可复现性、漂移和工作流集成等系统级属性几乎不提供任何洞察。一个在实际测试中表现完美的模型，在生产环境中可能因为系统级故障而持续崩溃。**

### 1.2 现有评估方法的不足

| 局限类型 | 具体表现 |
|---------|---------|
| **静态测试集偏差** | accuracy、perplexity 等指标只能反映固定测试集上的性能，无法预测 prompt 扰动、时间漂移下的行为 |
| **非确定性忽视** | 单次运行不足以捕捉模型行为的内在随机性，输出变异性在多步推理中可达 20-30% |
| **系统级因素缺失** | 不评估工具编排、版本漂移、成本驱动降级等部署级问题 |
| **长期一致性缺失** | 不评估跨轮次一致性、跨版本稳定性 |
| **幻觉 vs 系统故障混淆** | 当前分类只关注幻觉/偏见/安全，不涵盖推理漂移、内存管理、多智能体通信等系统级故障 |

### 1.3 核心研究问题

> **How can we trust AI agents' responses not only at the first proof-of-concept, but during their lifecycle in production settings?**

这一问题的答案不在于提升模型本身的准确率，而在于理解、预测和缓解系统级故障模式。

---

## 2. 系统级故障模式分类体系（15 种）

> **[Figure 2: LLM System Failure Taxonomy]** — 此图来自 PDF 第 3 页。展示了三个维度的 15 种故障模式的层次化分类树：左侧为 Reasoning Failures（5 种）、中间为 Input and Context Failures（5 种）、右侧为 System and Operational Failures（5 种）。

三个维度的关键区别在于故障发生的位置：
- **Reasoning failures：模型内部推理错误（即使 prompt 正确也会发生）**
- **Input & context failures：输入层问题（prompt 模糊、对抗、冲突等）**
- **System & operational failures：部署层问题（工具调用、多智能体通信、成本约束等）**

| 维度          | 故障模式                                                     | 描述                                    | 与传统 ML 故障的区别                      |
| ----------- | -------------------------------------------------------- | ------------------------------------- | --------------------------------- |
| **推理层**     | 1. Hallucinations & factual inaccuracies                 | 生成流利但非事实的内容，在系统中静默传播                  | 传统 ML 的"错误分类"是显式的，LLM 的幻觉是隐式的、流利的 |
|             | 2. Logical inconsistency & self-contradiction            | 多轮对话中前后矛盾，内部世界模型不一致                   | 传统 ML 无"对话历史"概念，无跨轮一致性要求          |
|             | 3. Multi-step planning collapse & looping                | 链式/工具流超出模型稳定推理深度，陷入循环                 | 传统 ML 通常单步推理，无长程规划                |
|             | 4. Overconfidence & calibration failure                  | 过度自信、不表达不确定性，下游组件将语言置信度误认为认知确定性       | 传统 ML 有概率输出，LLM 倾向于确定性表达          |
|             | 5. Failure to follow task constraints                    | 高级指令与上下文/系统反馈冲突时，模型趋向推断目标而非遵守约束       | 传统 ML 的输出空间固定，LLM 输出空间开放          |
| **输入/上下文层** | 6. Ambiguous or incomplete prompts                       | 模糊/不完整提示引发连锁失败，模型做出单一解释而不寻求澄清         | prompt 即输入，其质量直接决定可靠性             |
|             | 7. Prompt injection & adversarial inputs                 | 恶意输入覆盖正常行为，通过 prompt 层实现攻击            | 传统对抗攻击扰动输入像素/特征，这里是语义注入           |
|             | 8. Loss of context & truncation                          | 上下文窗口中的早期对话历史被挤出，导致"失忆"               | LLM 特有的有限上下文问题                    |
|             | 9. Domain mismatch / OOD inputs                          | 输入来自陌生或高度专业领域，预训练知识不足                 | 传统 ML 也有 OOD 问题，但 LLM 的开放输出使其更难检测 |
|             | 10. Conflicting or overlapping instructions              | 矛盾指令导致"任务脱轨"，模型在不可兼容目标间震荡             | 复合 prompt 的系统级冲突                  |
| **系统/运营层**  | 11. Tool / API invocation errors                         | 生成不存在工具名、无效参数或不按 API 规范调用             | 系统-工具交互故障                         |
|             | 12. External tool failure & runtime breakdowns           | API 失败、schema 变更、速率限制等外部故障传递          | 传统 ML 无外部工具依赖                     |
|             | 13. Communication breakdowns in multi-agent workflows    | 多智能体间共享记忆消失或被覆盖，导致"对话重置"              | 多智能体系统的特有故障                       |
|             | 14. Misalignment with application logic & business rules | 输出符合指令但违反领域约束或商业规则                    | 语义正确但系统级错误                        |
|             | 15. Cost-driven degradation & accuracy trade-offs        | 成本压力导致缩短上下文、降低采样、fallback 到弱模型等隐性性能降级 | LLM 特有的推理成本驱动决策                   |

> **[Figure 3: Human vs AI]** — 此图来自 PDF 第 3 页。对比了人类和 LLM 在回答"Why is the sky blue?"时的差异：人类考虑社交层级、先验知识、说话习惯、对齐要求、上下文、数据偏见等多重因素，回答"I am not completely sure, but I think that..."; LLM 则直接给出"I am very confident that the answer is..."。这揭示了 LLM 缺乏人类的多维推理和自我怀疑能力。

---

## 3. 评估差距分析（The Evaluation Gap）

### 3.1 现有 Benchmark 的系统性缺陷

| 缺陷 | 说明 |
|------|------|
| **固定测试集误导** | 固定测试集上的 accuracy 无法反映 prompt 扰动下的行为变化 |
| **单点评估不充分** | 单次 inference 无法捕捉模型的随机性输出，相同 prompt 多次运行结果可能截然不同 |
| **表现 vs 可靠性混淆** | 传统指标（BLEU, ROUGE, accuracy）衡量知识或推理能力，不衡量稳定性、可复现性、长期行为一致性 |
| **幻觉中心主义** | 现有分类聚焦幻觉/偏见/安全，但忽略了 tool-use 可靠性、版本漂移、多智能体协调等系统级问题 |

### 3.2 非确定性（Non-determinism）问题

- 温度、top-p、采样参数导致相同输入产生不同输出
- 研究发现：仅颠倒 prompt 中条目的顺序，近 50% 的比较对发生翻转（near 50% of pairs of comparisons flipped upon reversal in response order）
- 这严重削弱了基于 A/B 测试的系统升级策略

### 3.3 漂移（Drift）的多种类型

| 漂移类型 | 描述 |
|---------|------|
| **Version drift** | 模型更新/微调导致格式、推理风格或工具调用顺序改变 |
| **Data drift** | 真实输入分布偏离训练/验证分布 |
| **Behavior drift** | 相同 prompt 因随机采样或内部变更产生不同输出 |
| **Prompt drift** | prompt 措辞微调导致输出剧烈变化 |

---

## 4. 部署现实：生产差距（Deployment Realities）

### 4.1 可观测性（Observability）与监控不足

- 传统基础设施监控只追踪延迟和错误率，不追踪行为漂移
- 缺乏纵向分布监控（longitudinal monitoring of distribution shifts）
- 需要引入 output-variance tracking、formatting-change detection、行为指标纵向采样等能力

### 4.2 成本约束（Cost & Latency Constraints）

- 生产系统必须在推理成本预算内运行，导致：
  - 缩短上下文窗口
  - 降低采样温度/token 数
  - 使用更小的模型或 aggressive caching
  - 牺牲精度换取成本
- 这种 trade-off 是隐性的，标准监控不会暴露

### 4.3 版本控制与合规审计（Reproducibility & Auditability）

- 没有版本化的 prompt、检索日志、上下文快照和工具调用 trace，几周后无法复现特定结果
- 在金融、医疗、法律等受监管领域，这削弱了可审计性和可信赖性

> **[Figure 4: Incorrect tool failures in Gorilla]** — 此图来自 PDF 第 4 页。热力图展示了不同模态（Image, Language, Audio, Video, Tabular, Multimodal, RL, General）的工具调用失败模式。其中 Image→Language 的 ground truth 与 predicted 之间存在显著不匹配，说明多模态工具调用中模态映射错误是重要失败来源。

### 4.4 Prompt 工程的可复现性

> **[Figure 5: Prompt Engineering]** — 此图来自 PDF 第 7 页。展示了 prompt 工程的系统级流程，包括知识库检索、prompt 模板构建、LLM 推理到响应生成的完整链路。强调了 prompt 格式标准化和模块化组件对响应稳定性的影响。

---

## 5. 可靠 LLM 系统的设计原则

### 5.1 输入标准化（Input Canonicalization）

- LLM 输入需要**标准化**（canonicalized）：推理前重新格式化、重新排序、去噪
- 工作流应基于**版本化的 prompt 模板**，而非动态组装的临时指令
- prompt 中的自由格式指令是稳定性杀手

### 5.2 验证层（Verification Mechanisms）

- **Verifier layers** 仍处于起步阶段但至关重要
- 显式验证难以检测幻觉，中间验证可以限制多阶段 LLM pipeline 中错误推理的传播
- 需要 schema validation、一致性重运行（re-runs for consistency）等中间检查

### 5.3 监控体系（Monitoring for Operational Reliability）

| 监控维度 | 具体要求 |
|---------|---------|
| **输出方差追踪** | 相同输入的多次运行方差应在阈值内 |
| **格式变更检测** | 结构化输出格式的突然变化应触发告警 |
| **行为漂移监控** | 纵向采样行为指标，不只检查基础设施健康 |
| **分布偏移检测** | 通过时间序列输入分布监测发现数据漂移 |

### 5.4 成本治理（Cost Governance）

- 需要显式权衡：token 预算、模型选择、采样策略对准确率和可靠性的影响
- 避免隐性"精度降级"：监控指标应暴露成本驱动选择带来的可靠性损失

---

## 6. 局限性与未来方向

1. **缺乏标准化评估指标**：需要 stability、drift、reproducibility 和 cost 的标准化度量
2. **Benchmark 设计**：现有 leaderboard 评估的是小型孤立问答任务，真实部署需要长程规划、工具链和智能体交互
3. **漂移检测**：需要系统性地研究模型输出随时间的微妙变化，即使 prompt 保持不变
4. **工具使用可靠性**：现代 LLM 系统高度依赖 API、检索和外部服务，但系统化的工具调用正确性验证仍然缺失
5. **可观测性框架**：传统基础设施遥测不披露对齐错误、幻觉或逻辑不匹配，需要新的信号和仪表板
6. **成本-可靠性建模**：token 预算约束下的可靠性建模仍处于起步阶段

---

## 7. 个人思考

1. **范式转换的价值**：本文最有价值的贡献不在于提出了 15 种新故障（其中许多已有零星讨论），而在于**将它们组织成一个系统级分类框架**，并明确指出了"这不是模型问题，而是系统工程问题"的范式转换。这种视角的转换决定了后续研究和工程实践的方向。

2. **对评估社区的冲击**：文中指出"的评估差距"可能是最刺耳也最真实的批评——当前 AI 社区大量精力投入到创造新 SOTA，但在生产可靠性方面的评估几乎为零。"一个在固定测试集上表现完美的模型，在生产环境中可能因为系统级故障而持续崩溃"这一论断令人警醒。

3. **与非确定性共存**：LLM 的非确定性是设计特性而非故障，但传统软件工程假设确定性行为。如何为本质上非确定性的组件建立可靠系统，这是一个根本性的工程挑战。文中提到的"单次运行不足以评估"和"近 50% 的比较对会因顺序颠倒而翻转"等数据极具说服力。

4. **成本驱动的隐性降级**："成本驱动降级"是一个被严重低估的问题。许多生产系统因为预算压力而默默降低可靠性，但现有监控不暴露这种 trade-off。这提示我们：**可靠性监控必须显式纳入成本维度**。

5. **与当前 LLM 研发的关联**：类似 Anthropic 的 Constitutional AI、OpenAI 的 RLHF 等对齐工作主要解决推理层故障（幻觉、偏见），但本文指出更大的可靠性挑战在输入层和系统层。这意味着即使模型本身"完美"，系统仍可能不可靠。

---

## 8. 关键引用

```bibtex
@article{vinay2025failure,
  title={Failure Modes in LLM Systems: A System-Level Taxonomy for Reliable AI Applications},
  author={Vinay, Vaishali},
  journal={arXiv preprint arXiv:2511.19933},
  year={2025}
}
```
