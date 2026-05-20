---
created: 2026-05-19
---


paper： https://www.alphaxiv.org/abs/2605.10938?chatId=019e29b0-36ac-7ee3-8077-27c72a0b50c0

Code: https://github.com/lillian039/ELF

- ELF 证明了只要对离散领域进行极简的适配，连续扩散语言模型就能变得非常有效。它不仅在生成质量和效率上实现了平衡，还为未来的连续时间扩散语言模型提供了一个充满前景的简洁路径。



以下按照论文章节顺序，为您总结 **ELF (Embedded Language Flows)** 的核心内容：

---

## 1. 引言 (Introduction)
目前的扩散语言模型（DLM）主要分为离散和连续两类。尽管离散 DLM 近期表现强劲，但连续 DLM 是否因设计不足而落后仍是开放问题。
*   **核心挑战**：如何处理连续嵌入空间与离散 Token 空间之间的接口。
*   **基本理念**：提出 **ELF**，一种基于流匹配（Flow Matching）的极简设计，主要在连续空间运行，仅在最后一步进行离散化。

---

## 2. 背景与相关工作 (Background & Related Work)
*   **扩散/流模型**：回顾了从 DDPM 到流匹配（Flow Matching）的演进，指出流匹配在图像/视频领域已成主流。
*   **现有 DLM 局限**：
    *   **连续型**：往往在每一步都引入 Token 级的交叉熵损失，限制了流动力学的灵活性。
    *   **潜空间型**：通常依赖复杂的压缩潜空间和独立的解码器。
*   **ELF 的定位**：不使用潜空间压缩，直接在原始嵌入空间操作，且无需独立解码器。

---

## 3. ELF 核心框架 (Embedded Language Flows)

### 3.1 核心机制
*   **嵌入映射**：默认使用冻结的预训练 T5 编码器将 Token 转换为 768 维连续向量。
*   **流匹配与 $x$-预测**：采用线性插值路径。为了适应高维表示，ELF 预测干净数据 $x$ 而非速度 $v$。
    > Predicting clean embeddings (i.e., x) aligns naturally with the objective of predicting clean discrete tokens at the final step. [Method](https://alphaxiv.org/abs/2605.10938v1?page=4)
*   **权重共享的解码**：在 $t=1$ 时，同一个网络切换到“解码模式”，通过 Unembedding 矩阵输出 Token。

### 3.2 引导与控制
*   **自条件 (Self-conditioning)**：训练时 50% 概率预测两次，第二次预测以第一次的结果为条件。
*   **无分类器引导 (CFG)**：利用自条件信号作为引导，在推理时通过调节引导比例 $\omega$ 来平衡生成质量和多样性。

---

## 4. 实验验证 (Experiments)

### 4.1 消融实验 (Ablations)
*   **表示选择**：实验证明**上下文嵌入**（如 T5 编码器）显著优于非上下文嵌入（如单纯的 Lookup Table）。
*   **采样器对比**：随机性的 **SDE 采样器**在少步生成（如 32 步）中表现远超 ODE 采样器。
    > SDE-inspired sampler consistently achieves lower generative perplexity in fewer steps. [Ablations](https://alphaxiv.org/abs/2605.10938v1?page=7)

### 4.2 系统级对比 (System-Level Comparison)
*   **无条件生成**：在 OWT 数据集上，ELF-B (105M) 仅用 **32 步**就达到了比 MDLM 和 Duo 等模型更好的生成困惑度（Gen. PPL）。
*   **效率优势**：ELF 仅使用了约为其他模型 **1/10 的训练 Token 量**，且无需昂贵的蒸馏过程。
    > ELF achieves lower generative perplexity with fewer sampling steps than prior DLMs, without using distillation. [Introduction](https://alphaxiv.org/abs/2605.10938v1?page=1)
*   **条件生成**：在机器翻译（WMT14）和摘要（XSum）任务中，ELF 的表现也优于同规模的自回归和扩散基座模型。

---

## 5. 结论 (Conclusion)
ELF 证明了只要对离散领域进行极简的适配，连续扩散语言模型就能变得非常有效。它不仅在生成质量和效率上实现了平衡，还为未来的连续时间扩散语言模型提供了一个充满前景的简洁路径。

