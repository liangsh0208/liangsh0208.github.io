eagle 系列也是 MTP 的一个具体实现。 可以参考这个链接的文章。[MTP-Multi-Token Prediction](MTP-Multi-Token%20Prediction.md)




**论文信息**  
- **标题**: **EAGLE-3: Extreme Acceleration of LLMs via Speculative Decoding with Training-time Test**  
- **作者团队**: 该项目由来自北京大学（PKU）、智源人工智能研究院（BAAI）等机构的研究人员共同开发。  
- **arXiv 链接**: [arXiv:2412.00000](https://www.google.com/search?q=https://arxiv.org/abs/2412.00000)（注：由于是 2024 年底至 2025 年初发布的成果，建议直接在 arXiv 搜索 "EAGLE-3 speculative decoding"）。  
- **开源代码**: [https://github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)（该仓库涵盖了从 EAGLE 到 EAGLE-3 的持续更新）。


![424](Pasted%20image%2020260515115452.png)

# 简介
**EAGLE-3**（Extrapolation Algorithm for Greater Language-Model Efficiency, Version 3）是目前大模型推理加速领域最前沿的**投机采样（Speculative Decoding）**算法之一。它由轻量级的“草稿头”（Draft Head）插件和创新的训练策略组成，旨在解决标准自回归生成速度慢的问题。

相比于早期的 EAGLE 或 Medusa，EAGLE-3 的核心改进在于**多层特征融合**、**从特征预测回归到 Token 预测**以及**训练时测试（Training-Time Test）**机制。

### 1. 核心原理与架构

EAGLE-3 的本质是在不改变目标模型（Target Model）权重的前提下，在其上方附加一个极轻量级的预测层（通常仅占参数量的 2-5%）。

#### A. 多层特征融合 (Multi-layer Feature Fusion)

传统的投机采样（如 EAGLE-1）通常只使用目标模型最后一层的隐藏状态（Hidden States）。

- **EAGLE-3 的做法：** 它从目标模型的**不同深度**（底层、中层、高层）提取特征并进行融合。
    
- **优势：** 这种做法能够同时捕获底层语义信息和高层抽象逻辑，为“草稿头”提供更丰富的上下文背景，从而显著提高预测准确率。
    

#### B. 从特征预测转向 Token 预测 (Direct Token Prediction)

- **早期 EAGLE：** 试图预测下一个 Token 的“特征向量”（Feature Regression），然后再映射回 Token。
    
- **EAGLE-3：** 回归到了直接预测 **Token 概率分布** 的方式。
    
- **优势：** 简化了预测链条，减少了特征预测带来的误差累积，且更易于通过增大训练数据规模来提升性能。
    

### 2. 关键创新：训练时测试 (Training-Time Test, TTT)

这是 EAGLE-3 能够超越前代产品的核心技术：

- **分布不一致问题：** 在实际推理时，投机算法是基于自己生成的“草稿 Token”继续往下预测的（递归过程）；但在传统训练中，模型通常是基于“真实 Token”（Teacher Forcing）训练的。
    
- **TTT 机制：** 在训练阶段就模拟推理时的递归过程。让“草稿头”在训练时就尝试基于自己前一步的预测输出作为下一步的输入。
    
- **效果：** 这让模型学会了如何处理自身预测带来的偏差，大幅提升了在多步投机（Tree-based speculative decoding）中的**接受率（Acceptance Rate）**。
    

### 3. 工作流程：草稿 -> 验证 -> 树形搜索

1. **生成草稿树（Drafting）：** “草稿头”利用目标模型的隐藏状态，快速生成一棵候选 Token 树（多个可能的后续序列）。
    
2. **并行验证（Verification）：** 目标模型（如 Llama 3 或 Qwen）在单次前向传播中，利用 **Tree Attention** 同时验证这棵树上的所有路径。
    
3. **接受与回滚：** 目标模型找出与自己预测最匹配的最长路径。被验证通过的 Token 直接输出，不匹配的则被丢弃，并由目标模型补上一个正确的 Token。
    

### 4. 性能表现与优势

|特性|表现|
|---|---|
|**加速比**|在 Llama-3 或 Qwen 系列上通常可达 **2x - 5x** 的速度提升。|
|**无损精度**|由于最终由目标模型进行验证，其输出结果与原始模型**完全一致**。|
|**易用性**|目前已成为 vLLM、SGLang 和 TensorRT-LLM 等主流框架的标配。|
|**扩展性**|随着训练数据量的增加，加速效果呈现明显的正向 Scaling Law。|

### 总结

EAGLE-3 通过“借用”主模型的深度特征，并配合专门设计的轻量级 Transformer 层，实现了一种既快又准的“草稿机制”。对于像你这样从事 **LLM 分布式训练与优化** 的工程师来说，EAGLE-3 的重要性在于它几乎是目前生产环境下**投机采样**的工业标准，能显著降低高并发推理时的首字延迟（TTFT）和每秒 Token 数（TPS）。



# eagle 系列演进

|特性|EAGLE (v1)|EAGLE-2|EAGLE-3|P-EAGLE (最新)|
|---|---|---|---|---|
|核心创新点|特征外推 (Extrapolation)：预测 Hidden State 而非 Token。|动态树 (Dynamic Draft Tree)：基于置信度动态调整采样树结构。|TTT (Training-time Test)：解决训练与推理分布不一致。|并行采样 (Parallel)：一次前向传播生成所有草稿 Token。|
|草稿头输入|目标模型最后一层特征。|目标模型最后一层特征。|多层特征融合（低、中、高层混合）。|目标模型多层融合特征。|
|预测目标|特征回归 (Feature Regression)。|特征回归 (Feature Regression)。|直接预测 Token。|直接预测 Token。|
|训练策略|标准 Teacher Forcing。|同 v1（主要改进在搜索策略）。|模拟推理过程的递归训练。|并行监督训练。|
|典型加速比|2.0x - 3.0x|3.0x - 4.2x|3.0x - 6.5x|在 B200 等硬件上进一步提升 20-40%。|
|主要优势|比传统小模型投机更快。|搜索效率更高，对上下文更敏感。|极其稳健，即便预测步数多，准确率也不崩。|消除草稿头自身的串行延迟。|

# 与 deepseek MTP 的区别

作为算法工程师，你可能已经注意到 **DeepSeek-V3 的 MTP (Multi-Token Prediction)** 和 **EAGLE-3** 都是为了解决同一个问题：LLM 推理时的访存受限（Memory-bound）导致的速度瓶颈。

但在架构设计、训练目标和工程落地层面，两者有非常显著的异同。

### 1. 核心技术异同对比

|维度|**DeepSeek-V3 MTP**|**EAGLE-3**|
|---|---|---|
|**角色定位**|**内生能力**。是模型预训练的一部分，属于原生架构。|**外挂插件**。是针对已有模型（如 Llama/Qwen）训练的轻量预测头。|
|**预测方式**|**串行链式 (Sequential)**。第 n 个 MTP 模块依赖第 n-1 个的输出。|**特征外推 + 递归**。利用多层特征融合进行递归预测。|
|**训练目标**|**联合优化**。MTP 损失与主模型 Next-Token 损失共同训练，提升表征能力。|**独立优化**。主模型权重冻结，仅训练极轻量的预测头。|
|**输入来源**|主模型隐层输出 + 前一个 MTP 的 Embedding。|**多层特征融合**（低、中、高层特征拼接）。|
|**推理架构**|典型的“主模型 + 附加头”投机采样。|典型的“主模型 + 辅助模型”投机采样。|

### 2. 深度分析：相同点

- **本质逻辑一致：** 两者在推理阶段都遵循 **Speculative Decoding (投机采样)** 的逻辑。即“先低成本草拟多个 Token，再由主模型一次性并行验证”。
    
- **无损加速：** 无论 MTP 还是 EAGLE-3，最终都由目标模型（Target Model）进行 Logits 验证，因此输出结果与原始模型完全等价，不会损失精度。
    
- **硬件友好：** 都充分利用了现代 GPU 在验证阶段的并行计算能力，将受限于带宽的解码过程转化为受限于计算的过程。
    

### 3. 关键不同点（算法工程师视角）

#### A. 训练哲学：原生 vs 后置

- **DeepSeek MTP：** DeepSeek 认为 MTP 应该在**预训练阶段**就介入。论文指出，训练 MTP 不仅能加速推理，还能增强主模型的语义表征能力（尤其是对于逻辑和代码），因为模型被迫要“想得更远”。
    
- **EAGLE-3：** 属于**后置加速方案**。它的存在是为了让那些没有原生 MTP 能力的模型（如 Llama 3、Qwen 2.5）也能获得类似的加速效果。它的创新点在于 **TTT (Training-time Test)**，通过模拟推理时的递归过程来减少分布偏移。
    

#### B. 预测路径的设计

- **MTP 的“链式”结构：** DeepSeek 的 MTP 模块是串行的。预测第 2 个 Token 时，它会将第 1 个预测 Token 的 Embedding 与主模型的隐层特征结合。这种设计保持了完整的因果链。
    
- **EAGLE-3 的“特征融合”：** EAGLE-3 不仅仅看主模型的最后一层，它会融合多个中间层的特征。这类似于一种“全局信息采样”，能够捕捉到主模型在不同抽象层级上的意图，从而在复杂的投机任务中获得更高的接受率（Acceptance Rate）。
    

#### C. 工程部署

- **MTP：** 由于是模型内置的，权重的布局（Layout）在训练时就确定了。在 vLLM 或 SGLang 中，MTP 通常作为单独的 `mtp_heads` 加载，显存占用固定。
    
- **EAGLE-3：** 更加灵活。你可以针对同一个基座模型训练不同尺寸、不同精度的 EAGLE 头。甚至可以根据业务场景（如纯代码 vs 纯文学）切换不同的加速头。
    

### 4. 性能表现 (2026年数据参考)

- **DeepSeek-V3 原生 MTP：** 在典型的 Chat 任务下，接受率通常在 **80%-90%**，能带来约 **1.8x - 2.1x** 的端到端加速。
    
- **EAGLE-3：** 针对 Qwen 或 Llama 适配后，在部分长文本场景下加速比可达 **3x - 5x**。这主要归功于其多层特征融合带来的极高预测准确度，允许它一次性尝试投机更多的 Token（Draft Steps 更大）。
    

### 总结建议

如果你正在开发 **原生大模型**，DeepSeek 的 **MTP 方案** 是提升模型上限和推理速度的最佳预训练策略。 如果你是在优化 **已有开源模型** 的推理流水线，**EAGLE-3** 则是目前最强、兼容性最好的插件式加速方案。


# 论文阅读
![](Pasted%20image%2020260515120652.png)


## eagle1
paper: https://www.alphaxiv.org/abs/2401.15077

![441](Pasted%20image%2020260515124240.png)

- draft model 的输入有2个：隐层 feature 和 token embedding。
- draft model 是一层 layer + LM head【复用原始模型】
- 最后会生成一个树形结构。 在验证阶段会基于原始模型，主节点接受/拒绝
- 

- 与其它算法的差异示意图：
![523](Pasted%20image%2020260515133953.png)


## eagle2
https://www.alphaxiv.org/abs/2406.16858

- 展示了投机采样和 egale 的差异。 以及验证步骤。

![](Pasted%20image%2020260515134655.png)
- 简单示例，两个版本的差异。 eagle 是按固定形状tree，构建备选token。  eagle 是动态构建，更灵活有效。

![363](Pasted%20image%2020260515135509.png)

Figure 5 通过大量实验数据的波动规律，指出了**静态草稿树无法适应多变的推理场景**，从而为 EAGLE-2 这种能够根据上下文实时“变形”的动态算法铺平了道路。

