---
created: 2026-05-09
---



# VLM 模型架构

一个标准的VLM由三个紧密协作的模块构成[](https://developer.aliyun.com/article/1693391)[](https://developer.aliyun.com/article/1682879)：

- **视觉编码器（Vision Encoder）**：负责“看”。主流方案是**ViT（Vision Transformer）**，它将图像切分成小块并转换为特征向量，部分模型也使用CNN（如ResNet）[](https://developer.baidu.com/article/detail.html?id=6874046)[](https://developer.baidu.com/article/detail.html?id=6859956)。
    
- **语言模型（Language Model）**：负责“思考”和“回答”。作为系统的大脑，常使用**LLaMA、Qwen、GPT**等预训练模型来理解指令并生成文本[](https://developer.aliyun.com/article/1693391)。
    
- **连接模块（Connector/Projector）** ：作用是“翻译”。它将视觉特征映射到语言模型能理解的语义空间，实现跨模态对齐。最简单的方式就是一个**MLP（多层感知机）**[](https://developer.aliyun.com/article/1682879)。


## CLIP
Learning Transferable Visual Models From Natural Language Supervision
paper: https://www.alphaxiv.org/abs/2103.00020?chatId=019e0aa4-b1e1-7d0e-98a1-e77772f54699

CLIP（Contrastive Language-Image Pre-training）是由 OpenAI 提出的一种基于大规模天然语言监督的视觉表征学习算法。其核心思想是利用互联网上海量的“图像-文本”对，通过对比学习（Contrastive Learning）将视觉概念与自然语言描述建立联系。

### 核心设计思路：从固定分类到自然语言监督

传统的计算机视觉模型（如在 ImageNet 上训练的模型）通常被限制在预测一组固定的、预定义的类别（如 1000 个类）。这种方式限制了模型的通用性，因为识别新物体需要额外的标注数据和重新训练。

CLIP 则采用**自然语言**作为监督信号：

- **数据集规模**：使用了从互联网收集的 400 亿个（图像，文本）配对数据集（WIT）。
- **任务定义**：不要求模型预测具体的类别标签，而是让模型预测“哪一段文本与哪一张图片最匹配”。


### 模型设计架构
![](Pasted%20image%2020260509140822.png)
架构包含两个主要编码器：一个**图像编码器（Image Encoder）**（如 ResNet 或 ViT）和一个**文本编码器（Text Encoder）**（Transformer）。

> 对比预训练（Contrastive Pre-training）

在训练过程中，模型接收一个包含 $N$ 个（图像，文本）对的 Batch。

1. **特征提取**：图像编码器将 $N$ 张图像转换为向量 $I_1, I_2, ..., I_N$；文本编码器将 $N$ 个文本转换为向量 $T_1, T_2, ..., T_N$。
2. **多模态嵌入空间**：通过线性投影将这些向量映射到同一个共享的嵌入空间。
3. **计算相似度**：计算这 $N$ 张图片与 $N$ 个文本两两之间的余弦相似度，形成一个 $N \times N$ 的相似度矩阵。
4. **对比损失（Contrastive Loss）**：训练目标是最大化 $N$ 个真实配对（对角线上的元素 $I_i \cdot T_i$）的相似度，同时最小化其他 $N^2 - N$ 个错误配对的相似度。


模型实现伪代码：
![473](Pasted%20image%2020260509141218.png)

### 下游应用方法
#### 零样本迁移模式 (无需微调，最推荐)

这是 CLIP 最具代表性的用法。它不需要任务特定的训练数据，直接利用预训练时学到的“图像-文本”关联能力进行推理。

- **步骤**：
    1. **定义类别文本**：将任务的标签转换为句子（如“一张汽车的照片”、“一张飞机的照片”）。
    2. **生成分类权重**：将这些句子输入文本编码器，得到一组特征向量。
    3. **相似度匹配**：将待识别的图片输入图像编码器，计算其向量与所有类别文本向量的相似度，相似度最高的即为预测类别。
- **适用场景**：通用物体识别、跨域任务、由于没有标注数据而无法训练的任务。

> “The learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes.” 

---

#### 特征提取模式 (Linear Probe)

如果你拥有一定量的标注数据，并追求在该任务上的极限精度，可以采用这种模式。

- **做法**：
    - 冻结 CLIP 的图像编码器（不改变其参数）。
    - 将图像输入编码器提取出高维特征。
    - 在特征层之后接一个简单的**逻辑回归（Logistic Regression）**或**线性层**，只训练这层权重。
- **效果**：实验表明，CLIP 的线性探针（Linear Probe）在多个数据集上甚至超过了在 ImageNet 上全监督预训练的模型。


## Flamingo
Flamingo: a Visual Language Model for Few-Shot Learning
paper: https://www.alphaxiv.org/abs/2204.14198



![](Pasted%20image%2020260509142416.png)

![](Pasted%20image%2020260509143249.png)

<alphaxiv-thinking-title title="Analyzing Flamingo architecture components" />Flamingo 是一种视觉语言模型 (VLM) 家族，其核心设计理念是通过创新的架构组件将预训练且**冻结**的视觉编码器与语言模型连接起来。这种设计使其能够处理交错的图像、视频和文本序列，并具备强大的少样本 (few-shot) 学习能力。

以下是 Flamingo 模型架构的详细分解：

### 1. 视觉处理管道 (Visual Processing)

Flamingo 的视觉部分负责将原始像素转化为语言模型可以理解的“视觉标记”。

*   **视觉编码器 (Vision Encoder)**：采用预训练并冻结的 **Normalizer-Free ResNet (NFNet-F6)**。它首先在图文对数据集上通过对比学习目标进行预训练，用于提取空间特征。对于视频，模型以 1 FPS 采样帧并独立编码，随后加入学习到的时间嵌入。
*   **Perceiver Resampler**：这是连接视觉与语言的关键。由于视觉编码器输出的特征数量随分辨率或视频帧数变化，Perceiver Resampler 通过一组固定的学习潜变量 (Latent Queries) 对这些特征进行交叉注意力处理，最终输出固定数量 (64个) 的视觉标记。

> "This module connects the vision encoder to the frozen language model... It takes as input a variable number of image or video features from the vision encoder and produces a fixed number of visual outputs (64)" [Perceiver Resampler](https://alphaxiv.org/abs/2204.14198?page=5)

---

### 2. 视觉调节的语言模型 (Conditioning the LM)

Flamingo 并不从头训练语言模型，而是利用强大的预训练语言模型（如 Chinchilla），并通过插入新层来引入视觉信息。

#### GATED XATTN-DENSE 层
这是 Flamingo 最核心的创新。在冻结的语言模型 (LM) 层之间，研究者插入了可训练的交叉注意力层。

*   **交叉注意力**：查询 (Query) 来自语言特征，键 (Key) 和值 (Value) 来自 Perceiver Resampler 输出的视觉标记。
*   **Tanh 门控机制**：为了保证在初始化时模型性能不下降，新层采用了 $tanh(\alpha)$ 门控。初始时 $\alpha=0$，意味着 $tanh(0)=0$，此时模型完全等同于原始的预训练 LM。

> "To ensure that at initialization, the conditioned model yields the same results as the original language model, we use a tanh-gating mechanism... At initialization, the model output matches that of the pretrained LM, improving training stability and final performance." [Gated Layers](https://alphaxiv.org/abs/2204.14198?page=5)

---

### 3. 多模态交错处理与因果掩码

Flamingo 能够处理极其复杂的输入格式（例如：图像1 + 文本 + 图像2 + 文本）。

*   **多视觉输入支持**：模型通过一个特定的函数 $\phi$ 来定义掩码。在预测某个文本标记时，模型只交叉关注在该标记之前出现的**最后一张**图像或视频。这种设计限制了计算量，同时保证了时序上的逻辑一致性。
*   **交错格式**：通过在文本中插入 `<image>` 标签和 `<EOC>` (End of Chunk) 标记，模型可以像处理纯文本一样处理多模态序列。

---

### 4. 模型规模对比

Flamingo 提供了三种不同规模的版本，主要区别在于冻结的语言模型大小以及插入 GATED XATTN-DENSE 层的频率：

| 模型名称 | 冻结 LM 规模 | 总参数量 | 交叉注意力层插入频率 |
| :--- | :--- | :--- | :--- |
| **Flamingo-3B** | 1.4B (Chinchilla) | 3.2B | 每层 LM 之前插入 |
| **Flamingo-9B** | 7B (Chinchilla) | 9.3B | 每 4 层 LM 插入一次 |
| **Flamingo-80B** | 70B (Chinchilla) | 80B | 每 7 层 LM 插入一次 |



![](Pasted%20image%2020260509143205.png)


## BLIP-2
paper: https://www.alphaxiv.org/abs/2301.12597?chatId=019e0b71-eee3-715e-8b52-7ce9df0ea0de

论文中模型架构的示意图，方便直接理解。

![469](Pasted%20image%2020260509143712.png)

### - Qformer的模型架构
- 主要是用于 image encoder 与 LLM 的映射器。
![](Pasted%20image%2020260509143816.png)


- 编码后的图像embedding 如何插入到LLM 继续
![](Pasted%20image%2020260509143854.png)


### - 论文中的VQA finetune架构
![479](Pasted%20image%2020260509144005.png)


## LLAVA

Visual Instruction Tuning
paper: https://www.alphaxiv.org/abs/2304.08485


- 模型架构： 基于CLIP 做图像编码，然后过一个MLP 映射层，拼接 文本 token 输入LLM 做生成。

![](Pasted%20image%2020260509145646.png)


基于大模型做了个总结：
这篇论文介绍了 **LLaVA (Large Language and Vision Assistant)**，这是将**指令微调（Instruction Tuning）**扩展到多模态领域的开创性工作。其核心目标是构建一个能够像人类一样理解图像并遵循复杂指令的通用视觉助手。

---

### 核心贡献与创新点

#### 1. 多模态指令数据的自动生成
针对多模态指令数据匮乏的问题，作者提出了一种创新的数据生成流水线。他们利用**仅纯文本的 GPT-4**，通过输入图像的文本描述（如标题和检测框坐标）作为媒介，生成了 158K 独特的语言-图像指令遵循样本。
*   **数据类型**：包含多轮对话、详细描述和复杂的逻辑推理。
*   **创新之处**：这种方法将现有的图像-文本对转化为适合微调助理模型的指令格式。

> “我们提出了一个数据重构的视角和流水线，使用 ChatGPT/GPT-4 将图像-文本对转换为适当的指令遵循格式。” [Contributions](https://alphaxiv.org/abs/2304.08485v2?page=2)

#### 2. LLaVA 模型架构
LLaVA 的结构简洁而高效，主要由三个部分组成：
*   **视觉编码器**：采用预训练的 **CLIP ViT-L/14**。
*   **语言模型**：采用开源的 **Vicuna**（基于 LLaMA 微调的领先文本模型）。
*   **连接层（Projection）**：使用一个简单的**线性映射层**将视觉特征对齐到语言模型的词嵌入空间。

$$H_v = W \cdot Z_v$$

这种轻量级的连接设计允许模型快速迭代并有效利用现有的强大预训练模型。 [Architecture](https://alphaxiv.org/abs/2304.08485v2?page=4)

---

### 训练流程 (Training)

为了让模型学会如何“看图说话”并听从指令，训练分为两个阶段，层层递进：

#### **阶段 1：特征对齐预训练 (Pre-training for Feature Alignment)**

- **目标**：让语言模型开始理解视觉信号，即训练一个兼容的“视觉分词器”。
- **数据**：使用从 CC3M 过滤出的 59.5 万个图像-文本对，转换为简单的单轮对话（例如：问题是“请简要描述这张图”，回答是原有的 Caption）。
- **策略**：**冻结视觉编码器和语言模型**。
- **训练对象**：**只训练投影矩阵 $W$**。
- **意义**：这个阶段模型只是在学习如何把图片里的物体和对应的单词对应起来，而不改变模型原有的对话逻辑。

#### **阶段 2：端到端微调 (Fine-tuning End-to-End)**

- **目标**：让模型真正成为一个能够理解复杂意图并进行推理的助理。
- **数据**：使用作者生成的 158K 高质量指令遵循数据（包含对话、详细描述、逻辑推理）。
- **策略**：保持视觉编码器冻结，但**同时更新投影矩阵 $W$ 和语言模型 $\phi$ 的权重**。
- **应用场景**：
    1. **多模态聊天机器人**：在混合了对话和描述的数据上微调，提升日常交互能力。
    2. **Science QA**：在特定科学问答数据集上微调，增强模型在专业领域的推理能力。

> “在训练中，我们保持视觉编码器和 LLM 的权重冻结，仅最大化可训练参数 $\theta = W$（投影矩阵）的似然概率。”

---

### 实验结果与表现

#### 视觉对话能力
在专门构建的 **LLaVA-Bench** 测试中，LLaVA 展示了出色的对话和推理能力。即使在未见过的图像上，它也能表现出类似于 GPT-4 的行为。

| 评估基准 | LLaVA 相对得分 (对比 GPT-4) |
| :--- | :--- |
| LLaVA-Bench (COCO) | **85.1%** |
| 复杂推理能力 | **81.7%** (In-the-wild) |

#### ScienceQA 科学问答
LLaVA 在多模态科学问答任务上表现卓越。当与 GPT-4 结合使用（作为评判者）时，达到了 **92.53%** 的准确率，刷新了当时的 SOTA 记录。 [ScienceQA Results](https://alphaxiv.org/abs/2304.08485v2?page=9)

> “我们的实验表明，LLaVA 展示了令人印象深刻的多模态聊天能力，有时在未见过的图像/指令上表现出多模态 GPT-4 的行为。” [Abstract](https://alphaxiv.org/abs/2304.08485v2?page=1)

---

### 总结
LLaVA 证明了**视觉指令微调**是提升大型多模态模型通用能力的关键。它不仅提供了一个高性能的开源模型，还贡献了一套完整的自动化数据生成流程，为后续如 LLaVA-1.5 等更强大的模型奠定了基础。



## qwen3-vl

paper: https://www.alphaxiv.org/abs/2511.21631


![](Pasted%20image%2020260509152930.png)

- 基于大模型做个总结，主要是了解模型结构即可。 具体细节后续再补。
<alphaxiv-thinking-title title="Fetching more paper details" />

<alphaxiv-thinking-title title="Composing detailed response" />

以下是 Qwen3-VL 模型架构和训练流程的完整详解：

---

### 一、模型整体架构

Qwen3-VL 采用**视觉编码器 + MLP 融合层 + 大语言模型**的经典三模块架构，但在每个模块上均进行了关键升级。

> "The Qwen3-VL framework integrates a vision encoder and a language model decoder to process multimodal inputs, including text, images, and video." [Architecture Overview](https://alphaxiv.org/abs/2511.21631v2?page=3)

#### 1.1 视觉编码器（Vision Encoder）

- 基于 **SigLIP-2** 架构，并通过动态分辨率输入进行持续训练
- 采用 **2D-RoPE** 进行位置编码，并根据输入尺寸对绝对位置嵌入做插值
- 根据模型规模使用不同配置：
  - 2B、4B 轻量模型：使用 **SigLIP2-Large (300M)**
  - 其余模型：默认使用 **SigLIP2-SO-400M** [Vision Encoder](https://alphaxiv.org/abs/2511.21631v2?page=3)

#### 1.2 MLP 视觉-语言融合层（VL Merger）

- 使用**两层 MLP** 将视觉编码器输出的 2×2 视觉特征压缩为单个视觉 token
- 压缩后的维度与 LLM 的隐藏维度对齐
- 额外部署了**专用 Merger 模块**来配合 DeepStack 机制，将不同层次的视觉特征分别注入 LLM [MLP Merger](https://alphaxiv.org/abs/2511.21631v2?page=3)

#### 1.3 大语言模型（LLM 骨干）

Qwen3 系列作为语言模型底座，提供多种规格供选择：

| 类型 | 型号 |
|------|------|
| 致密模型 | 2B / 4B / 8B / 32B |
| 混合专家模型 | 30B-A3B / 235B-A22B |

---

### 二、三大架构创新

#### 2.1 交错式 MRoPE（Interleaved MRoPE）

原始 MRoPE 将嵌入维度划分为时间（$t$）、水平（$h$）和垂直（$w$）三个子空间，分别分配不同的旋转频率。这导致**频谱不平衡**，在长视频理解任务上表现较差。

新方案将 $t$、$h$、$w$ 三个分量以**交错方式**均匀分布在嵌入维度上，确保每个时空轴都均匀覆盖低频和高频波段，从而显著提升了长视频的位置建模能力。 [Interleaved MRoPE](https://alphaxiv.org/abs/2511.21631v2?page=3)

#### 2.2 DeepStack 多层视觉融合

DeepStack 是本版本最重要的架构创新之一。不同于原始 DeepStack 堆叠多尺度输入，Qwen3-VL 将其扩展为**从 ViT 中间层提取视觉 token**：

- 选取视觉编码器**三个不同深度**的特征层
- 各层特征由专用 Merger 投影为视觉 token
- 这些 token 直接**加到 LLM 前三层的对应隐藏状态上**

这种设计保留了从低层纹理到高层语义的丰富视觉信息，同时不会增加上下文长度。 [DeepStack](https://alphaxiv.org/abs/2511.21631v2?page=4)

#### 2.3 基于文本的视频时间戳对齐

原 Qwen2.5-VL 使用 T-RoPE 将帧的绝对时间直接绑定到位置 ID 上，存在两大缺陷：
1. 长视频会产生极大且稀疏的时间位置 ID，降低理解能力
2. 训练时需要大量均匀帧率分布的数据，成本很高

新方案用**文本字符串格式的时间戳 token**（如 `<3.0 seconds>`）来标记每段视频帧组，同时支持秒和时:分:秒两种格式，实现了更精确的时间感知。 [Video Timestamp](https://alphaxiv.org/abs/2511.21631v2?page=4)

---

### 三、预训练流程（4 阶段）

| 阶段 | 目标 | Token 预算 | 序列长度 |
|------|------|-----------|---------|
| S0 | 视觉-语言对齐（仅训练 Merger） | 67B | 8,192 |
| S1 | 全参数多模态预训练 | ~1T | 8,192 |
| S2 | 长上下文预训练 | ~1T | 32,768 |
| S3 | 超长上下文适配 | 100B | 262,144 |

- **S0**：冻结视觉编码器和 LLM，仅训练 MLP Merger，建立跨模态连接的初步基础
- **S1**：解冻所有参数，以约 1T token 的大规模多样化数据进行联合端到端训练
- **S2**：序列长度扩展至 32K，增加视频和 Agent 数据比例，提升长上下文能力
- **S3**：序列长度跃升至 262K，使用专门为超长文档和长视频任务精心构建的 100B token 数据集 [Stage S3](https://alphaxiv.org/abs/2511.21631v2?page=5)

---

### 四、后训练流程（3 阶段）

#### 4.1 监督微调（SFT）

- 分两阶段进行：先以 32K 上下文训练，再扩展至 256K 上下文
- 数据按需求**分叉为两条路径**：
  - 标准格式 → **非思考型（Non-thinking）模型**
  - CoT 格式 → **思考型（Thinking）模型**（显式建模推理过程）
- SFT 数据集约 **120 万样本**，其中 1/3 为纯文本，2/3 为图文和视频-文本对 [SFT Data](https://alphaxiv.org/abs/2511.21631v2?page=10)

#### 4.2 强对弱蒸馏（Strong-to-Weak Distillation）

- 使用**强力教师模型**向学生模型进行知识蒸馏
- 关键之处：蒸馏**仅使用纯文本数据**来微调 LLM 骨干，却在文本和多模态任务上均能带来推理能力的显著提升 [Distillation](https://alphaxiv.org/abs/2511.21631v2?page=9)

#### 4.3 强化学习（RL）

分为两个维度：
- **推理 RL**：专注于提升数学、逻辑等复杂推理任务的能力
- **通用 RL**：覆盖 OCR、视觉定位（Grounding）、指令遵循等多个领域的细粒度能力 [Reinforcement Learning](https://alphaxiv.org/abs/2511.21631v2?page=9)

---

### 五、训练优化：平方根重权化

在优化层面，模型从**按样本平均损失**改为**按 token 的平方根归一化损失**。这一方法更好地平衡了纯文本数据和多模态数据的训练贡献，在提升多模态性能的同时不会损害语言能力。 [Loss Reweighting](https://alphaxiv.org/abs/2511.21631v2?page=1)




## kimi-vl

paper: https://www.alphaxiv.org/abs/2504.07491

![](Pasted%20image%2020260509161544.png)



## Kimi-VL 模型架构详解

Kimi-VL 的架构由三个核心部分组成：**原生分辨率视觉编码器（MoonViT）**、**MLP 投影层**，以及**混合专家（MoE）语言模型**。

---

### 1. MoonViT：原生分辨率视觉编码器

MoonViT 是 Kimi-VL 最具创新性的组件之一，其核心目标是**原生处理任意分辨率的图像**，无需传统的图像裁剪拼接操作（如 LLaVA-OneVision 所采用的方式）。

**关键设计：**
- **图像打包（Packing）**：借鉴 NaViT 的方法，将图像分割为 patch，展平后顺序拼接成一维序列，从而支持同一批次中不同分辨率图像的混合处理。
- **位置编码**：采用双重位置编码策略——
  - 从 SigLIP-SO-400M 继承的**可学习固定尺寸绝对位置嵌入**（经插值保留原有能力）
  - 额外引入沿高度和宽度维度的 **2D RoPE**，增强高分辨率图像的精细位置感知
- **注意力机制**：与语言模型共享相同的计算算子，支持 FlashAttention 的变长序列注意力机制，保证不同分辨率图像下训练吞吐量不受损
- **参数规模**：约 400M 参数，初始化自 SigLIP-SO-400M

> "We incorporate the packing method from NaViT, where images are divided into patches, flattened, and sequentially concatenated into 1D sequences." [MoonViT Design](https://alphaxiv.org/abs/2504.07491?page=3)

在 Kimi-VL-A3B-Thinking-2506 中，MoonViT 进一步被继续训练，最大支持单张图像 **320 万像素**输入（是原始限制的 4 倍），大幅提升高分辨率感知能力。

---

### 2. MLP 投影层

MLP 投影层负责**连接视觉编码器与语言模型**，结构简洁高效：

1. **Pixel Shuffle 操作**：对 MoonViT 输出的图像特征进行空间压缩——在空间维度做 $2\times2$ 下采样，同时将通道维度相应扩展
2. **两层 MLP**：将 pixel shuffle 后的特征投影到语言模型的嵌入维度

这一设计在降低视觉 token 数量（减少计算开销）的同时保留了足够的视觉信息。

---

### 3. MoE 语言模型（Moonlight）

语言解码器采用 Kimi 自研的 **Moonlight MoE 模型**，架构类似于 DeepSeek-V3。

| 属性 | 数值 |
|------|------|
| 激活参数量 | **2.8B** |
| 总参数量 | **16B** |
| 上下文窗口 | **128K tokens** |
| MoE 结构 | 共享专家 + 非共享专家 + 路由器 |

**MoE 结构特点：**
- 每个 MoE FFN 层由**共享专家**（Shared Experts）和**非共享专家**（Non-shared Experts）组成，通过路由器动态选择激活哪些专家
- 仅激活 2.8B 参数即可处理推理，实现了参数效率与模型容量之间的平衡

> "The language model of Kimi-VL utilizes our Moonlight model, an MoE language model with 2.8B activated parameters, 16B total parameters, and an architecture similar to DeepSeek-V3." [MoE LLM](https://alphaxiv.org/abs/2504.07491?page=4)

---

### 4. 优化器：增强版 Muon

整个训练过程（包括视觉编码器、投影层、语言模型）统一使用**增强版 Muon 优化器**：
- 在原始 Muon 基础上加入了**权重衰减**，并精细调整了每参数的更新尺度
- 采用 **ZeRO-1** 策略进行分布式实现，优化内存效率并降低通信开销

---

### 架构总结

```
输入图像（任意分辨率）
      ↓
   MoonViT（400M）
   原生分辨率编码 + 2D RoPE
      ↓
   MLP 投影层
   Pixel Shuffle + 2层MLP
      ↓
   Moonlight MoE LLM（2.8B激活 / 16B总量）
   128K上下文 + 共享/非共享专家
      ↓
   文本输出
```

这一架构设计的核心优势在于：以极低的激活参数量（仅 ~3B），同时覆盖高分辨率视觉感知、长上下文理解和复杂推理等能力。[Architecture Overview](https://alphaxiv.org/abs/2504.07491?page=3)


> 训练策略。
> ## Kimi-VL 训练流程详解

整体训练分为**预训练**和**后训练**两大阶段，合计消耗约 4.4T tokens（不含初始纯文本预训练）。

---

## 一、预训练阶段

### Stage 1：ViT 独立训练（2T + 0.1T tokens）

MoonViT 先独立训练，数据包括图文对（alt text、合成描述、grounding bbox、OCR文本），采用双重训练目标：

$$L = L_{\text{siglip}} + \lambda L_{\text{caption}}, \quad \lambda = 2$$

- $L_{\text{siglip}}$：对比损失（SigLIP变体）
- $L_{\text{caption}}$：条件于图像特征的下一token预测损失

训练完 2T tokens 后，额外用 0.1T tokens 将 MoonViT **对齐到 MoE 语言模型**（仅更新 MoonViT 和 MLP 投影层），显著降低初始困惑度，为后续联合训练铺路。[ViT Training](https://alphaxiv.org/abs/2504.07491?page=5)

---

### Stage 2：联合预训练（1.4T tokens）

语言模型从已处理 5.2T tokens 的 Moonlight 检查点加载，与 MoonViT 一起联合训练：
- 初始阶段仅使用纯文本数据，随后**逐渐提升多模态数据比例**
- 多模态数据涵盖：图文描述、交错数据、OCR、知识、视频、Agent 数据六大类
- 通过渐进式策略，保留语言能力的同时整合视觉理解能力

---

### Stage 3：联合冷却训练（0.6T tokens）

使用**高质量**语言和多模态数据进行精炼：

- **语言侧**：高质量预训练语料子集 + 合成数学/代码 QA 对（通过拒绝采样过滤）
- **多模态侧**：学术视觉数据转 QA 对 + 高质量子集回放
- QA 对占比刻意控制在**较低比例**，避免过拟合特定问答模式

> "Unlike post-training stages, these language and multimodal QA pairs in the cooldown stage are only included for activating specific abilities and henceforth facilitating learning high-quality data, thus, we keep their ratio at a low portion to avoid overfitting these QA patterns." [Cooldown Strategy](https://alphaxiv.org/abs/2504.07491?page=5)

---

### Stage 4：联合长上下文激活（0.3T tokens）

将上下文长度从 **8K 扩展至 128K**，RoPE 基频从 50,000 重置为 800,000，分两个子阶段各扩展 4 倍：

- 长数据占比提升至 **25%**，其余 75% 回放短数据以防止遗忘
- 长数据类型：长文本 + 长交错数据 + 长视频 + 长文档
- 最终在文本和视频 Needle-in-a-Haystack 测试中，128K 范围内均达到高召回率

---

## 二、后训练阶段

### Stage 5：联合监督微调 SFT

使用指令数据对基础模型进行微调，采用 ChatML 格式：

| 子阶段 | 序列长度 | 学习率 |
|--------|----------|--------|
| 第一轮 | 32K | $2\times10^{-5}$ → $2\times10^{-6}$ |
| 第二轮 | 128K | re-warmup 至 $1\times10^{-5}$ → $1\times10^{-6}$ |

- 监督信号**仅施加于回答部分和特殊 token**，系统/用户提示被 mask
- 通过 format-aware packing 将多个样本打包进单个序列以提升训练效率

---

### Stage 6：长链思维 SFT（Long-CoT SFT）

为激活深度推理能力，构建高质量长 CoT 热身数据集：
- 通过提示工程（类似拒绝采样）从 **Kimi k1.5** 采样多条推理路径，过滤错误路径
- 数据刻意涵盖四类认知过程：
  - **Planning**（系统规划步骤）
  - **Evaluation**（评估中间步骤）
  - **Reflection**（反思与修正）
  - **Exploration**（探索替代方案）

---

### Stage 7：强化学习（RL）

采用**在线策略镜像下降**变体作为 RL 算法，优化目标为：

$$\max_\theta \mathbb{E}_{(x,y^*)\sim D}\left[\mathbb{E}_{(y,z)\sim\pi_\theta}[r(x,y,y^*)] - \tau \text{KL}(\pi_\theta(x) \| \pi_{\theta_i}(x))\right]$$

- 奖励函数 $r \in \{0, 1\}$ 基于答案正确性（规则判断）
- $\tau$ 控制 KL 正则化强度，防止策略更新过激

**关键训练技巧：**
- **长度惩罚**：对过长回答施加奖励惩罚，缓解"过度思考"问题
- **课程采样（Curriculum Sampling）**：根据难度标签聚焦高价值样本
- **优先采样（Prioritized Sampling）**：根据每个实例的历史成功率动态调整训练优先级

> "We implement a length-based reward to penalize excessively long responses, mitigating the overthinking problem where the model generates redundant reasoning chains." [RL Length Penalty](https://alphaxiv.org/abs/2504.07491?page=7)

---

## 训练流程总览

```
纯文本预训练（5.2T）
        ↓
ViT 独立训练（2T + 0.1T对齐）
        ↓
联合预训练（1.4T）
        ↓
联合冷却训练（0.6T）
        ↓
联合长上下文激活（0.3T）
        ↓
联合 SFT（32K + 128K两阶段）→ Kimi-VL-A3B-Instruct
        ↓
长 CoT SFT + 强化学习 → Kimi-VL-A3B-Thinking / Thinking-2506
```

整个训练设计的核心理念是**始终保持联合训练**（语言+多模态同步训练），避免引入多模态能力的同时损失纯文本能力。[Training Philosophy](https://alphaxiv.org/abs/2504.07491?page=4)