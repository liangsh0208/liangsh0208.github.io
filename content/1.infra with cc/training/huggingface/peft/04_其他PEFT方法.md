---
created: 2026-05-06
---

# 04_其他PEFT方法

## 文档元信息

| 属性 | 内容 |
|------|------|
| 源码定位 | `/Users/danchen/Documents/1.RL_fw/huggingface/peft/src/peft/tuners/` |
| 覆盖方法 | IA3, AdaLoRA, P-Tuning, Prompt Tuning 等 |
| 注册机制 | `register_peft_method()` 统一管理 |

---

## 1. IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

### 1.1 核心思想

IA3 通过学习**缩放向量**来抑制/放大内部激活，而非添加新参数：

$$
h' = h \cdot l_k \quad \text{(keys)} \\
o' = o \cdot l_v \quad \text{(values)} \\
FF' = FF \cdot l_{ff} \quad \text{(feed-forward)}
$$

### 1.2 IA3Config 配置

```python
from peft import IA3Config, get_peft_model

ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],  # 标记为 FFN
    modules_to_save=["classifier"],
)

model = get_peft_model(model, ia3_config)
```

### 1.3 与 LoRA 对比

| 特性 | LoRA | IA3 |
|------|------|-----|
| **参数形式** | 低秩矩阵 A, B | 缩放向量 l |
| **参数数量** | 2 * r * (in + out) | in + out |
| **数学运算** | 矩阵乘法 | 逐元素乘法 |
| **推荐层** | q_proj, v_proj | k_proj, v_proj, down_proj |

### 1.4 模型层映射

```python
# IA3 推荐的目标模块映射
TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING = {
    "llama": ["k_proj", "v_proj", "down_proj"],
    "qwen2": ["k_proj", "v_proj", "down_proj"],
    "t5": ["k", "v", "wi_1"],
    # ...
}
```

---

## 2. AdaLoRA (Adaptive Low Rank Adaptation)

### 2.1 核心思想

**自适应预算分配**：根据重要性分数动态调整各层的秩：

- 对重要层分配更多参数（更高秩）
- 对不重要层分配更少参数（更低秩）
- 总参数预算固定

### 2.2 AdaLoRAConfig 配置

```python
from peft import AdaLoraConfig, get_peft_model

adalora_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                    # 初始秩
    target_modules=["q_proj", "v_proj"],
    # 自适应预算参数
    target_r=32,             # 目标最终秩
    init_r=64,               # 初始化秩
    tinit=2000,              # 预热步数
    tfinal=10000,            # 预算分配结束步数
    deltaT=200,              # 调整间隔
    # 正则化参数
    beta1=0.85,
    beta2=0.85,
    orth_reg_weight=0.5,     # 正交正则化权重
)

model = get_peft_model(model, adalora_config)
```

### 2.3 预算分配机制

```python
# 重要性分数计算
importance = |S_i,j| * (||U_i,j||^2 + ||V_i,j||^2)

# 其中 S 是奇异值，U/V 是左右奇异向量
```

---

## 3. Prompt Tuning

### 3.1 核心思想

在输入序列前添加**可训练的虚拟 token**（软提示）， frozen 整个预训练模型：

```
输入: [<soft_prompt_tokens>, <input_tokens>]
```

### 3.2 PromptTuningConfig 配置

```python
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,              # 软提示长度
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="分类任务：",  # 初始化文本
    tokenizer_name_or_path="bert-base-uncased",
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
)

model = get_peft_model(model, prompt_config)
```

---

## 4. P-Tuning v2

### 4.1 核心思想

在**每一层**的输入中插入可训练的前缀，而非仅在输入层：

```python
# 第 i 层的前缀
prefix_tokens_i = [<virtual_tokens>]  # 可训练
```

### 4.2 P-tuning 配置

```python
from peft import PrefixTuningConfig, get_peft_model

p_tuning_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=20,              # 每层前缀长度
    prefix_projection=True,             # 启用 MLP 投影
    encoder_hidden_size=512,            # MLP 隐藏层大小
    num_layers=24,                      # 模型层数
    num_attention_heads=16,
)

model = get_peft_model(model, p_tuning_config)
```

### 4.3 Prompt Tuning vs P-Tuning

| 特性 | Prompt Tuning | P-Tuning v2 |
|------|---------------|-------------|
| **插入位置** | 仅输入层 | 每一层 |
| **参数位置** | 嵌入层 | 每层注意力前 |
| **表达能力** | 较弱 | 较强 |
| **参数量** | 较少 | 较多 |

---

## 5. 其他支持的方法

### 5.1 BOFT (Born-Again Feature Tuning)

```python
from peft import BOFTConfig

boft_config = BOFTConfig(
    boft_block_size=16,
    boft_n_butterfly_factor=4,
    target_modules=["q_proj", "v_proj"],
)
```

### 5.2 LoHa (Low-Rank Hadamard Product)

```python
from peft import LoHaConfig

loha_config = LoHaConfig(
    r=8,
    alpha=16,
    rank_dropout=0.0,
    module_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
)
```

### 5.3 VeRA (Vector-based Random Matrix Adaptation)

```python
from peft import VeraConfig

vera_config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    projection_prng_seed=0,
)
```

---

## 6. 方法选择指南

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| **通用微调** | LoRA | 简单有效，社区支持好 |
| **极显存受限** | QLoRA | 4-bit 量化 + LoRA |
| **更高精度** | DoRA | 幅度-方向解耦 |
| **参数自适应** | AdaLoRA | 动态分配参数预算 |
| **不修改模型** | Prompt Tuning | 纯软提示 |
| **更强表达** | P-Tuning v2 | 多层前缀 |
| **更少参数** | IA3 | 缩放向量而非低秩矩阵 |

---

## 7. 相关论文

| 方法 | 论文 |
|------|------|
| AdaLoRA | [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512) |
| (IA)³ | [Few-Shot Parameter-Efficient Fine-Tuning TO Improve Generalization](https://arxiv.org/abs/2205.05638) |
| Prompt Tuning | [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) |
| P-Tuning | [GPT Understands, Too](https://arxiv.org/abs/2103.10385) |

---

## 相关文档

- [00_整体架构与设计理念.md](1.infra%20with%20cc/training/huggingface/peft/00_整体架构与设计理念.md) - 架构概览
- [08_实战配置指南.md](08_实战配置指南.md) - 训练脚本

---

*文档生成日期: 2026-04-20*
