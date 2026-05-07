# Models 模块

**路径**：`verl/models/`

Models 模块负责模型的定义、权重加载、格式转换，以及针对 verl 训练场景的 monkey patch。

---

## 1. 目录结构

```
models/
├── registry.py                  # 模型注册表
├── weight_loader_registry.py    # 权重加载器注册表
├── transformers/                # HuggingFace 兼容模型
│   ├── llama.py                 # LLaMA / LLaMA2 / LLaMA3
│   ├── qwen2.py                 # Qwen2
│   ├── qwen2_vl.py              # Qwen2-VL（视觉语言模型）
│   ├── qwen3_5.py               # Qwen3.5
│   ├── qwen3_vl.py              # Qwen3-VL
│   ├── glm4v.py                 # GLM-4V
│   ├── kimi_vl.py               # Kimi-VL
│   ├── apertus.py               # Apertus
│   ├── dense_common.py          # 通用 Dense LLM 前向逻辑
│   ├── monkey_patch.py          # 通用 monkey patch（FlashAttn、Remove Padding 等）
│   ├── npu_patch.py             # NPU 专用 patch（华为昇腾）
│   └── tiled_mlp.py             # Tiled MLP（显存优化）
├── mcore/                       # Megatron-Core 模型
│   ├── bridge.py                # HuggingFace → Megatron-Core 转换层
│   ├── mbridge.py               # 双向桥接（HF ↔ MCCore）
│   ├── loader.py                # 权重加载
│   ├── saver.py                 # 权重保存
│   ├── weight_converter.py      # 权重格式转换
│   ├── config_converter.py      # 配置格式转换
│   ├── model_forward.py         # 标准前向传播
│   ├── model_forward_1f1b_overlap.py  # 1F1B 流水线调度
│   ├── model_forward_fused.py   # Fused 前向（减少通信）
│   ├── model_initializer.py     # 模型初始化
│   ├── mtp_patch.py             # Multi-Token Prediction patch
│   ├── patch.py                 # 其他 Megatron patch
│   ├── registry.py              # Megatron 模型注册
│   ├── util.py                  # 工具函数
│   └── qwen2_5_vl/              # Qwen2.5-VL Megatron 专用实现
└── diffusers_model/             # 扩散模型支持
```

---

## 2. HuggingFace 兼容模型（`transformers/`）

### 2.1 Monkey Patch

**文件**：`verl/models/transformers/monkey_patch.py`

verl 对标准 HuggingFace 模型应用多种 monkey patch 以提升训练效率：

```python
def apply_monkey_patch(model, config):
    """根据配置应用对应的 patch"""

    # 1. FlashAttention2 替换标准 Attention（大幅降低内存，加速计算）
    if config.get("use_flash_attention", True):
        replace_attention_with_flash_attn(model)

    # 2. Remove Padding：去掉 padding token 的计算
    if config.get("use_remove_padding", False):
        apply_remove_padding_patch(model)

    # 3. 位置编码扩展（RoPE Scaling）
    if config.get("rope_scaling"):
        apply_rope_scaling(model, config.rope_scaling)

    # 4. Tiled MLP：大 hidden size 时分块计算 MLP
    if config.get("use_tiled_mlp", False):
        apply_tiled_mlp(model)
```

### 2.2 Dense Common（`dense_common.py`）

封装了通用 Dense LLM 的前向传播逻辑，适用于 LLaMA、Qwen 等标准 decoder-only 架构：

```python
def forward_with_log_probs(model, input_ids, attention_mask, labels=None):
    """
    统一的前向传播接口，返回：
    - logits：(bs, seq_len, vocab_size)
    - log_probs：(bs, seq_len)，每个位置的 log_prob
    - entropy：(bs, seq_len)，每个位置的熵
    """
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits

    # 计算 log_prob（针对实际 token）
    log_probs = logprobs_from_logits(logits[:, :-1], input_ids[:, 1:])

    # 计算熵（度量策略的确定性）
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

    return logits, log_probs, entropy
```

### 2.3 视觉语言模型（VLM）

**Qwen2-VL / Qwen3-VL**（`qwen2_vl.py`、`qwen3_vl.py`）：
- 输入包含 `pixel_values`、`image_grid_thw`（图像 patch 位置）
- 通过 ViT 提取图像特征，插入到 token 序列中
- 特殊处理图像 patch 的 position_ids

```python
def prepare_vl_inputs(batch):
    """处理多模态输入"""
    text_inputs = tokenize(batch["text"])
    image_inputs = process_images(batch["images"])  # ViT 预处理

    # 合并文本和图像 token
    input_ids, attention_mask, position_ids = merge_text_image_tokens(
        text_inputs, image_inputs
    )
    return input_ids, attention_mask, position_ids, image_inputs["pixel_values"]
```

---

## 3. Megatron-Core 集成（`mcore/`）

### 3.1 权重格式转换

HuggingFace 和 Megatron-Core 使用不同的权重命名和分布方式：

| 维度 | HuggingFace | Megatron-Core |
|------|-------------|---------------|
| 命名 | `model.layers.0.self_attn.q_proj.weight` | `decoder.layers.0.self_attention.linear_qkv.weight` |
| 结构 | Q/K/V 分开存储 | Q/K/V 合并存储 |
| 分片 | 不分片 | 按 TP 维度分片 |
| 位置编码 | RoPE 在 Attention 内部 | RoPE 独立模块 |

**权重转换流程**（`weight_converter.py`）：

```python
class HF2MCoreConverter:
    def convert(self, hf_state_dict: dict) -> dict:
        mcore_state_dict = {}
        for hf_name, tensor in hf_state_dict.items():
            mcore_name = self.name_mapping[hf_name]  # 名称映射
            tensor = self.reshape(tensor, hf_name)   # 形状变换（QKV 合并）
            tensor = self.shard(tensor, mcore_name)  # 按 TP 分片
            mcore_state_dict[mcore_name] = tensor
        return mcore_state_dict
```

### 3.2 Bridge 层

```python
class MCoreBridge:
    """连接 HuggingFace 接口和 Megatron-Core 计算的桥接层"""

    def __init__(self, hf_model_config, mcore_config):
        # 创建 Megatron-Core 模型
        self.mcore_model = GPTModel(mcore_config)
        # 加载 HuggingFace 权重
        self.load_hf_weights(hf_model_config.path)

    def forward(self, tokens, position_ids, attention_mask):
        """将 HuggingFace 格式输入转换为 Megatron-Core 格式"""
        return self.mcore_model(tokens, position_ids, attention_mask)

    def state_dict_for_hf(self):
        """将 Megatron-Core 权重转换回 HuggingFace 格式（用于保存）"""
        return self.converter.mcore_to_hf(self.mcore_model.state_dict())
```

### 3.3 1F1B 流水线调度

**文件**：`model_forward_1f1b_overlap.py`

1F1B（One Forward One Backward）调度通过流水线并行和通信 overlap 减少等待时间：

```
GPU 0 (PP stage 0):  F0  B0  F1  B1  F2  B2 ...
GPU 1 (PP stage 1):      F0  B0  F1  B1  F2 ...
                     (F=forward, B=backward)
```

与朴素 PP 相比，减少了 GPU idle time（bubble）。

### 3.4 Qwen2.5-VL Megatron 实现

**文件**：`verl/models/mcore/qwen2_5_vl/`

针对 Qwen2.5-VL 的 Megatron-Core 专用实现，包含：
- ViT 的张量并行分片
- 图像 token 和文本 token 的合并处理
- 多模态位置编码

---

## 4. 模型注册机制

**文件**：`verl/models/registry.py`

```python
MODEL_REGISTRY: dict[str, type] = {}

def register_model(model_type: str):
    """注册 HuggingFace 模型类"""
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator

@register_model("llama")
class LlamaForCausalLMWithValueHead(LlamaForCausalLM):
    """带 Value Head 的 LLaMA（Critic 用）"""
    ...
```

**权重加载器注册**（`weight_loader_registry.py`）：

```python
# 不同后端使用不同的权重加载方式
WEIGHT_LOADER_REGISTRY = {
    "hf": HFWeightLoader,        # HuggingFace 格式
    "dtensor": DTensorWeightLoader,  # PyTorch DTensor 分片格式
    "megatron": MegatronWeightLoader, # Megatron 格式（TP/PP 分片）
}
```

---

## 5. 扩散模型（`diffusers_model/`）

支持基于扩散模型的 RL 训练（如 DDPO）：

```python
class DiffusionModel:
    """扩散模型 RL 训练的封装"""

    def generate(self, prompt: str, num_steps: int = 50):
        """扩散采样过程（等价于 LLM 的 rollout）"""
        x = torch.randn(1, 4, 64, 64)  # 初始噪声
        for t in reversed(range(num_steps)):
            # 预测噪声
            noise_pred = self.unet(x, t, prompt_embeds)
            # 去噪一步
            x = self.scheduler.step(noise_pred, t, x).prev_sample
        return x  # 生成的图像

    def compute_log_prob(self, trajectory):
        """计算扩散轨迹的 log_prob（用于 PPO 更新）"""
        ...
```

奖励函数：`utils/reward_score/jpeg_compressibility.py`（图像压缩率作为奖励）。

---

## 6. 模型加载工具函数

**文件**：`verl/utils/model.py`

```python
def create_model_with_value_head(config):
    """根据配置创建带 Value Head 的 Critic 模型"""
    base_model = AutoModelForCausalLM.from_pretrained(config.path)
    value_head = ValueHead(base_model.config.hidden_size)
    return ModelWithValueHead(base_model, value_head)

def normalize_token_ids(token_ids, tokenizer):
    """标准化 token ids（处理不同 tokenizer 的边界 token）"""
    # 去掉多余的 pad_token，确保 eos_token 在正确位置
    ...
```
