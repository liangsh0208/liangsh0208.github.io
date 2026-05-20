---
created: 2026-05-06
---

# Bridge 模式与模型转换

> 本文档深入分析 Megatron-Bridge 的核心 Bridge 模式，详细讲解 Hugging Face 和 Megatron-Core 之间的模型转换机制。

---

## 目录

1. [Bridge 模式概述](#1-bridge-模式概述)
2. [MegatronModelBridge 基类](#2-megatronmodelbridge-基类)
3. [参数映射系统](#3-参数映射系统)
4. [LlamaBridge 实例分析](#4-llamabridge-实例分析)
5. [权重转换流程](#5-权重转换流程)
6. [并行感知转换](#6-并行感知转换)
7. [扩展新模型](#7-扩展新模型)

---

## 1. Bridge 模式概述

### 1.1 设计理念

Bridge 模式解决了 Hugging Face 和 Megatron-Core 之间的生态系统碎片化问题：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Hugging Face 生态                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ config.json  │  │ model.safetensors │ tokenizer.json │          │
│  │ (模型配置)    │  │ (模型权重)    │  │ (分词器)      │            │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Bridge 转换
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Megatron-Core 生态                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ TransformerConfig + 分布式并行格式                             │  │
│  │ iter_0000001/mp_rank_00/model_optim_rng.pt                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Bridge 模式的核心价值

| 特性 | 描述 |
|------|------|
| **双向转换** | HF ↔ Megatron 无缝转换 |
| **并行感知** | 自动处理 TP/PP/VP/EP 分布式切分 |
| **在线转换** | 无需中间检查点，逐参数流式转换 |
| **架构自动检测** | 根据模型配置自动选择正确的 Bridge |
| **验证机制** | 提供数值校验确保转换正确性 |

### 1.3 架构层次

```
models/
├── conversion/                    # 转换核心
│   ├── auto_bridge.py            # 自动桥接器
│   ├── model_bridge.py           # Bridge 基类
│   ├── param_mapping.py          # 参数映射定义
│   └── mapping_registry.py       # 映射注册表
├── <model_family>/               # 各模型族实现
│   ├── bridge.py                 # 具体 Bridge 实现
│   ├── <model>_provider.py       # 模型提供者
│   └── hf_pretrained/            # HF 模型定义 (可选)
```

---

## 2. MegatronModelBridge 基类

### 2.1 类定义

**文件位置**: `src/megatron/bridge/models/conversion/model_bridge.py`

```python
class MegatronModelBridge(
    abc.ABC, 
    Generic[MappingT, ModelProviderTarget, MegatronModel]
):
    """
    模型桥接的抽象基类。
    
    定义 HF 模型和 Megatron 模型之间转换的标准接口。
    每个 Bridge 负责特定模型架构的转换逻辑。
    """
    
    # 类属性
    ModelProviderClass: Type[ModelProviderMixin]  # 模型提供者类
    HFModelClass: Type[PreTrainedModel]            # HF 模型类
```

### 2.2 核心抽象方法

```python
@abc.abstractmethod
def init_megatron_model(self) -> MegatronModule:
    """初始化 Megatron 模型结构 (不含权重)"""
    pass

@abc.abstractmethod
def convert_hf_to_megatron(
    self, 
    hf_model: PreTrainedModel,
    megatron_model: MegatronModule
) -> None:
    """将 HF 权重转换并加载到 Megatron 模型"""
    pass

@abc.abstractmethod
def convert_megatron_to_hf(
    self,
    megatron_model: MegatronModule,
    hf_config: PretrainedConfig
) -> Dict[str, torch.Tensor]:
    """将 Megatron 权重转换为 HF 格式"""
    pass
```

### 2.3 注册机制

```python
# 装饰器注册
@MegatronModelBridge.register_bridge(
    source=LlamaForCausalLM,  # HF 模型类
    target=GPTModel,          # Megatron 模型类
    model_type="llama"        # 模型类型标识
)
class LlamaBridge(MegatronModelBridge):
    """Llama 模型的桥接实现"""
    pass

# 注册表内部结构
_registry = {
    LlamaForCausalLM: LlamaBridge,
    Qwen2ForCausalLM: Qwen2Bridge,
    GemmaForCausalLM: GemmaBridge,
    ...
}
```

### 2.4 权重转换任务

```python
@dataclass(frozen=True)
class WeightConversionTask(Generic[MappingT]):
    """统一的权重转换任务封装"""
    
    param_name: str           # 本地参数名 (无 module. 前缀)
    global_param_name: str   # 全局参数名 (考虑 PP/EP 偏移)
    mapping: MappingT        # 参数映射规则实例
    pp_rank: Optional[int]   # 流水线并行 rank (保存时需要)
    vp_stage: Optional[int]  # 虚拟流水线阶段 (加载时需要)
    megatron_module: Optional[nn.Module]  # Megatron 模块引用
    param_weight: Optional[torch.Tensor]  # 参数张量
```

---

## 3. 参数映射系统

### 3.1 MegatronParamMapping 基类

```python
class MegatronParamMapping(abc.ABC):
    """参数映射的基类，定义 HF 和 Megatron 参数之间的转换规则"""
    
    @abc.abstractmethod
    def hf_to_megatron(
        self, 
        hf_weights: torch.Tensor,
        megatron_module: nn.Module
    ) -> torch.Tensor:
        """将 HF 权重转换为 Megatron 格式"""
        pass
    
    @abc.abstractmethod
    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor,
        megatron_module: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """将 Megatron 权重转换为 HF 格式"""
        pass
```

### 3.2 AutoMapping - 简单 1:1 映射

```python
class AutoMapping(MegatronParamMapping):
    """自动映射：参数名不同，但形状和值相同"""
    
    def __init__(
        self, 
        megatron_param: str,   # Megatron 参数名 (支持通配符 *)
        hf_param: str          # HF 参数名 (支持通配符 *)
    ):
        self.megatron_param = megatron_param
        self.hf_param = hf_param
    
    def hf_to_megatron(self, hf_weights, megatron_module):
        """直接返回 HF 权重"""
        return hf_weights
    
    def megatron_to_hf(self, megatron_weights, megatron_module):
        """直接返回 Megatron 权重"""
        return {self.hf_param: megatron_weights}
```

**示例用法**:

```python
# Llama 模型的简单映射
AutoMapping(
    megatron_param="embedding.word_embeddings.weight",
    hf_param="model.embed_tokens.weight"
)

AutoMapping(
    megatron_param="decoder.layers.*.self_attention.linear_proj.weight",
    hf_param="model.layers.*.self_attn.o_proj.weight"
)
```

### 3.3 QKVMapping - QKV 合并映射

```python
class QKVMapping(MegatronParamMapping):
    """
    QKV 映射：将 HF 的三个独立 Q/K/V 矩阵合并为 Megatron 的单个 QKV 矩阵
    
    HF 格式: 
        - model.layers.*.self_attn.q_proj.weight [hidden, hidden]
        - model.layers.*.self_attn.k_proj.weight [hidden, kv_hidden]
        - model.layers.*.self_attn.v_proj.weight [hidden, kv_hidden]
    
    Megatron 格式:
        - decoder.layers.*.self_attention.linear_qkv.weight [3*hidden, hidden]
    """
    
    def __init__(
        self, 
        megatron_param: str,
        q: str,  # Q 投影参数名
        k: str,  # K 投影参数名
        v: str   # V 投影参数名
    ):
        self.megatron_param = megatron_param
        self.q = q
        self.k = k
        self.v = v
    
    def hf_to_megatron(self, hf_weights, megatron_module):
        """合并 Q/K/V 为 QKV"""
        # hf_weights 是包含 q, k, v 三个张量的元组
        q, k, v = hf_weights
        return torch.cat([q, k, v], dim=0)
    
    def megatron_to_hf(self, megatron_weights, megatron_module):
        """拆分 QKV 为 Q/K/V"""
        # 根据 GQA 配置确定切分点
        hidden_size = megatron_weights.shape[1]
        q_size = hidden_size  # Q 的大小
        kv_size = ...         # K/V 的大小 (考虑 GQA)
        
        q = megatron_weights[:q_size, :]
        k = megatron_weights[q_size:q_size+kv_size, :]
        v = megatron_weights[q_size+kv_size:, :]
        
        return {
            self.q: q,
            self.k: k,
            self.v: v
        }
```

### 3.4 GatedMLPMapping - 门控 MLP 映射

```python
class GatedMLPMapping(MegatronParamMapping):
    """
    门控 MLP 映射：将 HF 的 gate_proj 和 up_proj 合并为 linear_fc1
    
    HF 格式 (SwiGLU):
        - model.layers.*.mlp.gate_proj.weight [ffn_hidden, hidden]
        - model.layers.*.mlp.up_proj.weight [ffn_hidden, hidden]
    
    Megatron 格式:
        - decoder.layers.*.mlp.linear_fc1.weight [2*ffn_hidden, hidden]
    """
    
    def __init__(
        self,
        megatron_param: str,
        gate: str,  # gate_proj 参数名
        up: str     # up_proj 参数名
    ):
        self.megatron_param = megatron_param
        self.gate = gate
        self.up = up
    
    def hf_to_megatron(self, hf_weights, megatron_module):
        """合并 gate 和 up"""
        gate, up = hf_weights
        return torch.cat([gate, up], dim=0)
    
    def megatron_to_hf(self, megatron_weights, megatron_module):
        """拆分为 gate 和 up"""
        ffn_hidden = megatron_weights.shape[0] // 2
        gate = megatron_weights[:ffn_hidden, :]
        up = megatron_weights[ffn_hidden:, :]
        return {self.gate: gate, self.up: up}
```

### 3.5 MegatronMappingRegistry

```python
class MegatronMappingRegistry:
    """
    映射注册表：管理模型的所有参数映射规则
    """
    
    def __init__(self, *mappings: MegatronParamMapping):
        self.mappings = list(mappings)
    
    def get_mapping(self, param_name: str) -> Optional[MegatronParamMapping]:
        """根据参数名获取映射规则"""
        for mapping in self.mappings:
            if self._match(mapping.megatron_param, param_name):
                return mapping
        return None
    
    def _match(self, pattern: str, name: str) -> bool:
        """通配符匹配"""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)
```

---

## 4. LlamaBridge 实例分析

### 4.1 类定义

**文件位置**: `src/megatron/bridge/models/llama/llama_bridge.py`

```python
@MegatronModelBridge.register_bridge(
    source=LlamaForCausalLM,
    target=GPTModel,
    model_type="llama"
)
class LlamaBridge(MegatronModelBridge):
    """
    Llama 模型桥接器。
    
    通过 AutoBridge 自动选择，不直接使用。
    """
```

### 4.2 Provider Bridge 方法

```python
def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
    """
    将 HF Llama 配置转换为 Megatron GPTModelProvider
    
    Args:
        hf_pretrained: HuggingFace PreTrainedCausalLM 实例
        
    Returns:
        配置好的 GPTModelProvider
    """
    # 调用父类方法完成基础转换
    provider = super().provider_bridge(hf_pretrained)
    
    # Llama 特定的 Megatron 配置
    provider.normalization = "RMSNorm"        # 使用 RMSNorm
    provider.gated_linear_unit = True          # 启用 SwiGLU
    provider.hidden_dropout = 0.0              # 无 dropout
    provider.bias_activation_fusion = True     # 融合 bias-activation
    provider.masked_softmax_fusion = True      # 融合 masked softmax
    provider.persist_layer_norm = True         # 持久化 LayerNorm
    provider.bias_dropout_fusion = True        # 融合 bias-dropout
    provider.apply_rope_fusion = True          # 融合 RoPE
    provider.rotary_percent = 1.0              # 完整 RoPE

    # 处理 RoPE scaling (Llama 3.1/3.2)
    hf_config = hf_pretrained.config
    hf_rope_scaling = getattr(hf_config, "rope_scaling", None)
    if hf_rope_scaling is not None and hf_rope_scaling.get("rope_type") == "llama3":
        provider.rope_scaling = True
        provider.rope_scaling_factor = hf_rope_scaling.get("factor", 8.0)

    return provider
```

### 4.3 Megatron 到 HF 配置转换

```python
@classmethod
def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
    """
    将 Megatron GPTModelProvider 配置转换为 HF LlamaConfig 字典
    
    Args:
        provider: GPTModelProvider 实例
        
    Returns:
        LlamaConfig 参数字典
    """
    # 调用父类方法完成基础转换
    hf_config = super(LlamaBridge, cls).megatron_to_hf_config(provider)

    # 处理 RoPE scaling
    if provider.rope_scaling:
        hf_config["rope_scaling"] = {
            "rope_type": "llama3",
            "factor": provider.rope_scaling_factor,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }

    return hf_config
```

### 4.4 映射注册表

```python
def mapping_registry(self) -> MegatronMappingRegistry:
    """返回 Llama 模型的参数映射注册表"""
    
    # 简单 1:1 映射
    param_mappings = {
        # Embedding
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        
        # Output
        "output_layer.weight": "lm_head.weight",
        
        # Final LayerNorm
        "decoder.final_layernorm.weight": "model.norm.weight",
        
        # Attention LayerNorm (TE 实现)
        "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": 
            "model.layers.*.input_layernorm.weight",
        
        # Attention LayerNorm (本地实现)
        "decoder.layers.*.input_layernorm.weight": 
            "model.layers.*.input_layernorm.weight",
        
        # MLP LayerNorm (TE 实现)
        "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": 
            "model.layers.*.post_attention_layernorm.weight",
        
        # MLP LayerNorm (本地实现)
        "decoder.layers.*.pre_mlp_layernorm.weight": 
            "model.layers.*.post_attention_layernorm.weight",
        
        # Attention Output Projection
        "decoder.layers.*.self_attention.linear_proj.weight": 
            "model.layers.*.self_attn.o_proj.weight",
        
        # MLP Down Projection
        "decoder.layers.*.mlp.linear_fc2.weight": 
            "model.layers.*.mlp.down_proj.weight",
    }
    
    mapping_list = []
    for megatron_param, hf_param in param_mappings.items():
        mapping_list.append(AutoMapping(megatron_param, hf_param))
    
    # 特殊映射
    mapping_list.extend([
        # QKV 合并
        QKVMapping(
            megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
            q="model.layers.*.self_attn.q_proj.weight",
            k="model.layers.*.self_attn.k_proj.weight",
            v="model.layers.*.self_attn.v_proj.weight",
        ),
        
        # Gated MLP
        GatedMLPMapping(
            megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
            gate="model.layers.*.mlp.gate_proj.weight",
            up="model.layers.*.mlp.up_proj.weight",
        ),
    ])

    return MegatronMappingRegistry(*mapping_list)
```

---

## 5. 权重转换流程

### 5.1 HF → Megatron 转换流程

```python
def convert_hf_to_megatron(hf_model, megatron_model, bridge):
    """
    HF 到 Megatron 权重转换流程
    """
    # 1. 获取参数映射注册表
    registry = bridge.mapping_registry()
    
    # 2. 遍历 Megatron 模型参数
    for name, param in megatron_model.named_parameters():
        # 3. 查找对应的映射规则
        mapping = registry.get_mapping(name)
        
        if mapping is None:
            # 无映射规则，跳过或报错
            continue
        
        # 4. 获取 HF 权重
        hf_weights = get_hf_weights(hf_model, mapping.hf_param)
        
        # 5. 应用转换
        megatron_weights = mapping.hf_to_megatron(hf_weights, megatron_model)
        
        # 6. 处理并行切分 (TP/EP)
        if needs_tensor_parallel_split(name):
            megatron_weights = split_for_tp(megatron_weights)
        
        # 7. 加载权重
        param.data.copy_(megatron_weights)
```

### 5.2 Megatron → HF 转换流程

```python
def convert_megatron_to_hf(megatron_model, hf_config, bridge):
    """
    Megatron 到 HF 权重转换流程
    """
    hf_weights = {}
    
    # 1. 获取参数映射注册表
    registry = bridge.mapping_registry()
    
    # 2. 收集 Megatron 权重 (考虑 TP 合并)
    megatron_weights = collect_megatron_weights(megatron_model)
    
    # 3. 遍历映射规则
    for mapping in registry.mappings:
        # 4. 从 Megatron 权重中提取
        megatron_weight = megatron_weights.get(mapping.megatron_param)
        
        if megatron_weight is None:
            continue
        
        # 5. 应用逆转换
        hf_dict = mapping.megatron_to_hf(megatron_weight, megatron_model)
        
        # 6. 合并到输出字典
        hf_weights.update(hf_dict)
    
    return hf_weights
```

---

## 6. 并行感知转换

### 6.1 张量并行 (TP) 处理

```python
def split_for_tensor_parallel(weight, tp_rank, tp_size):
    """
    将权重切分到多个 TP rank
    
    Args:
        weight: 完整权重张量
        tp_rank: 当前 TP rank
        tp_size: TP world size
    """
    if is_column_parallel(weight):
        # 列并行：按输出维度切分
        return torch.chunk(weight, tp_size, dim=0)[tp_rank]
    else:
        # 行并行：按输入维度切分
        return torch.chunk(weight, tp_size, dim=1)[tp_rank]

def gather_from_tensor_parallel(weight, tp_group):
    """
    从多个 TP rank 收集完整权重
    """
    return torch.distributed.all_gather(weight, group=tp_group)
```

### 6.2 流水线并行 (PP) 处理

```python
def _megatron_local_name_to_global(
    models, 
    config: TransformerConfig, 
    param_name: str,
    vp_stage: Optional[int] = None
) -> str:
    """
    将本地参数名转换为全局参数名 (考虑 PP 和 EP)
    """
    # 处理 PP 层编号偏移
    if "layers." in param_name:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        if get_pg_size(pp_group) > 1:
            # 获取全局层编号
            local_layer_number = int(param_name.split("layers.")[1].split(".")[0])
            global_layer_number = layer_module.layer_number - 1
            param_name = param_name.replace(
                f"layers.{local_layer_number}.",
                f"layers.{global_layer_number}."
            )
    
    # 处理 EP 专家编号偏移
    if ".mlp.experts.linear_fc" in param_name:
        ep_group = parallel_state.get_expert_model_parallel_group()
        if get_pg_size(ep_group) > 1:
            num_experts = config.num_moe_experts
            num_experts_per_rank = num_experts // ep_group.size()
            local_expert_number = int(param_name.split(".linear_fc")[-1])
            global_expert_number = num_experts_per_rank * ep_group.rank() + local_expert_number
            param_name = param_name.replace(f".{local_expert_number}", f".{global_expert_number}")
    
    return param_name
```

### 6.3 专家并行 (EP) 处理

```python
def handle_expert_weights(weight, ep_rank, ep_size, num_experts):
    """
    处理 MoE 专家权重的 EP 分布
    """
    num_experts_per_rank = num_experts // ep_size
    start_expert = ep_rank * num_experts_per_rank
    end_expert = start_expert + num_experts_per_rank
    
    # 切分专家
    expert_weights = []
    for i in range(start_expert, end_expert):
        expert_weights.append(weight[i])
    
    return torch.stack(expert_weights)
```

---

## 7. 扩展新模型

### 7.1 创建新 Bridge 的步骤

1. **创建 Bridge 类**:

```python
# src/megatron/bridge/models/my_model/my_model_bridge.py

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping

@MegatronModelBridge.register_bridge(
    source=MyModelforCausalLM,
    target=GPTModel,
    model_type="my_model"
)
class MyModelBridge(MegatronModelBridge):
    """My Model 桥接器"""
    
    def provider_bridge(self, hf_pretrained):
        """配置 Megatron 模型参数"""
        provider = super().provider_bridge(hf_pretrained)
        
        # 设置模型特定配置
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        ...
        
        return provider
    
    def mapping_registry(self):
        """定义参数映射"""
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            ...
        ]
        return MegatronMappingRegistry(*mappings)
```

2. **导出 Bridge**:

```python
# src/megatron/bridge/models/my_model/__init__.py
from .my_model_bridge import MyModelBridge

__all__ = ["MyModelBridge"]
```

3. **注册到 AutoBridge**:

```python
# src/megatron/bridge/models/conversion/model_bridge.py
from megatron.bridge.models.my_model import MyModelBridge
```

### 7.2 测试验证

```python
# tests/unit_tests/models/my_model/test_my_model_bridge.py

def test_weight_conversion():
    """测试权重转换的正确性"""
    # 1. 加载 HF 模型
    hf_model = load_hf_model("my-org/my-model")
    
    # 2. 转换到 Megatron
    bridge = AutoBridge.from_hf_pretrained("my-org/my-model")
    provider = bridge.to_megatron_provider()
    megatron_model = provider.provide_distributed_model()
    
    # 3. 权重转换
    bridge.convert_hf_to_megatron(hf_model, megatron_model)
    
    # 4. 数值验证
    with torch.no_grad():
        hf_output = hf_model(input_ids)
        megatron_output = megatron_model(input_ids)
        
        assert torch.allclose(hf_output, megatron_output, atol=1e-5)
```

---

## 总结

本章详细介绍了 Megatron-Bridge 的核心 Bridge 模式：

| 组件 | 职责 |
|------|------|
| `MegatronModelBridge` | 抽象基类，定义转换接口 |
| `AutoMapping` | 简单 1:1 参数映射 |
| `QKVMapping` | Q/K/V 到 QKV 合并映射 |
| `GatedMLPMapping` | gate/up 到 fc1 合并映射 |
| `MegatronMappingRegistry` | 映射规则注册表 |
| `AutoBridge` | 自动选择正确的 Bridge |

关键设计模式：
- **注册机制**：通过装饰器自动注册 Bridge
- **双向转换**：支持 HF ↔ Megatron 双向转换
- **并行感知**：自动处理 TP/PP/EP 分布式切分
- **参数流式处理**：逐参数转换，内存高效

---

## 相关模块

| 模块 | 关系说明 |
|------|---------|
| [AutoBridge](02_AutoBridge自动桥接.md) | 自动检测模型类型并选择正确的 Bridge |
| [ModelProvider](03_ModelProvider模式.md) | Bridge 的 `to_megatron_provider()` 返回 ModelProvider |
| [检查点系统](08_检查点系统.md) | Bridge 使用检查点系统保存/加载模型 |
| [PEFT](06_PEFT参数高效微调.md) | Bridge 的 `save_hf_adapter()` 导出 LoRA |
| [训练配方](09_训练配方Recipes.md) | 配方使用 Bridge 加载预训练模型 |