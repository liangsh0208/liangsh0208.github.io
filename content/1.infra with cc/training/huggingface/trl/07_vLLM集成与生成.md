# vLLM 集成与生成

> **【源码定位】**
> - vLLM 生成: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/generation/vllm_generation.py`
> - vLLM 客户端: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/generation/vllm_client.py`
> - vLLM 服务脚本: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/scripts/vllm_serve.py`

---

## 1. vLLM 集成概述

vLLM 是一个高性能的 LLM 推理引擎，TRL 提供原生支持，可以显著加速 GRPO 等需要频繁生成的训练过程。

### 集成优势

- **PagedAttention**: 更高效的 KV Cache 管理
- **Continuous Batching**: 动态批处理
- **与训练 Colocate**: 共用 GPU，自动管理显存
- **分布式支持**: Tensor Parallelism

---

## 2. 两种使用模式

### Colocate 模式 (推荐)

vLLM 与训练进程共用 GPU：

```python
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,  # vLLM 占用 30%
    vllm_tensor_parallel_size=1,
)
```

**特点**：
- 自动管理 GPU 显存分配
- 权重自动同步
- 无需额外服务

### Server 模式

连接到独立运行的 vLLM 服务：

```bash
# 1. 启动 vLLM 服务
trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

```python
# 2. 训练配置
training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="server",
    vllm_server_url="http://localhost:8000",
)
```

**特点**：
- 生成和训练完全分离
- 可使用多 GPU 专门用于生成
- 需要手动启动服务

---

## 3. VLLMGeneration 类

**源码位置**: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/generation/vllm_generation.py`

### 核心功能

```python
class VLLMGeneration:
    """
    vLLM 生成后端封装
    处理 vLLM 初始化、生成、权重同步
    """
    
    def __init__(self, mode: str, model_name: str, ...):
        """
        Args:
            mode: "colocate" 或 "server"
            model_name: 模型名称或路径
        """
    
    def generate(self, prompts: list[str], **generation_kwargs) -> list[str]:
        """批量生成文本"""
    
    def sync_weights(self, model_state_dict: dict):
        """同步模型权重 (仅 colocate 模式)"""
        # PEFT 模型权重自动合并
        # DeepSpeed ZeRO-3/FSDP 参数收集
```

### 权重同步流程

```
训练模型更新
    ↓
收集分布式参数 (FSDP/DeepSpeed)
    ↓
PEFT 权重合并 (如适用)
    ↓
同步到 vLLM
    ↓
生成新 completions
```

---

## 4. Generator 通用接口

### 获取 Generator

```python
from trl import GRPOTrainer, GRPOConfig

# 自动生成器选择
config = GRPOConfig(
    use_vllm=True,   # 使用 vLLM
    # use_vllm=False,  # 使用 Transformers 生成
)

# GRPOTrainer 内部自动处理
trainer = GRPOTrainer(...)
```

### 生成参数配置

```python
from trl import GRPOConfig

config = GRPOConfig(
    # 采样参数
    temperature=0.9,       # 温度
    top_p=0.95,           # Nucleus 采样
    top_k=50,             # Top-k 采样
    repetition_penalty=1.0, # 重复惩罚
    
    # 生成长度
    max_completion_length=2048,
    
    # vLLM 特定
    vllm_gpu_memory_utilization=0.3,
    vllm_tensor_parallel_size=1,
    vllm_max_model_len=None,  # 最大模型长度限制
)
```

---

## 5. vLLM 服务脚本

### 命令行启动

```bash
# 基础启动
trl vllm-serve --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct

# 高级配置
trl vllm-serve \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --dtype bfloat16
```

### 脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name_or_path` | 模型路径 | 必填 |
| `--port` | 服务端口 | 8000 |
| `--host` | 绑定地址 | 127.0.0.1 |
| `--tensor_parallel_size` | TP 并行度 | 1 |
| `--gpu_memory_utilization` | GPU 显存占用 | 0.9 |
| `--max_model_len` | 最大序列长度 | 模型配置 |
| `--dtype` | 数据类型 | auto |

---

## 6. PEFT 模型支持

### LoRA 模型生成

vLLM 自动生成器自动处理 PEFT 权重：

```python
from peft import get_peft_model, LoraConfig
from trl import GRPOTrainer, GRPOConfig

# 配置 LoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)

# GRPOTrainer 自动处理权重合并和同步
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    peft_config=peft_config,
    args=GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
    ),
    ...
)
```

### 权重同步流程

```python
# GRPOTrainer 内部自动执行:

# 1. 获取 PEFT 模型状态
peft_state_dict = model.state_dict()

# 2. 合并 LoRA 权重到基础模型
merged_weights = merge_lora_weights(base_model, peft_state_dict)

# 3. 处理分布式情况
if is_deepspeed_zero3 or is_fsdp:
    merged_weights = gather_distributed_parameters(merged_weights)

# 4. 同步到 vLLM
vllm_generator.sync_weights(merged_weights)
```

---

## 7. 显存优化策略

### Colocate 模式显存分配

```python
from trl import GRPOConfig

config = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,  # vLLM 使用 30%
    # 训练使用剩余 70%
)
```

### 显存不足时的调整

```python
# 1. 减少 vLLM 占用
GRPOConfig(vllm_gpu_memory_utilization=0.2)

# 2. 减少生成长度
GRPOConfig(max_completion_length=512)

# 3. 减少每组生成数量
GRPOConfig(num_generations=4)

# 4. 使用 Server 模式分离 GPU
GRPOConfig(vllm_mode="server")  # 在另一台机器/GPU 上运行 vLLM
```

---

## 8. 实战配置示例

### 单卡 Colocate 配置

```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="./output/grpo-vllm",
    
    # 生成参数
    num_generations=8,
    max_completion_length=1024,
    temperature=0.9,
    
    # 训练参数
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    
    # vLLM 配置
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.25,  # 单卡谨慎分配
    vllm_tensor_parallel_size=1,
)
```

### 多卡分布式配置

```python
from trl import GRPOConfig

config = GRPOConfig(
    # vLLM 使用 TP=2
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,
    vllm_tensor_parallel_size=2,
    
    # 训练使用 DDP/FSDP
    # torchrun --nproc_per_node=4 ...
)
```

### Server 模式配置

```bash
# 节点 1: vLLM 服务 (GPU 0,1)
trl vllm-serve \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --tensor_parallel_size 2 \
    --port 8000
```

```python
# 节点 2: 训练 (GPU 2,3)
from trl import GRPOConfig

config = GRPOConfig(
    use_vllm=True,
    vllm_mode="server",
    vllm_server_url="http://node1:8000",
)
```

---

## 9. 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `OOM during generation` | vLLM 显存设置过高 | 降低 `vllm_gpu_memory_utilization` |
| `vLLM connection error` | 服务未启动 | 检查 vLLM 服务状态和端口 |
| `Weight sync timeout` | 分布式参数收集慢 | 增加超时时间或减少模型大小 |
| `CUDA OOM after sync` | 内存碎片 | 设置 `torch_empty_cache_steps` |
| `Incorrect generations` | 权重不同步 | 检查 PEFT 合并逻辑 |
