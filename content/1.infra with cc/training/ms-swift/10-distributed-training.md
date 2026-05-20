---
created: 2026-05-09
---

# ms-swift 分布式训练与并行策略

ms-swift 支持从单机单卡到大规模集群的多种分布式训练策略，涵盖数据并行、序列并行以及完整的 Megatron-LM 大规模并行体系。

---

## 1. 分布式策略全景

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        分布式训练策略矩阵                                │
├──────────────┬────────────────────────────────────────────────────────┤
│ 数据并行      │ DDP (PyTorch)                                          │
│              │ DeepSpeed ZeRO2 / ZeRO3                                │
│              │ FSDP / FSDP2 (Fully Sharded Data Parallel)             │
├──────────────┼────────────────────────────────────────────────────────┤
│ 序列并行      │ Ulysses (`swift/sequence_parallel/ulysses.py`)        │
│              │ Ring Attention (`swift/sequence_parallel/...`)         │
├──────────────┼────────────────────────────────────────────────────────┤
│ Megatron     │ TP  (Tensor Parallel)                                  │
│ 大规模并行    │ PP  (Pipeline Parallel)                                │
│              │ SP  (Sequence Parallel, Megatron 原生)                  │
│              │ CP  (Context Parallel)                                 │
│              │ ETP (Expert Tensor Parallel)                           │
│              │ EP  (Expert Parallel)                                  │
│              │ VPP (Virtual PP)                                       │
└──────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. 数据并行

### 2.1 DDP 自动包装

**文件**: `swift/cli/main.py` (71-95 lines)

```python
def cli_main():
    torchrun_args = get_torchrun_args()  # 读取 NPROC_PER_NODE, NNODES
    if torchrun_args is not None:
        # 自动包装为 torchrun
        args = [python, '-m', 'torch.distributed.run', 
                '--nproc_per_node', nproc, '--nnodes', nnodes, ...,
                file_path, *argv]
    subprocess.run(args)
```

使用方式（零配置）：
```bash
export NPROC_PER_NODE=8
export NNODES=2
export NODE_RANK=0
swift sft --model xxx --dataset xxx ...
```

### 2.2 DeepSpeed 集成

```bash
swift sft --model xxx --dataset xxx \
    --deepspeed default-zero2 \
    # 或 --deepspeed default-zero3
```

**ZeRO3 参数 gather 补丁** (`swift/trainers/mixin.py`):
```python
def _fix_zero3_gather_all_parameters(self):
    """修复 DeepSpeed ZeRO3 下参数聚合"""
    if is_deepspeed_zero3_enabled():
        from deepspeed.runtime.zero import GatheredParameters
        # 在 save/load/EVAL 时自动 gather 参数
```

**ZeRO3 加载检查点补丁**:
```python
def _patch_deepspeed_load_checkpoint(self):
    # 兼容 resume_only_model 模式
    def deepspeed_load_checkpoint(*args, **kwargs):
        try:
            return origin(*args, **kwargs)
        except Exception as e:
            logger.warning(f'Failed: {e}')
```

### 2.3 Flash Checkpoint

**文件**: `swift/trainers/mixin.py`

通过 DLRover 实现秒级 checkpoint：

```python
def _prepare_flash_ckpt(self):
    if self.args.use_flash_ckpt:
        import dlrover.trainer.torch.flash_checkpoint.hf_trainer
        # 自动替换 save_checkpoint 为异步 flash checkpoint
```

### 2.4 FSDP / FSDP2

```bash
swift sft --model xxx --dataset xxx \
    --fsdp auto_wrap \
    --fsdp_config fsdp_config.json

# FSDP2
swift sft --model xxx --dataset xxx \
    --fsdp full_shard
```

**Activation Offloading** (`swift/callbacks/activation_cpu_offload.py`):

```python
class ActivationCpuOffloadCallback:
    """FSDP 激活值异步卸载到 CPU，降低显存占用"""
    def on_step_begin(self, args, state, control, **kwargs):
        # 在 forward 前将上一层的激活值异步卸载
```

---

## 3. 序列并行

### 3.1 Ulysses 序列并行

**文件**: `swift/sequence_parallel/ulysses.py` (~40KB)

基于 DeepSpeed Ulysses 的序列并行实现：将序列维度按 head 维度切分到多个 GPU。

```python
class DistributedAttention(torch.nn.Module):
    def __init__(self, local_attention, sequence_parallel, scatter_idx=2, gather_idx=1):
        self.local_attn = local_attention
        self.sequence_parallel = sequence_parallel
    
    def forward(self, query, key, value, attention_mask, *args, **kwargs):
        # 1. all2all: scatter 序列维度
        query = _SeqAllToAll.apply(self.sequence_parallel.group, query, scatter_idx, gather_idx)
        key = _SeqAllToAll.apply(self.sequence_parallel.group, key, scatter_idx, gather_idx)
        value = _SeqAllToAll.apply(self.sequence_parallel.group, value, scatter_idx, gather_idx)
        
        # 2. 本地 attention 计算
        attn_output = self.local_attn(query, key, value, attention_mask, *args, **kwargs)
        
        # 3. all2all: gather 序列维度
        attn_output = _SeqAllToAll.apply(self.sequence_parallel.group, attn_output, gather_idx, scatter_idx)
        return attn_output
```

**使用方式**:
```bash
swift sft --model xxx --dataset xxx \
    --sequence_parallel_size 4  # 4 GPU 切分序列
```

### 3.2 Ring Attention

**文件**: `swift/sequence_parallel/zigzag_ring_attn.py` (~25KB)

基于 Ring Attention 的序列并行：

```python
def zigzag_ring_attention(query, key, value, ...):
    """环形注意力：每个 GPU 只计算序列的一部分，通过环形通信传递 KV"""
    local_output = None
    for i in range(world_size):
        # 从邻居获取 KV
        k, v = receive_from_prev_neighbor()
        # 本地 attention
        local_output += local_attention(q, k, v)
        # 发送给下一个邻居
        send_to_next_neighbor(k, v)
```

**Ulysses + Ring Attention 组合**:
```bash
# 2025.09 新增：两者可联合使用
# sequence_parallel_size 不再受 head 数限制
swift sft --model xxx --dataset xxx --sequence_parallel_size 8
```

### 3.3 GatherLoss — 序列并行损失聚合

**文件**: `swift/sequence_parallel/` 相关模块

```python
class GatherLoss:
    """在序列并行环境下收集各 rank 的 loss"""
    def __call__(self, loss):
        if sequence_parallel.world_size > 1:
            dist.all_reduce(loss, group=sequence_parallel.group)
            loss /= sequence_parallel.world_size
        return loss
```

---

## 4. Megatron-LM 集成

### 4.1 独立子系统

`swift/megatron/` 是一个**完整的独立子系统**，拥有：
- 自己的 CLI 入口 (`megatron`)
- 自己的参数体系 (`swift/megatron/arguments/`)
- 自己的 Pipeline (`swift/megatron/pipelines/`)
- 自己的 Trainer (`swift/megatron/trainers/`)
- 自己的模型工具 (`swift/megatron/model/`)

### 4.2 Mcore-Bridge

**文件**: `swift/megatron/` + `mcore-bridge` 依赖

Mcore-Bridge 将 Megatron-LM 的并行接口桥接到 transformers 模型格式，使用户无需手动修改 Megatron 配置：

```
Transformers Model ──▶ Mcore-Bridge ──▶ Megatron Model
                (自动并行化)          (TP/PP/SP/EP)
```

依赖: `mcore-bridge>=1.2.0`, `megatron-core>=0.15`

### 4.3 并行策略配置

```bash
megatron sft \
    --model Qwen/Qwen3-4B \
    --dataset xxx \
    --tensor_model_parallel_size 2 \      # TP
    --pipeline_model_parallel_size 2 \  # PP
    --sequence_parallel \                  # SP
    --context_parallel_size 2 \           # CP
    --expert_model_parallel_size 2 \      # EP (MoE)
    --num_layers_per_virtual_pipeline_stage 2  # VPP
```

### 4.4 Megatron 支持的训练任务

| 任务 | Pipeline | Trainer |
|-----|----------|---------|
| CPT | `MegatronSft` | `MegatronTrainer` |
| SFT | `MegatronSft` | `MegatronTrainer` |
| GRPO | `MegatronRLHF` | `MegatronGRPOTrainer` |
| DPO | `MegatronRLHF` | `MegatronDPOTrainer` |
| KTO | `MegatronRLHF` | `MegatronKTOTrainer` |
| Reward | `MegatronSft` | `MegatronRewardTrainer` |
| Embedding | `MegatronSft` | `MegatronEmbeddingTrainer` |
| Reranker | `MegatronSft` | `MegatronRerankerTrainer` |

### 4.5 Megatron SFT Pipeline

**文件**: `swift/megatron/pipelines/train/sft.py`

```python
class MegatronSft(SwiftSft):
    args_class = MegatronSftArguments
    
    def __init__(self, args):
        # 1. MindSpeed / NPU 适配 (华为昇腾)
        if is_torch_npu_available():
            patch_mindspeed_te_cp_implementation(megatron_args)
            repatch(megatron_args)
        
        # 2. 虚拟设备加载模型元信息（不加载实际权重）
        with torch.device('meta'):
            self.model, self.processor = args.get_model_processor(
                return_dummy_model=True)
        
        # 3. 初始化 Megatron 迭代器
        args.init_iters(train_dataset, val_dataset)
        
        # 4. Megatron Trainer 训练
        trainer = self.prepare_trainer()
        trainer.train(train_dataset, val_dataset)
    
    def prepare_trainer(self):
        if args.task_type == 'embedding':
            return MegatronEmbeddingTrainer(args, self.template)
        elif args.task_type == 'reranker':
            return MegatronRerankerTrainer(args, self.template)
        else:
            return MegatronTrainer(args, self.template)
```

### 4.6 模型转换

**文件**: `swift/megatron/utils/convert.py`

```python
def convert_hf_to_megatron(hf_model_path, megatron_model_path, ...):
    """将 HF 检查点转换为 Megatron 格式"""

def convert_megatron_to_hf(megatron_model_path, hf_model_path, ...):
    """将 Megatron 检查点转回 HF 格式"""
```

---

## 5. Ray 分布式支持

**文件**: `swift/ray/`

```python
class RayHelper:
    """Ray 分布式训练辅助类"""
    @staticmethod
    def worker(group):
        # Ray remote worker 装饰器
    
    @staticmethod
    def function(group):
        # Ray remote function 装饰器
```

在 Pipeline 中通过 `@RayHelper.worker` 和 `@RayHelper.function` 标注分布式函数。

---

## 6. 关键代码路径索引

| 功能 | 关键文件/函数 |
|-----|-------------|
| 自动 torchrun 包装 | `swift/cli/main.py::get_torchrun_args()` |
| DeepSpeed ZeRO3 gather | `swift/trainers/mixin.py::_fix_zero3_gather_all_parameters` |
| DeepSpeed 加载补丁 | `swift/trainers/mixin.py::_patch_deepspeed_load_checkpoint` |
| Flash Checkpoint | `swift/trainers/mixin.py::_prepare_flash_ckpt` |
| Activation Offload | `swift/callbacks/activation_cpu_offload.py` |
| Ulysses 序列并行 | `swift/sequence_parallel/ulysses.py::DistributedAttention` |
| Ring Attention | `swift/sequence_parallel/zigzag_ring_attn.py` |
| 序列并行准备 | `swift/sequence_parallel/__init__.py::sequence_parallel.prepare()` |
| Megatron SFT | `swift/megatron/pipelines/train/sft.py::MegatronSft` |
| Megatron Trainer | `swift/megatron/trainers/trainer.py::MegatronTrainer` |
| Megatron GRPO | `swift/megatron/trainers/grpo_trainer.py::MegatronGRPOTrainer` |
| Megatron 参数 | `swift/megatron/arguments/base_args.py::MegatronSftArguments` |
| 模型转换 | `swift/megatron/utils/convert.py` |
| Ray 辅助 | `swift/ray/` 目录 |
