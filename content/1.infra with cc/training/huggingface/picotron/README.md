# picotron 架构文档

> **源码位置**: `/Users/danchen/Documents/1.RL_fw/huggingface/picotron`  
> **前置知识**: PyTorch分布式训练、Transformer架构、CUDA编程基础

---

## 文档导航

| 文档 | 主题 | 核心代码文件 |
|------|------|-------------|
| [00_整体架构与设计理念](1.infra%20with%20cc/training/huggingface/picotron/00_整体架构与设计理念.md) | 4D并行概述、极简设计原则 | - |
| [01_进程组管理](01_进程组管理.md) | ProcessGroupManager、4D进程网格 | `process_group_manager.py` |
| [02_张量并行TP](02_张量并行TP.md) | 列并行、行并行、AllReduce | `tensor_parallel/` |
| [03_流水线并行PP](03_流水线并行PP.md) | AFAB/1F1B调度引擎 | `pipeline_parallel/` |
| [04_上下文并行CP](04_上下文并行CP.md) | Ring Attention、在线Softmax | `context_parallel/` |
| [05_数据并行DP](05_数据并行DP.md) | 梯度同步、Bucket优化 | `data_parallel/` |
| [06_模型实现](06_模型实现.md) | Llama架构、GQA、SwiGLU | `model.py` |
| [07_Checkpoint与延迟初始化](07_Checkpoint与延迟初始化.md) | meta device、零内存加载 | `checkpoint.py` |
| [08_训练脚本解析](08_训练脚本解析.md) | train.py、配置生成 | `train.py` |

---

## 快速参考

### 4D并行组合速查

```
总GPU数 = DP × PP × CP × TP

常见配置:
- 数据并行: DP=8, PP=1, CP=1, TP=1 (8卡单节点)
- 3D并行: DP=4, TP=2, PP=2 (16卡)
- 4D长序列: DP=2, TP=2, PP=2, CP=2 (16卡 + 32K序列)
```

### 约束条件速查

```python
# 1. 总进程数
assert world_size == dp_size * pp_size * cp_size * tp_size

# 2. Context Parallel
assert seq_length % cp_size == 0

# 3. Tensor Parallel  
assert num_attention_heads % tp_size == 0
assert num_key_value_heads % tp_size == 0
assert hidden_size % tp_size == 0

# 4. Pipeline Parallel
# 层数分配可能不均匀
layers_per_gpu[i] = num_layers // pp_size + (1 if i < num_layers % pp_size else 0)
```

### 核心文件行数

| 文件 | 行数 | 核心职责 |
|-----|------|----------|
| `train.py` | ~280 | 训练主循环 |
| `model.py` | ~300 | Llama模型定义 |
| `process_group_manager.py` | ~70 | 4D进程组管理 |
| `tensor_parallel.py` | ~190 | TP实现 |
| `pipeline_parallel.py` | ~220 | PP及调度器 |
| `data_parallel.py` | ~170 | DP带bucket优化 |
| `context_parallel.py` | ~200 | Ring Attention |
| `checkpoint.py` | ~290 | Lazy初始化与存储 |

---

## 启动命令模板

```bash
# 1. 生成配置
python create_config.py \
    --out_dir tmp \
    --exp_name experiment \
    --dp 2 --tp 2 --pp 2 \
    --seq_len 8192 \
    --model_name meta-llama/Llama-2-7b-hf

# 2. 启动训练
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --config tmp/experiment/config.json

# 3. 启用Flash Attention加速
FLASH_ATTEN=1 torchrun ... train.py ...
```

---

*文档重构于 2026-04-20*
