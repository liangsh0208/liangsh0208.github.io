---
title: "Slime 代码走读 — 07. 工具层与性能分析"
date: 2026-06-15T12:00:00+08:00
draft: false
---

> 本文档覆盖 slime 的工具层实现：`flops_utils.py`（TFLOPS 计算）、`timer.py`（性能计时）、`arguments.py`（参数系统）、`metric_utils.py`（指标统计），以及性能分析的实践方法。

---

## 一、TFLOPS 计算：`slime/utils/flops_utils.py`

slime 中仅计算了前向（fwd）FLOPs，训练时通过 ×3 估算总 FLOPs（fwd + 2×bwd）。

### 1.1 Transformer Layer FLOPs 公式

```python
def calculate_embedding_flops(seqlen, hidden_size):
    return 2 * seqlen * hidden_size

def calculate_lm_head_flops(seqlen, hidden_size, vocab_size):
    return 2 * seqlen * hidden_size * vocab_size

def calculate_qkv_projection_flops(args, seqlen, hidden_size, num_heads, num_query_groups):
    if args.q_lora_rank is None:
        q_flops = 2 * seqlen * hidden_size * num_heads * args.kv_channels
    else:
        # DeepSeek MLA / Qwen3 的压缩 Q projection
        q_flops = 2 * seqlen * args.q_lora_rank * (hidden_size + num_heads * args.qk_head_dim)
    
    if args.kv_lora_rank is None:
        kv_flops = 2 * 2 * seqlen * hidden_size * num_query_groups * args.kv_channels
    else:
        kv_flops = 2 * seqlen * (
            args.kv_lora_rank * (hidden_size + num_heads * args.qk_head_dim)
            + hidden_size * args.qk_pos_emb_head_dim
        )
    return q_flops + kv_flops

def calculate_attention_flops(args, seqlen, num_heads):
    # QK^T with causal mask (only upper triangular)
    if args.qk_pos_emb_head_dim:
        flops = 2 * num_heads * seqlen * seqlen * (args.qk_head_dim + args.qk_pos_emb_head_dim) / 2
    else:
        flops = 2 * num_heads * seqlen * seqlen * args.kv_channels / 2
    
    # A*V
    if args.v_head_dim:
        flops += num_heads * seqlen * seqlen * args.v_head_dim
    else:
        flops += num_heads * seqlen * seqlen * args.kv_channels
    return flops

def calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size):
    return 2 * seqlen * hidden_size * ffn_hidden_size * 3  # gate + up + down
```

### 1.2 总 FLOPs 计算

```python
def calculate_fwd_flops(seqlens, args):
    total_flops = 0
    for seqlen in seqlens:
        # Dense layers 的 FLOPs
        total_flops += calculate_layer_flops(args, seqlen, hidden_size, ...) * num_dense_layers
        # MoE layers 的 FLOPs（router_topk 个 expert 参与）
        total_flops += calculate_layer_flops(args, seqlen, hidden_size, ..., moe_ffn) * num_moe_layers
        # LM Head
        total_flops += calculate_lm_head_flops(seqlen, hidden_size, vocab_size)
    return total_flops
```

**注意 MoE 的处理**：
- `moe_ffn = moe_ffn_hidden_size * moe_router_topk + shared_expert_ffn`
- 每个 token 经过 `topk` 个 expert，所以 MLP 的 hidden_size 需要乘 `topk`

### 1.3 性能指标计算

```python
# train_metric_utils.py
def log_perf_data_raw(rollout_id, args, is_primary_rank, compute_total_fwd_flops):
    total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)
    
    # log_probs / ref_log_probs forward
    log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]
    
    # actor train (fwd + bwd)
    log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
    log_dict["perf/actor_train_tok_per_s"] = sum(seq_lens) / log_dict["perf/actor_train_time"]
    
    # step time = wait + train
    total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
    log_dict["perf/step_time"] = total_time
    log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time
```

### 1.4 `calculate_fwd_flops` 局限性

> ⚠️ 这个 FLOPs 计算**只包含矩阵乘法的浮点运算**，不包括：
> - Softmax
> - LayerNorm / RMSNorm
> - dropout / activation functions
> - CUDA Graph 开销
> - KV cache 管理
> - MoE router gating
> 
> 实际 MFU 会被低估。但在同类任务间横向对比时仍有参考价值。

---

## 二、性能计时器：`slime/utils/timer.py`

```python
import time
from megatron.core import mpu

class Timer(metaclass=SingletonMeta):
    """Singleton 计时器，收集所有阶段的累积时间"""
    def __init__(self):
        self.timers = {}      # name -> total_elapsed
        self.start_time = {}  # name -> start_timestamp
        self.seq_lens = []    # 用于 FLOPS 计算的序列长度
    
    def start(self, name):
        self.start_time[name] = time.time()
        if mpu.get_rank() == 0:
            logger.info(f"Timer {name} start")
    
    def end(self, name):
        elapsed = time.time() - self.start_time[name]
        self.timers[name] = self.timers.get(name, 0) + elapsed
        if mpu.get_rank() == 0:
            logger.info(f"Timer {name} end ({elapsed:.1f}s)")
    
    def reset(self, name=None):
        if name is None:
            self.timers = {}
        elif name in self.timers:
            del self.timers[name]
    
    def log_dict(self):
        return {f"perf/{key}_time": val for key, val in self.timers.items()}
```

### 2.1 装饰器和上下文管理器

```python
# 装饰器
@timer
def my_func():
    pass

# 上下文管理器
with timer("my_block"):
    do_something()
```

### 2.2 `inverse_timer` — 先 end 再 start

```python
# 用于 "train_wait" 的统计：上个阶段结束 → train 开始之间的间隔
@with_defer(lambda: Timer().start("train_wait"))
def init(self, ...):
    ...
    # init 结束后，Timer().start("train_wait") 会被自动调用
    # 表示 "从 init 结束到 train 开始" 期间的等待时间
```

### 2.3 `Timer` 的时序

```python
# MegatronTrainRayActor.train_actor() 中的计时

with inverse_timer("train_wait"), timer("train"):
    # train_wait end, train start
    
    timer("log_probs")  # ref log_probs
    timer("log_probs")  # current log_probs
    
    compute_advantages_and_returns()  # 无 timer，很快
    
    timer("actor_train")  # actual training
    
    # train end → log_perf_data() 调用
```

**注意**：Timer 是单例，所以多个 rollout 的时间是**累积**的。在 `log_perf_data_raw()` 中会先 `deepcopy` 再 `reset()`。

---

## 三、参数系统：`slime/utils/arguments.py`

这是一个 **1777 行**的参数文件，分为多个子系统。

### 3.1 参数组织架构

```python
def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_slime_arguments(parser):
        # ① Ray / 集群参数
        add_cluster_arguments(parser)
        # ② 训练后端参数 (Megatron)
        add_train_arguments(parser)
        # ③ Rollout / SGLang 参数
        add_rollout_arguments(parser)
        # ④ RL 算法参数
        add_rl_arguments(parser)
        # ⑤ Checkpoint / 保存参数
        add_checkpoint_arguments(parser)
        # ⑥ 评测参数
        add_eval_arguments(parser)
        # ⑦ 日志 / profiling 参数
        add_logging_arguments(parser)
        # ⑧ 自定义函数路径参数
        add_custom_fn_path_arguments(parser)
    return add_slime_arguments
```

### 3.2 关键参数分类

#### 集群参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--actor-num-nodes` | 1 | 训练 Actor 的节点数 |
| `--actor-num-gpus-per-node` | 8 | 每节点 Actor GPU |
| `--rollout-num-gpus` | None | Rollout 总 GPU（colocate 时自动设=actor） |
| `--colocate` | False | 训练和推理同 GPU |
| `--offload-train` | None | train 时 offlaod 模型到 CPU |
| `--offload-rollout` | None | rollout 时 offload 推理引擎 |

#### Megatron 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--qkv-format` | "thd" | "thd" (packed) 或 "bshd" (b+padding) |
| `--optimizer-cpu-offload` | False | Adam states 放 CPU |
| `--accumulate-allreduce-grads-in-fp32` | False | grad allreduce 用 fp32 |
| `--overlap-grad-reduce` | False | 通信计算 overlap |
| `--sequence-parallel` | False | layernorm 等 sequence-parallel |

#### Rollout / SGLang 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rollout-batch-size` | 32 | 每 rollout 的 prompt 数 |
| `--n-samples-per-prompt` | 4 | 每 prompt 生成几条 response |
| `--rollout-max-response-len` | 1024 | 生成长度上限 |
| `--rollout-temperature` | 1.0 | 采样温度 |
| `--sglang-mem-fraction-static` | 0.8 | SGLang 显存占比 |
| `--use-dynamic-batch-size` | False | 按 max_tokens_per_gpu 切 microbatch |

#### RL 算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator` | "ppo" | "grpo" / "ppo" / "reinforce_plus_plus" |
| `--use-kl-loss` | False | 是否加 KL 正则 |
| `--kl-loss-coef` | 0.001 | KL loss 权重 |
| `--kl-loss-type` | "k1" | "k1" / "k2" / "k3" / "low_var_kl" |
| `--eps-clip` | 0.2 | PPO 裁剪下界 |
| `--eps-clip-high` | 0.2 | PPO 裁剪上界（asymmetric） |
| `--normalize-advantages` | False | 是否对 advantage 做 whitening |
| `--use-tis` | False | Truncated Importance Sampling |
| `--use-opsm` | False | Off-Policy Sequence Masking |

#### Checkpoint 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save-interval` | 100 | 每 N rollout 保存 |
| `--save-hf` | None | HuggingFace 格式保存路径模板 |
| `--load` | None | Megatron checkpoint 路径 |
| `--hf-checkpoint` | None | HuggingFace 预训练权重路径 |
| `--ref-load` | None | Reference model 路径 |

---

## 四、指标统计：`slime/utils/metric_utils.py`

### 4.1 `compute_statistics()`

```python
def compute_statistics(values: list[float]):
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }
```

### 4.2 `compute_pass_rate()`

```python
def compute_pass_rate(flat_rewards, group_size, num_groups=None):
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)
    
    # pass@1, pass@2, pass@4, ... (until group_size)
    pass_k_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]
    
    for k in pass_k_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)
        log_dict[f"pass@{k}"] = np.mean(pass_k_estimates)
    
    return log_dict

def _estimate_pass_at_k(num_samples, num_correct, k):
    """1 - C(n-c, k) / C(n, k)"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

### 4.3 `has_repetition()` — 复读检测

```python
def has_repetition(text):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    return False

def compression_ratio(data, algorithm="zlib", level=9):
    compressed = zlib.compress(data.encode("utf-8"), level)
    return len(data) / len(compressed)  # 压缩比 > 10 判定为重复
```

> 原理：文本重复（如 "AAAA..."）会被 zlib 极度压缩，压缩比极高。

---

## 五、Profiling：`slime/utils/profile_utils.py`

### 5.1 `TrainProfiler`

```python
class TrainProfiler:
    def __init__(self, args):
        if args.use_pytorch_profiler:
            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                with_flops=True,
            )
    
    def step(self, rollout_id):
        if args.profile_every_n_rollout and rollout_id % args.profile_every_n_rollout == 0:
            prof.export_chrome_trace(f"trace_{rollout_id}.json")
```

### 5.2 关键性能诊断方法

```bash
# ① 查看 pipeline bubble
python3 -m torch.profiler chrome_trace trace.json
# 在 Chrome trace 中搜索 "NCCL" 和 "cudaLaunchKernel"

# ② 查看 wait_time 来源
# 分析 train.log 中 perf/wait_time_ratio 和 perf/train_wait_time

# ③ 分析 weight sync 开销
# 关注 perf/update_weights_time
# 量化时 int4/fp4 的 post_process 可能是大头

# ④ 分析 rollout 生成效率
# 关注 perf/rollout_time 对比 perf/tokens_per_gpu_per_sec
# 如果 tokens_per_gpu_per_sec 远低于 H20 理论，检查 KV cache hit rate
```

---

## 六、命令行启动流程详解：`shell/run-qwen3-32B.sh`

```bash
#!/bin/bash
# 4 节点，32 GPU，Qwen3-32B

# ① 环境变量
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ② Megatron Parallel 配置
CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${OUTDIR}"
   --save "${OUTDIR}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --global-batch-size 256
   --balance-data          # DP rank 间均衡 token 数
)

PERF_ARGS=(
   --tensor-model-parallel-size 8   # TP=8
   --sequence-parallel              # 启用 sequence parallel
   --recompute-granularity full     # 全重计算
   --use-dynamic-batch-size         # dynamic microbatch
   --max-tokens-per-gpu 20480
)

# ③ GRPO 算法
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss        # 计算 KL 但不施加（coef=0）
   --kl-loss-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ④ SGLang 配置
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)  # 预编译 batch sizes
)

# ⑤ 启动
train.py \
    --actor-num-nodes 4 \
    --actor-num-gpus-per-node 8 \
    --colocate \
    --logs \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
```

---

## 七、性能分析速查表

| 问题 | 诊断指标 | 代码位置 | 优化方向 |
|------|---------|---------|---------|
| train 等待时间太长 | `perf/wait_time_ratio` | `train_metric_utils.py:42` | 减少 Ray `ray.get()` 的阻塞，检查 SGLang generate 是否成为瓶颈 |
| actor_train_tflops 低 | `perf/actor_train_tflops` | `flops_utils.py:66` | 检查 microbatch 是否够大、是否 padding 占比太高、CP 是否引入额外通信 |
| generate 太慢 | `perf/tokens_per_gpu_per_sec` | `rollout.py:1204` | 增大 SGLang 并发、启用 CUDA Graph、检查 Prefix Cache hit rate |
| weight sync 太慢 | `perf/update_weights_time` | `actor.py:535` | 减少 DP 数、量化（int4/fp8）、colocate 走内存而非网络 |
| OOM | N/A | `model.py` | 增大 `recompute-granularity`、启用 CPU offload、减小 `max-tokens-per-gpu`、减小 microbatch |
| 梯度爆炸 | `train/grad_norm` > 1e3 | `model.py:648` | 减小学习率、增大 weight decay、检查 advantage 是否超大值 |
| rollout_log_probs 和 ref 差异大 | `train/train_rollout_logprob_abs_diff` | `loss.py:796` | 检查权重是否在 generate 后被意外更新；async 模式下此值应小 |

---

## 八、关键设计洞察

1. **`Timer` 是 Singleton** — 跨模块跨函数共享同一个计时器实例，确保所有阶段的时间被统一收集
2. **`flops_utils.py` 的理论 TFLOPs 不要和端到端 throughput 混用** — 后者包含所有等待时间，前者只计算 matmul
3. **`arguments.py` 的 1777 行是"自我保护文档"** — 几乎所有设计选择都有对应的 flag
4. **`metric_utils.py` 的 `has_repetition()` 用 zlib 压缩比检测复读** — 比 LLM-based 检测快得多
5. **`profile_utils.py` 的 PyTorch profiler 产出 chrome trace** — 是定位 NCCL 通信瓶颈的终极工具
