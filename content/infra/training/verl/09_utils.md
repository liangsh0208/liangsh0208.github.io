# Utils 工具库

**路径**：`verl/utils/`

utils 是 verl 的通用基础设施层，提供数据集、日志、分布式通信、内存管理、奖励函数等各类工具。

---

## 1. 数据集（`utils/dataset/`）

### 1.1 `RLDataset`（`rl_dataset.py`）

RL 训练数据集，从 parquet / jsonl 文件读取预处理好的 prompt：

```python
class RLDataset(Dataset):
    """
    每条样本包含：
    - input_ids：tokenize 后的 prompt token ids（左 padding）
    - attention_mask：prompt 的 attention mask
    - position_ids：位置编码
    - reward_model：{"ground_truth": "...", "style": "..."}
    - data_source：数据集名称（用于路由到对应奖励函数）
    - extra_info：额外信息（如多选题的选项）
    """

    def __init__(self, data_files, tokenizer, config, max_samples=-1):
        self.data = load_datasets(data_files)  # 加载所有 parquet/jsonl
        self.tokenizer = tokenizer
        self.max_prompt_length = config.max_prompt_length

    def __getitem__(self, idx):
        sample = self.data[idx]
        # tokenize prompt，左 padding 到 max_prompt_length
        input_ids = tokenize_with_left_padding(
            sample["prompt"],
            self.tokenizer,
            self.max_prompt_length,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "reward_model": sample["reward_model"],
            "data_source": sample["data_source"],
        }
```

### 1.2 多轮 SFT 数据集（`multiturn_sft_dataset.py`）

用于 SFT 训练多轮对话的数据集，支持 system prompt、历史对话拼接：

```python
class MultiTurnSFTDataset(Dataset):
    """支持多轮对话 SFT 训练"""

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        # 拼接所有轮次：system + user_1 + assistant_1 + ... + assistant_n
        full_text = apply_chat_template(messages, self.tokenizer)
        # 只对 assistant 部分计算 loss（input_ids 中 prompt 部分的 label = -100）
        labels = mask_prompt_labels(full_text, messages)
        return {"input_ids": full_text, "labels": labels}
```

### 1.3 RM 数据集（`rm_dataset.py`）

Reward Model 训练数据集，包含 chosen / rejected 对：

```python
class RMDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "chosen_input_ids": ...,
            "rejected_input_ids": ...,
        }
```

### 1.4 数据集工具（`dataset_utils.py`）

```python
class DatasetPadMode(Enum):
    LEFT = "left"    # 左 padding（适合 Causal LM）
    RIGHT = "right"  # 右 padding

def pad_sequence_to_length(sequence, target_length, pad_value, mode):
    """将序列 padding 到指定长度"""
    ...

def collate_fn(data_list: list[dict]) -> dict:
    """将 list of dicts 转换为 batched tensors"""
    tensors = {k: torch.stack([d[k] for d in data_list]) for k in tensor_keys}
    non_tensors = {k: np.array([d[k] for d in data_list]) for k in non_tensor_keys}
    return {**tensors, **non_tensors}
```

---

## 2. 日志（`utils/logger/`）

```python
class Tracking:
    """统一的训练日志接口，支持多个后端"""

    def __init__(self, project_name, experiment_name, default_backend, config):
        # 支持的后端：wandb / tensorboard / mlflow / console
        self.backends = [create_backend(b) for b in default_backend]

    def log(self, data: dict, step: int):
        """记录 metrics"""
        for backend in self.backends:
            backend.log(data, step)
```

支持的日志后端：
- **WandB**：推荐用于实验跟踪
- **TensorBoard**：通过 `SummaryWriter` 记录
- **控制台**：直接打印（调试用）
- **MLflow**：企业级实验管理

---

## 3. 分布式工具（`utils/distributed.py`）

```python
def initialize_global_process_group_ray():
    """在 Ray actor 中初始化 torch.distributed 进程组"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())
```

---

## 4. FSDP 工具（`utils/fsdp_utils.py`）

FSDP2 相关的工具函数集合：

```python
def apply_fsdp2(model, mixed_precision, cpu_offload, wrap_policy):
    """对模型应用 FSDP2 分片"""
    fully_shard(
        model,
        policy=wrap_policy,        # 决定哪些 submodule 独立分片
        mixed_precision=mixed_precision,
        offload_policy=cpu_offload,
    )
    return model

def get_fsdp_wrap_policy(model):
    """自动生成 Transformer layer 级别的包裹策略"""
    # 每个 Transformer block 独立分片（最常用的策略）
    transformer_cls = get_transformer_layer_class(model)
    return ModuleWrapPolicy({transformer_cls})

def fsdp2_clip_grad_norm_(model, max_norm):
    """FSDP2 兼容的梯度裁剪（分布式版本）"""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.stack([g.norm() for g in grads]).norm()
    scale = max_norm / (total_norm + 1e-6)
    if scale < 1.0:
        for g in grads:
            g.mul_(scale)
```

---

## 5. TensorDict 工具（`utils/tensordict_utils.py`）

```python
def concat_dict_to_tensordict(list_of_dicts: list[dict]) -> TensorDict:
    """将多个 dict 合并为一个 TensorDict"""
    return TensorDict({
        k: torch.cat([d[k] for d in list_of_dicts], dim=0)
        for k in list_of_dicts[0].keys()
    })

def maybe_fix_3d_position_ids(data: TensorDict):
    """处理 3D position_ids（Qwen2-VL 等多模态模型需要）"""
    if "position_ids" in data and data["position_ids"].dim() == 3:
        # 将 (bs, 3, seq_len) 压缩为 (bs, seq_len)
        data["position_ids"] = data["position_ids"][:, 0, :]
```

---

## 6. 序列长度均衡（`utils/seqlen_balancing.py`）

在 DP 训练中，各 rank 的序列长度不均衡会导致慢 rank 阻塞所有 rank：

```python
def get_seqlen_balanced_partitions(
    batch: DataProto,
    num_partitions: int,
) -> list[DataProto]:
    """
    将 batch 中的样本重新排列，使每个 partition 的总 token 数尽量相等。
    使用贪心算法：按序列长度降序排序，依次分配给 token 数最少的 partition。
    """
    seqlens = compute_seqlens(batch)
    sorted_indices = sorted(range(len(seqlens)), key=lambda i: -seqlens[i])
    partitions = [[] for _ in range(num_partitions)]
    partition_tokens = [0] * num_partitions

    for idx in sorted_indices:
        # 分配给当前 token 数最少的 partition
        min_partition = min(range(num_partitions), key=lambda p: partition_tokens[p])
        partitions[min_partition].append(idx)
        partition_tokens[min_partition] += seqlens[idx]

    return [batch[indices] for indices in partitions]
```

---

## 7. Torch 数学工具（`utils/torch_functional.py`）

```python
def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算每个位置的 log_prob：log P(label_t | context)
    输入：logits (bs, seq_len, vocab_size)，labels (bs, seq_len)
    输出：log_probs (bs, seq_len)
    """
    log_softmax = F.log_softmax(logits, dim=-1)
    return torch.gather(log_softmax, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

def masked_mean(tensor, mask, axis=-1):
    """只对 mask=1 的位置求均值"""
    return (tensor * mask).sum(axis) / mask.sum(axis).clamp(min=1)

def masked_whiten(tensor, mask, shift_mean=True):
    """标准化：减均值除以标准差，只作用于 mask=1 的位置"""
    mean = masked_mean(tensor, mask)
    std = masked_mean((tensor - mean) ** 2, mask).sqrt()
    if shift_mean:
        return (tensor - mean) / (std + 1e-8)
    return tensor / (std + 1e-8)
```

---

## 8. 内存管理（`utils/memory_utils.py`）

```python
def log_gpu_memory_usage(tag: str):
    """打印当前 GPU 显存使用情况（调试用）"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{tag}] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def release_memory(model):
    """强制释放模型显存（推理完成后）"""
    model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()
```

---

## 9. 性能分析（`utils/profiler/`）

```python
class GPUMemoryLogger:
    """追踪每个操作前后的显存变化"""
    def __enter__(self):
        self.before = torch.cuda.memory_allocated()
    def __exit__(self, ...):
        after = torch.cuda.memory_allocated()
        delta = (after - self.before) / 1e9
        logger.debug(f"Memory delta: {delta:+.3f} GB")
```

支持的 profiler 后端：
- `nsys`（NVIDIA Nsight Systems）：用于 CUDA kernel 分析
- `torch.profiler`：PyTorch 内置性能分析器

---

## 10. Megatron 工具（`utils/megatron_utils.py`、`utils/megatron/`）

```python
def initialize_megatron(config):
    """初始化 Megatron 全局状态（TP/PP 进程组）"""
    # 设置 tensor_model_parallel_size / pipeline_model_parallel_size
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
    )

def get_lm_forward_output_and_loss_func(model):
    """获取 Megatron 标准前向传播函数"""
    ...
```

---

## 11. 其他重要工具

| 文件 | 功能 |
|------|------|
| `config.py` | Hydra 配置工具，`omega_conf_to_dataclass()` |
| `import_utils.py` | 动态加载类（`load_class_from_fqn`），处理可选依赖 |
| `tracking.py` | 实验跟踪（`Tracking`、`ValidationGenerationsLogger`）|
| `fs.py` | 文件系统抽象（本地 / HDFS / S3）|
| `hdfs_io.py` | HDFS 读写 |
| `net_utils.py` | 网络工具（获取空闲端口等）|
| `py_functional.py` | Python 函数式工具（`DynamicEnum`、`append_to_dict` 等）|
| `tokenizer.py` | Tokenizer 工具（`normalize_token_ids`）|
| `attention_utils.py` | Attention 相关工具（`unpad_input`、`pad_input`）|
| `rollout_trace.py` | Rollout 追踪（记录每步 rollout 的详细信息）|
| `fp8_utils.py` | FP8 量化工具（ModelOpt 集成）|
| `groupwise.py` | 分组操作工具（GDPO 等算法用）|
