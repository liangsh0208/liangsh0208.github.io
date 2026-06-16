---
title: "Slime 代码走读 — 05. 数据流水线：Dataset → Sample → DataIterator → get_batch"
date: 2026-06-15T12:00:00+08:00
draft: false
---

> 本文档走读 slime 的数据流水线，覆盖从原始 prompt 文件到 Megatron microbatch 的完整过程：Dataset 加载 → Sample 结构 → rollout 数据生成 → process_rollout_data → DataIterator → get_batch → CP padding & mask。

---

## 一、Sample 数据结构

```python
# slime/utils/types.py
@dataclass
class Sample:
    group_index: int | None = None       # prompt 分组索引
    index: int | None = None              # 全局 sample ID
    prompt: str | list[dict] = ""         # 纯文本或 conversation list
    tokens: list[int] = field(default_factory=list)  # prompt + response token IDs
    multimodal_inputs: dict | None = None   # 原始多模态输入（images, audio等）
    multimodal_train_inputs: dict | None = None  # 处理后 tensor
    response: str = ""                    # 生成文本
    response_length: int = 0             # response token 数
    label: str | None = None             # 参考答案
    reward: float | dict | None = None   # reward/rm 输出
    loss_mask: list[int] | None = None   # response token 的 mask（0/1）
    rollout_log_probs: list[float] | None = None   # SGLang 返回的 logp
    rollout_routed_experts: list[list[int]] | None = None  # MoE routing
    status: Status = Status.PENDING     # pending / completed / truncated / aborted / failed
    metadata: dict = field(default_factory=dict)
    teacher_log_probs: list[float] | None = None
    
    # 性能统计
    non_generation_time: float = 0.0
    
    # Speculative decoding
    spec_info: SpecInfo = field(default_factory=SpecInfo)
    
    # Prefix cache
    prefix_cache_info: PrefixCacheInfo = field(default_factory=PrefixCacheInfo)
    
    # Consistent hashing
    session_id: str | None = None
```

**Sample 的生命周期**：

```
Dataset (JSONL) → Sample (prompt only, status=PENDING)
  │
  ▼ rollout generate
  Sample (prompt + response, status=COMPLETED/TRUNCATED, tokens filled, reward filled)
  │
  ▼ _convert_samples_to_train_data
  RolloutBatch (dict[str, list]): tokens/response_lengths/loss_masks/rollout_log_probs/...
  │
  ▼ _split_train_data_by_dp
  rank_data: Box(ray.put({tokens: [...], advantages: [...], ...}))
  │
  ▼ actor.train() → _get_rollout_data
  RolloutBatch on GPU (list of Tensor)
  │
  ▼ get_data_iterator → get_batch
  microbatch: dict("tokens", "packed_seq_params", "full_loss_masks", "advantages", ...)
```

---

## 二、Dataset 加载：`slime/utils/data.py`

### 2.1 `Dataset.__init__()`

```python
class Dataset:
    def __init__(self, path, tokenizer, processor, max_length, 
                 prompt_key="text", label_key=None, 
                 apply_chat_template=False, multimodal_keys=None):
        origin_samples = []
        for data in read_file(path):  # 支持 .jsonl / .parquet / @slice
            # 构建 messages（支持多模态占位符替换）
            prompt = _build_messages(data, prompt_key, as_conversation=apply_chat_template, multimodal_keys)
            
            # 应用 chat template
            if apply_chat_template:
                output_prompt = tokenizer.apply_chat_template(
                    prompt, tools=tools, tokenize=False, add_generation_prompt=True
                )
            else:
                output_prompt = prompt
            
            # 多模态 inputs
            if processor:
                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                multimodal_inputs = None
            
            origin_samples.append(Sample(
                prompt=output_prompt,
                label=data[label_key] if label_key else None,
                metadata=data.get("metadata", {}),
                multimodal_inputs=multimodal_inputs,
            ))
        
        # 过滤超长 prompt
        if max_length is not None:
            self.origin_samples = filter_long_prompt(origin_samples, tokenizer, processor, max_length)
```

### 2.2 `read_file()` — 通用文件读取

```python
def read_file(path):
    path, row_slice = _parse_generalized_path(path)  # 支持 "path.jsonl@[0:1000]"
    
    if path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            yield json.loads(line.strip())
    elif path.endswith(".parquet"):
        for batch in pq.read_table(path).iter_batches():
            yield from batch.to_pylist()
```

### 2.3 `_build_messages()` — 多模态 prompt 构建

```python
def _build_messages(data, prompt_key, as_conversation, multimodal_keys=None):
    prompt = data.get(prompt_key)
    
    if isinstance(prompt, str):
        if not as_conversation:
            return prompt  # 纯文本
        prompt = [{"role": "user", "content": prompt}]  # 转成 conversation
    
    if multimodal_keys:
        # 占位符替换："<image>" → {"type": "image", "image": data["image_url"]}
        pattern = "(<image>|<video>|<audio>)"
        for message in prompt:
            for segment in re.split(pattern, message["content"]):
                if segment in multimodals:
                    content_list.append({"type": type, ...})
            message["content"] = content_list
    
    return prompt
```

---

## 三、Rollout → Megatron 数据转换

### 3.1 `_convert_samples_to_train_data()`

```python
def _convert_samples_to_train_data(self, samples):
    raw_rewards, rewards = self._post_process_rewards(samples)
    
    train_data = {
        "tokens":          [sample.tokens for sample in samples],
        "response_lengths":[sample.response_length for sample in samples],
        "rewards":         rewards,         # group-normed（如 GRPO）
        "raw_reward":      raw_rewards,
        "truncated":       [1 if sample.status == TRUNCATED else 0],
        "sample_indices":  [sample.index for sample in samples],
        "loss_masks":       sample.loss_mask or [1]*response_length,
    }
    
    # 可选字段
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]
    
    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
    
    if samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]
    
    if samples[0].multimodal_train_inputs is not None:
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]
    
    return train_data
```

### 3.2 `_post_process_rewards()` — GRPO 归一化

```python
def _post_process_rewards(self, samples):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    
    if args.advantage_estimator in ["grpo", "gspo"] and args.rewards_normalization:
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        # reshape to [group_size, n_samples_per_prompt]
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        
        # group mean subtraction (GRPO)
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean
        
        # optional std division
        if args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)
        
        rewards = rewards.flatten().tolist()
    else:
        rewards = raw_rewards
    
    return raw_rewards, rewards
```

### 3.3 `_split_train_data_by_dp()` — DP 分发

```python
def _split_train_data_by_dp(self, data, dp_size):
    total_lengths = [len(t) for t in data["tokens"]]
    
    if self.args.balance_data:
        # 均衡各 DP rank 的总 token 数（减少 P2P / PP 等待）
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        # Round-robin
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]
    
    for i in range(dp_size):
        rollout_data = {}
        partition = partitions[i]
        for key in keys_to_partition:
            rollout_data[key] = [data[key][j] for j in partition]
        
        # 全局字段不需要 partition（如 total_lengths, raw_reward）
        for key_global in ["raw_reward", "total_lengths"]:
            rollout_data[key_global] = data[key_global]
        
        # dynamic global batch size
        if hasattr(self, "_dynamic_global_batch_size"):
            rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
        
        rollout_data_refs.append(Box(ray.put(rollout_data)))
    
    return rollout_data_refs
```

---

## 四、`process_rollout_data()` — Ray→GPU 传输

```python
def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)  # 从 Object Store 取
    
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]
    
    # 记录到 Timer（用于 FLOPS 统计）
    Timer().seq_lens = total_lengths
    
    # 只保留当前 rank 的 samples 的 total_length
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]
    
    return rollout_data
```

---

## 五、`DataIterator` — Microbatch 迭代器

```python
# slime/backends/megatron_utils/data.py
class DataIterator:
    def __init__(self, rollout_data, micro_batch_size=None, micro_batch_indices=None):
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size       # 固定 microbatch 模式
        self.micro_batch_indices = micro_batch_indices # dynamic batch 模式
        assert micro_batch_size is None or micro_batch_indices is None
        self.offset = 0

    def get_next(self, keys):
        batch = {}
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            elif self.micro_batch_indices is not None:
                indices = self.micro_batch_indices[self.offset]
                batch[key] = [vals[i] for i in indices]
            else:
                batch[key] = vals[self.offset : self.offset + self.micro_batch_size]
        
        self.offset += 1 if self.micro_batch_indices else self.micro_batch_size
        return batch

    def reset(self):
        self.offset = 0
```

**两种模式**：

| 模式 | 触发条件 | 说明 |
|------|---------|------|
| 固定 microbatch | `micro_batch_size` 设值 | 每 microbatch 固定 N 个 sample |
| Dynamic batch | `micro_batch_indices` 设值 | 每 microbatch 的索引列表由 `get_seqlen_balanced_partitions()` 计算 |

---

## 六、`get_data_iterator()` — 构建 Microbatch 调度

```python
def get_data_iterator(args, model, rollout_data):
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dp_group = mpu.get_data_parallel_group()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    cp_size = mpu.get_context_parallel_world_size()
    
    num_local_samples = len(rollout_data["total_lengths"])
    global_batch_size = rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
    num_local_gbs = global_batch_size // dp_size
    num_steps_per_rollout = num_local_samples // num_local_gbs
    
    if not args.use_dynamic_batch_size:
        # 固定 microbatch：每个 step 的 microbatch 数相同
        num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
        data_iterator = _generate_data_iterator(rollout_data, micro_batch_size=args.micro_batch_size)
    else:
        # Dynamic batch：按 max_tokens_per_gpu 自动计算 microbatch
        assert args.max_tokens_per_gpu is not None
        num_microbatches = []
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            num_mbs = get_minimum_num_micro_batch_size(
                samples[start:end], args.max_tokens_per_gpu * cp_size
            )
            num_microbatches.append(num_mbs)
        
        # Allreduce max（所有 DP rank 对齐 microbatch 数）
        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device="cuda")
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)
        num_microbatches = num_microbatches.tolist()
        
        # Seqlen balancing：让各 microbatch 的 token 数接近
        micro_batch_indices = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            partitions = get_seqlen_balanced_partitions(
                rollout_data["total_lengths"][start:end], num_mbs, equal_size=False
            )
            micro_batch_indices.extend(partitions)
        
        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)
    
    return data_iterator, num_microbatches
```

### `get_minimum_num_micro_batch_size()` — First Fit 算法

```python
def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    batches = []
    for length in total_lengths:
        for i in range(len(batches))
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)
    return len(batches)  # 最少需要多少 microbatch
```

> 这是一个经典的 bin packing 问题。First Fit 保证 `num_microbatches` 不超过最优解的 2 倍。

---

## 七、`get_batch()` — Microbatch → CUDA Tensor

```python
def get_batch(data_iterator, keys, pad_multiplier=128, qkv_format="thd", allgather_cp=False):
    batch = data_iterator.get_next(keys)
    tokens = batch["tokens"]
    
    # CP 前保存原始 token
    batch["unconcat_tokens"] = tokens
    
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    
    if qkv_format == "thd":
        if allgather_cp:
            # DSA 模式：先全局 concatenation，再 slice once
            cu_seqlens_list = [0]
            for t in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))
            tokens = torch.cat(tokens)
            
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=0)
            
            cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device="cuda")
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        else:
            # 常规 CP：先 slice，再 concat
            tokens = [slice_with_cp(t, 0, qkv_format) for t in tokens]
            cu_seqlens = [0]
            for t in tokens:
                cu_seqlens.append(cu_seqlens[-1] + t.size(0))
            tokens = torch.cat(tokens)
            
            # padding 到 pad_size 倍数
            pad = (pad_size - tokens.size(0) % pad_size) % pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=0)
                cu_seqlens.append(cu_seqlens[-1] + pad)
            
            # thd 格式：cu_seqlens 需 × cp_size
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size
        
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )
        tokens = tokens.unsqueeze(0)  # [1, T_padded]
    
    elif qkv_format == "bshd":
        max_seqlen = batch["max_seq_lens"][0]
        tokens = [slice_with_cp(t, 0, qkv_format, max_seqlen) for t in tokens]
        tokens = torch.stack(tokens)      # [batch_size, max_seq_len]
        packed_seq_params = None
    
    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params
    
    # Loss masks（对齐到 token stream）
    loss_masks = []
    for loss_mask, total_length, response_length in zip(batch["loss_masks"], batch["total_lengths"], batch["response_lengths"]):
        prompt_length = total_length - response_length
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        if allgather_cp:
            loss_masks.append(loss_mask)
        else:
            loss_mask = slice_with_cp(loss_mask, 0, qkv_format)
        loss_masks.append(loss_mask)
    
    if qkv_format == "thd" and allgather_cp:
        loss_masks = torch.cat(loss_masks)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    
    batch["full_loss_masks"] = loss_masks
    
    return batch
```

### 关键图解：`qkv_format="thd"` 的 CP 切分

```
原始序列 [prompt0, response0, prompt1, response1, ...] → concat 后

Total sequence:     ├───────────────────────────────────────┤
CP 模式 (zigzag):   ├─ chunk_0 ─┤.............├─ chunk_1 ─┤
                                      ↑
                    cp_rank=0 拿: chunk_0 + chunk_1（靠后的段）
                    cp_rank=1 拿: chunk_0' + chunk_1'
                    
DSA 模式 (allgather_cp): 
                    先全局 concat → pad → 按长度均分 → 每个 rank 拿一段
                    
thd PackedSeqParams:
                    cu_seqlens = [0, T1, T2, T3, ...]  (cumulative sequence lengths)
                    max_seqlen = max(Ti)
                    qkv_format = "thd" (Total, Head, Dim)
```

---

## 八、`slice_with_cp()` — Context Parallel Token 切分

```python
def slice_with_cp(tokens, pad_value, qkv_format="thd", max_seq_len=None):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    
    if cp_size == 1:
        return tokens
    
    token_len = len(tokens)
    chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    
    pad = 2 * cp_size * chunk_size - token_len
    tokens = F.pad(tokens, (0, pad), value=pad_value)
    
    # zigzag 切分
    start_1 = chunk_size * cp_rank
    end_1   = chunk_size * (cp_rank + 1)
    start_2 = chunk_size * (2 * cp_size - cp_rank - 1)
    end_2   = chunk_size * (2 * cp_size - cp_rank)
    
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])
```

> **zigzag ring attention**：通过把序列切成交错的 chunk，实现 ring allreduce 时的错位通信，减少 bubble。

---

## 九、数据流水线总结

```
JSONL / Parquet
  │
  ├── read_file() → dict rows
  │     └── _parse_generalized_path("data.jsonl@[0:10000]")
  │
  ├── Dataset.__init__()
  │     ├── _build_messages() [multimodal 占位符替换]
  │     ├── apply_chat_template() [tokenizer]
  │     └── filter_long_prompt() [max_length 过滤]
  │
  Sample(prompt="...", label="...", metadata={})
      │
      └── data_source.get_samples(rollout_id) → list[list[Sample]]
          │
          └── rollout generate() → Sample(response="...", tokens=[...], reward=1.0)
              │
              └── _convert_samples_to_train_data()
                  ├── _post_process_rewards() [GRPO group norm]
                  └── RolloutBatch (dict of lists)
                      │
                      └── _split_train_data_by_dp(dp_size=4)
                          ├── balance_data=True: get_seqlen_balanced_partitions()
                          └── Box(ray.put(rank_data)) [Ray Object Store]
                              │
                              └── process_rollout_data(dp_rank, dp_size)
                                  ├── ray.get() → CPU dict
                                  ├── torch.tensor().to(cuda) [GPU transfer]
                                  └── slice_log_prob_with_cp() [CP restore]
                                      │
                                      └── get_data_iterator()
                                          ├── fixed mode: [num_microbatches] * steps
                                          └── dynamic mode: get_minimum_num_micro_batch_size()
                                              └── get_seqlen_balanced_partitions()
                                              │
                                              └── get_batch()
                                                  ├── slice_with_cp() [CP zigzag]
                                                  ├── build PackedSeqParams()
                                                  ├── pad loss_masks
                                                  └── build "full_loss_masks"
                                                      │
                                                      └── Megatron Pipeline Engine
                                                          │
                                                          └── model(input_ids=tokens, packed_seq_params=...)
```
