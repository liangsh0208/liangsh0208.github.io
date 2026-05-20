---
created: 2026-05-06
---

# Rollout Worker 模块

**路径**：`verl/workers/rollout/`

Rollout Worker 使用当前策略对输入 prompt 进行推理生成，产出 token 序列及对应的 log 概率，是 RL 训练数据的来源。

---

## 1. 接口设计

**文件**：`verl/workers/rollout/base.py`

```python
class BaseRollout:
    """所有 Rollout 实现的抽象基类"""

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        输入：DataProto，包含：
            - batch["input_ids"]：prompt token ids，(bs, prompt_len)
            - batch["attention_mask"]：prompt attention mask
            - meta_info["temperature"]：采样温度
            - meta_info["do_sample"]：是否采样（False = 贪心解码）

        输出：DataProto，新增字段：
            - batch["responses"]：生成的 token ids，(bs, response_len)
            - batch["rollout_log_probs"]：每个 token 的 log 概率，(bs, response_len)
            - batch["attention_mask"]：完整序列（prompt+response）的 mask
        """
        raise NotImplementedError
```

---

## 2. vLLM Rollout

**文件**：`verl/workers/rollout/vllm_rollout/vllm_rollout.py`

vLLM 是 verl 最常用的推理后端，支持 PagedAttention、连续批处理等优化。

### 2.1 两种模式

**同步模式（`VLLMRollout`）**：
- Rollout Worker 和 Actor Worker 共享 GPU（Hybrid Engine）
- 训练时 vLLM 引擎处于睡眠状态（通过 `sleep_level` 释放 KV cache）
- 推理时将 FSDP 分片的参数合并后加载到 vLLM

**异步服务器模式（`ServerAdapter`）**：
- vLLM 作为独立的 HTTP/gRPC 服务运行
- Actor 训练完后，通过权重传输协议更新服务器权重
- 推理请求异步发送，适合长序列生成

```python
class ServerAdapter(BaseRollout):
    """客户端适配器，向独立的 vLLM 服务器发请求"""

    def update_weights(self, weights: dict):
        """通知 vLLM 服务器更新权重"""
        self.server_handle.update_weights.remote(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """向 vLLM 服务器发送生成请求"""
        response = requests.post(self.server_url, json=prompts)
        return DataProto.from_dict(response.json())
```

### 2.2 权重同步机制

训练后，Actor 的新权重需要同步给 vLLM 引擎。verl 支持多种传输方式：

**直接内存传输（IPC）**：
```python
if is_support_ipc():
    # 通过 CUDA IPC（进程间通信）直接共享 GPU 内存，零拷贝
    weights = get_weights_via_ipc(actor_model)
    vllm_engine.update_weights(weights)
```

**Bucketed Weight Transfer**：
```python
class BucketedWeightSender:
    """将权重按 bucket 分批传输，避免一次性占用过多内存"""

    def send(self, weights: dict[str, torch.Tensor]):
        for bucket in self._split_into_buckets(weights):
            self.send_bucket(bucket)
```

**NCCL 广播**（多节点场景）：
- Actor 侧的 rank 0 将完整权重通过 NCCL all-gather 收集
- 然后广播给 vLLM 的所有 TP rank

### 2.3 Sleep Level（显存管理）

vLLM 的 KV cache 会占用大量显存。在 Hybrid Engine 模式下，训练时 vLLM 需要释放显存：

```python
# 推理完成后，vLLM 进入睡眠
vllm_engine.sleep(level=VLLM_SLEEP_LEVEL)
# 训练完成后，唤醒 vLLM 并加载新权重
vllm_engine.wake_up()
vllm_engine.update_weights(new_weights)
```

---

## 3. SGLang Rollout

**文件**：`verl/workers/rollout/sglang_rollout/sglang_rollout.py`

SGLang 是另一个高性能推理引擎，特别擅长多轮对话（RadixAttention 前缀缓存）和工具调用场景。

### 3.1 HTTP 服务引擎

```python
class HttpServerEngine:
    """将 SGLang 服务器封装为本地引擎接口"""

    def generate(self, prompts, sampling_params) -> list[str]:
        payload = {
            "text": prompts,
            "sampling_params": sampling_params,
            "return_logprob": True,  # 返回 log 概率
        }
        response = requests.post(f"{self.server_url}/generate", json=payload)
        return response.json()
```

### 3.2 异步 SGLang 服务器

**文件**：`verl/workers/rollout/sglang_rollout/async_sglang_server.py`

支持 multi-turn 场景（`sglang_multiturn`），每轮对话后调用外部环境/工具，将环境响应拼接到对话历史继续生成：

```python
async def async_generate_sequences(prompts, tools, env):
    for turn in range(max_turns):
        # 生成当前轮回复
        responses = await sglang_server.generate(prompts)

        # 调用工具或环境
        tool_results = await env.step(responses)

        # 将工具结果拼接到对话历史
        prompts = concat_messages(prompts, responses, tool_results)

        if all_done(responses):
            break
```

---

## 4. TRT-LLM Rollout

**文件**：`verl/workers/rollout/trtllm_rollout/`

TensorRT-LLM 后端，适合 FP8 量化推理和 NVIDIA Hopper GPU 上的极致性能优化。

---

## 5. Naive / HuggingFace Rollout

**文件**：`verl/workers/rollout/naive/naive_rollout.py`、`verl/workers/rollout/hf_rollout.py`

基于 HuggingFace `transformers.generate()` 的参考实现：
- 无需安装 vLLM/SGLang
- 适合单机调试、功能验证
- 性能较低，不适合生产训练

```python
class HFRollout(BaseRollout):
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        with torch.no_grad():
            output = self.model.generate(
                input_ids=prompts.batch["input_ids"],
                attention_mask=prompts.batch["attention_mask"],
                max_new_tokens=self.config.response_length,
                temperature=self.config.temperature,
                do_sample=True,
            )
        # 提取 log_probs
        logits = self.model(output).logits
        log_probs = logprobs_from_logits(logits, output)
        ...
```

---

## 6. Rollout Replica

**文件**：`verl/workers/rollout/replica.py`

支持多个 Rollout 副本并行生成（提高 rollout 吞吐量），然后合并结果：

```python
class RolloutReplica:
    """管理多个 Rollout 实例，实现负载均衡"""

    def generate_sequences_parallel(self, prompts: DataProto) -> DataProto:
        # 按副本数切分 batch
        chunks = prompts.chunk(len(self.replicas))
        # 并发生成
        futures = [
            replica.generate_sequences.remote(chunk)
            for replica, chunk in zip(self.replicas, chunks)
        ]
        # 合并结果
        results = ray.get(futures)
        return DataProto.concat(results)
```

---

## 7. Rollout 配置（`RolloutConfig`）

```yaml
rollout:
  name: "vllm"                        # vllm / sglang / hf / trtllm
  n: 8                                # 每个 prompt 生成几条回答（GRPO 需要 n>1）
  temperature: 1.0                    # 采样温度
  top_p: 1.0
  top_k: -1
  response_length: 2048               # 最大生成长度
  tensor_model_parallel_size: 1       # vLLM TP size
  gpu_memory_utilization: 0.5         # vLLM 显存使用率（Hybrid Engine 用较小值）

  # Hybrid Engine 权重同步
  load_format: "dtensor"              # dtensor / hf / megatron

  # 睡眠级别（Hybrid Engine）
  sleep_level: 1                      # 0=不睡眠, 1=释放 KV cache, 2=释放所有显存

  # 多轮交互
  multi_turn:
    enable: false
    max_turns: 5
```

---

## 8. Rollout 数据格式（`schemas.py`）

```python
@dataclass
class GenerationOutput:
    """vLLM 生成结果的结构化表示"""
    sequences: torch.Tensor          # (bs, prompt_len + response_len)
    sequences_str: list[str]         # 解码后的字符串
    rollout_log_probs: torch.Tensor  # (bs, response_len)，每个 token 的 log_prob
    attention_mask: torch.Tensor     # (bs, prompt_len + response_len)
    position_ids: torch.Tensor       # 位置编码

class RolloutRequestOutput:
    """单条请求的输出"""
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]  # 多个候选（n>1 时）
    finish_reason: str               # stop/length/abort
```

---

## 9. Rollout Skip（调试功能）

**文件**：`verl/utils/rollout_skip.py`

在调试时可以跳过真实的 rollout，用随机数据代替，加速调试循环：

```python
class RolloutSkip:
    def wrap_generate_sequences(self):
        """用随机生成替换真实 rollout"""
        original_fn = self.rollout_manager.generate_sequences
        def mock_generate(batch):
            return generate_random_sequences(batch)
        self.rollout_manager.generate_sequences = mock_generate
```
