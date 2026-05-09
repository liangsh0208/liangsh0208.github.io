# ms-swift 推理与部署：多后端引擎架构

ms-swift 支持多种推理加速引擎，通过统一的抽象层屏蔽后端差异，使同一份模型和模板代码可在 transformers、vLLM、SGLang、LMDeploy 之间无缝切换。部署层面提供 OpenAI-Compatible 的 FastAPI 服务。

---

## 1. 整体架构

```
                                              ┌──────────────────────┐
                                              │   User Request       │
                                              │   (chat/completion)  │
                                              └──────────┬───────────┘
                                                         │
                              ┌─────────────────────────┼─────────────────────────┐
                              │                         │                         │
                              ▼                         ▼                         ▼
                    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
                    │  SwiftInfer     │      │  SwiftDeploy    │      │  InferClient    │
                    │  (推理管道)      │      │  (部署服务)      │      │  (远程客户端)    │
                    └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
                             │                        │                        │
                             ▼                        ▼                        ▼
                    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
                    │  BaseInferEngine│      │  FastAPI App    │      │  OpenAI API     │
                    │  + InferEngine  │      │                 │      │                 │
                    └────────┬────────┘      └─────────────────┘      └─────────────────┘
                             │
         ┌───────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Transformers    │ │ VllmEngine      │ │ SglangEngine    │ │ LmdeployEngine  │
│ Engine          │ │                 │ │                 │ │                 │
│ (HF generate)   │ │ (vLLM backend)  │ │ (SGLang backend)│ │ (LMDeploy bkd)  │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │                   │
         │          ┌────────┴────────┐         │                   │
         │          ▼                 ▼         │                   │
         │  ┌─────────────┐   ┌─────────────┐  │                   │
         │  │ GRPOVllm    │   │ VllmEngine  │  │                   │
         │  │ Engine      │   │             │  │                   │
         │  └─────────────┘   └─────────────┘  │                   │
```

---

## 2. 抽象层设计

### 2.1 InferEngine 基类

**文件**: `swift/infer_engine/infer_engine.py`

```python
class InferEngine:
    """推理引擎抽象基类"""
    def __init__(self, template):
        self.template = template
        self.max_batch_size = 0  # 0: no limit
    
    def infer(self, infer_requests, request_config, use_tqdm=True, **kwargs):
        """同步推理入口"""
        raise NotImplementedError
    
    def infer_async(self, infer_requests, request_config):
        """异步推理入口"""
        raise NotImplementedError
```

### 2.2 核心抽象能力

| 能力 | 说明 |
|-----|------|
| `infer()` | 批量同步推理 |
| `infer_async()` | 异步流式推理 |
| `_batch_encode()` | 批量编码（ThreadPool 加速）|
| `_post_process()` | 后处理（tool-call/stop-word/usage）|
| `switch_adapter()` | 动态切换 LoRA adapter |

---

## 3. TransformersEngine — 原生推理

**文件**: `swift/infer_engine/transformers_engine.py` (~27KB)

基于 HuggingFace `transformers` 的原生推理引擎，兼容所有 Swift 模型和 adapter。

### 3.1 初始化

```python
class TransformersEngine(InferEngine):
    def __init__(self, model, template=None, adapters=None, max_batch_size=1, ...):
        if isinstance(model, str):
            # 从 model_id 加载
            self.model, processor = self._get_model_processor(model, **kwargs)
        elif isinstance(model, nn.Module):
            # 直接使用传入的模型实例
            self.model = model
        
        super().__init__(template)
        
        # 加载 adapters
        for adapter in self.adapters:
            self._add_adapter(adapter)
        
        self._queue = Queue()
        self._task_pool = {}
```

### 3.2 推理流程

```python
def infer(self, infer_requests, request_config, use_tqdm=True):
    # 1. 批量编码（多线程）
    inputs_list = self._batch_encode(infer_requests)
    
    # 2. 调用 model.generate()
    generation_config = self._get_generation_config(request_config)
    outputs = self.model.generate(**inputs, generation_config=generation_config)
    
    # 3. 解码输出
    responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 4. 后处理（stop-word / tool-call）
    parsed_responses = self._post_process(responses, request_config)
    
    return parsed_responses
```

### 3.3 关键特性

- **多 adapter 支持**: 通过 `add_adapter()` 动态切换 LoRA
- **logits streaming**: 支持流式输出 token-by-token
- **batch inference worker**: 后台线程处理批量请求
- **device map 兼容**: 支持 `device_map='auto'` 的多 GPU 推理

---

## 4. VllmEngine — vLLM 后端

**文件**: `swift/infer_engine/vllm_engine.py` (~38KB)

基于 vLLM 的 `LLMEngine` / `AsyncLLMEngine`，支持PagedAttention和高吞吐推理。

### 4.1 初始化

```python
class VllmEngine(InferEngine):
    def __init__(self, model_id_or_path, template, torch_dtype, ...):
        from vllm import LLM, SamplingParams
        
        # 1. 初始化 vLLM
        self.llm = LLM(
            model=model_id_or_path,
            dtype=torch_dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            ...
        )
        
        # 2. 注册 LoRA adapters
        for adapter in self.adapters:
            self.llm.add_lora(adapter)
    
    def infer(self, infer_requests, request_config, **kwargs):
        sampling_params = self._to_sampling_params(request_config)
        
        # vLLM 批量生成
        outputs = self.llm.generate(prompts, sampling_params)
        
        return self._parse_vllm_outputs(outputs)
```

### 4.2 vLLM 特殊能力

| 能力 | 说明 |
|-----|------|
| **LoRA 并发** | 同时服务多个 LoRA adapter |
| **Tensor Parallel** | `--vllm_tensor_parallel_size` |
| **Data Parallel** | `--vllm_data_parallel_size` |
| **Chunked Prefill** | 长序列分段处理 |
| **Prefix Caching** | 共享前缀缓存 |

### 4.3 GRPOVllmEngine

**文件**: `swift/infer_engine/grpo_vllm_engine.py` (~6KB)

专为 GRPO 训练中的 rollout 生成优化的 vLLM 封装：

```python
class GRPOVllmEngine(VllmEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 针对 group generation 优化的 batching
    
    def generate_for_grpo(self, prompts, num_generations):
        # 对同一 prompt 生成多个 completion
        # 返回 (completions, logprobs)
```

---

## 5. SGLang / LMDeploy 后端

### 5.1 SGLangEngine

**文件**: `swift/infer_engine/sglang_engine.py` (~14KB)

```python
class SglangEngine(InferEngine):
    def __init__(self, model_id_or_path, ...):
        from sglang import Runtime
        self.runtime = Runtime(model_path=model_id_or_path, ...)
    
    def infer(self, infer_requests, request_config, **kwargs):
        # 使用 SGLang 的 RadixAttention 和结构化生成
        ...
```

### 5.2 LmdeployEngine

**文件**: `swift/infer_engine/lmdeploy_engine.py` (~16KB)

```python
class LmdeployEngine(InferEngine):
    def __init__(self, model_id_or_path, ...):
        from lmdeploy import TurbomindEngineConfig, pipeline
        self.pipeline = pipeline(model_id_or_path, ...)
    
    def infer(self, infer_requests, request_config, **kwargs):
        # 使用 LMDeploy 的 turbomind 推理
        ...
```

---

## 6. InferClient — 远程推理客户端

**文件**: `swift/infer_engine/infer_client.py` (~7KB)

OpenAI-Compatible 远程推理客户端：

```python
class InferClient(InferEngine):
    def __init__(self, base_url, api_key=None, ...):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
    
    def infer(self, infer_requests, request_config, **kwargs):
        # 调用远程 /v1/chat/completions
        responses = []
        for req in infer_requests:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=req.messages,
                max_tokens=request_config.max_tokens,
                temperature=request_config.temperature,
                stream=request_config.stream,
            )
            responses.append(resp)
        return responses
```

应用场景：
- 推理与训练分离架构
- 调用商业 API（OpenAI/Claude）做蒸馏
- 多节点推理负载均衡

---

## 7. SwiftInfer — 推理管道

**文件**: `swift/pipelines/infer/infer.py`

```python
class SwiftInfer(SwiftPipeline):
    def __init__(self, args):
        super().__init__(args)
        if args.merge_lora:
            merge_lora(args, device_map='cpu')
        
        # 按 infer_backend 选择引擎
        self.infer_engine = self.get_infer_engine(args, self.template)
    
    @staticmethod
    def get_infer_engine(args, template):
        if args.infer_backend == 'transformers':
            return TransformersEngine(...)
        elif args.infer_backend == 'vllm':
            return VllmEngine(...)
        elif args.infer_backend == 'sglang':
            return SglangEngine(...)
        elif args.infer_backend == 'lmdeploy':
            return LmdeployEngine(...)
```

### 7.1 推理模式

| 模式 | 说明 |
|-----|------|
| **Dataset Inference** | 对数据集批量推理，结果保存到 jsonl |
| **CLI Inference** | 交互式命令行对话 |
| **Generation Config** | 从 args 自动构建 GenerationConfig |

---

## 8. SwiftDeploy — 部署服务

**文件**: `swift/pipelines/infer/deploy.py`

基于 FastAPI 的 OpenAI-Compatible 服务：

```python
class SwiftDeploy(SwiftInfer):
    def run(self):
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.post('/v1/chat/completions')
        async def chat_completions(request: ChatCompletionRequest):
            return await self.infer_engine.infer_async(request)
        
        @app.post('/v1/completions')
        async def completions(request: CompletionRequest):
            return await self.infer_engine.infer_async(request)
        
        @app.post('/v1/embeddings')
        async def embeddings(request: EmbeddingRequest):
            return self.infer_engine.embed(request)
        
        @app.get('/health')
        def health():
            return {'status': 'ok'}
        
        uvicorn.run(app, host=self.args.host, port=self.args.port)
```

### 8.1 API 端点

| 端点 | 功能 |
|-----|------|
| `POST /v1/chat/completions` | 对话补全（OpenAI 兼容）|
| `POST /v1/completions` | 文本补全 |
| `POST /v1/embeddings` | 文本向量化 |
| `GET /v1/models` | 列出可用模型 |
| `GET /health` | 健康检查 |
| `GET /ping` | 存活探测 |

### 8.2 多引擎并发部署

```bash
# vLLM 部署
swift deploy --model Qwen3-4B --infer_backend vllm \
    --vllm_tensor_parallel_size 2 \
    --vllm_data_parallel_size 2

# SGLang 部署
swift deploy --model Qwen3-4B --infer_backend sglang

# LMDeploy 部署
swift deploy --model Qwen3-4B --infer_backend lmdeploy
```

---

## 9. 后处理与工具

### 9.1 Tool-call 解析

```python
# 从模型输出中提取 tool_call 块
def parse_tool_calls(response_text):
    """解析 <tool_call>...</tool_call> 或 JSON 格式"""
    tool_calls = []
    if '<tool_call>' in response_text:
        # XML 格式
        tool_calls = extract_xml_tool_calls(response_text)
    elif '```json' in response_text:
        # JSON 代码块
        tool_calls = extract_json_tool_calls(response_text)
    return tool_calls
```

### 9.2 Stop-word 处理

```python
class StopWordsCriteria(StoppingCriteria):
    """自定义停止词条件"""
    def __init__(self, stop_words, tokenizer):
        self.stop_words = stop_words
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        # 检测是否生成停止词
        for stop_word in self.stop_words:
            if self._check_match(input_ids, stop_word):
                return True
        return False
```

### 9.3 Usage Info 追踪

```python
@dataclass
class UsageInfo:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

---

## 10. 协议与数据结构

**文件**: `swift/infer_engine/protocol.py`

```python
@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: bool = False
    tools: Optional[List[Tool]] = None

@dataclass
class ChatCompletionResponse:
    id: str
    object: str = 'chat.completion'
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
```

---

## 11. 关键代码路径索引

| 功能 | 关键文件/函数 |
|-----|-------------|
| InferEngine 基类 | `swift/infer_engine/infer_engine.py::InferEngine` |
| TransformersEngine | `swift/infer_engine/transformers_engine.py::TransformersEngine` |
| VllmEngine | `swift/infer_engine/vllm_engine.py::VllmEngine` |
| GRPOVllmEngine | `swift/infer_engine/grpo_vllm_engine.py::GRPOVllmEngine` |
| SglangEngine | `swift/infer_engine/sglang_engine.py::SglangEngine` |
| LmdeployEngine | `swift/infer_engine/lmdeploy_engine.py::LmdeployEngine` |
| InferClient | `swift/infer_engine/infer_client.py::InferClient` |
| 推理管道 | `swift/pipelines/infer/infer.py::SwiftInfer` |
| 部署服务 | `swift/pipelines/infer/deploy.py::SwiftDeploy` |
| OpenAI 协议 | `swift/infer_engine/protocol.py` |
| AdapterRequest | `swift/infer_engine/utils.py::AdapterRequest` |
| 生成配置 | `swift/infer_engine/utils.py::prepare_generation_config` |
| 工具解析 | `swift/infer_engine/utils.py` (tool-call parsing) |
| 停止词 | `swift/template/utils.py::StopWordsCriteria` |
