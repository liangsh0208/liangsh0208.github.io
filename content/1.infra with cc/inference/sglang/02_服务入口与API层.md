---
created: 2026-05-06
---

# 第二章：服务入口与API层

## 一、概述

本章详细分析 SGLang 的 HTTP 服务架构、API 设计以及请求/响应数据结构。SGLang 提供了 OpenAI 兼容的 API 接口，使用户可以无缝迁移现有应用。

## 二、HTTP 服务架构

### 2.1 FastAPI 应用结构

**文件位置**：`python/sglang/srt/entrypoints/http_server.py`

```python
# FastAPI 应用初始化
app = FastAPI(
    title="SGLang",
    description="SGLang inference server",
    version=version.__version__,
)

# 路由注册
@app.post("/generate")
async def generate_request(...): ...

@app.post("/embedding")
async def embedding_request(...): ...

@app.post("/v1/chat/completions")
async def chat_completions(...): ...

@app.post("/v1/completions")
async def completions(...): ...
```

### 2.2 服务启动流程

```python
def launch_server(
    server_args: ServerArgs,
    pipe_finish,
    ...
):
    # 1. 创建引擎
    engine = Engine(**server_args.to_dict())

    # 2. 创建 TokenizerManager
    tokenizer_manager = TokenizerManager(server_args, port_args)

    # 3. 启动 HTTP 服务
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        ...
    )
```

### 2.3 核心路由

| 路由 | 方法 | 功能 |
|------|------|------|
| `/generate` | POST | 通用文本生成 |
| `/generate_batch` | POST | 批量文本生成 |
| `/v1/chat/completions` | POST | OpenAI Chat API |
| `/v1/completions` | POST | OpenAI Completions API |
| `/v1/embeddings` | POST | OpenAI Embeddings API |
| `/v1/models` | GET | 模型列表 |
| `/health` | GET | 健康检查 |
| `/health_generate` | POST | 带生成的健康检查 |
| `/flush_cache` | POST | 清空前缀缓存 |
| `/get_server_info` | GET | 获取服务器信息 |
| `/update_weights` | POST | 在线更新模型权重 |
| `/start_profile` | POST | 启动性能分析 |
| `/stop_profile` | POST | 停止性能分析 |
| `/tokenize` | POST | 分词 |
| `/detokenize` | POST | 反分词 |

### 2.4 gRPC 服务入口

除 HTTP 服务外，SGLang 还提供 gRPC 服务入口，适合高吞吐内部 RPC 场景：

**文件位置**：`python/sglang/srt/grpc/`

```python
# gRPC 服务定义与实现
class SGLangGrpcServer:
    """gRPC 推理服务"""
    def __init__(self, server_args: ServerArgs):
        self.tokenizer_manager = TokenizerManager(server_args)
        # 绑定 gRPC 方法到 tokenizer_manager
```

gRPC 入口与 HTTP 入口共享同一套 `TokenizerManager` 和 `Scheduler`，只是协议层不同。

## 三、请求数据结构

### 3.1 GenerateReqInput

**文件位置**：`python/sglang/srt/managers/io_struct.py`

```python
@dataclasses.dataclass
class GenerateReqInput:
    """文本生成请求输入"""

    # ========== 输入内容 ==========
    text: Optional[Union[str, List[str]]] = None     # 输入文本
    input_ids: Optional[Union[List[int], List[List[int]]]] = None  # 输入 token IDs

    # ========== 采样参数 ==========
    sampling_params: Optional[Union[SamplingParams, Dict]] = None  # 采样参数
    # 或单独指定：
    max_new_tokens: int = 128                        # 最大生成 token 数
    temperature: float = 1.0                         # 温度参数
    top_p: float = 1.0                               # Top-p 采样
    top_k: int = -1                                  # Top-k 采样
    min_p: float = 0.0                               # Min-p 采样
    frequency_penalty: float = 0.0                   # 频率惩罚
    presence_penalty: float = 0.0                    # 存在惩罚
    repetition_penalty: float = 1.0                  # 重复惩罚

    # ========== 停止条件 ==========
    stop: Optional[List[str]] = None                 # 停止词
    stop_token_ids: Optional[List[int]] = None       # 停止 token IDs

    # ========== 约束解码 ==========
    json_schema: Optional[str] = None                # JSON Schema
    regex: Optional[str] = None                      # 正则表达式
    ebnf: Optional[str] = None                       # EBNF 语法

    # ========== 输出控制 ==========
    n: int = 1                                       # 返回结果数量
    return_logprob: bool = False                     # 返回对数概率
    top_logprobs_num: int = 0                        # 返回 top-k 对数概率
    stream: bool = False                             # 流式输出
    skip_special_tokens: bool = True                 # 跳过特殊 token

    # ========== 多模态 ==========
    image_data: Optional[Union[str, List[str]]] = None   # 图片数据 (base64/URL)
    audio_data: Optional[Union[str, List[str]]] = None   # 音频数据

    # ==========LoRA ==========
    lora_path: Optional[str] = None                  # LoRA 适配器路径

    # ========== 会话管理 ==========
    session_id: Optional[str] = None                 # 会话 ID
    regenerate: bool = False                         # 重新生成
```

### 3.2 EmbeddingReqInput

```python
@dataclasses.dataclass
class EmbeddingReqInput:
    """文本嵌入请求输入"""

    text: Optional[Union[str, List[str]]] = None     # 输入文本
    input_ids: Optional[List[int]] = None            # 输入 token IDs
```

### 3.3 TokenizedGenerateReqInput

```python
@dataclasses.dataclass
class TokenizedGenerateReqInput:
    """分词后的生成请求 (内部使用)"""

    # 从 GenerateReqInput 转换
    rid: str                                         # 请求 ID
    origin_input_ids: List[int]                      # 原始输入 token IDs
    sampling_params: SamplingParams                  # 采样参数
    return_logprob: bool                             # 返回对数概率
    stream: bool                                     # 流式输出
    lora_path: Optional[str]                         # LoRA 路径

    # 多模态数据
    image_inputs: Optional[List] = None
    audio_inputs: Optional[List] = None

    # 会话信息
    session_id: Optional[str] = None
```

## 四、响应数据结构

### 4.1 BatchStrOutput

```python
@dataclasses.dataclass
class BatchStrOutput:
    """文本输出响应"""

    rid: str                                    # 请求 ID
    output_str: str                             # 输出文本
    input_ids: List[int]                        # 输入 token IDs
    output_ids: List[int]                       # 输出 token IDs
    finish_reason: Dict                         # 完成原因

    # 对数概率 (如请求)
    input_token_logprobs: Optional[List] = None
    output_token_logprobs: Optional[List] = None
    output_top_logprobs: Optional[List] = None
```

### 4.2 BatchTokenIDOutput

```python
@dataclasses.dataclass
class BatchTokenIDOutput:
    """Token ID 输出响应 (内部使用)"""

    rid: str
    output_ids: List[int]
    finish_reason: Dict
    ...
```

### 4.3 BatchEmbeddingOutput

```python
@dataclasses.dataclass
class BatchEmbeddingOutput:
    """嵌入输出响应"""

    rid: str
    embedding: List[float]                       # 嵌入向量
```

## 五、OpenAI 兼容 API

### 5.1 Chat Completions API

**文件位置**：`python/sglang/srt/entrypoints/openai/`

```python
# 请求格式
{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": true
}

# 响应格式 (非流式)
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 9,
        "total_tokens": 19
    }
}
```

### 5.2 协议转换

```python
class OpenAIServingChat:
    """OpenAI Chat API 适配器"""

    def convert_request(self, request: ChatCompletionRequest) -> GenerateReqInput:
        """将 OpenAI 请求转换为内部格式"""

        # 1. 组装 prompt
        prompt = self._build_prompt(request.messages)

        # 2. 映射采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            stop=request.stop,
            ...
        )

        # 3. 创建内部请求
        return GenerateReqInput(
            text=prompt,
            sampling_params=sampling_params,
            stream=request.stream,
            ...
        )

    def convert_response(self, output: BatchStrOutput) -> ChatCompletionResponse:
        """将内部响应转换为 OpenAI 格式"""
        ...
```

### 5.3 流式响应 (SSE)

```python
async def generate_stream(
    tokenizer_manager: TokenizerManager,
    request: GenerateReqInput,
):
    """流式生成响应"""

    # 1. 发送请求
    async for output in tokenizer_manager.generate_request(request):
        # 2. 构建 SSE 事件
        chunk = {
            "id": output.rid,
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {"content": output.output_str},
                "finish_reason": output.finish_reason
            }]
        }

        # 3. 发送 SSE 格式数据
        yield f"data: {json.dumps(chunk)}\n\n"

    # 4. 结束标记
    yield "data: [DONE]\n\n"
```

## 六、请求处理流程

### 6.1 完整流程图

```
HTTP Request
     │
     ▼
┌─────────────────┐
│  FastAPI Router │  路由分发
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Request Parser  │  参数解析、校验
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Protocol Adapter│  协议转换 (OpenAI → 内部)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│TokenizerManager │  分词、多模态处理
│ generate_request│
└────────┬────────┘
         │ ZMQ PUSH
         ▼
┌─────────────────┐
│   Scheduler     │  调度、批处理
└────────┬────────┘
         │ ZMQ PUSH
         ▼
┌─────────────────┐
│ Detokenizer     │  Token → 文本
└────────┬────────┘
         │ ZMQ PUSH
         ▼
┌─────────────────┐
│Response Builder │  构建响应
└────────┬────────┘
         │
         ▼
HTTP Response (stream/non-stream)
```

### 6.2 generate 端点实现

```python
@app.post("/generate")
async def generate_request(
    request: GenerateReqInput,
    token_manager: TokenizerManager = Depends(get_tokenizer_manager),
):
    """文本生成端点"""

    if request.stream:
        # 流式响应
        return StreamingResponse(
            generate_stream(token_manager, request),
            media_type="text/event-stream",
        )
    else:
        # 非流式响应
        output = await token_manager.generate_request(request)
        return JSONResponse(content=output.to_dict())
```

## 七、多模态输入处理

### 7.1 图片输入

```python
# 请求格式
{
    "text": "Describe this image:",
    "image_data": [
        "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    ]
}

# 或使用 URL
{
    "text": "What's in this image?",
    "image_data": [
        "https://example.com/image.jpg"
    ]
}
```

### 7.2 多模态处理流程

```
image_data (base64/URL)
       │
       ▼
┌─────────────────┐
│  Image Decoder  │  解码图片
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vision Encoder  │  提取视觉特征
│   (CLIP/ViT)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Projector│  投影到语言空间
└────────┬────────┘
         │
         ▼
 image_inputs (tensor)
```

## 八、错误处理

### 8.1 错误响应格式

```python
class ErrorResponse:
    """错误响应"""

    def __init__(self, message: str, code: int, type: str):
        self.error = {
            "message": message,
            "type": type,
            "code": code
        }

# 示例响应
{
    "error": {
        "message": "Model not found",
        "type": "NotFoundError",
        "code": 404
    }
}
```

### 8.2 异常处理中间件

```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": {"message": str(exc), "type": "BadRequest"}}
    )

@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_handler(request: Request, exc: torch.cuda.OutOfMemoryError):
    return JSONResponse(
        status_code=503,
        content={"error": {"message": "GPU out of memory", "type": "ServiceUnavailable"}}
    )
```

## 九、性能优化

### 9.1 异步处理

```python
# 使用 asyncio 实现非阻塞 I/O
async def generate_request(request: GenerateReqInput):
    # 异步发送到 scheduler
    await tokenizer_manager.send_to_scheduler.send_pyobj(request)

    # 异步接收响应
    output = await tokenizer_manager.recv_from_detokenizer.recv_pyobj()
    return output
```

### 9.2 批量请求处理

```python
@app.post("/generate_batch")
async def generate_batch_request(requests: List[GenerateReqInput]):
    """批量生成端点"""

    # 并发发送所有请求
    tasks = [process_request(r) for r in requests]
    outputs = await asyncio.gather(*tasks)
    return outputs
```

## 十、API 使用示例

### 10.1 基本文本生成

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "max_new_tokens": 32,
        "temperature": 0.7,
    }
)
print(response.json()["output_str"])
```

### 10.2 流式输出

```python
import sseclient

response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True,
    },
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    print(event.data)
```

### 10.3 JSON 结构化输出

```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Generate a user profile:",
        "json_schema": """
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"}
            },
            "required": ["name", "age"]
        }
        """
    }
)
```

## 十一、总结

### 11.1 API 层设计要点

1. **OpenAI 兼容**：提供标准 OpenAI API，便于迁移
2. **灵活输入**：支持文本/token ID 两种输入方式
3. **流式支持**：SSE 实现流式输出
4. **多模态**：原生支持图片、音频输入
5. **约束解码**：支持 JSON Schema、正则等约束

### 11.2 核心数据流

```
HTTP Request → GenerateReqInput → TokenizedGenerateReqInput
                                                 ↓
                                          TokenizerManager
                                                 ↓
                                          Scheduler
                                                 ↓
                                          Detokenizer
                                                 ↓
HTTP Response ← BatchStrOutput ← BatchTokenIDOutput
```

---

**上一章**：[服务启动与初始化](01_服务启动与初始化.md)

**下一章**：[请求预处理](03_请求预处理.md)