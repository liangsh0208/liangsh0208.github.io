# ms-swift 项目架构全景

> **版本**: ms-swift v4.x (main branch)  
> **项目全称**: SWIFT — Scalable lightWeight Infrastructure for Fine-Tuning  
> **社区**: ModelScope (阿里巴巴达摩院)  
> **开源协议**: Apache 2.0  

---

## 1. 项目定位与设计理念

ms-swift 是一个**面向大语言模型（LLM）与多模态大模型（MLLM）的全生命周期训推框架**，覆盖从预训练（Pre-training）、微调（SFT）、人类偏好对齐（RLHF）、模型量化、推理加速到部署服务的完整链路。

### 1.1 核心设计哲学

| 设计原则 | 具体体现 |
|---------|---------|
| **模型无关化** | 不重新实现 Transformer 架构，而是以 **HuggingFace `transformers`** 为底座，通过注册表 + 补丁机制适配 1000+ 模型 |
| **统一模板系统** | 所有模型共享一套 `Template` 数据格式化层，将原始对话数据 → token IDs 的转换与模型架构解耦 |
| **训练范式统一** | SFT / Pretrain / DPO / GRPO / PPO / KTO / Embedding 等任务共享同一套 Pipeline → Trainer 基座 |
| **推理后端可插拔** | 一条命令切换 `transformers` / `vllm` / `sglang` / `lmdeploy` 推理引擎 |
| **分布式透明化** | CLI 自动检测 `NPROC_PER_NODE` / `NNODES` 环境变量，自动包装 `torch.distributed.run` |
| **Megatron 平权化** | 通过 `mcore-bridge` 桥接，使 Megatron-LM 的大规模并行训练（TP/PP/SP/CP/EP）使用体验与 transformers 一致 |

### 1.2 依赖生态

```
                    ┌─────────────────────────────────────┐
                    │           ms-swift (上层框架)          │
                    └──────────────┬──────────────────────┘
                                   │
        ┌────────────┬─────────────┼─────────────┬────────────┐
        ▼            ▼             ▼             ▼            ▼
   transformers    trl           peft        accelerate    modelscope
   (模型基座)    (RLHF基类)    (PEFT方法)   (分布式训练)  (模型/数据集Hub)
        │            │             │             │            │
        └────────────┴─────────────┴─────────────┴────────────┘
                                   │
        ┌────────────┬─────────────┼─────────────┬────────────┐
        ▼            ▼             ▼             ▼            ▼
       vllm        sglang      lmdeploy    deepspeed    megatron-core
     (推理加速)   (推理加速)   (推理加速)   (ZeRO并行)   (大规模并行)
```

**核心版本约束**（取自 `setup.py` 与 `requirements/`）:
- `transformers>=4.33,<5.9`
- `trl>=0.15,<1.0`
- `peft>=0.11,<0.20`
- `datasets>=3.0`
- `modelscope>=1.23`
- `deepspeed>=0.14`
- `vllm>=0.5.1`
- `megatron-core>=0.15` (可选)

---

## 2. 顶层目录结构

```
ms-swift/
├── swift/                      # 核心源码（见第3节详细展开）
│   ├── cli/                    # CLI入口点
│   ├── model/                  # 模型注册、加载、Patch
│   ├── template/               # 模板系统（数据格式化）
│   ├── dataset/                # 数据集加载与预处理
│   ├── tuners/                 # PEFT/轻量微调实现
│   ├── trainers/               # 基础训练器（SFT/Pretrain/Embedding/Reranker）
│   ├── rlhf_trainers/          # 对齐训练器（DPO/GRPO/PPO/KTO等）
│   ├── infer_engine/           # 多后端推理引擎
│   ├── pipelines/              # 高层任务管道
│   ├── sequence_parallel/      # 序列并行（Ulysses / Ring Attention）
│   ├── megatron/               # Megatron-LM 集成子系统
│   ├── rewards/                # 奖励模型插件（ORM / PRM）
│   ├── rollout/                # RL Rollout环境（multi-turn / gym）
│   └── utils/                  # 通用工具
├── examples/                   # 端到端使用示例（train/infer/deploy/export/eval）
│   ├── train/                  # 训练示例（按任务/技术细分）
│   ├── infer/                  # 推理示例（按后端细分）
│   ├── megatron/               # Megatron训练示例
│   └── deploy/                 # 部署示例
├── docs/                       # 官方文档（ReadTheDocs）
├── tests/                      # 单元测试与集成测试
├── requirements/               # 分模块依赖文件
└── setup.py                    # 包定义，入口: `swift`, `megatron`
```

---

## 3. 核心模块职责速览

### 3.1 CLI 层 (`swift/cli/`)

`swift/cli/main.py::cli_main()` 是所有命令的中央调度器：

```python
ROUTE_MAPPING = {
    'pt': 'swift.cli.pt',           # 预训练
    'sft': 'swift.cli.sft',         # 监督微调
    'rlhf': 'swift.cli.rlhf',       # 对齐训练
    'infer': 'swift.cli.infer',     # 推理
    'deploy': 'swift.cli.deploy',   # 部署
    'export': 'swift.cli.export',   # 导出/量化
    'eval': 'swift.cli.eval',       # 评估
    'app': 'swift.cli.app',         # Web UI
    'merge-lora': 'swift.cli.merge_lora',
    'sample': 'swift.cli.sample',
    'rollout': 'swift.cli.rollout',
}
```

**自动分布式包装**: 当检测到环境变量 `NPROC_PER_NODE` 或 `NNODES` 时，CLI 自动将命令包裹为 `python -m torch.distributed.run` 执行。

### 3.2 Pipeline 层 (`swift/pipelines/`)

Pipeline 是 CLI 与底层 Trainer 之间的**编排层**，负责：
1. 解析参数（`Arguments` dataclass）
2. 加载模型与 Processor
3. 初始化 Template
4. 加载与预处理数据集
5. 准备 Tuner（LoRA/Adapter/Full）
6. 实例化 Trainer 并启动训练

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│ CLI (swift) │───▶│  SwiftPipeline│───▶│   Trainer    │───▶│  Checkpoint │
│  sft/rlhf   │    │ Sft/RLHF/Infer│    │ + SwiftMixin │    │   /Deploy   │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

核心 Pipeline 类：

| Pipeline | 文件 | 用途 |
|---------|------|------|
| `SwiftSft` | `swift/pipelines/train/sft.py` | SFT / Pretrain 训练管道 |
| `SwiftRLHF` | `swift/pipelines/train/rlhf.py` | RLHF 训练管道（DPO/GRPO/PPO等）|
| `SwiftInfer` | `swift/pipelines/infer/infer.py` | 推理管道 |
| `SwiftDeploy` | `swift/pipelines/infer/deploy.py` | 部署服务（FastAPI）|
| `SwiftExport` | `swift/pipelines/export/` | 导出/合并/量化 |

### 3.3 模型层 (`swift/model/`)

模型层是 ms-swift 的**元数据驱动注册系统**，核心设计：

- **`ModelMeta`** (`model_meta.py`): 描述一个模型家族的元数据（model_type, template, loader, arch等）
- **`MODEL_MAPPING`** (`register.py`): 全局 `Dict[str, ModelMeta]` 注册表
- **`ModelLoader`**: 统一的模型加载入口，内部调用 `AutoModelForCausalLM.from_pretrained()`
- **`patcher.py`**: 动态补丁，解决 device_map、dummy model、seq_cls head 等问题
- **`model_arch.py`**: 定义 `ModelKeys` / `MultiModelKeys`，用于 LoRA target 选择和 freeze 控制

详见 [`model-system.md`](./04-model-system.md)。

### 3.4 模板层 (`swift/template/`)

`Template` (`swift/template/base.py`) 是整个框架的**数据格式化中枢**：

- 将原始对话（system/user/assistant/tool_call）编码为模型可接受的 token 序列
- 支持多模态输入：image / video / audio / bbox
- 根据训练模式生成不同的 data collator：`causal_lm` / `rlhf` / `kto` / `seq_cls` / `embedding` / `reranker`
- 支持 `padding_free` / `packing` / `sequence_parallel`

详见 [`data-system.md`](./05-data-system.md)。

### 3.5 PEFT 层 (`swift/tuners/`)

轻量微调方法的统一封装：

- **`SwiftModel`** (`base.py`): 统一包装器，支持多 adapter 切换、merge/unmerge
- **原生实现**: LoRA (`lora.py`)、Adapter (`adapter.py`)、Prompt Tuning (`prompt.py`)、ReFT (`reft.py`)、LLaMA-Pro (`llamapro.py`)、LongLoRA (`longlora/`)
- **PEFT 桥接** (`peft.py`): AdaLoRA、BOFT、LoHa、LoKr、OFT、Vera 等通过 HuggingFace `peft` 间接支持

详见 [`tuners-peft.md`](./06-tuners-peft.md)。

### 3.6 训练器层 (`swift/trainers/` & `swift/rlhf_trainers/`)

训练器继承体系：

```
HuggingFace Trainer
        │
        ├─── SwiftMixin ───┬─── Seq2SeqTrainer (causal_lm)
        │                    ├─── Trainer (seq_cls)
        │                    ├─── EmbeddingTrainer
        │                    └─── RerankerTrainer
        │
        ├─── RLHFTrainerMixin ─┬─── DPOTrainer
        │                      ├─── GRPOTrainer
        │                      ├─── PPOTrainer
        │                      ├─── KTOTrainer
        │                      ├─── ORPOTrainer
        │                      ├─── CPOTrainer
        │                      ├─── GKDTrainer
        │                      └─── RewardTrainer
```

`SwiftMixin` (`swift/trainers/mixin.py`, ~61KB) 是所有训练器的**公共能力注入器**：
- 数据 collator 构建（从 Template 自动生成）
- 梯度累积修正、Loss Scaling
- DeepSpeed Zero3 参数 gather 补丁
- Flash Checkpoint（DLRover 集成）
- 序列并行 Hook 注入

详见 [`training-system.md`](./07-training-system.md) 与 [`rlhf-trainers.md`](./08-rlhf-trainers.md)。

### 3.7 推理引擎层 (`swift/infer_engine/`)

统一推理抽象：

| 引擎 | 文件 | 后端 |
|-----|------|------|
| `TransformersEngine` | `transformers_engine.py` | 原生 HF `model.generate` |
| `VllmEngine` | `vllm_engine.py` | vLLM `LLMEngine` / `AsyncLLMEngine` |
| `GRPOVllmEngine` | `grpo_vllm_engine.py` | vLLM 专用 GRPO rollout |
| `SglangEngine` | `sglang_engine.py` | SGLang 后端 |
| `LmdeployEngine` | `lmdeploy_engine.py` | LMDeploy 后端 |
| `InferClient` | `infer_client.py` | 远程 OpenAI-Compatible 客户端 |

详见 [`inference-system.md`](./09-inference-system.md)。

### 3.8 Megatron 子系统 (`swift/megatron/`)

完整的 Megatron-LM 集成，独立 CLI 入口 `megatron`：
- 自己的参数体系 (`swift/megatron/arguments/`)
- 自己的 Pipeline (`swift/megatron/pipelines/train/`)
- 自己的 Trainer (`swift/megatron/trainers/`)
- 并行策略：TP / PP / SP / CP / ETP / EP / VPP
- 支持 CPT / SFT / GRPO / DPO / KTO / Reward / Embedding / Reranker

详见 [`megatron-training.md`](./03-megatron-training.md)。

---

## 4. 核心数据流全景图

### 4.1 Transformers 训练全流程

```
User CLI
    │
    ▼
swift sft --model Qwen3-4B --dataset xxx ...
    │
    ▼
swift/cli/main.py::cli_main()
    │
    ▼  (自动包裹 torch.distributed.run)
swift/cli/sft.py ──▶ swift/pipelines/sft_main()
    │
    ▼
SwiftSft.__init__(args)
    ├── _prepare_model_tokenizer()
    │   └── args.get_model_processor()
    │       └── swift/model/register.py::ModelLoader.get_model()
    │           └── AutoModelForCausalLM.from_pretrained() + patcher
    │
    ├── _prepare_template()
    │   └── args.get_template(processor)
    │       └── swift/template/register.py (按 model_type 匹配 Template)
    │
    ├── _prepare_dataset()
    │   ├── swift/dataset/loader.py::load_dataset()
    │   ├── EncodePreprocessor (template.encode)
    │   └── LazyLLMDataset / PackingDataset (可选)
    │
    ├── TunerMixin.prepare_model()  (若使用 LoRA/Adapter)
    │   └── swift/tuners/base.py::Swift.prepare_model()
    │
    └── TrainerFactory.get_trainer_cls(args)
        └── 按 task_type 路由到 Seq2SeqTrainer / DPOTrainer / GRPOTrainer...
            └── trainer.train()
```

详见 [`transformers-training.md`](./02-transformers-training.md)。

### 4.2 RLHF 训练流程（以 GRPO 为例）

```
SwiftRLHF._prepare_model_tokenizer()
    ├── policy model (可训练)
    ├── ref model (冻结，用于 KL 散度)
    ├── reward model(s) (冻结，用于打分)
    └── vllm_client (可选，异步生成 completions)

GRPOTrainer.__init__()
    ├── 继承自: RolloutTrainerMixin + SwiftMixin + HFGRPOTrainer
    ├── _prepare_algorithm_params()  # 算法超参
    ├── prepare_rollout()            # vLLM / transformers 生成初始化
    ├── _prepare_rewards()           # 注册 reward_funcs / reward_models
    └── _prepare_liger_loss()        # Liger-Kernel 加速

GRPOTrainer._prepare_inputs()
    ├── 生成 completions (通过 vLLM 或 TransformersEngine)
    ├── 计算 per_token_logps (策略 vs 参考)
    ├── 调用 reward_funcs / reward_models 打分
    └── 组装带 advantage 的训练 batch

GRPOTrainer.compute_loss()
    └── GRPO loss = advantage * (logpi_policy - logpi_ref) + KL_penalty
```

### 4.3 推理/部署流程

```
SwiftInfer.__init__()
    ├── prepare_model_template() / get_infer_engine()
    └── 按 infer_backend 选择引擎:
        ├── transformers ──▶ TransformersEngine(model.generate)
        ├── vllm ──▶ VllmEngine(LLMEngine)
        ├── sglang ──▶ SglangEngine
        └── lmdeploy ──▶ LmdeployEngine

SwiftDeploy (FastAPI)
    ├── /v1/chat/completions  (OpenAI-Compatible)
    ├── /v1/completions
    └── /v1/embeddings
```

---

## 5. 训练范式全景

### 5.1 监督学习

| 任务 | Pipeline | Trainer | 说明 |
|-----|----------|---------|------|
| **SFT** | `SwiftSft` | `Seq2SeqTrainer` | 指令微调 |
| **Pretrain** | `SwiftSft` | `Seq2SeqTrainer` | 预训练/继续预训练 |
| **Seq Cls** | `SwiftSft` | `Trainer` | 序列分类 |
| **Embedding** | `SwiftSft` | `EmbeddingTrainer` | Embedding 模型训练 |
| **Reranker** | `SwiftSft` | `RerankerTrainer` | 重排序模型训练 |

### 5.2 人类偏好对齐（RLHF）

| 算法 | Trainer | 说明 |
|-----|---------|------|
| **DPO** | `DPOTrainer` | Direct Preference Optimization |
| **IPO** | `DPOTrainer` (loss_type=ipo) | Identity Preference Optimization |
| **SimPO** | `DPOTrainer` (loss_type=simpo) | Simple Preference Optimization |
| **ORPO** | `ORPOTrainer` | Odds Ratio Preference Optimization |
| **KTO** | `KTOTrainer` | Kahneman-Tversky Optimization |
| **CPO** | `CPOTrainer` | Contrastive Preference Optimization |
| **PPO** | `PPOTrainer` | Proximal Policy Optimization |
| **GRPO** | `GRPOTrainer` | Group Relative Policy Optimization |
| **DAPO** | `GRPOTrainer` (algorithm=dapo) | Decoupled Advantage Preference Optimization |
| **GSPO** | `GRPOTrainer` (algorithm=gspo) | Group-based Self-Play Optimization |
| **SAPO** | `GRPOTrainer` (algorithm=sapo) | Self-Adaptive Preference Optimization |
| **CISPO** | `GRPOTrainer` (algorithm=cispo) | Contrastive Iterative Self-Play Optimization |
| **RLOO** | `GRPOTrainer` (algorithm=rloo) | REINFORCE Leave-One-Out |
| **Reinforce++** | `GRPOTrainer` (algorithm=reinforce++) | REINFORCE with baseline |
| **CHORD** | `GRPOTrainer` (algorithm=chord) | Collaborative Hindsight Optimization for RL |
| **GKD** | `GKDTrainer` | Generalized Knowledge Distillation |
| **Reward Model** | `RewardTrainer` | 奖励模型训练 |

### 5.3 轻量微调（PEFT）方法

| 方法 | 实现位置 | 说明 |
|-----|---------|------|
| LoRA | `swift/tuners/lora.py` | 原生实现，支持 LoRA+ |
| QLoRA | 通过 bitsandbytes + LoRA | 4-bit/8-bit 量化训练 |
| DoRA | 通过 peft 桥接 | Weight-Decomposed LoRA |
| Adapter | `swift/tuners/adapter.py` | 原生 Bottleneck Adapter |
| Prompt Tuning | `swift/tuners/prompt.py` | 原生 Prompt Tuning |
| Prefix Tuning | 通过 peft 桥接 | 前缀调优 |
| ReFT | `swift/tuners/reft.py` | Representation Fine-Tuning |
| LLaMA-Pro | `swift/tuners/llamapro.py` | 扩展层深度调优 |
| LongLoRA | `swift/tuners/longlora/` | 长上下文 LoRA |
| NEFTune | `swift/tuners/neftune.py` | Noisy Embedding Fine-Tuning |
| AdaLoRA / BOFT / LoHa / LoKr / OFT / Vera | `swift/tuners/peft.py` | 通过 HuggingFace `peft` 桥接 |

---

## 6. 分布式与并行策略

```
┌─────────────────────────────────────────────────────────────────┐
│                     分布式训练矩阵                               │
├──────────────┬──────────────────────────────────────────────────┤
│ 数据并行      │ DDP (torch.nn.parallel.DistributedDataParallel)   │
│              │ DeepSpeed ZeRO2 / ZeRO3                          │
│              │ FSDP / FSDP2                                     │
├──────────────┼──────────────────────────────────────────────────┤
│ 序列并行      │ Ulysses (`swift/sequence_parallel/ulysses.py`)  │
│              │ Ring Attention (`swift/sequence_parallel/...`)   │
├──────────────┼──────────────────────────────────────────────────┤
│ Megatron并行  │ TP (Tensor Parallel)                             │
│              │ PP (Pipeline Parallel)                           │
│              │ SP (Sequence Parallel)                           │
│              │ CP (Context Parallel)                              │
│              │ ETP (Expert Tensor Parallel)                       │
│              │ EP (Expert Parallel)                             │
│              │ VPP (Virtual Pipeline Parallel)                  │
└──────────────┴──────────────────────────────────────────────────┘
```

详见 [`distributed-training.md`](./10-distributed-training.md)。

---

## 7. 推理加速与部署

| 能力 | 实现 |
|-----|------|
| 多后端推理 | transformers / vLLM / SGLang / LMDeploy |
| LoRA 动态加载 | vLLM `AdapterRequest` 支持多 LoRA 并发 |
| 批量推理 | `InferEngine.infer()` 自动批处理 |
| 流式输出 | 所有引擎支持 `stream=True` |
| OpenAI API | `SwiftDeploy` 提供 FastAPI 服务，兼容 `/v1/chat/completions` |
| 工具调用 | `InferEngine` 内置 tool-call 解析 |

详见 [`inference-system.md`](./09-inference-system.md)。

---

## 8. 量化与导出

| 技术 | 说明 |
|-----|------|
| **LoRA Merge** | 将 LoRA 权重合并回基模型 |
| **GPTQ** | 4-bit 权重量化 |
| **AWQ** | Activation-aware Weight Quantization |
| **FP8** | NVIDIA Hopper 架构 FP8 量化 |
| **BNB** | BitsAndBytes 8-bit/4-bit |
| **Ollama 导出** | 导出为 Ollama 格式 |

详见 [`export-deploy.md`](./11-export-deploy.md)。

---

## 9. 文档索引

| 文档 | 主题 |
|-----|------|
| `ms-swift-overview.md` | 本文档：项目架构全景 |
| [`model-system.md`](./04-model-system.md) | 模型注册、加载与架构映射 |
| [`tuners-peft.md`](./06-tuners-peft.md) | PEFT 方法实现 |
| [`data-system.md`](./05-data-system.md) | 模板系统与数据管道 |
| [`training-system.md`](./07-training-system.md) | 基础训练器与 Mixin 架构 |
| [`rlhf-trainers.md`](./08-rlhf-trainers.md) | RLHF 对齐训练算法 |
| [`inference-system.md`](./09-inference-system.md) | 推理引擎与部署 |
| [`distributed-training.md`](./10-distributed-training.md) | 分布式与序列并行 |
| [`export-deploy.md`](./11-export-deploy.md) | 导出量化与部署 |
| [`transformers-training.md`](./02-transformers-training.md) | Transformers 训练全流程 |
| [`megatron-training.md`](./03-megatron-training.md) | Megatron 训练全流程 |
