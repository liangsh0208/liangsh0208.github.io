# TransformerEngine 详解

> NVIDIA TransformerEngine — 面向大模型训练与推理的低精度加速库  
> 官方仓库：https://github.com/NVIDIA/TransformerEngine  
> 适用 GPU：Hopper（H100）/ Ada（L40S）/ Blackwell（B200）

---

## 模块文档索引

| # | 文档 | 内容 |
|---|------|------|
| 01 | [FP8 量化系统](./01_FP8量化系统.md) | Recipe 种类、Scaling Factor 计算、amax 管理、fp8_autocast |
| 02 | [低精度 Tensor 类型](./02_低精度Tensor类型.md) | Float8Tensor、MXFP8、NVFP4、GroupedTensor、Quantizer 接口 |
| 03 | [GEMM 矩阵乘法](./03_GEMM矩阵乘法.md) | cuBLASLt FP8 GEMM、Grouped GEMM、Split Accumulator、融合 Epilogue |
| 04 | [Normalization 归一化](./04_Normalization归一化.md) | LayerNorm/RMSNorm CUDA kernel、融合 FP8 Cast、反向推导 |
| 05 | [Fused Attention](./05_FusedAttention融合注意力.md) | Flash Attention、FP8 Attention、Ring Attention、KV Cache |
| 06 | [PyTorch 模块层](./06_PyTorch模块层.md) | Linear、LayerNormMLP、TransformerLayer、fp8_model_init |
| 07 | [算子图系统 ops](./07_算子图系统_ops.md) | BasicOperation、FusedOperation、OperationFuser 自动融合 |
| 08 | [分布式训练](./08_分布式训练.md) | TP、SP、UserBuffers 通信重叠、FSDP2、Activation Checkpoint |
| 09 | [JAX 前端与 MoE](./09_JAX前端与MoE支持.md) | JAX/Flax 模块、XLA Custom Call、Token Permutation、Expert Parallelism |

---

## 1. 是什么

Transformer Engine（TE）是 NVIDIA 开源的一个**加速 Transformer 模型的 GPU 库**，核心价值在于：

- 在 NVIDIA GPU 上无缝启用 **FP8 低精度训练与推理**，显著提升吞吐、降低显存占用
- 提供一套**高度融合的算子**（fused kernels），减少内存带宽瓶颈
- 同时支持 **PyTorch** 和 **JAX/Flax** 两大主流框架
- 暴露**框架无关的 C++ API**，方便其他深度学习框架接入 FP8

### 支持的数据格式

| 格式 | 精度 | 适用 GPU |
|------|------|---------|
| FP32 / BF16 / FP16 | 标准精度 | Ampere 及以上 |
| FP8 E4M3 / E5M2 | 8-bit 浮点 | Hopper、Ada、Blackwell |
| MXFP8 | Microscaling FP8（OCP MX 规格） | Blackwell |
| NVFP4 | 4-bit 浮点（精度接近 FP16） | Blackwell |

---

## 2. 核心优势

### 2.1 FP8 自动混合精度

类比 PyTorch 的 `torch.autocast`，TE 提供 `fp8_autocast` context manager，用户无需手动管理量化细节：

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

model = te.TransformerLayer(hidden_size=1024, ffn_hidden_size=4096, num_attention_heads=16)

fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = model(input_tensor, attention_mask)
```

TE 在模块内部自动维护每个 Tensor 的 **amax 历史** 和 **scaling factor**，无需用户介入。

### 2.2 关键性能收益

- **FP8 GEMM**：相比 BF16，H100 上矩阵乘法吞吐提升约 **2×**
- **融合算子**：LayerNorm + Linear、Linear + 激活 + Linear 等多步操作合并为单个 CUDA kernel，减少显存读写
- **通信与计算重叠**：AllReduce / AllGather 与 GEMM 同时执行，隐藏通信延迟

---

## 3. 仓库结构

```
TransformerEngine/
├── transformer_engine/       # Python 包主体
│   ├── common/               # C++/CUDA 内核（框架无关）
│   ├── pytorch/              # PyTorch 前端
│   ├── jax/                  # JAX 前端
│   └── debug/                # 数值调试工具
├── 3rdparty/
│   ├── cudnn-frontend/       # cuDNN Graph API 封装
│   ├── cutlass/              # NVIDIA CUTLASS GEMM 模板
│   └── googletest/           # C++ 测试框架
├── tests/                    # 测试（cpp / pytorch / jax）
├── benchmarks/               # 性能基准测试
├── examples/                 # 使用示例
├── docs/                     # Sphinx 文档
├── build_tools/              # 构建辅助脚本
└── qa/                       # QA Pipeline（L0~L3 分级测试）
```

---

## 4. 核心模块详解

### 4.1 `common/` — C++/CUDA 内核层

编译为共享库 `libtransformer_engine.so`，是整个库的算子底层，所有框架前端共用。

#### 主要子模块

| 子模块 | 功能 |
|--------|------|
| `cast/` | 精度转换内核（FP32 ↔ FP8/FP16/BF16），支持 fp8 / mxfp8 / nvfp4 |
| `gemm/` | 矩阵乘法，基于 cuBLASLt + CUTLASS，支持分组 GEMM（MoE） |
| `fused_attn/` | 融合注意力，基于 cuDNN Frontend，支持 FP8/F16/任意序列长度 |
| `normalization/` | LayerNorm、RMSNorm 的前向/反向 CUDA kernel |
| `recipe/` | FP8 Scaling 策略实现（见第 5 节） |
| `comm_gemm_overlap/` | AllReduce/AllGather 与 GEMM 重叠执行（UserBuffers） |
| `transpose/` | 转置 + cast-transpose 融合 |
| `activation/` | GELU / SiLU / SwiGLU 等激活函数内核 |
| `fused_rope/` | Rotary Position Embedding（RoPE）内核 |
| `fused_router/` | MoE Router 内核 |
| `permutation/` | MoE token 排列（permute/unpermute） |
| `hadamard_transform/` | Hadamard 变换（用于随机特征等） |
| `newton_schulz/` | Newton-Schulz 矩阵正交化（用于 Muon 优化器） |
| `triton/` | 部分算子的 Triton JIT 实现 |

#### 公共头文件（C++ API）

所有对外暴露的 C 接口位于 `common/include/transformer_engine/`，关键头文件：

```
transformer_engine.h    — 核心 Tensor 结构体 & 数据类型定义
cast.h                  — 精度转换 API
gemm.h                  — GEMM API
normalization.h         — 归一化 API
fused_attn.h            — 注意力 API
recipe.h                — Scaling recipe API
comm_gemm_overlap.h     — 通信-计算重叠 API
```

---

### 4.2 `pytorch/` — PyTorch 前端

三个抽象层次，从高到低：

#### 层一：完整 Transformer 模块

```
transformer.py         → TransformerLayer（完整 encoder/decoder block）
attention/
├── multi_head_attention.py  → MultiheadAttention（支持 GQA/MQA）
├── dot_product_attention/   → DotProductAttention
└── rope.py                  → RotaryPositionEmbedding
```

#### 层二：可复用子模块（`module/`）

| 模块 | 说明 |
|------|------|
| `Linear` | FP8 感知的线性层，替代 `nn.Linear` |
| `LayerNorm` / `RMSNorm` | 归一化层 |
| `LayerNormLinear` | 融合 LayerNorm + Linear（一次 kernel 完成） |
| `LayerNormMLP` | 融合 LayerNorm + FC1 + 激活 + FC2 |
| `GroupedLinear` | MoE 分组矩阵乘（多个 expert 并行） |

#### 层三：算子图 API（`ops/`）

更底层的**可组合算子系统**，适合构建自定义层：

```
ops/
├── basic/        # 原子操作：linear、layernorm、activation、quantize、
│                 #           all_gather、reduce_scatter、dropout 等
├── fused/        # 自动融合后的复合操作：
│                 #   forward_linear_bias_activation
│                 #   backward_linear_add
│                 #   userbuffers_forward_linear（通信重叠）等
├── fuser.py      # 将 basic ops 序列自动识别并替换为 fused ops
├── op.py         # BasicOperation 基类
└── sequential.py # SequentialOperation（ops 链）
```

#### 低精度 Tensor 类型（`tensor/`）

TE 定义了自己的 Tensor 子类来携带量化元数据：

| 类型 | 格式 | 关键元数据 |
|------|------|-----------|
| `Float8Tensor` | FP8 E4M3/E5M2（per-tensor） | scale, amax |
| `Float8BlockwiseTensor` | FP8 块级缩放 | per-block scale |
| `MXFloat8Tensor` | MXFP8（Blackwell） | scaling block size |
| `NVFloat4Tensor` | NVFP4（Blackwell） | scale |
| `GroupedTensor` | MoE 分组 Tensor | group offsets |

#### 其他功能模块

| 文件 | 功能 |
|------|------|
| `quantization.py` | `fp8_autocast`、`autocast`、`fp8_model_init` context manager |
| `fp8.py` | FP8 amax history / scale 管理 |
| `distributed.py` | 张量并行、序列并行、activation checkpointing |
| `graph.py` | CUDA Graph 支持（`make_graphed_callables`） |
| `optimizers/` | FP8 感知的 multi-tensor Adam/SGD |
| `cpu_offload.py` | CPU offloading（参数卸载到 CPU） |
| `router.py` | MoE Token Router |
| `newton_schulz.py` | Muon 优化器所需矩阵正交化 |
| `export.py` / `onnx_extensions.py` | ONNX 导出支持 |
| `cross_entropy.py` | 并行 Cross Entropy（张量并行友好） |

---

### 4.3 `jax/` — JAX 前端

```
jax/
├── flax/
│   ├── module.py      # Flax Dense、LayerNorm 等基础模块
│   └── transformer.py # TransformerLayer（Flax 实现）
├── attention.py       # 多头注意力
├── dense.py           # 全连接层
├── layernorm.py / layernorm_mlp.py
├── sharding.py        # XLA 分布式张量切分（TP/SP）
├── quantize/          # JAX FP8 量化
├── cpp_extensions/    # C++ 内核 JAX 绑定（XLA custom call）
└── triton_extensions/ # Triton 扩展
```

---

### 4.4 `debug/` — 数值调试工具

用于排查 FP8 训练中的精度问题：
- 数值统计收集（activation / weight 的 amax、分布）
- 精度对比（FP8 vs BF16 逐层对比）
- 自定义调试特性注入

---

## 5. FP8 Scaling Recipe 系统

FP8 的核心挑战是确定每个 Tensor 的 **scaling factor**（将 FP32 值映射到 FP8 范围）。TE 提供多种策略：

### 5.1 DelayedScaling（最常用）

```python
recipe.DelayedScaling(
    margin=0,             # amax 裕量（对数空间偏移）
    interval=1,           # 每隔 N 步更新一次 scale
    fp8_format=recipe.Format.E4M3,  # 或 E5M2
    amax_history_len=16,  # 保留历史 amax 的窗口长度
    amax_compute_algo="max",  # 或 "most_recent"
)
```

**原理**：维护一个 amax 历史窗口，用历史最大绝对值估计当前 scale，每隔若干步更新一次。训练稳定，适合绝大多数场景。

### 5.2 CurrentScaling

每步实时计算当前 Tensor 的 amax 并立即更新 scale。精度更准确，但需要额外的 amax 归约操作。

### 5.3 BlockScaling（FP8 块级缩放）

将 Tensor 切分为若干块（如 128 个元素为一组），每块独立计算 scale。数值范围覆盖更精准，适合权重分布不均匀的情况。

### 5.4 MXFP8（Blackwell 专用）

符合 OCP MX 规格的 Microscaling FP8，硬件原生支持块级量化，无需软件管理 scale。

### 5.5 NVFP4（Blackwell 专用）

4-bit 格式，精度接近 FP16，速度和效率极高。已用于 NVIDIA Nemotron 3 训练。

---

## 6. 通信与计算重叠

在张量并行训练中，AllReduce / AllGather 通常是流水线瓶颈。TE 通过 **UserBuffers** 机制实现通信与 GEMM 的并行执行：

```python
# 初始化 UserBuffers 通信缓冲区
te.initialize_ub(
    shape=[batch * seq_len, hidden_size],
    tp_size=tensor_parallel_size,
    use_fp8=True,
)
```

`userbuffers_forward_linear` / `userbuffers_backward_linear` 等 fused ops 会自动将 GEMM 计算切片，使每片 GEMM 完成后立即触发对应通信，实现接近 100% 的通信隐藏。

---

## 7. 架构总览

```
用户代码 (PyTorch / JAX / Flax)
         │
         ▼
高层模块  TransformerLayer / MultiheadAttention / LayerNormMLP
         │
         ▼
算子层   ops.Sequential / module.Linear / attention.*
         │
         ▼
C++ 绑定  pytorch/csrc/extensions  |  jax/csrc/extensions
         │
         ▼
内核层   libtransformer_engine.so（common/）
         │  cast / gemm / fused_attn / normalization / recipe / ...
         ▼
硬件加速  cuBLASLt + cuDNN Frontend + CUTLASS + NVSHMEM
         │
         ▼
GPU 硬件  Hopper H100 / Ada L40S / Blackwell B200
```

---

## 8. 快速上手

### PyTorch — 替换线性层

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# 原始代码：model = nn.Linear(768, 3072)
model = te.Linear(768, 3072, bias=True)
inp = torch.randn(2048, 768, device="cuda")

fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.E4M3)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model(inp)

out.sum().backward()
```

### PyTorch — 完整 TransformerLayer

```python
layer = te.TransformerLayer(
    hidden_size=1024,
    ffn_hidden_size=4096,
    num_attention_heads=16,
    layer_type="encoder",       # 或 "decoder"
    self_attn_mask_type="causal",
)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output, *_ = layer(hidden_states, attention_mask)
```

### JAX/Flax

```python
import transformer_engine.jax.flax as te_flax

class MyTransformer(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        return te_flax.TransformerLayer(
            hidden_size=1024,
            mlp_hidden_size=4096,
            num_attention_heads=16,
        )(x, mask)
```

---

## 9. 主要集成生态

| 框架/项目 | 集成方式 |
|-----------|---------|
| NVIDIA NeMo | 原生集成，TE 是 NeMo LLM 训练的默认后端 |
| Megatron-LM / Megatron-Core | 深度集成，支持 TP/PP/SP + FP8 |
| HuggingFace Transformers | 通过 `te.TransformerLayer` 替换标准层 |
| DeepSpeed | 部分算子集成 |
| PyTorch FSDP | 完整支持（见 `examples/pytorch/fsdp`） |

---

## 10. 构建与安装

```bash
# 从源码构建（需要 CUDA Toolkit >= 11.8）
git clone --recurse-submodules https://github.com/NVIDIA/TransformerEngine
cd TransformerEngine
pip install .

# 仅安装 PyTorch 扩展
NVTE_FRAMEWORK=pytorch pip install .

# 仅安装 JAX 扩展
NVTE_FRAMEWORK=jax pip install .
```

**关键依赖**：
- CUDA >= 11.8（FP8 需要 Hopper，即 CUDA >= 11.8 + sm_90）
- PyTorch >= 2.1（PyTorch 前端）
- cuDNN >= 8.9（fused attention）
- CUTLASS（submodule，随仓库携带）

---

## 11. 测试体系

```
tests/
├── cpp/               C++ 内核单元测试（GoogleTest）
├── cpp_distributed/   分布式 C++ 测试（comm-GEMM overlap）
├── pytorch/           PyTorch 集成测试（~30 个测试文件）
│   ├── test_sanity.py          基础功能验证
│   ├── test_numerics.py        数值精度验证
│   ├── test_fusible_ops.py     算子融合验证
│   ├── test_quantized_tensor.py FP8 Tensor 验证
│   ├── test_cuda_graphs.py     CUDA Graph 验证
│   ├── attention/              注意力专项测试
│   └── distributed/            分布式专项测试
└── jax/               JAX 集成测试

qa/（CI 分级）
├── L0_*   单元测试（本地可运行）
├── L1_*   集成测试（需要多 GPU）
├── L2_*   大规模分布式测试
└── L3_*   Flash Attention 版本兼容性
```

---

## 参考资料

- [官方文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [GitHub 仓库](https://github.com/NVIDIA/TransformerEngine)
- [FP8 训练论文](https://arxiv.org/abs/2209.05433) — "FP8 Formats for Deep Learning"
- [NVFP4 博客](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- [FP8 RL 训练实践](https://lmsys.org/blog/2025-11-25-fp8-rl/)
