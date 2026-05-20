---
created: 2026-05-06
---

# MQA Logits 与特殊算子

> 源码路径：  
> `deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh` — SM90 MQA logits kernel  
> `deep_gemm/include/deep_gemm/impls/smxx_fp8_mqa_logits.hpp` — 封装层  
> `deep_gemm/include/deep_gemm/impls/smxx_fp8_paged_mqa_logits.hpp` — paged 版本  
> `deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh` — HyperConnection kernel  
> `csrc/apis/einsum.hpp` — Einsum API  
> `tests/test_attention.py` — MQA 测试  
> `tests/test_hyperconnection.py` — HC 测试

---

## 1. FP8 MQA Logits（DeepSeek v3.2 Indexer）

### 1.1 背景

DeepSeek v3.2 引入了"Lightning Indexer"（闪电索引器），使用 MQA（Multi-Query Attention）风格的相关性打分，将每个 query token 与 KV 库中的候选 token 进行相关性评分，输出一个 logit 值。

计算逻辑：

```python
# 对于 query token i 和 kv token j：
kv_j = kv[j, :] * kv_sf[j]        # [head_dim]，FP8 dequantize
out_ij = (q[i, :, :] @ kv_j)      # [num_heads]，multi-head 内积
out_ij = out_ij.relu() * weights[i, :]  # 按 head 加权后 ReLU
out_ij = out_ij.sum()              # 标量：最终 logit
```

输出 `out[i, j]`：形状 `[seq_len, seq_len_kv]`，每个元素是一对 (query, key) 的相关性打分。

### 1.2 Python 接口

```python
# 非 paged 版本（prefilling）
deep_gemm.fp8_mqa_logits(
    q,                    # [seq_len, num_heads, head_dim]，E4M3
    kv,                   # ([seq_len_kv, head_dim], [seq_len_kv])，FP8 + float SF
    weights,              # [seq_len, num_heads]，float
    cu_seq_len_k_start,   # [seq_len]，int32：每个 q token 对应的 kv 起始位置
    cu_seq_len_k_end,     # [seq_len]，int32：每个 q token 对应的 kv 结束位置
    clean_logits=True,    # 是否将未填充的 logits 设为 -inf
)
# 返回：[seq_len, seq_len_kv]

# paged 版本（decoding，KV cache 分页管理）
deep_gemm.fp8_paged_mqa_logits(
    q,
    kv_pages, kv_page_scales,  # 分页的 KV cache
    weights,
    metadata,              # 通过 get_paged_mqa_logits_metadata 预计算
    clean_logits=True,
)
```

### 1.3 Kernel 实现原理

```cuda
// deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,   // 是否压缩 logits 输出
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,  // tile 大小
          uint32_t kNumQStages, uint32_t kNumKVStages,  // 流水线 stage
          ...>
void sm90_fp8_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv, ...)
```

核心思路：

1. **每个 block 处理一段 Q**（`BLOCK_Q` 个 query）和所有有效 KV
2. **Q 只加载一次**到 SMEM，KV 流水 load
3. 对每对 `(Q_block, KV_block)` 执行 `BLOCK_Q×kNumHeads × kHeadDim` 的矩阵乘法（实质上是 FP8 GEMM）
4. 结果乘以 weights 并 ReLU，累加到 logits 输出

```cuda
// 使用与普通 FP8 GEMM 相同的 WGMMA 指令，但矩阵形状是 (BLOCK_Q * kNumHeads) × BLOCK_KV
using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_Q * kNumHeads>::type;

// Q 只加载 kNumQStages 次（Q token 少，多级 stage 浪费 SMEM）
// KV 流水 load（KV token 多）
```

---

## 2. HyperConnection（HC）TF32 Pre-Norm GEMM

### 2.1 背景

HyperConnection 是 DeepSeek 提出的一种新型残差连接方式，代替标准的 LayerNorm + 线性层，通过预归一化 GEMM 实现。

### 2.2 Kernel 特点

```cuda
// deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh
template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,       // K 维度切割数（split-K）
          uint32_t kSwizzleCDMode,   // 输出 swizzle 模式
          uint32_t kNumStages,
          uint32_t kNumMathThreads, uint32_t kNumTMAThreads>
void sm90_tf32_hc_prenorm_gemm_impl(
    const uint32_t shape_m,
    const TmaDescriptor tensor_map_a,  // A 是 BF16（激活值）
    const TmaDescriptor tensor_map_b,  // B 是 FP32（权重）
    const TmaDescriptor tensor_map_d,  // D 是 FP32（输出）
    float* sqr_sum)                    // 输出每行的平方和（用于后续归一化）
```

关键特点：

1. **混合精度**：A 是 BF16，B 是 FP32 → TF32 MMA（使用 `TF32MMASelector`）
2. **同时计算平方和**：在 GEMM 输出的同时累计每行 L2 norm，用于 Pre-Norm
3. **Split-K 支持**：K 维度过大时分割为 kNumSplits 份，每份结果通过原子加累积
4. **Swizzle 输出**：D 矩阵使用特殊 swizzle 布局，适配后续操作的访存模式

### 2.3 Swizzle 实现

HC kernel 有一个独特的 swizzle 函数用于避免 SMEM bank conflict：

```cuda
template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
CUTLASS_DEVICE uint32_t get_swizzled_bank_group_idx(
    const uint32_t& offset, const uint32_t& lane_idx) {
    constexpr uint32_t kGroupsInSwizzleRange = kSwizzleMode / kSwizzleBase;
    const auto bank_group_idx = offset + lane_idx * kGroupsInSwizzleRange;
    auto row = (bank_group_idx / kNumBankGroups);
    auto col = (bank_group_idx % kNumBankGroups);
    col ^= row % kGroupsInSwizzleRange;  // XOR 扰动
    return (row * kNumBankGroups + col) % kGroupsInSwizzleRange;
}
```

这是 XOR swizzle 的变体，确保一个 warp 中的 32 个线程访问不同的 SMEM bank（每个 bank group 16 字节）。

---

## 3. Einsum 算子

```python
# csrc/apis/einsum.hpp
deep_gemm.einsum(a, b, d, op='bmk_bnk_mn')  # batch 化矩阵乘法
deep_gemm.fp8_einsum(a, b, d, op='bmk_bnk_mn')  # FP8 版本
```

支持的 einsum 模式（`bmk_bnk_mn`）：

```
bmk,bnk->mn：对所有 batch 维度 b 求和，结果是 [m, n]
即：d[m, n] = sum_b( sum_k( a[b, m, k] * b[b, n, k] ) )
```

对应的 CUDA kernel（`sm90_bmk_bnk_mn.cuh` / `sm100_bmk_bnk_mn.cuh`）实际上是一个批量 GEMM + 跨 batch 累加，使用 `GemmType::Batched` 调度器模式。

---

## 4. FP8 FP4 版 MQA Logits（SM100）

SM100 引入了 FP4 版本的 MQA logits kernel：

```python
deep_gemm.fp8_fp4_mqa_logits(
    q,         # E4M3 query
    kv,        # (FP4 kv, UE8M0 scales)
    weights,
    cu_seq_len_k_start, cu_seq_len_k_end,
    clean_logits=True,
)
```

对应的 SM100 kernel（`sm100_fp4_mqa_logits.cuh`）使用 UMMA `tcgen05.mma` 指令，能以更高吞吐处理 FP4 KV cache，特别适合长上下文场景。

---

## 5. 辅助工具函数

### 5.1 缩放因子布局变换

```python
# 将 PyTorch tensor 的 SF 变换到 TMA 要求的布局
deep_gemm.transform_sf_into_required_layout(sf_tensor, m, n, k)

# 获取 TMA 对齐后的 tensor（MN-major 布局）
aligned_tensor = deep_gemm.get_mn_major_tma_aligned_tensor(tensor)

# SM100 专用：FP32 → UE8M0 打包 + MN-major 对齐
packed_sf = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_fp32)

# K-grouped 场景的 SF 打包
packed_k_grouped = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(...)
```

### 5.2 TMA 对齐大小查询

```python
# 查询 TMA 要求的对齐字节数
align_bytes = deep_gemm.get_tma_aligned_size(dtype, dim_size)
```

### 5.3 全局配置控制

```python
deep_gemm.set_num_sms(108)           # 限制 SM 使用数量（用于多任务共享）
deep_gemm.get_num_sms()              # 读取当前 SM 数量

deep_gemm.set_tc_util(80)            # 设置 TC 利用率估计（%），影响 heuristic
deep_gemm.get_tc_util()

deep_gemm.set_pdl(True)              # 启用 Programmatic Dependent Launch
deep_gemm.get_pdl()

# M/K 对齐设置（MoE contiguous 布局）
deep_gemm.set_mk_alignment_for_contiguous_layout(128)
alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
theoretical = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()

# Block N 必须是 multiple_of 的倍数
deep_gemm.set_block_size_multiple_of(16)
```

---

## 6. FP8 Skip Head GEMM

```python
deep_gemm.fp8_gemm_nt_skip_head_mid(...)
```

这是一个用于特定 attention 变种的特殊 GEMM，通过 `EpilogueHeadSplits` 在输出时跳过某些 head 段：

```cuda
// deep_gemm/include/deep_gemm/epilogue/transform.cuh
template <uint32_t kLeft, uint32_t kMid, uint32_t kRight>
struct EpilogueHeadSplits: EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        // 在 N 维度索引上插入一段 "gap"（kMid 长度），实现跳过
        return n_idx + (n_idx + kRight) / (kLeft + kRight) * kMid;
    }
};
```

这用于 GQA（Grouped Query Attention）中某些特殊的头部分布变换。

---

## 7. 测试框架

```python
# deep_gemm/testing/__init__.py
from .bench import bench, bench_kineto
from .numeric import calc_diff
from .utils import ignore_env, get_arch_major

# 典型测试模式
t = bench_kineto(
    lambda: deep_gemm.fp8_gemm_nt(a, b, d),
    'gemm_',              # 匹配的 kernel 名字前缀
    suppress_kineto_output=True
)
cublas_t = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a[0], b[0], d), ...)

print(f'DeepGEMM {t*1e6:.1f}us | {2*m*n*k/t/1e12:.0f} TFLOPS | '
      f'{cublas_t/t:.2f}x cuBLAS')
```

`calc_diff` 计算相对误差：

```python
def calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max() / b.abs().max()
```
