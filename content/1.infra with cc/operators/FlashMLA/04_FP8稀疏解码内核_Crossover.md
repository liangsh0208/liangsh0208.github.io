---
created: 2026-05-06
---

# FP8 稀疏解码内核：Crossover 技术

> 源码路径：`csrc/sm90/decode/sparse_fp8/`  
> 关键文件：`splitkv_mla.h`、`components/dequant.h`、`components/helpers.h`  
> 技术博客：`docs/20250929-hopper-fp8-sparse-deep-dive.md`

---

## 1. 背景：为什么需要 FP8 KV Cache

DeepSeek-V3.2 将上下文长度从 64K 扩展到 **128K**，单条请求的 KV Cache：

```
576 bytes/元素 × 2 (K+V) × 128 头 × 128K token × 2 bytes/BF16
→ 单请求 KV Cache ≈ 8.72 GiB
```

这导致：
- GPU 显存不足 → OOM 或 batch size 极小
- 有效 batch size 小 → GPU 利用率低

**FP8 量化方案**：将 NoPE 部分（512 elements）量化为 FP8，保留 RoPE 部分（64 elements）为 BF16，显存缩减约 **40%**。

---

## 2. FP8 KV Cache 格式

每个 token 的 KV Cache 占 **656 字节**，结构如下：

```
┌─────────────────────────────────────────────────────────────┐
│  512 bytes：NoPE 部分（float8_e4m3 × 512）                   │
│  ┌────────┬────────┬────────┬────────┐                       │
│  │128 fp8 │128 fp8 │128 fp8 │128 fp8 │ ← 4 个量化块           │
│  └────────┴────────┴────────┴────────┘                       │
├─────────────────────────────────────────────────────────────┤
│  16 bytes：量化 scale 因子（float32 × 4）                    │
│  scale[0] 对应 fp8[0:128]                                    │
│  scale[1] 对应 fp8[128:256]                                  │
│  scale[2] 对应 fp8[256:384]                                  │
│  scale[3] 对应 fp8[384:512]                                  │
├─────────────────────────────────────────────────────────────┤
│  128 bytes：RoPE 部分（bfloat16 × 64，不量化）               │
└─────────────────────────────────────────────────────────────┘
总计：512 + 16 + 128 = 656 bytes/token
```

**量化策略**：
- 粒度：1×128（tile-level），每 128 个元素共享一个 scale
- 格式：FP8 E4M3（最大绝对值 448）
- RoPE 不量化：因为 RoPE 对精度敏感（位置依赖性强）

---

## 3. 去量化实现

H800 没有直接的 FP8→BF16 指令，需要多步转换：

```cpp
// csrc/sm90/decode/sparse_fp8/components/dequant.h

__device__ __forceinline__
bf16x8 cvt_fp8x8_bf16x8(const fp8x8 &inputs, const __nv_bfloat162 &scale_bf162) {
    #define DEQUANT_FP8x4(OUTPUT_BF16_LO, OUTPUT_BF16_HI, FP8x4) \
    { \
        float4 fp32x4 = (float4)(FP8x4);                           /* ① FP8 → FP32 */ \
        OUTPUT_BF16_LO = __float22bfloat162_rn({fp32x4.x, fp32x4.y}) * scale_bf162; /* ② FP32 → BF16 × scale */ \
        OUTPUT_BF16_HI = __float22bfloat162_rn({fp32x4.z, fp32x4.w}) * scale_bf162; \
    }
    
    bf16x8 result;
    DEQUANT_FP8x4(result.a01, result.a23, inputs.lo);  // 处理 4 个 FP8
    DEQUANT_FP8x4(result.a45, result.a67, inputs.hi);  // 再处理 4 个 FP8
    return result;
}
```

**去量化 4 步骤**：
1. FP8 E4M3 → FP32（GPU 内置转换）
2. FP32 → BF16（精度截断）
3. BF16 × BF16 scale（乘以量化 scale）
4. 结果为 BF16，用于后续 WGMMA

每个 token（512 fp8）去量化需要约 **50 时钟周期**，而对应的 GEMM 只需约 **34 时钟周期** → **去量化成为瓶颈**。

---

## 4. Crossover 技术：利用 Distributed Shared Memory

### 4.1 核心思想

MLA 是 MQA 模式：128 个 query 头共享同一个 KV 头。每个 token 的 KV 数据对所有 128 个 query 头是相同的。

每个 CTA（CUDA Thread Block）处理 64 个 query 头，每次 launch 需要两个 CTA（处理 128 头）。这两个 CTA 需要处理**完全相同**的 KV 数据。

**Crossover 方案**：两个 CTA 各自加载一半 KV 数据并去量化，然后通过 DSM 交换，最终每个 CTA 都拥有完整的去量化 KV 数据。去量化工作量减半！

### 4.2 实现步骤

将 CTA 组织为大小为 2 的**集群（Cluster）**：

```
Cluster: {CTA_A, CTA_B}
  CTA_A 负责 query head 0~63
  CTA_B 负责 query head 64~127
```

每个 CTA 执行：

```
Step 1: 从 global memory 加载一半 KV（256 fp8 + 2 scale）
        CTA_A 加载 token[0:256], CTA_B 加载 token[256:512]

Step 2: 去量化（CUDA Core，约 25 cycles per CTA）
        CTA_A: fp8[0:256] → bf16[0:256]
        CTA_B: fp8[256:512] → bf16[256:512]

Step 3: 写到自己的 shared memory（本地部分）

Step 4: 用 st.async 写到对方的 shared memory（DSM 交叉写）
        CTA_A: smem_A[256:512] ← CTA_B 的数据（通过 DSM）
        CTA_B: smem_B[0:256] ← CTA_A 的数据（通过 DSM）

Step 5: 等待 cluster transaction barrier 同步

Step 6: 每个 CTA 的 shared memory 中都有完整 512 bf16 KV，执行 WGMMA
```

等效的去量化时间减少约 **50%**（从 50 cycles → 25 cycles/CTA）。

### 4.3 DSM 写入代码（伪代码）

```cpp
// 集群内跨 CTA 写入
namespace cde = cute::experimental::cluster_data_exchange;

// CTA_A 将自己去量化的 [0:256] 写入 CTA_B 的 shared memory
cde::store_async(
    dst_smem_ptr_in_cta_b,   // 目标：CTA_B 的 shared memory 地址
    src_local_data,           // 源：CTA_A 本地已去量化的数据
    barrier                   // cluster transaction barrier
);

// 等待双方都完成传输
barrier.arrive_and_wait();
```

---

## 5. 稀疏注意力（Sparse Attention）

### 5.1 稀疏 indices

稀疏解码不访问全部 KV cache，而是通过预先计算的 indices 只访问最相关的 tokens：

```python
# indices: (batch_size, seq_len_q, topk)
# indices[i][j][k] = token t 在 paged KV cache 中的位置
#   = page_block_idx * block_size + offset_within_block
indices = torch.randint(0, num_kv_tokens, (batch, seq_q, topk))

out, lse = flash_mla_with_kvcache(
    q,
    k_cache,
    block_table=None,   # 稀疏模式不需要
    cache_seqlens=None,
    head_dim_v=512,
    tile_scheduler_metadata=sched_meta,
    is_fp8_kvcache=True,
    indices=indices,    # 稀疏索引
    topk_length=actual_topk,  # 每个请求的实际 topk
)
```

### 5.2 topk_length 优化

不同请求可能有不同的实际 topk（例如短序列不需要完整 topk），通过 `topk_length` 指定实际访问的索引数量，避免无效的显存访问：

```python
# 如果 q1 实际只需要 1000 个 tokens
topk_length = torch.tensor([2048, 1000, 2048, ...], dtype=torch.int32)
```

---

## 6. Sparse Decode 参数结构体

```cpp
struct SparseAttnDecodeParams {
    // 基础维度
    int b, s_q, h_q, h_kv;
    int d_qk, d_v;
    int topk, page_block_size;
    
    // 输入张量
    cutlass::bfloat16_t* q;    // [b, s_q, h_q, d_qk]
    cutlass::bfloat16_t* kv;   // [num_blocks, page_block_size, d_qk]
                                // 注意：kv 包含 FP8+scale+RoPE 的混合格式
    int* indices;              // [b, s_q, topk]
    int* topk_length;          // [b], 实际 topk，可为 nullptr
    float* attn_sink;          // [h_q], 注意力 sink 权重
    
    // 输出
    float* lse;               // [b, s_q, h_q]
    cutlass::bfloat16_t* out; // [b, s_q, h_q, d_v]
    
    // 额外稀疏 KV（双稀疏源）
    cutlass::bfloat16_t* extra_kv;
    int* extra_indices;
    
    // Split-KV 累积缓冲区
    float* lse_accum;          // [num_splits, s_q, h_q]
    float* o_accum;            // [num_splits, s_q, h_q, d_v]
    DecodingSchedMeta* tile_scheduler_metadata_ptr;
    int* num_splits_ptr;
};
```

---

## 7. attn_sink 机制

某些推理系统使用 **attention sink**（来自 LLM 中注意力聚集在初始 token 的现象）：

```python
# attn_sink: [h_q], float32
# 若提供，最终输出按以下方式缩放：
out_scaled = out * exp(lse) / (exp(lse) + exp(attn_sink))
```

这允许推理系统显式保留对初始 tokens 的固定注意力权重，用于稳定长上下文推理中的数值表现。

---

## 8. 性能数据

H800 SXM5，`b=128, h_q=128, s_q=2, topk=2048`：

| 方案 | 性能 |
|------|------|
| 旧版 FP8 稀疏解码（无 Crossover）| ~250 TFlops |
| 新版 FP8 稀疏解码（Crossover）| ~**410 TFlops** |
| 参考：BF16 密集解码（计算限制）| ~640 TFlops |

当 topk=32768 时，新版内核可达 **~460 TFlops**（overhead 占比降低）。

**等效序列长度分析**：topk=2048 的稀疏解码，运行时间约等于密集解码 seq_len=3000 时的时间。当实际 seq_len > 3000 时，稀疏解码性能优势越来越明显，这体现了 DeepSeek Sparse Attention 算法的效果。
