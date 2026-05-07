# FP8 GEMM 核心实现（SM90 Hopper）

> 源码路径：  
> `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh` — 主 kernel  
> `deep_gemm/include/deep_gemm/mma/sm90.cuh` — WGMMA 封装  
> `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` — block 调度器  
> `deep_gemm/include/deep_gemm/common/tma_copy.cuh` — TMA 拷贝工具

---

## 1. 概述

SM90（Hopper）FP8 GEMM 的核心 kernel 是 `sm90_fp8_gemm_1d1d_impl`，命名中的 **1D1D** 表示：

- **A 的 scaling factor（SFA）**：每行（1D on M 维）一个 FP32 scale
- **B 的 scaling factor（SFB）**：每列（1D on N 维）一个 FP32 scale

即每 128 个 K 通道共享一个 scale（per-128-channel 量化），这是 FP8 E4M3 量化的标准格式。

计算公式：

```
D[i,j] += SFA[i] * SFB[j] * sum_k(A[i,k] * B[j,k])
```

> 还有 `sm90_fp8_gemm_1d2d`：SFB 在 K 方向也有细粒度，用于 B 矩阵的 per-column-per-block-k 量化，主要用于 wgrad（权重梯度）。

---

## 2. kernel 模板签名

```cuda
// deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d1d_impl(...)
```

关键模板参数说明：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `SHAPE_M/N/K` | 编译期常数维度（0 表示运行时传入） | 0 或具体值 |
| `BLOCK_M/N/K` | tile 大小（`BLOCK_K` 固定为 128） | M:64/128, N:64~256 |
| `kSwizzleAMode/BMode` | A/B 共享内存 swizzle 字节数 | 64/128 |
| `kNumStages` | 流水线 stage 数 | 3~8 |
| `kNumTMAThreads` | 专门做 TMA load 的线程数 | 128（1 warp group） |
| `kNumMathThreads` | 做 WGMMA 的线程数 | 128（M≤64）或 256（M>64） |
| `kNumTMAMulticast` | TMA multicast 个数（利用 cluster） | 1 或 2 |

---

## 3. 双 warpgroup 设计：生产者-消费者模型

Hopper 的核心创新是 **Warpgroup（128 线程）** + **异步 WGMMA 指令**。kernel 内部将线程分为两组：

```
blockDim = kNumTMAThreads(128) + kNumMathThreads(128 or 256)

warp_idx >= kNumMathThreads/32:  TMA warpgroup（生产者）
warp_idx <  kNumMathThreads/32:  Math warpgroup（消费者）
```

**TMA warpgroup（生产者）**：只有 1 个 warp 实际工作（`elect_one_sync()`），负责：
1. 检查 `empty_barriers` 确认 smem 已被消费
2. 发出 TMA load 指令（异步，不阻塞 CPU/GPU 后续指令）
3. `full_barrier.arrive_and_expect_tx()` 告知 TMA 预期字节数

**Math warpgroup（消费者）**：负责：
1. `full_barriers[stage_idx]->wait(phase)` 等待 TMA 完成
2. 读取 SFA/SFB 缩放因子
3. 发出 WGMMA 指令（异步矩阵乘法）
4. `empty_barrier_arrive()` 释放 smem 给生产者

---

## 4. 多级流水线（软件流水 + 阶段 barrier）

```cuda
// 共享内存布局（按 stage 分槽）
auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
    return smem_buffer + SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE;
});

// barrier 对：full（TMA 已写入）+ empty（math 已消费）
auto full_barriers  = PatternVisitor([&](i) { return barriers + i; });
auto empty_barriers = PatternVisitor([&](i) { return barriers + kNumStages + i; });

// Pipeline stage 和 phase（交替 0/1 防止 ABA 问题）
const auto get_pipeline = [=](const uint32_t& iter_idx) {
    return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1};
};
```

**kNumStages 个 smem 槽**同时持有不同 K 块的数据，形成深度为 kNumStages 的流水线，隐藏 TMA latency（约 300+ 时钟周期）。

---

## 5. TMA（Tensor Memory Accelerator）拷贝

TMA 是 Hopper 专有的高速内存拷贝引擎，绕过 L1 cache 直接操作 SMEM，支持 2D/3D 张量布局和 swizzle。

```cuda
// deep_gemm/include/deep_gemm/common/tma_copy.cuh
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode, ...>
CUTLASS_DEVICE void copy(void const* desc_ptr, ClusterTransactionBarrier* barrier_ptr,
                          dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
                          const uint32_t& num_tma_multicast = 1) {
    if (num_tma_multicast == 1) {
        // 单 CTA：普通 TMA load
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr, ..., smem_ptr, inner_idx, outer_idx);
    } else {
        // 多 CTA multicast：只有 cluster 中 rank 0 发出 TMA，数据广播到所有 CTA
        if (cute::block_rank_in_cluster() == 0) {
            cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, barrier_ptr,
                (1 << num_tma_multicast) - 1, ..., smem_ptr, inner_idx, outer_idx);
        }
    }
}
```

**TMA Multicast** 利用 SM90 的 cluster（最多 8 个 SM 组成一个 cluster，共享 SMEM）：
- 当 2 个 SM 处理相邻的 N tile 时，B 矩阵数据相同，可以只加载一次广播两个 SM
- 减少 L2 带宽消耗约 50%

---

## 6. WGMMA（Warp Group Matrix Multiply Accumulate）

```cuda
// deep_gemm/include/deep_gemm/mma/sm90.cuh
template <int N_, typename MMA>
struct FP8MMA {
    CUTLASS_DEVICE static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b,
                                      float* d, bool scale_d) {
        // 展开为 N/2 个 FMA 指令
        MMA::fma(desc_a, desc_b, d[0], d[1], ..., scale_d ? One : Zero);
    }

    static constexpr int M = 64;   // 每次 WGMMA 处理 64 行
    static constexpr int N = N_;   // N 维度 8~256
    static constexpr int K = 32;   // 每次处理 32 个 K 元素
    static constexpr int kNumAccum = M * N / 128;  // 累加器个数
};
```

**FP8MMASelector** 在编译期根据 BLOCK_N 选择最优 WGMMA 指令：

```cuda
template <int N>
struct FP8MMASelector {
    static constexpr auto select_mma() {
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 192) return MMA_64x192x32_F32E4M3E4M3_SS_TN();
        // ... 8~256，步长 8
    }
};
```

WGMMA 指令格式：`MMA_{M}x{N}x{K}_{accum_type}{a_type}{b_type}_{src_type}_{layout}`
- `SS`：A 和 B 都从 SMEM 读（Shared memory → Shared memory）
- `TN`：A 是 K-major（行主序），B 是 K-major

---

## 7. FP8 缩放因子的处理（promote with scales）

WGMMA 输出是 FP8×FP8 的原始累加值（单位量化，未乘 scale）。在每个 K 块计算完成后立即乘以 scale：

```cuda
// 读取 A 行对应的 scale（2 行共享 warp 内的数据）
auto scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + r_0);  // row r_0
auto scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + r_1);  // row r_1 = r_0 + 8

// 读取 B 列对应的 scale（每 4 个累加器共享一对 scale）
scales_b[i] = ptx::ld_shared(smem_sfb[stage_idx] + i * 8 + col_idx * 2);

// 将原始累加值乘以 scale 后累积到 final_accum
final_accum[i*4+0] += scale_a_0 * scale_b_0 * accum[i*4+0];
final_accum[i*4+1] += scale_a_0 * scale_b_1 * accum[i*4+1];
final_accum[i*4+2] += scale_a_1 * scale_b_0 * accum[i*4+2];
final_accum[i*4+3] += scale_a_1 * scale_b_1 * accum[i*4+3];
```

这种方式在每个 K 块后立即"dequantize"，避免精度损失的累积。

---

## 8. 输出写回（TMA Store）

计算完成后，结果通过 TMA Store 写回到全局内存，支持 REDUCE_ADD（累加到 C 矩阵）：

```cuda
// 先将结果写入 SMEM（float 格式）
ptx::st_shared(smem_d_0 + i * 4, {final_accum[i*4+0], final_accum[i*4+1]});
ptx::st_shared(smem_d_1 + i * 4, {final_accum[i*4+2], final_accum[i*4+3]});

// TMA store fence（确保 SMEM 写入对 TMA 可见）
cute::tma_store_fence();

// 使用 TMA REDUCE_ADD 写回全局内存（支持 D = C + GEMM 的累加语义）
cute::SM90_TMA_REDUCE_ADD_2D::copy(
    &tensor_map_cd, smem_d_0, n_block_idx * BLOCK_N,
    current_group_idx * shape_m + m_block_idx * BLOCK_M + r_0);
cute::tma_store_arrive();
```

---

## 9. SMEM 布局与 swizzle

共享内存的精确布局（以 A 矩阵为例）：

```
smem_buffer:
  [SMEM_TENSOR_MAP_SIZE]   ← K 分组 GEMM 时的 TMA descriptor（128 字节×2）
  [SMEM_D_SIZE]            ← 输出缓冲 BLOCK_M × BLOCK_N × 4 字节
  [kNumStages × SMEM_A]    ← A 矩阵多级缓冲
  [kNumStages × SMEM_B]    ← B 矩阵多级缓冲
  [kNumStages × SMEM_SFA]  ← A 缩放因子（BLOCK_M × 4 字节）
  [kNumStages × SMEM_SFB]  ← B 缩放因子（BLOCK_N × 4 字节，128 字节对齐）
  [2×kNumStages × barrier] ← full/empty barrier 对
```

**Swizzle 模式**决定如何在 bank 维度交错数据：

| swizzle_mode | 说明 | 适用场景 |
|-------------|------|----------|
| 0 | 不 swizzle（INTERLEAVE） | 小 block |
| 64 | 64 字节 swizzle（B64） | 中等 block |
| 128 | 128 字节 swizzle（B128） | 标准 FP8（BLOCK_K=128, 1字节/元素） |

> `SMEM_D_SIZE` 要求对齐到 1024 字节（`__align__(1024)`），这是 TMA swizzle 的硬件要求。

---

## 10. 寄存器数量控制

SM90 每个 SM 有 65536 个 32 位寄存器，warpgroup 间需要精确分配以避免溢出：

```cuda
// TMA warpgroup 使用较少寄存器（主要等待，不做算术）
constexpr uint32_t kNumTMARegisters  = (kNumPipelineUnrolls == 0 ? 40 : 24);
// Math warpgroup 使用较多寄存器（大量累加器）
constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);

// 运行时切换
cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();    // math wg
cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();   // tma wg
```

寄存器分配影响 occupancy（每 SM 同时运行的 warpgroup 数）。DeepGEMM 选择 `kNumMathRegisters=232/240`，优先最大化计算密度而非 occupancy。

---

## 11. PDL（Programmatic Dependent Launch）

```cuda
// 在 scheduler.get_next_block() 之前等待前驱 kernel 完成
cudaGridDependencySynchronize();
```

PDL 允许在 CUDA graph 中重叠计算：当本 kernel 只依赖前驱 kernel 的部分输出时，可以在前驱 kernel 还在运行时就提前启动。`cudaGridDependencySynchronize()` 是 CUDA 12.9+ 的新原语，在真正需要数据时才等待。
