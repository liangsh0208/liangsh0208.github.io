# SM100 FP8/FP4 GEMM（Blackwell）

> 源码路径：  
> `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh` — 主 kernel  
> `deep_gemm/include/deep_gemm/mma/sm100.cuh` — UMMA/tcgen05 封装  
> `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh` — 内联 tcgen05 PTX  
> `deep_gemm/include/deep_gemm/epilogue/sm100_store_cd.cuh` — 输出写回  
> `csrc/jit_kernels/heuristics/sm100.hpp` — Blackwell 启发式

---

## 1. Blackwell 架构新特性

SM100（Blackwell B200）相比 Hopper 引入了若干根本性变化：

| 特性 | SM90 Hopper | SM100 Blackwell |
|------|-------------|-----------------|
| MMA 指令 | `wgmma`（WGMMA） | `tcgen05.mma`（UMMA） |
| 累加器位置 | 寄存器（RF） | **Tensor Memory（TMEM）** |
| FP4 原生支持 | 无 | 有（MXF8F6F4 family） |
| Scaling factor 格式 | FP32（软件应用） | UE8M0 packed（UTCCP 硬件加速） |
| CTA cluster | 最多 8 SM | 2-SM cta_group（内置） |
| 输出 swizzle | 手动 | UMMA 自动管理 |

---

## 2. UMMA 指令（tcgen05）

Blackwell 的矩阵乘法指令是 `tcgen05.mma`（Tensor Core Generation 5）：

```cuda
// deep_gemm/include/deep_gemm/ptx/tcgen05.cuh

// 单 CTA（cta_group::1），FP8×FP4 或 FP8×FP8
struct SM100_MMA_MXF8F6F4_SS {
    CUTLASS_DEVICE static void fma(
        uint64_t const& desc_a,   // A 矩阵 SMEM descriptor
        uint64_t const& desc_b,   // B 矩阵 SMEM descriptor
        uint32_t const& tmem_c,   // 累加器 Tensor Memory 地址
        uint32_t const& scale_c,  // scale C（1=累加，0=清零）
        uint64_t const& desc,     // MMA shape descriptor
        uint32_t const& tmem_sfa, // SFA Tensor Memory 地址（UTCCP）
        uint32_t const& tmem_sfb  // SFB Tensor Memory 地址（UTCCP）
    ) {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
            " [%0], %1, %2, %3, [%5], [%6], p; \n\t"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
               "r"(desc >> 32), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

// 双 CTA（cta_group::2），2 个 SM 协同计算 256×N block
struct SM100_MMA_MXF8F6F4_2x1SM_SS {
    CUTLASS_DEVICE static void fma(...) {
        asm volatile(
            "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale ...");
    }
};
```

关键差异：`cta_group::2` 模式下，2 个 SM 同时执行相同的 MMA 指令，共同处理 `UMMA_M=256` 行的 block（每个 SM 负责 128 行），需要 Tensor Memory 分配时使用 `Allocator2Sm`。

---

## 3. Tensor Memory（TMEM）

Tensor Memory 是 Blackwell 独有的片上存储层级，位于 SMEM 和 RF 之间，专为存放 MMA 累加器设计：

```cuda
// deep_gemm/include/deep_gemm/common/utils.cuh
// TMEM 列数必须是 32/64/128/256/512 之一
template <uint32_t kNumCols>
CUTLASS_DEVICE constexpr uint32_t get_num_aligned_tmem_cols() {
    if constexpr (kNumCols <=  32) return  32;
    if constexpr (kNumCols <=  64) return  64;
    if constexpr (kNumCols <= 128) return 128;
    if constexpr (kNumCols <= 256) return 256;
    return 512;
}
```

TMEM 布局（在 `sm100_fp8_fp4_gemm_1d1d.cuh` 中）：

```cuda
// TMEM 列分配：累加器 + SFA + SFB
constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
constexpr uint32_t kNumSFATmemCols   = SF_BLOCK_M / 32;  // UE8M0 packed
constexpr uint32_t kNumSFBTmemCols   = SF_BLOCK_N / 32;
constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<
    kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();

// 分配 TMEM（需要 cluster sync）
kNumMulticast > 1 ? cute::cluster_sync() : void();
Allocator tmem_allocator;  // Allocator1Sm 或 Allocator2Sm
const auto tmem_base = tmem_allocator.allocate(kNumTmemCols);
```

---

## 4. UTCCP（Unified Tensor Core Compute Pipeline）

SM100 的 UTCCP 硬件单元能在 MMA 计算期间**自动完成缩放因子的应用**，无需像 SM90 那样在软件中手动 dequantize：

缩放因子格式变化：
- SM90：FP32 格式，每个 scale 4 字节
- SM100：**UE8M0 packed 格式**，4 个 scale 打包进 1 个 `uint32_t`（每个 scale 8 位，无尾数，纯指数）

Python 侧的布局转换：

```python
# deep_gemm/mega/__init__.py
def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    # 重排为 UTCCP 要求的布局：
    # [G, MN/128, 4, 32, packed_SF_K] -> 交换 axis(2,3) -> [G, MN, packed_SF_K]
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
               .transpose(2, 3)
               .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)
```

这种布局使得 128 个连续的元素（一个 UMMA 的 M 粒度）对应的 scale 在内存中是连续的，UTCCP 可以高效读取。

---

## 5. kernel 结构：Non-Epilogue + Epilogue 分离

SM100 kernel 的线程分工不同于 SM90：

```cuda
template <..., uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads, ...>
__launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
void sm100_fp8_fp4_gemm_1d1d_impl(...)
```

- **Non-epilogue 线程**（前 `kNumNonEpilogueThreads` 个）：负责 TMA load 和 UMMA 计算，结果写入 TMEM
- **Epilogue 线程**（后 `kNumEpilogueThreads` 个）：负责从 TMEM 读取结果、应用后处理、通过 TMA 写回 GMEM

Epilogue 线程分工允许 UMMA 和输出写回**并行执行**（流水线 epilogue stages = 2）。

---

## 6. 全布局支持（NT/TN/NN/TT）

SM100 的一大增强是支持所有 4 种矩阵布局，而 SM90 FP8 仅支持 NT（非转置 A，转置 B）：

```cuda
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB, ...>
```

- `cute::UMMA::Major::K`：K-major（行主序 A / 列主序 B，即标准情况）
- `cute::UMMA::Major::MN`：MN-major（列主序 A / 行主序 B）

**Swap A/B 模式**（`kSwapAB=true`）：内部交换 A 和 B 的角色，实现 TN/TT 布局的高效处理，BLOCK_N 固定为 `LAYOUT_AD_M=128`。

---

## 7. 共享内存布局

```cuda
// SMEM sizes（单 stage）
constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);  // UE8M0 packed
constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);

// 对 2-CTA multicast 的处理
constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
```

当 `kNumMulticast=2` 时，每个 CTA 只加载一半的数据（`LOAD_BLOCK_M = BLOCK_M/2`），另一半通过 TMA 2-SM multicast 从 cluster peer 获取，减少 L2 带宽。

---

## 8. SF 粒度参数（kGranKA / kGranKB）

SM100 支持两种缩放粒度：

```cuda
constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
```

| `kGranK` | 含义 | 使用场景 |
|----------|------|----------|
| 32 | 每 32 个 K 通道一个 scale（细粒度） | FP4 矩阵（MXF4 格式） |
| 128 | 每 128 个 K 通道一个 scale（粗粒度） | FP8 矩阵（MXF8 格式） |

当 `kGranK=128` 时，UMMA 一次迭代（`BLOCK_K=128`）只需一个 SF，加载 4 次 stage 才换一次 SF；当 `kGranK=32` 时，每个 BLOCK_K 对应 4 个 SF，每次 stage 就需要加载新 SF。

---

## 9. 性能对比

| 维度 | SM90 FP8（1D1D） | SM100 FP8/FP4（1D1D） |
|------|----------------|----------------------|
| UMMA 大小 | 64×N×32 | 128×N×128（单 CTA）/ 256×N×128（双 CTA） |
| Scale 应用 | 软件（每 K-block 后手动） | 硬件 UTCCP（自动） |
| 输出 | FP32 SMEM → TMA REDUCE | TMEM → Epilogue → TMA store |
| 最大 TFLOPS | ~1550（H800） | 理论更高（B200 峰值 ~4500 TFLOPS）|
| FP4 支持 | 无 | 有（FP8×FP4，throughput ×2） |
