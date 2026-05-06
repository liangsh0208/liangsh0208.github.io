# 分组 GEMM 与 MoE 调度

> 源码路径：  
> `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` — 通用 block 调度器  
> `csrc/jit_kernels/heuristics/sm90.hpp` — SM90 heuristics（含分组逻辑）  
> `csrc/apis/gemm.hpp` — Python API 分发层  
> `tests/test_fp8_fp4.py` — 分组 GEMM 测试

---

## 1. 分组 GEMM 的需求背景

MoE（Mixture of Experts）模型中，每个 token 由 top-k 个专家处理。推理时：

- **Prefilling**（前缀填充）：batch 中所有 token 各自路由到不同专家，每个专家收到的 token 数量不同但已知
- **Decoding**（解码）：CUDA graph 下 CPU 不知道每个专家的 token 数，需要动态处理

DeepGEMM 提供两种分组 GEMM 模式和一种特殊的 K 维分组模式：

| 模式 | 适用场景 | grouped_layout 含义 |
|------|----------|---------------------|
| M-grouped Contiguous | Prefilling forward | 前缀和数组（psum） |
| M-grouped Masked | Decoding（CUDA graph） | 每组实际 M 大小 |
| K-grouped Contiguous | Training backward（wgrad） | 每组 K 大小 |

---

## 2. M-grouped Contiguous（连续布局）

**设计思路**：将所有专家的输入 token 在 M 维度上拼接成一个大张量。`grouped_layout` 是一个 `[M]` 的 int 数组，每个元素表示该行 token 属于的专家 ID（用于反查分组边界）。

```
总输入张量 A: [M_total, K]
专家 0 占 M0 行，专家 1 占 M1 行，...

grouped_layout[row] = expert_id (for standard contiguous)
或
grouped_layout[i] = cumsum_M_i (for psum layout)
```

API 调用：

```python
# 前向推理 contiguous MoE GEMM
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    a,              # (a_tensor, a_scale): FP8 矩阵 + 缩放因子
    b,              # (b_tensor, b_scale): 共享权重矩阵
    d,              # 输出
    grouped_layout, # 每行对应的 expert_id 或前缀和
)
```

**对齐要求**：每个专家的 M 必须对齐到 `get_mk_alignment_for_contiguous_layout()`（等于 BLOCK_M）：

```python
alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
```

### Scheduler 实现（Contiguous 模式）

```cuda
// deep_gemm/include/deep_gemm/scheduler/gemm.cuh
CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    if constexpr (kGemmType == GemmType::MGroupedContiguous) {
        // 通过 grouped_layout 判断当前 m_block 属于哪个 group
        const auto offset = grouped_layout[m_block_idx * BLOCK_M];
        // 偏移量决定这个 block 该从哪个 B 矩阵 row 读取
        return offset * shape_n + block_idx * block_size;
    }
}
```

**Psum layout**（`MGroupedContiguousWithPsumLayout`）：`grouped_layout[i]` 是第 i 组的累计行数，Scheduler 每次推进一组时通过 `current_psum_m - last_psum_m` 算出当前组大小。

---

## 3. M-grouped Masked（Masked 布局）

**设计思路**：在 CUDA graph 下，GPU 不需要等 CPU 知道 token 分布，而是预先给每个专家分配固定上限的 M slots，用 `mask` 指示哪些 slot 是有效的。

```
grouped_layout[expert_i] = num_valid_tokens_for_expert_i
（仅在 kernel 运行时由前驱 kernel 填写，CPU 无感知）
```

```python
# 解码阶段 masked MoE GEMM
deep_gemm.m_grouped_fp8_gemm_nt_masked(
    a,              # (a_tensor, a_scale)
    b,              # (b_tensor, b_scale)  
    d,              # 输出
    grouped_layout, # int32 数组，每个专家实际有效的 M
    expected_m,     # 每个专家分配的 M 上限
)
```

### Scheduler 实现（Masked 模式）

```cuda
CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    if constexpr (kGemmType == GemmType::MGroupedMasked) {
        while (true) {
            if (current_group_idx == kNumGroups) return false;

            // 从 grouped_layout 读取当前组的实际 M
            num_m_blocks = ceil_div(grouped_layout[current_group_idx], BLOCK_M);
            const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
            if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                break;

            current_group_idx++, current_m_cumsum = current_m_block_cumsum;
        }
        get_swizzled_block_idx(...);
    }
}
```

计算有效性检查：

```cuda
CUTLASS_DEVICE bool is_computation_valid(const uint32_t& m_block_idx, ...) const {
    if constexpr (kGemmType == GemmType::MGroupedMasked)
        return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
}
```

---

## 4. K-grouped Contiguous（权重梯度场景）

MoE 训练中，计算权重梯度需要：

```
dW_expert_i = dY_expert_i.T @ X_expert_i
```

不同专家收到不同数量的 token（K 维度不同），但权重矩阵形状 M×N 相同，因此是在 K 维度上分组的 TN GEMM。

```python
# 训练 backward 权重梯度
deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
    a,            # 梯度 dY，拼接：[sum_K, M]
    sfa,          # dY 的缩放因子
    b,            # 激活值 X，拼接：[sum_K, N]
    sfb,          # X 的缩放因子
    d,            # 输出权重梯度：[num_groups, M, N]
    grouped_layout, # int 数组，每组的 K 大小
)
```

### 动态 TMA descriptor 更新

K 分组 GEMM 的挑战是每组的 K 不同，需要动态更新 TMA descriptor：

```cuda
if (kGemmType == GemmType::KGroupedContiguous && last_group_idx != scheduler.current_group_idx) {
    // 原地修改 SMEM 中的 TMA descriptor（更换 global 地址和 stride）
    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_a,
        gmem_a_ptr + current_k_offset * shape_m);
    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a,
        scheduler.current_shape_k, scheduler.current_shape_k);

    // 同步到 GMEM buffer（其他 SM 也需要看到更新）
    *(gmem_tensor_map_a) = *(smem_tensor_map_a);
    ptx::tensor_map_release_gpu();

    // 立即重新获取（fence 确保顺序）
    ptx::tensor_map_acquire_gpu(gmem_tensor_map_a);
}
```

---

## 5. Block Swizzle 调度优化

**问题**：简单按行优先分配 block 会导致 L2 cache 利用率低（相邻 block 不共享 A 或 B 数据）。

**解决方案**：将 block 分成"超级块（super group）"，在一个 super group 内的 block 共享 A 数据（或 B 数据），利用 TMA multicast 和 L2 spatial locality。

```cuda
CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t& block_idx,
                                            uint32_t& m_block_idx, uint32_t& n_block_idx) {
    // kNum1DBlocksPerGroup = 8 或 16（启发式选择，从两者中选 L2 usage 更小的）
    const auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
    const auto group_idx = block_idx / num_blocks_per_group;
    auto first_block_idx = group_idx * kNum1DBlocksPerGroup;

    // kIsMulticastOnA=true：在 N 维度分组（A 数据在 M block 间共享）
    if constexpr (kIsMulticastOnA) {
        m_block_idx = in_group_idx / num_blocks_in_group;
        n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
    } else {
        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        n_block_idx = in_group_idx / num_blocks_in_group;
    }
}
```

`kNum1DBlocksPerGroup` 的选择逻辑（L2 usage 最小化）：

```cpp
// csrc/jit_kernels/heuristics
template <...>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    uint32_t num_best_blocks = 0, min_usage = MAX;
    for (const auto candidate: {8u, 16u}) {
        // kIsMulticastOnA: 在 N 上分组，L2 pressure = N_tiles×N + ceil(SMs/N_tiles)×M
        const auto usage = kIsMulticastOnA ?
            candidate * BLOCK_N + ceil_div(kNumSMs, candidate) * BLOCK_M :
            candidate * BLOCK_M + ceil_div(kNumSMs, candidate) * BLOCK_N;
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}
```

---

## 6. TMA Multicast 与 SM90 奇数行修正

当使用 TMA multicast（2 个 SM 组成 cluster）时，若 N 维度 block 数为奇数，最后一组只有 1 个 SM，无法 multicast：

```cuda
// SM90 动态修正（SM100 不需要，因为 2-CTA 模式是静态的）
#if __CUDA_ARCH__ < 1000
if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
    // 奇数组：前部分用 multicast，最后一个单独处理
    if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
        num_blocks_in_group = num_blocks_in_group ^ 1; // 偶数部分
    } else {
        // 最后一个 block：重映射到不用 multicast 的模式
        in_group_idx -= (num_blocks_in_group ^ 1) * secondary_num_blocks;
        first_block_idx += num_blocks_in_group ^ 1;
        num_blocks_in_group = 1;
    }
}
#endif
```

---

## 7. 分组 GEMM 的 M/K 对齐要求

**M 对齐（Contiguous 布局）**：每个专家的 token 数必须对齐到 BLOCK_M：

```python
# Python 侧：使用 padding 确保对齐
from deep_gemm.utils.math import align
padded_m_per_expert = align(actual_m, block_m)
```

**理论最小对齐**（`get_theoretical_mk_alignment_for_contiguous_layout()`）：考虑到 TMA multicast 的约束，实际最小对齐通常等于 `BLOCK_M × cluster_size`（如 128 × 2 = 256）。

**K 对齐（K-grouped 布局）**：每组的 K 必须是 128 的倍数（对应 FP8 的 per-128-channel scaling）：

```cpp
DG_HOST_ASSERT(ks[i] % 128 == 0);  // csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp
```

---

## 8. Python 侧数据准备（以 Contiguous 为例）

```python
import torch
import deep_gemm

num_experts, n, k = 8, 4096, 4096
alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

# 每个专家随机生成 token 数，并对齐
tokens_per_expert = [align(random.randint(100, 500), alignment) for _ in range(num_experts)]
m_total = sum(tokens_per_expert)

# 拼接 A 矩阵（all experts concatenated on M dim）
a_tensor = torch.zeros(m_total, k, dtype=torch.float8_e4m3fn, device='cuda')
a_scale  = torch.ones(m_total, k // 128, dtype=torch.float, device='cuda')

# 构造 grouped_layout：每行标记所属专家
grouped_layout = torch.zeros(m_total, dtype=torch.int32, device='cuda')
offset = 0
for i, m_i in enumerate(tokens_per_expert):
    grouped_layout[offset:offset+m_i] = i
    offset += m_i

# B 矩阵：各专家共享同一个 [num_experts, n, k] 的权重
b_tensor = torch.zeros(num_experts, n, k, dtype=torch.float8_e4m3fn, device='cuda')
b_scale  = torch.ones(num_experts, n, k // 128, dtype=torch.float, device='cuda')
d = torch.zeros(m_total, n, dtype=torch.float, device='cuda')

deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a_tensor, a_scale), (b_tensor, b_scale), d, grouped_layout)
```
