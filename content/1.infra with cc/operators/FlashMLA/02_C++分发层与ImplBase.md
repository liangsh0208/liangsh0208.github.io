---
created: 2026-05-06
---

# C++ 分发层与 ImplBase 架构

> 源码路径：`csrc/api/`  
> 关键文件：`api.cpp`、`common.h`、`dense_decode.h`、`sparse_decode.h`、`sparse_fwd.h`、`params.h`

---

## 1. 整体设计思路

FlashMLA 的 C++ 分发层承担三项职责：

1. **Python → CUDA 桥接**：通过 pybind11 将 PyTorch Tensor 转换为 CUDA kernel 可直接读写的原始指针和参数结构体。
2. **多内核实现的统一分发**：基于 GPU 架构（SM90/SM100）、头数（64/128）、head_dim（512/576）等特性，动态选择最合适的 CUDA kernel 实现。
3. **运行时特性验证**：保证所选实现类支持当前请求的全部特性组合，防止调用不兼容的内核。

---

## 2. 基础类型与架构检测

### 2.1 类型别名（defines.h）

```cpp
// csrc/defines.h
using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

// WGMMA 相关
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::NamedBarrier;

// 向量化数据类型（用于向量化内存访问）
struct float8 {
    float2 a01, a23, a45, a67;   // 8 个 float
};

struct bf16x8 {
    __nv_bfloat162 a01, a23, a45, a67;   // 8 个 bfloat16（4 个 bfloat162）
};
```

### 2.2 架构检测（common.h）

```cpp
// csrc/api/common.h
struct Arch {
    int major;
    int minor;
    int num_sms;
    cudaDeviceProp* device_prop;

    Arch() {
        device_prop = at::cuda::getCurrentDeviceProperties();
        major = device_prop->major;
        minor = device_prop->minor;
        num_sms = device_prop->multiProcessorCount;
    }

    bool is_sm90a() const { return major == 9 && minor == 0; }   // Hopper
    bool is_sm100f() const { return major == 10; }                // Blackwell
};
```

在每次 kernel 调用时，分发层都会先创建 `Arch` 对象查询当前 GPU 架构，再据此选择对应的 kernel 实现。

---

## 3. Dispatch 宏系统

`common.h` 定义了一套模板 dispatch 宏，将运行时参数转换为编译期常量，从而让 CUDA kernel 可以针对不同配置做模板特化：

### 3.1 头数 Dispatch

```cpp
#define DISPATCH_NUM_HEADS(NUM_HEADS, CONSTEXPR_NAME, ...) \
    [&] () { \
        if (NUM_HEADS == 128) { \
            static constexpr int CONSTEXPR_NAME = 128; \
            return __VA_ARGS__(); \
        } else if (NUM_HEADS == 64) { \
            static constexpr int CONSTEXPR_NAME = 64; \
            return __VA_ARGS__(); \
        } else { \
            TORCH_CHECK(false, "Unsupported num_heads_q: ", NUM_HEADS); \
        } \
    } ();
```

### 3.2 Head Dim Dispatch

```cpp
#define DISPATCH_HEAD_DIM(HEAD_DIM, CONSTEXPR_NAME, ...) \
[&] () { \
    if (HEAD_DIM == 576) {                         // DeepSeek-V3.x：NoPE(512) + RoPE(64)
        static constexpr int CONSTEXPR_NAME = 576; \
        return __VA_ARGS__(); \
    } else if (HEAD_DIM == 512) {                  // MODEL1
        static constexpr int CONSTEXPR_NAME = 512; \
        return __VA_ARGS__(); \
    } else { \
        TORCH_CHECK(false, "Unsupported head_dim_qk: ", HEAD_DIM); \
    } \
} ();
```

### 3.3 模型类型 Dispatch

```cpp
#define DISPATCH_MODEL_TYPE(MODEL_TYPE, CONSTEXPR_NAME, ...) \
[&] () { \
    if (MODEL_TYPE == ModelType::V32) {              // DeepSeek-V3.2：d_qk=576
        static constexpr ModelType CONSTEXPR_NAME = ModelType::V32; \
        return __VA_ARGS__(); \
    } else if (MODEL_TYPE == ModelType::MODEL1) {    // MODEL1：d_qk=512
        static constexpr ModelType CONSTEXPR_NAME = ModelType::MODEL1; \
        return __VA_ARGS__(); \
    } ... \
} ();
```

这些宏的用法示例：

```cpp
// 稀疏解码时，根据 h_q 和 model_type 分发
DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
    DISPATCH_NUM_HEADS(params.h_q, NUM_HEADS, [&]() {
        sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<
            MODEL_TYPE, NUM_HEADS  // 编译期常量，影响寄存器分配和 tile 大小
        >(params);
    });
});
```

---

## 4. 参数结构体（params.h）

所有 CUDA kernel 的参数通过专用的 POD 结构体传递，避免复杂的 C++ 对象穿越 CUDA kernel 边界。

### 4.1 密集解码参数

```cpp
// csrc/params.h
struct DenseAttnDecodeParams {
    using index_t = int64_t;
    
    // 维度参数
    int b;           // batch size
    int s_q;         // query 序列长度（MTP 开启时 > 1）
    int q_seq_per_hk;  // = h_q / h_k * s_q（合并后的 query 数）
    int d, d_v;      // K 维度（576）和 V 维度（512）
    int h_q, h_k;    // Q 头数和 KV 头数
    int num_blocks;
    int q_head_per_hk;  // = h_q / h_k（每 KV 头对应的 Q 头数）
    bool is_causal;
    float scale_softmax, scale_softmax_log2;  // 后者 = scale × log2(e)，用于 exp2 优化
    
    // 数据指针
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ o_ptr;
    float *__restrict__ softmax_lse_ptr;
    
    // Stride（元素数，非字节数）
    index_t q_batch_stride, k_batch_stride, o_batch_stride;
    index_t q_row_stride,   k_row_stride,   o_row_stride;
    index_t q_head_stride,  k_head_stride,  o_head_stride;
    
    // Paged KV Cache
    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;           // 通常为 64
    int *__restrict__ seqlens_k_ptr;
    
    // Tile Scheduler（Split-KV 相关）
    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;
    int total_num_splits;
    float *__restrict__ softmax_lseaccum_ptr;  // Split 中间 LSE 累积缓冲区
    float *__restrict__ oaccum_ptr;            // Split 中间输出累积缓冲区
    
    cudaStream_t stream;
};
```

### 4.2 调度元数据结构

```cpp
// 每个 SM 分组的调度元数据（8×4 = 32 字节，内存对齐）
struct __align__(4*8) DecodingSchedMeta {
    int begin_req_idx, end_req_idx;       // 负责的请求范围（闭区间）
    int begin_block_idx, end_block_idx;   // 负责的 block 范围（左闭右开）
    int begin_split_idx;                  // split 起始索引
    int is_first_req_splitted;            // 是否负责某请求的第一个 split
    int is_last_req_splitted;             // 是否负责某请求的最后一个 split
    int _pad[1];                          // 对齐填充
};
```

### 4.3 稀疏解码参数

```cpp
struct SparseAttnDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_qk, d_v;
    float sm_scale, sm_scale_div_log2;
    int num_blocks, page_block_size, topk;
    ModelType model_type;
    
    // 主 KV cache 指针
    cutlass::bfloat16_t* __restrict__ q;    // [b, s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;   // [num_blocks, page_block_size, d_qk]（FP8 格式）
    int* __restrict__ indices;              // [b, s_q, topk]
    int* __restrict__ topk_length;         // [b]，nullable
    float* __restrict__ attn_sink;         // [h_q]，nullable
    
    // 输出
    float* __restrict__ lse;               // [b, s_q, h_q]
    cutlass::bfloat16_t* __restrict__ out; // [b, s_q, h_q, d_v]
    
    // 额外 KV cache（双 KV cache 支持）
    int extra_num_blocks, extra_page_block_size, extra_topk;
    cutlass::bfloat16_t* __restrict__ extra_kv;
    int* __restrict__ extra_indices;
    int* __restrict__ extra_topk_length;
    
    // 各维度 stride
    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_kv_block, stride_kv_row;
    // ... 更多 stride
    
    // Split-KV 相关
    float* __restrict__ lse_accum;         // [num_splits, s_q, h_q]
    float* __restrict__ o_accum;           // [num_splits, s_q, h_q, d_v]
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
    int* __restrict__ num_splits_ptr;
    int num_sm_parts;
    
    cudaStream_t stream;
};
```

---

## 5. ImplBase 架构：可扩展的多实现分发

### 5.1 核心设计

`ImplBase` 是一个基于 C++ 模板的抽象基类，每个 kernel 实现类都继承它并声明自己支持的特性集合：

```cpp
// csrc/api/common.h
template<
    typename RunArgT_,   // 参数结构体类型（如 SparseAttnDecodeParams）
    typename FeatureT_   // 特性枚举类型（如 DecodeFeatures）
>
class ImplBase {
protected:
    using RunArgT = RunArgT_;
    using FeatureT = FeatureT_;
    
    // 子类必须实现：实际执行 kernel
    virtual inline void run_(const RunArgT &params,
                             const std::vector<FeatureT> &required_features) = 0;
    
    // 子类必须实现：返回支持的特性列表
    constexpr virtual inline std::span<const FeatureT> get_supported_features() const = 0;

public:
    // 检查所有 required_features 是否都被支持
    inline bool check_if_all_features_are_supported(
        const std::vector<FeatureT> &required_features);
    
    // 带错误信息的版本
    inline void check_if_all_features_are_supported_and_abort(
        const std::vector<FeatureT> &required_features);
    
    // 公开入口：先验证特性，再执行
    inline void run(const RunArgT &params,
                    const std::vector<FeatureT> &required_features) {
        check_if_all_features_are_supported_and_abort(required_features);
        run_(params, required_features);
    }
};
```

### 5.2 特性枚举

```cpp
// csrc/api/sparse_decode.h
enum class DecodeFeatures : int {
    HEAD_64,             // 64 个 Q 头
    HEAD_128,            // 128 个 Q 头
    HEAD_DIM_576,        // head_dim = 576（V3.2）
    HEAD_DIM_512,        // head_dim = 512（MODEL1）
    V32_KVCACHE_FORMAT,  // DeepSeek-V3.2 FP8 格式
    MODEL1_KVCACHE_FORMAT,
    ATTN_SINK,           // 需要 attention sink
    TOPK_LENGTH,         // topk_length 可变
    EXTRA_KVCACHE,       // 有额外 KV cache
    EXTRA_TOPK_LENGTH
};

// csrc/api/sparse_fwd.h
enum class FwdFeatures : int {
    HEAD_64, HEAD_128,
    HEAD_DIM_576, HEAD_DIM_512,
    ATTN_SINK, SINK_LSE, TOPK_LENGTH
};
```

### 5.3 实现类继承体系（稀疏解码）

```
DecodeImplBase (abstract)
├── Decode_Sm90_Impl
│   支持：HEAD_64/128, HEAD_DIM_512/576, V32/MODEL1, ATTN_SINK, TOPK_LENGTH, EXTRA_KV, EXTRA_TOPK_LEN
│   kernel：sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel
│
├── Decode_Sm100_Head64_Impl
│   支持：HEAD_64, HEAD_DIM_512/576, V32/MODEL1, 全部可选特性（不含 HEAD_128）
│   kernel：sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel
│
├── Decode_Sm100_Head64x2_Impl      ← V3.2 在 SM100 上的变通方案
│   支持：HEAD_128（通过调用 head64 kernel 两次实现）
│   原因：SM100 head128 kernel 不支持 d_qk=576，故对 V3.2 形状拆分执行
│
└── Decode_Sm100_Head128_Impl
    支持：HEAD_128, HEAD_DIM_512（仅！不支持 576）
    kernel：sm100::fwd_for_small_topk::head128（复用预填充 kernel）
```

### 5.4 运行时分发逻辑

```cpp
// sparse_decode.h 中的分发决策（简化）
DecodeImplBase* impl;
if (arch.is_sm100f()) {
    if (h_q == 64) {
        impl = new Decode_Sm100_Head64_Impl();
    } else if (h_q == 128) {
        if (d_qk == 576) {
            impl = new Decode_Sm100_Head64x2_Impl();  // 两次 head64
        } else if (d_qk == 512) {
            impl = new Decode_Sm100_Head128_Impl();
        }
    }
} else if (arch.is_sm90a()) {
    impl = new Decode_Sm90_Impl();
}

// 收集当前请求的特性
std::vector<DecodeFeatures> features;
if (h_q == 128)       features.push_back(DecodeFeatures::HEAD_128);
if (d_qk == 576)      features.push_back(DecodeFeatures::HEAD_DIM_576);
if (have_attn_sink)   features.push_back(DecodeFeatures::ATTN_SINK);
if (have_topk_length) features.push_back(DecodeFeatures::TOPK_LENGTH);
// ...

// 验证并执行（impl 内部调用对应 CUDA kernel）
impl->run(params, features);
```

这种设计使得**新增 GPU 架构或新增特性时，只需添加新的实现类，而不需要修改现有的分发逻辑**。

---

## 6. 密集解码 C++ 侧的 Q 张量变形

密集解码中，`dense_decode.h` 在调用 kernel 前对 Q 张量做了一次关键的维度重排，将 MQA 模式（多个 Q 头共享一个 KV 头）转换为更易于 WGMMA 计算的形式：

```cpp
// dense_decode.h（简化）
// 原始 Q 形状：[b, s_q, h_q, d_k]，其中 h_q = h_kv * q_head_per_hk
// 目标形状：[b, s_q * q_head_per_hk, h_kv, d_k]

const int num_q_heads_per_hk = num_heads_q / num_heads_k;  // = 128 / 1 = 128
const int q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk;  // 将多 Q 头展开到时间维度

q = q.view({batch_size, seqlen_q_ori, num_heads_k, num_q_heads_per_hk, head_size_k})
    .transpose(2, 3)
    .reshape({batch_size, q_seq_per_hk, num_heads_k, head_size_k});
```

变形前后对比（`b=2, s_q=1, h_q=128, h_kv=1, d=576`）：

| 状态 | 形状 | 含义 |
|------|------|------|
| 原始 | `[2, 1, 128, 576]` | batch × seq × Q头 × dim |
| 变形后 | `[2, 128, 1, 576]` | batch × (seq×Q头) × KV头 × dim |

这样 CUDA kernel 只需要处理 `[b, q_seq_per_hk, h_kv=1, d]` 形状，将 Q 头数融入序列维度，简化了 kernel 的实现。

---

## 7. Split-KV 内存分配

分发层负责分配 split-KV 的中间缓冲区，这是实现 SM 间负载均衡的关键：

```cpp
// 总 split 数 = batch_size + num_sm_parts（每个 SM 分组可能额外负责一个 split）
const int total_num_splits = batch_size + params.num_sm_parts;

at::Tensor lse_accum = torch::empty(
    {total_num_splits, num_heads, q_seq_per_hk},
    opts.dtype(at::kFloat)
);  // 每个 split 的 log-sum-exp 中间值

at::Tensor out_accum = torch::empty(
    {total_num_splits, num_heads, q_seq_per_hk, head_size_v},
    opts.dtype(at::kFloat)
);  // 每个 split 的输出中间值（FP32，精度更高）
```

最终由 `smxx::decode::run_flash_mla_combine_kernel` 将多个 split 的结果按 Online Softmax 的方式合并为最终输出（详见 `03_Dense解码内核.md`）。

---

## 8. 枚举名称反射

`common.h` 实现了一个轻量的编译期枚举反射机制，用于在运行时报错时打印友好的特性名称：

```cpp
// 利用 __PRETTY_FUNCTION__ 提取枚举值的字符串名称
template<auto value>
constexpr auto get_static_enum_name() {
    std::string_view name = __PRETTY_FUNCTION__;
    // 解析 "...= DecodeFeatures::HEAD_128]" 中的 "HEAD_128"
    // ...
}

// 示例输出：
// Required features:
//   - 1: HEAD_128
//   - 2: HEAD_DIM_576
// Supported features:
//   - 0: HEAD_64
//   - 2: HEAD_DIM_512
//   ...
```

当实现类不支持请求的特性组合时，会打印详细的不匹配信息，极大地方便了调试。
