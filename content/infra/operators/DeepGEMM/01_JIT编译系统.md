# JIT 编译系统

> 源码路径：  
> `csrc/jit/compiler.hpp` — 编译器抽象（NVCC / NVRTC）  
> `csrc/jit/cache.hpp` — 编译缓存  
> `csrc/jit/kernel_runtime.hpp` — kernel 加载与调用  
> `csrc/jit/device_runtime.hpp` — GPU 设备信息  
> `csrc/jit_kernels/impls/runtime_utils.hpp` — LaunchRuntime 基类

---

## 1. 设计动机

DeepGEMM 的核心哲学是"**安装时不编译，运行时编译**"。传统 CUDA 扩展在 `pip install` 时就要编译所有组合，导致：

1. 安装时间长（编译 FP8 kernel 动辄数分钟）
2. 预编译的 binary 对 GPU 型号、CUDA 版本不够灵活
3. 无法在运行时将矩阵维度（M/N/K）编译为常数，丧失 NVCC 常数折叠优化

JIT 系统解决了上述问题：每个 `(kernel_name, shape, config)` 组合**第一次调用时**即时编译，结果缓存到磁盘（默认 `~/.deep_gemm/`）。后续调用直接加载缓存 cubin，CPU 开销极低。

---

## 2. 系统架构

```
Python API 调用（如 fp8_gemm_nt）
         │
         ▼
csrc/apis/gemm.hpp   ← 参数验证、TMA descriptor 构造
         │
         ▼
csrc/jit_kernels/heuristics/sm90.hpp   ← 枚举候选配置，选最优
         │
         ▼
csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp
  ├── generate_impl()   ← 生成 .cu 源码字符串（填充模板参数）
  └── launch_impl()     ← 调用已加载的 kernel
         │
         ▼
csrc/jit/compiler.hpp (NVCCCompiler / NVRTCCompiler)
  └── build()   ← 签名哈希 → 缓存命中检查 → 编译 → 返回 KernelRuntime
         │
         ▼
csrc/jit/kernel_runtime.hpp
  └── launch_kernel()   ← cuLaunchKernel / cuLaunchKernelEx（带 cluster）
```

---

## 3. 编译器抽象：`Compiler` 基类

```cpp
// csrc/jit/compiler.hpp
class Compiler {
public:
    std::string signature, flags;  // 编译签名和 flags（用于 cache key）

    // 核心入口：根据签名哈希决定是否重新编译
    std::shared_ptr<KernelRuntime> build(const std::string& name,
                                          const std::string& code) const;

    virtual void compile(const std::string &code,
                         const std::filesystem::path& dir_path,
                         const std::filesystem::path &cubin_path,
                         const std::optional<std::filesystem::path> &ptx_path) const = 0;
};
```

`build()` 的缓存逻辑：

```cpp
const auto kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code);
const auto dir_path = cache_dir_path / "cache" /
    fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

// 命中内存缓存
if (const auto runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
    return runtime;

// 编译到临时目录，原子性 rename 到最终缓存目录（分布式文件系统安全）
const auto tmp_dir_path = make_tmp_dir() / get_uuid();
compile(code, tmp_dir_path, tmp_cubin_path);
std::filesystem::rename(tmp_dir_path, dir_path, error_code);
```

**两级缓存**：
- **L1 内存缓存**：进程内 `kernel_runtime_cache`，避免重复磁盘 I/O
- **L2 磁盘缓存**：`~/.deep_gemm/cache/kernel.<name>.<hash>/kernel.cubin`

---

## 4. 两种编译后端

### 4.1 NVCC 编译器（默认）

```cpp
class NVCCCompiler final: public Compiler {
    void compile(...) const override {
        // 将 code 写入临时 .cu 文件
        put(code_path, code);
        // 调用 nvcc，生成 .cubin
        const auto command = fmt::format(
            "cd {} && {} {} -cubin -o {} {}",
            compile_dir, nvcc_path, code_path, cubin_path, flags);
        call_external_command(command);
    }
};
```

NVCC 编译 flags 包含：

```
-std=c++20
--gpu-architecture=sm_90a         # 精确架构（支持特定指令集）
-O3 --expt-relaxed-constexpr
--expt-extended-lambda
--ptxas-options=--register-usage-level=10  # 优先减少寄存器溢出
```

> 注：SM90 需要 `sm_90a`（非 `sm_90`），CUDA 12.9 起会自动做 FFMA interleaving 优化（之前需要后处理 SASS）。

### 4.2 NVRTC 编译器（可选，`DG_JIT_USE_NVRTC=1`）

NVRTC 在进程内直接调用 `nvrtcCompileProgram()`，无需 fork 子进程，约快 10x 编译速度：

```cpp
class NVRTCCompiler final: public Compiler {
    void compile(...) const override {
        nvrtcProgram program;
        nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr);
        nvrtcCompileProgram(program, option_cstrs.size(), option_cstrs.data());
        // 获取 CUBIN 数据并写入磁盘
        nvrtcGetCUBINSize(program, &cubin_size);
        nvrtcGetCUBIN(program, cubin_data.data());
        put(cubin_path, cubin_data);
    }
};
```

NVRTC 在 CUDA 12.8+ 启用 PCH（预编译头），进一步加速重复编译。

---

## 5. 代码生成：以 SM90 FP8 GEMM 为例

每个 kernel 实现类有 `generate_impl()` 方法，将运行时已知的 config 填入 C++ 模板参数：

```cpp
// csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp
static std::string generate_impl(const Args& args) {
    return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d1d_impl<
        {M}, {N}, {K},          // 编译为常数（0 表示运行时传入）
        {kNumGroups},
        {BLOCK_M}, {BLOCK_N}, {BLOCK_K},
        {kSwizzleAMode}, {kSwizzleBMode},
        {kNumStages},
        {kNumTMAThreads}, {kNumMathThreads},
        {kNumTMAMulticast}, {kIsTMAMulticastOnA},
        {kNumSMs},
        {kGemmType}, {cd_dtype_t}
    >);
}};
)", ...);
}
```

当维度被"编译进"模板参数时（`M != 0`），NVCC 会将其视为编译期常数，开启循环展开、常数折叠等优化，显著提升性能。

控制哪些维度被编译为常数：

```python
# 告知 JIT：N 和 K 是常数，M 是变量
deep_gemm.set_ignore_compile_dims('m')   # 不把 M 编译为常数
```

---

## 6. Kernel 加载与调用

```cpp
// csrc/jit/kernel_runtime.hpp
class KernelRuntime {
    CUmodule module;
    CUfunction kernel_func;

    void load(const std::filesystem::path& dir_path) {
        cuModuleLoad(&module, cubin_path.c_str());
        cuModuleGetFunction(&kernel_func, module, "__instantiate_kernel");
        // 实际的 kernel 函数通过 __instantiate_kernel 的地址反推
    }
};

// 支持 SM90 cluster launch（kNumSMsPerCluster > 1）
static cudaError_t launch_kernel(const KernelHandle& kernel,
                                  const LaunchConfigHandle& config, ...) {
    if (config.num_sms_per_cluster == 1) {
        return cuLaunchKernel(func, grid, 1, 1, block, 1, 1, smem_size, stream, ...);
    } else {
        // 使用 cuLaunchKernelEx 配合 cudaLaunchAttributeClusterDimension
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {config.num_sms_per_cluster, 1, 1};
        return cuLaunchKernelEx(&launch_config, func, params, nullptr);
    }
}
```

---

## 7. 调试与诊断环境变量

| 环境变量 | 功能 |
|----------|------|
| `DG_JIT_DEBUG=1` | 打印编译命令、PTXAS 日志 |
| `DG_PRINT_CONFIGS=1` | 打印每个形状选择的 config |
| `DG_JIT_PTXAS_CHECK=1` | 断言 kernel 不使用 local memory（防止寄存器溢出） |
| `DG_JIT_DUMP_ASM=1` | 保存 PTX 和 SASS 到缓存目录 |
| `DG_JIT_WITH_LINEINFO=1` | 嵌入源码行信息，供 Nsight 分析 |
| `DG_JIT_PTXAS_VERBOSE=1` | 显示每个 kernel 的寄存器/共享内存使用量 |
| `DG_JIT_CACHE_DIR=<path>` | 自定义缓存目录 |
| `DG_JIT_USE_NVRTC=1` | 使用 NVRTC 代替 NVCC |
| `DG_JIT_NVCC_COMPILER=<path>` | 指定 NVCC 路径 |

---

## 8. 分布式文件系统安全性

在多机训练环境（如 NFS 挂载的家目录），多个 rank 可能同时触发同一 kernel 的 JIT 编译。DeepGEMM 的处理策略：

```cpp
// 1. 先编译到带 UUID 的临时目录
const auto tmp_dir_path = make_tmp_dir() / get_uuid();

// 2. 编译完后 fsync 整个目录（确保 NFS 可见）
fsync_dir(tmp_dir_path);

// 3. 原子 rename 到最终路径（rename 在 POSIX 系统是原子操作）
std::filesystem::rename(tmp_dir_path, dir_path, error_code);
if (error_code) {
    // 别的 rank 抢先完成了，清理我们的临时目录，用对方的结果
    safe_remove_all(tmp_dir_path);
}
```

这确保即使多个 rank 并发编译，最终只有一份正确的缓存，不会有竞态条件。

---

## 9. `LaunchRuntime<T>` CRTP 基类

```cpp
// csrc/jit_kernels/impls/runtime_utils.hpp
template <typename Derived>
class LaunchRuntime {
public:
    static std::string generate(const typename Derived::Args& args) {
        return Derived::generate_impl(args);
    }

    static void launch(const std::shared_ptr<KernelRuntime>& runtime,
                       const typename Derived::Args& args) {
        const auto handle = runtime->get_kernel_handle(...);
        const auto config_handle = runtime->get_launch_config_handle(...);
        Derived::launch_impl(handle, config_handle, args);
    }
};
```

通过 CRTP，每个具体 kernel（`SM90FP8Gemm1D1DRuntime`、`SM100FP8FP4Gemm1D1DRuntime` 等）只需实现 `generate_impl()` 和 `launch_impl()`，自动获得完整的 JIT 编译 + 缓存 + 调用流程。
