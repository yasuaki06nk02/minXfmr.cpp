# 20_CUDA_BACKEND.md

# CUDA Backend Design

## Purpose

This document describes the CUDA backend architecture for minXfmr.cpp.

The goal is not to build the fastest CUDA inference engine.

The goal is:

* preserve simplicity
* preserve readability
* preserve portability
* keep CPU backend as the reference implementation
* allow optional GPU acceleration

The CUDA backend must remain an implementation detail behind the backend interface.

Transformer code should not contain CUDA-specific logic.

---

# Design Philosophy

minXfmr.cpp follows a CPU-first design.

The CPU backend is always considered the reference implementation.

The CUDA backend is an optional acceleration layer.

The following principles apply:

* identical model behavior
* identical output tokens
* shared transformer code
* minimal backend-specific branching
* optional build configuration

The transformer implementation must not depend directly on CUDA APIs.

---

# Architecture

```text
Application
     │
     ▼
Transformer
     │
     ▼
Backend Interface
     │
     ├── CpuBackend
     │
     └── CudaBackend
```

The Transformer interacts only with the backend interface.

Backend selection occurs during runtime initialization.

---

# Initial Scope

The first CUDA implementation focuses only on the most expensive operations.

Supported:

* Linear
* MatMul
* Attention Score MatMul
* Attention Value MatMul

Not supported initially:

* Tensor Core optimization
* Flash Attention
* CUDA Graphs
* Quantized CUDA kernels
* Multi-GPU
* Tensor Parallelism

These features can be added later.

---

# Supported Hardware

Initial target platforms:

## Jetson Nano

```text
GPU Architecture : Maxwell
Compute Capability : 5.3
CUDA Version : 10.2
```

## RTX 3070

```text
GPU Architecture : Ampere
Compute Capability : 8.6
CUDA Version : 12.x
```

The first implementation must compile on both systems.

---

# CUDA Compatibility Strategy

The minimum supported CUDA version is:

```text
CUDA 10.2
```

This ensures compatibility with Jetson Nano.

The backend should avoid CUDA APIs that require newer toolkits.

Preferred APIs:

```cpp
cudaMalloc()
cudaFree()
cudaMemcpy()

cublasCreate()
cublasDestroy()

cublasSgemm()
cublasSgemv()
```

These APIs are available on both Jetson Nano and modern RTX GPUs.

---

# Backend Interface

The backend abstraction should remain small.

```cpp
class Backend {
public:
    virtual ~Backend() = default;

    virtual void linear(
        const Tensor& input,
        const Tensor& weight,
        const Tensor* bias,
        Tensor& output) = 0;

    virtual void rmsnorm(
        const Tensor& input,
        const Tensor& weight,
        Tensor& output,
        float eps) = 0;

    virtual void rope(
        Tensor& q,
        Tensor& k,
        int position) = 0;

    virtual void attention(
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& output) = 0;
};
```

Transformer code must use only this interface.

---

# Weight Loading Strategy

Model weights should be uploaded once.

```text
GGUF
   │
   ▼
CPU Memory
   │
   ▼
GPU Memory
```

Weights remain resident on the GPU.

Repeated CPU-to-GPU transfers must be avoided.

---

# Memory Ownership

Each backend owns its own memory.

CPU backend:

```cpp
std::vector<float>
```

CUDA backend:

```cpp
cudaMalloc()
cudaFree()
```

Tensor ownership must remain explicit.

Hidden memory allocation is discouraged.

---

# Tensor Design

Tensor objects should remain backend-independent.

Example:

```cpp
struct Tensor {
    void* data;
    DataType type;

    int ndim;
    int dims[4];

    DeviceType device;
};
```

Where:

```cpp
enum class DeviceType {
    CPU,
    CUDA
};
```

The transformer layer should not care where the memory lives.

---

# Phase 1

Initial CUDA implementation.

Features:

* backend abstraction
* CUDA build option
* cuBLAS integration
* GPU resident weights

Target:

```text
Qwen2.5-1.5B
```

Expected performance:

```text
Jetson Nano
2 - 4 tok/s
```

---

# Phase 2

Performance improvements.

Features:

* pinned memory
* CUDA streams
* overlap CPU and GPU work

Target:

```text
Jetson Nano
4 - 6 tok/s
```

---

# Phase 3

RTX optimization.

Features:

* FP16
* Tensor Core support
* cublasGemmEx()

Target:

```text
RTX 3070
30+ tok/s
```

---

# Phase 4

Advanced optimizations.

Possible future work:

* cuBLASLt
* CUDA Graphs
* Flash Attention
* Quantized CUDA kernels

These optimizations must not increase complexity of the transformer core.

---

# Build Configuration

Example:

```bash
cmake -DMINXFMR_ENABLE_CUDA=ON ..
```

Runtime selection:

```bash
minxfmr_cli model.gguf --backend cpu
```

```bash
minxfmr_cli model.gguf --backend cuda
```

If CUDA initialization fails:

```text
Fallback to CPU backend
```

The application must continue to function.

---

# Success Criteria

The CUDA backend is considered successful when:

* transformer code remains unchanged
* CPU backend remains the reference implementation
* Jetson Nano build succeeds
* RTX build succeeds
* CUDA backend can be disabled completely
* backend switching requires no transformer changes

The primary goal is maintainability.

Performance improvements are secondary.
