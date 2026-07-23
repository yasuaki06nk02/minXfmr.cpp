# 08_CPU_BACKEND.md

# minXfmr.cpp CPU Backend Design

> **Minimal CPU Compute Engine**

---

# Overview

CPU Backend provides the low-level numerical operations required by the Transformer.

The Transformer does not know how computation is performed.

Architecture:

```text id="h1y5s8"
Transformer

↓

Backend Runtime Interface

↓

CPU Backend

↓

CPU Instructions
```

---

# Design Philosophy

CPU Backend follows these rules:

* Simple implementation first
* Correctness before optimization
* No model knowledge
* No token knowledge
* No runtime state

---

# Responsibility

CPUBackend handles:

* Matrix multiplication
* Vector operations
* Element operations
* Quantized operations

---

# Non-Responsibilities

CPUBackend does not handle:

* Model loading
* Tokenization
* KV Cache
* Sampling
* Prompt processing

---

# Directory Structure

```text id="qz7mcm"
src/backend/

├── backend_runtime.h

├── backend_runtime.cpp

└── cpu/

    ├── cpu_backend.h

    └── cpu_backend.cpp
```

---

# CPU Backend Interface (Current)

## Header

```cpp id="mfw93n"
bool cpu_matmul(const Tensor* A, const Tensor* B, Tensor* out);
bool cpu_add(const Tensor* a, const Tensor* b, Tensor* out);

float* cpu_workspace(size_t n);
void cpu_workspace_reset(bool shrink = false);

bool cpu_matvec(const float* vec, const float* mat, float* out, size_t K, size_t N);
bool cpu_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride);

bool cpu_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride);
bool cpu_vec_dot_rows_ring(const float* vec, const float* ring, size_t head, size_t seq_max, size_t len, size_t K, size_t row_stride, float* out);
bool cpu_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride);
```

---

# Runtime Selection

Backend selection is handled by backend runtime using `MINXFMR_BACKEND`.

CPU backend remains the fallback/reference path when CUDA is unavailable.

---

# Tensor Operations

The backend operates only on:

```cpp id="9jgbgz"
Tensor
```

Example:

```text id="5y3z6w"
Tensor A

+

Tensor B

↓

Tensor C
```

---

# Matrix Multiplication

## Importance

The most expensive operation in Transformer inference.

---

Basic operation:

```text id="wz7tqj"
C = A × B
```

---

Example:

```text id="wfj1n9"
A

[M x K]


B

[K x N]


↓

C

[M x N]
```

---

# MVP Implementation

Simple loop version.

```cpp id="rrd1vh"
for(i)

 for(j)

  for(k)

    C[i][j] +=
       A[i][k] *
       B[k][j];
```

---

# Optimization Roadmap

## Phase 1

Naive C++ implementation.

Goal:

Correct output.

---

## Phase 2

Memory optimization.

Techniques:

* Cache friendly layout
* Loop reorder
* Blocking

---

## Phase 3

SIMD.

Targets:

* ARM NEON
* x86 AVX

---

# Vector Operations

Required operations:

```text id="t6a7ga"
Add

Multiply

Scale

Copy

Dot Product
```

---

# Example

Vector add:

```text id="6rx83s"
A[i] + B[i]

↓

C[i]
```

---

# Activation Functions

Some operations are required by FFN.

Examples:

```text id="g2l3rm"
SiLU

GELU
```

---

# Softmax

Attention requires:

```text id="q66f8q"
softmax(x)
```

Formula:

```text id="avmr7s"
exp(x)

/

sum(exp(x))
```

---

# RMSNorm

Although conceptually part of Transformer,

the calculation may use backend operations.

Example:

```text id="6r4q6h"
sum(x²)

sqrt()

scale()
```

---

# Quantized Operations

CPUBackend handles:

```text id="g2e8t9"
F32 MatMul

Q4_K MatMul
```

---

# Quantized MatMul

Flow:

```text id="j5h7d7"
Q4_K Weight

+

F32 Activation

↓

CPUBackend

↓

Output
```

---

# Data Type Dispatch

Example:

```cpp id="u5db8g"
switch(type)
{

case F32:

    matmul_f32();

break;


case Q4_K:

    matmul_q4_k();

break;

}
```

---

# Memory Access Rules

The backend must avoid:

```cpp id="o7qk7v"
malloc()

inside loops
```

---

Prefer:

```text id="r5dby6"
Allocate

↓

Reuse

↓

Release
```

---

# Workspace Buffer

Current implementation:

```cpp id="0m2p9h"
float* cpu_workspace(size_t n);
void cpu_workspace_reset(bool shrink = false);
```

---

Purpose:

Thread-local reusable temporary calculation memory.

---

# ARM64 Optimization

Android and Jetson Nano use:

```text id="2l2t9s"
ARM64
```

---

Future optimization:

```text id="4n1h0n"
CPUBackend

↓

ARMBackend

↓

NEON
```

---

# Backend Interface Design

Future:

```text id="r2b6v1"
Backend

├── CPUBackend

├── VulkanBackend

└── CUDABackend
```

---

# Important Rule

Transformer must not contain:

```cpp id="j8l4y5"
#ifdef ARM
```

or:

```cpp id="i4f8g6"
#ifdef CUDA
```

---

Hardware differences belong only in Backend.

---

# Testing

CPUBackend tests:

## Matrix Multiplication

Input:

```text id="h5r0g4"
A

B
```

Expected:

```text id="qx7h4n"
C
```

---

## Numerical Accuracy

Compare:

```text id="2s83z7"
Reference CPU

vs

Optimized CPU
```

---

# Performance Measurement

Measure:

* Matrix multiplication time
* Token/sec
* Memory bandwidth

---

# Jetson Nano Validation

Target:

```text id="6z9nq3"
Jetson Nano

↓

Ubuntu

↓

CPU only

↓

CPUBackend
```

---

# Android Validation

Target:

```text id="o9y1x8"
Android

↓

ARM64

↓

JNI

↓

CPUBackend
```

---

# Optimization Priority

Order:

1. Correctness
2. Memory layout
3. MatMul speed
4. SIMD
5. Quantized kernels

---

# Forbidden Design

Avoid:

## Large Math Framework Dependency

Example:

```text
Eigen

BLAS

OpenBLAS
```

Reason:

* Binary size increase
* Dependency complexity

---

## Model-aware Backend

Bad:

```cpp
backend.run_llama_attention()
```

---

Good:

```cpp
backend.matmul()
```

---

# Final Goal

CPUBackend should be:

* Small
* Portable
* Replaceable
* Understandable

The same backend design should run:

```text
Jetson Nano

↓

Android

↓

Embedded ARM Device
```

without changing the Transformer engine.
