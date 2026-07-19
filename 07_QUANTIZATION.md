# 07_QUANTIZATION.md

# minXfmr.cpp Quantization Design

> **Minimal Quantized Transformer Runtime**

---

# Overview

Quantization reduces model memory size and improves runtime efficiency by representing weights with fewer bits.

minXfmr.cpp treats quantization as a property of model tensors.

The runtime architecture should support quantized tensors without changing the Transformer design.

---

# Design Philosophy

The Transformer should not know whether a tensor is:

* FP32
* FP16
* Q8
* Q4

The abstraction is:

```text
Tensor

↓

Backend

↓

Computation
```

---

# Why Quantization Matters

Large language models require significant memory.

Example:

```text
7B Model

FP16

≈14GB


Q4

≈4GB
```

---

For Android:

Memory reduction is often more important than raw speed.

---

# Supported Quantization Roadmap

## MVP

Support:

```text
F32

FP16
```

Purpose:

* Verify correctness
* Simplify debugging

---

## Phase 2

Support:

```text
Q4
```

Purpose:

* Android deployment
* Small memory footprint

---

## Future

Support:

```text
Q8

Q6

Q5

Q3

Q2
```

---

# Tensor Type Design

A tensor contains its own type information.

```cpp
enum DataType
{
    F32,

    F16,

    Q8,

    Q6,

    Q5,

    Q4
};
```

---

# Tensor Abstraction

The Transformer sees:

```cpp
Tensor
```

not:

```cpp
Q4Tensor
```

---

Example:

```text
Attention

↓

Tensor

↓

Backend decides computation
```

---

# Quantized Tensor Structure

A quantized block contains:

```text
Quantized Values

+

Scale

+

Optional Zero Point
```

---

Example:

```text
Original:

1.25
0.98
0.43
0.11


↓

Q4


Stored:

0101
0011
0001
0000


+

Scale
```

---

# Block Quantization

Most LLM quantization uses block-based quantization.

Example:

```text
Weight Matrix


[values...]

        |

        v


Split into blocks


Block 0

Block 1

Block 2
```

---

Each block has:

```text
Scale

+

Compressed values
```

---

# Dequantization

The simplest method:

```text
Quantized Value

↓

Dequantize

↓

FP16/FP32

↓

Compute
```

---

Example:

```text
Q4

↓

FP16

↓

Matrix Multiplication
```

---

# Future Optimization

Avoid full dequantization.

Instead:

```text
Q4 Weight

+

FP16 Activation

↓

Quantized Matmul
```

---

# Backend Responsibility

Quantization belongs to backend.

Architecture:

```text
Transformer

↓

Tensor

↓

CPUBackend

↓

F32 Matmul

or

Q4 Matmul
```

---

# CPU Backend Interface

Example:

```cpp
class CPUBackend
{
public:

    void matmul(
        Tensor& a,
        Tensor& b,
        Tensor& out);

};
```

The backend checks:

```text
Tensor.type
```

and selects implementation.

---

# Memory Comparison

Example:

7B parameter model.

| Type | Bits | Approx Memory |
| ---- | ---- | ------------- |
| F32  | 32   | 28GB          |
| F16  | 16   | 14GB          |
| Q8   | 8    | 7GB           |
| Q4   | 4    | 4GB           |

---

# Android Target

Recommended initial target:

```text
7B class model

↓

Q4

↓

Android ARM64
```

---

# Jetson Nano Target

Jetson Nano:

* 4GB RAM
* ARM64 CPU

Recommended:

```text
Small model

+

Q4

+

CPU only
```

---

# Quantization Loading Flow

```text
GGUF

↓

Tensor Info

↓

DataType Check

↓

Allocate Memory

↓

Load Quantized Data

↓

Tensor
```

---

# Quantized Tensor Memory

Example:

```text
Tensor

{

data

type = Q4

block_size

scale

}
```

---

# Ownership

Same as normal tensors.

```text
Model

↓

Tensor

↓

Quantized Data
```

---

# Quantization Does Not Change

The following remain unchanged:

* API
* Context
* KV Cache
* Tokenizer
* Transformer flow

---

# Accuracy Considerations

Quantization introduces approximation error.

Effect:

```text
FP16

↓

Higher accuracy


Q4

↓

Lower memory

↓

Small quality loss
```

---

# Evaluation

Compare:

* Generated text quality
* Token/sec
* Memory usage

---

# MVP Quantization Implementation Order

## Step 1

Add DataType enum.

---

## Step 2

Load F16 tensors.

---

## Step 3

Implement F16 conversion.

---

## Step 4

Add Q4 tensor reader.

---

## Step 5

Implement Q4 dequantization.

---

## Step 6

Optimize Q4 matmul.

---

# Forbidden Design

Avoid:

## Quantization logic inside Transformer

Bad:

```cpp
Attention
{
    if(Q4)
}
```

---

Good:

```text
Transformer

↓

Backend

↓

Quantized Operation
```

---

# Future Hardware Optimization

Possible backends:

```text
CPU

 ├── ARM NEON

 ├── x86 SIMD


GPU

 ├── Vulkan

 └── CUDA
```

---

# Design Rule

Quantization is a storage and computation optimization.

It must not leak into the high-level inference design.

---

# Final Goal

minXfmr.cpp should achieve:

```text
Small Model

+

Small Runtime

+

Low Memory

+

Portable CPU Inference
```

Quantization is the key technology that makes this possible.
