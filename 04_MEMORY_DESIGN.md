# 04_MEMORY_DESIGN.md

# minXfmr.cpp Memory Design

> **Explicit Ownership and Minimal Memory Usage**

---

# Design Philosophy

minXfmr.cpp follows one memory rule:

> **Every allocated byte must have a clear owner.**

No hidden ownership.

No global memory pool.

No unexpected allocation.

---

# Memory Goals

The runtime should prioritize:

* Low memory usage
* Predictable allocation
* Simple lifetime management
* Android compatibility

---

# Memory Categories

Runtime memory is divided into five categories.

```text
Memory

├── Model Memory

├── Runtime Memory

├── KV Cache Memory

├── Temporary Compute Memory

└── Output Memory
```

---

# Memory Ownership Overview

```text
Context

│

├── Model

│    └── Tensor Memory

│

├── Tokenizer

│

├── KV Cache

│

└── Runtime Buffers
```

---

# Context Lifetime

The lifetime of all runtime memory is controlled by Context.

```text
minxfmr_open()

↓

Allocate Memory

↓

generate()

↓

reset()

↓

Reuse Memory

↓

close()

↓

Free Memory
```

---

# Model Memory

## Responsibility

Stores all model parameters.

Examples:

* Embedding weights
* Attention weights
* FFN weights
* Norm weights

---

## Ownership

Owner:

```text
Model
```

---

## Lifetime

```text
open()

↓

close()
```

Model memory is never released during generation.

---

# Tensor Memory

## Tensor Structure

```cpp
struct Tensor
{
    void* data;

    DataType type;

    Shape shape;

    size_t bytes;
};
```

---

# Tensor Ownership

MVP:

Model owns tensors.

Example:

```text
Model

 ├── Tensor 1

 ├── Tensor 2

 └── Tensor 3
```

---

# Tensor Loading

Flow:

```text
GGUF File

↓

GGUFReader

↓

Tensor Metadata

↓

Tensor Allocation

↓

Model
```

---

# Memory Mapping

Future optimization:

```text
GGUF

↓

mmap()

↓

Direct Tensor Access
```

MVP:

Load required tensors into memory.

---

# KV Cache Memory

## Purpose

Stores previous attention states.

Without KV Cache:

```text
Every token

↓

Recalculate previous tokens
```

With KV Cache:

```text
Previous tokens

↓

Reuse
```

---

# Ownership

Owner:

```text
Context
```

---

# Lifetime

```text
open()

↓

generate()

↓

reset()

↓

reuse

↓

close()
```

---

# Reset Behavior

reset() does NOT free memory.

It only clears contents.

Example:

Before:

```text
KV Cache

Token1
Token2
Token3
```

After:

```text
KV Cache

empty
```

Memory remains allocated.

---

# Temporary Compute Memory

## Purpose

Intermediate calculations.

Examples:

* Attention scores
* Matmul output
* Activation buffers

---

# Ownership

Owner:

```text
Runtime
```

---

# Allocation Strategy

Avoid:

```cpp
new/delete
```

inside token generation loop.

---

Prefer:

```text
Allocate once

↓

Reuse repeatedly
```

---

# Generation Memory Flow

One token generation:

```text
Input Token

↓

Embedding Buffer

↓

Attention Buffer

↓

FFN Buffer

↓

Logits Buffer

↓

Sampler
```

Temporary buffers are reused.

---

# Output Memory

Because minXfmr.cpp uses streaming:

```c
callback(token)
```

there is no large output buffer.

---

# Advantage

Memory does not grow with generated text.

Example:

1000 tokens:

Bad design:

```text
1000 tokens

↓

Large string buffer
```

minXfmr design:

```text
Token

↓

Callback

↓

Discard
```

---

# Allocation Rules

## Rule 1

No allocation during token generation if possible.

---

## Rule 2

All allocations happen during:

```text
open()
```

---

## Rule 3

reset() only clears state.

---

## Rule 4

close() releases all memory.

---

# Memory Error Handling

Allocation failure:

```cpp
nullptr
```

is returned.

No exception.

---

# Memory Debug Mode

Future:

Optional tracking:

```text
Allocation count

Memory usage

Peak memory
```

---

# Android Memory Considerations

Android devices have:

* Limited RAM
* Background memory pressure
* Process termination risk

Therefore:

The runtime must expose memory usage internally.

Future API:

```c
size_t minxfmr_memory_usage(ctx);
```

---

# Quantization Memory

Model size depends heavily on quantization.

Example:

```text
FP16

↓

Large memory


Q4

↓

Small memory
```

The runtime must treat quantization as a model property.

---

# Weight Memory vs Compute Memory

Separate:

```text
Model weights

+

Runtime workspace
```

Example:

```text
4GB Model

+

300MB Runtime
```

---

# Memory Optimization Roadmap

## MVP

* Simple allocation
* Clear ownership

---

## Future

* mmap GGUF
* Memory mapped weights
* Tensor sharing
* Buffer pooling
* ARM optimization

---

# Forbidden Design

Avoid:

## Global Memory

```cpp
global_model;
```

---

## Hidden Allocation

```cpp
generate()
{
    new Buffer();
}
```

---

## Shared Ownership Complexity

Avoid:

```cpp
shared_ptr everywhere
```

---

# Recommended Ownership Model

Prefer:

```text
Unique Owner

+

Raw Reference
```

Example:

```text
Context owns Model

Transformer references Model
```

---

# Memory Design Principle

The runtime should answer:

> "Who owns this memory?"

for every allocation.

If the answer is unclear,

the design is wrong.
