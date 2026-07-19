# 99_AI_IMPLEMENTATION_PROMPT.md

# minXfmr.cpp AI Implementation Prompt

> **Implementation Instructions for AI Coding Agent**

---

# Role

You are an expert C++ systems engineer.

Your task is to implement:

```
minXfmr.cpp
```

a minimal Transformer inference runtime.

The goal is not to create a feature-rich framework.

The goal is:

> Build the smallest understandable LLM runtime that can run on embedded CPUs and Android.

---

# Project Philosophy

Follow these principles:

```
Small

Simple

Portable

Understandable

Testable
```

---

# Important Restrictions

Do not:

* Copy llama.cpp architecture
* Add unnecessary abstractions
* Add large dependencies
* Add GPU code initially
* Create unnecessary classes
* Use complex inheritance

---

# Target Platforms

The same source code must run on:

```
Linux

↓

Jetson Nano

↓

Android ARM64
```

---

# Language Requirements

Use:

```
C++17
```

Avoid:

* C++20-only features
* Compiler-specific extensions
* Platform-dependent code in core

---

# Build Requirements

Use:

```
CMake
```

The project must build with:

```
cmake

make
```

---

# Public API Requirement

The only public API should initially be:

```c
typedef struct minxfmr_context minxfmr_context;


minxfmr_context*
minxfmr_open(
    const char* model_path
);


int
minxfmr_generate(
    minxfmr_context* ctx,

    const char* prompt,

    void (*callback)(
        const char* token
    )
);


void
minxfmr_reset(
    minxfmr_context* ctx
);


void
minxfmr_close(
    minxfmr_context* ctx
);
```

---

# API Philosophy

The application should only know:

```
open()

generate()

reset()

close()
```

Everything else is internal.

---

# Directory Structure

Maintain:

```
minxfmr.cpp

├── include/

│   └── minxfmr.h


├── src/

│   ├── api/

│   ├── runtime/

│   ├── model/

│   ├── gguf/

│   ├── tokenizer/

│   ├── transformer/

│   └── backend/

│       └── cpu/


├── tests/

└── examples/
```

---

# Class Design Rules

Use composition.

Avoid inheritance.

Preferred:

```
Context

 ├── Model

 ├── Tokenizer

 └── KVCache
```

---

# Required Core Classes

Implement:

```
Context

Model

GGUFReader

Tokenizer

Tensor

Transformer

Attention

FeedForward

RMSNorm

RoPE

KVCache

Sampler

CPUBackend
```

---

# Memory Rules

Every allocation must have an owner.

Follow:

```
Context owns runtime objects

Model owns tensors

KVCache owns attention cache

Backend owns temporary workspace
```

---

# Forbidden Memory Patterns

Do not use:

```
global variables

singleton model

hidden allocation during generation
```

---

# Tensor Design

Tensor should only contain:

```
data pointer

shape

datatype

size
```

Do not put inference logic inside Tensor.

---

# Backend Design

Transformer must not know CPU details.

Correct:

```
Transformer

↓

CPUBackend

↓

CPU Instructions
```

Wrong:

```
Transformer

↓

#ifdef ARM

#ifdef CUDA
```

---

# CPU Backend Requirements

Initial implementation:

Use simple C++.

Implement:

```
matmul

add

multiply

softmax

dot product
```

---

# Optimization Order

Do not optimize first.

Follow:

```
Correctness

↓

Memory

↓

MatMul

↓

SIMD

↓

Quantization
```

---

# Model Format

Use:

```
GGUF
```

Do not create a new format.

---

# GGUF Implementation Order

Implement:

1.

Header reading

2.

Metadata reading

3.

Tensor information

4.

Tensor loading

5.

Vocabulary loading

---

# Quantization Strategy

Initial support:

```
F32

FP16
```

Later:

```
Q4
```

Do not mix quantization logic into Transformer.

---

# Inference Flow

Implement:

```
Prompt

↓

Tokenizer

↓

Token IDs

↓

Embedding

↓

Transformer layers

↓

Logits

↓

Sampler

↓

Token callback
```

---

# Generation Loop

The core loop:

```cpp
while(!finished)
{

    logits =
        transformer.forward();


    token =
        sampler.sample(logits);


    callback(token);

}
```

---

# KV Cache

Implement:

```
open()

↓

allocate

generate()

↓

use

reset()

↓

clear

close()

↓

free
```

---

# Streaming Requirement

Do not store generated text.

Use:

```cpp
callback(token)
```

Output should be streamed.

---

# Error Handling

Avoid exceptions.

Use:

```
bool

nullptr

error code
```

---

# Testing Requirements

Create tests for:

## Tensor

* allocation
* access

## Backend

* matmul
* softmax

## GGUF

* header
* metadata

## Runtime

* open
* generate
* reset
* close

---

# Development Order

Follow:

```
Phase 0

Project setup


Phase 1

Tensor


Phase 2

CPU Backend


Phase 3

Transformer components


Phase 4

Full Transformer


Phase 5

Tokenizer


Phase 6

GGUF


Phase 7

Runtime


Phase 8

Generation


Phase 9

API


Phase 10

Quantization


Phase 11

Optimization


Phase 12

Jetson


Phase 13

Android
```

---

# Code Quality Rules

Every source file:

* Clear responsibility
* Small functions
* Comments explain why
* No unnecessary abstraction

---

# Function Size

Prefer:

```
< 50 lines
```

---

# Class Size

Prefer:

```
< 200 lines
```

---

# When Adding Code

Before creating something new, ask:

```
Can an existing class own this responsibility?
```

If yes:

Do not create a new class.

---

# Android Requirement

Android integration must use:

```
C++

↓

JNI

↓

Kotlin
```

JNI must remain thin.

---

# Jetson Requirement

The first hardware validation:

```
Jetson Nano

CPU only

No CUDA
```

---

# Success Criteria

The implementation is complete when:

```
✓ CMake build works

✓ Tensor tests pass

✓ CPU backend works

✓ GGUF model loads

✓ Transformer executes

✓ Text generation works

✓ Streaming works

✓ reset works

✓ close frees memory

✓ Jetson Nano runs

✓ Android ARM64 runs
```

---

# Final Instruction

Do not try to build a large AI framework.

Build the smallest possible Transformer runtime.

Every line of code must justify its existence.

The final product should feel closer to:

```
SQLite

+

llama.cpp simplicity

+

embedded runtime
```

than a large ML framework.

---

# End of Implementation Prompt
