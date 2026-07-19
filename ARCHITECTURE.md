# ARCHITECTURE.md

# minXfmr.cpp Architecture

> **Minimal Transformer Runtime Architecture**

---

# Design Goal

The architecture of minXfmr.cpp follows one principle:

> **Simple outside. Modular inside.**

Users interact with only four public APIs.

Internally, each module has exactly one responsibility.

---

# High-Level Architecture

```text
                +----------------------+
                |      C API           |
                |----------------------|
                | open()               |
                | generate()           |
                | reset()              |
                | close()              |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |      Runtime         |
                +----------+-----------+
                           |
       +-------------------+-------------------+
       |                   |                   |
       v                   v                   v
+--------------+   +---------------+   +---------------+
|    Model     |   |    Context    |   |   Backend     |
+--------------+   +---------------+   +---------------+
       |                   |                   |
       +-------------------+-------------------+
                           |
                           v
                +----------------------+
                |   Transformer Core   |
                +----------+-----------+
                           |
       +---------+---------+---------+---------+
       |         |         |         |         |
       v         v         v         v         v
 Tokenizer   Tensor     RoPE    Attention   Sampler
```

---

# Layer Structure

```text
Application

↓

C API

↓

Runtime

↓

Transformer

↓

Tensor

↓

CPU Backend
```

Each layer only communicates with the layer directly below it.

No cross-layer dependencies.

---

# Directory Structure

```text
minxfmr-cpp/

docs/

include/
    minxfmr.h

samples/

src/

    backend/
        cpu/

    model/
        gguf/
        tokenizer/

    runtime/

    transformer/

    tensor/

    platform/

    util/

tests/
```

---

# Module Responsibilities

## Runtime

Responsible for:

* Session management
* Generation flow
* Context lifetime

Does NOT perform mathematical computation.

---

## Model

Responsible for:

* GGUF loading
* Tensor metadata
* Vocabulary
* Hyperparameters

Does NOT execute inference.

---

## Tokenizer

Responsible for:

* Encode
* Decode

No model logic.

---

## Context

Responsible for:

* Runtime state
* Sequence information
* KV Cache ownership

---

## Tensor

Responsible for:

* Tensor storage
* Tensor shape
* Tensor access

Tensor contains data only.

No model logic.

---

## Transformer

Responsible for inference pipeline.

Contains:

* Embedding
* RMSNorm
* RoPE
* Attention
* Feed Forward
* Decoder Block

No file loading.

No tokenizer.

---

## KV Cache

Responsible for:

* Key storage
* Value storage
* Reset

Nothing else.

---

## Sampler

Responsible for:

* Greedy
* Temperature
* Top-K
* Top-P

Sampling only.

---

## Backend

Responsible for numerical computation.

MVP includes:

* CPU Backend

Future:

* NEON
* Vulkan

Backend never knows about language models.

It only computes tensors.

---

# Data Flow

```text
Prompt

↓

Tokenizer

↓

Input Tokens

↓

Transformer

↓

Logits

↓

Sampler

↓

Output Token

↓

Tokenizer

↓

Text
```

---

# Generation Flow

```text
open()

↓

Load GGUF

↓

Create Context

↓

Ready

↓

generate()

↓

Encode Prompt

↓

Transformer

↓

Sample Token

↓

Decode Token

↓

Repeat

↓

Return Text

↓

reset()

↓

Clear KV Cache

↓

Ready

↓

close()

↓

Release Everything
```

---

# Dependency Rules

Allowed:

```text
Runtime

↓

Model

↓

Transformer

↓

Tensor

↓

Backend
```

Forbidden:

```text
Backend

↓

Runtime
```

```text
Tensor

↓

Tokenizer
```

```text
Sampler

↓

GGUF
```

Modules must not depend on unrelated modules.

---

# Object Ownership

```text
Runtime

owns

↓

Model

↓

Context

↓

KV Cache
```

Transformer modules never own resources.

They operate on references only.

---

# Memory Management

Memory ownership is explicit.

No hidden allocations.

No global state.

No singleton.

---

# Thread Model

MVP:

Single-thread only.

Future versions may introduce worker threads without changing the public API.

---

# Error Flow

Errors propagate upward.

```text
Backend

↓

Transformer

↓

Runtime

↓

C API
```

No exceptions.

Return explicit status values internally.

---

# Platform Separation

Platform-specific code remains isolated.

```text
platform/

    linux/

    android/
```

No Android code inside runtime.

No JNI inside transformer.

---

# Future Extensions

New features must not modify existing architecture.

Instead, add isolated modules.

Examples:

* Vulkan Backend
* NEON Backend
* Metal Backend

Public API remains unchanged.

---

# Architecture Principle

Every module should answer one question:

> **What is my single responsibility?**

If the answer contains "and",

the module should probably be split.
