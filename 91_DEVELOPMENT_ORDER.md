# 91_DEVELOPMENT_ORDER.md

# minXfmr.cpp Development Order

> **Implementation Roadmap**

---

# Development Philosophy

minXfmr.cpp is developed incrementally.

Each step must:

* Compile successfully
* Have a clear responsibility
* Be independently testable

Do not implement the entire runtime at once.

---

# Phase 0: Project Foundation

## Goal

Create the minimum C++ project structure.

---

## Tasks

* [ ] Create repository
* [ ] Setup CMake
* [ ] Setup C++17
* [ ] Create include directory
* [ ] Create src directory
* [ ] Create test directory
* [ ] Create basic executable

---

## Expected Result

The project builds successfully.

Example:

```bash
mkdir build
cmake ..
make
```

---

# Phase 1: Public API Skeleton

## Goal

Create the stable external interface first.

---

## Tasks

* [ ] Create minxfmr.h
* [ ] Define minxfmr_context
* [ ] Implement minxfmr_open()
* [ ] Implement minxfmr_generate()
* [ ] Implement minxfmr_reset()
* [ ] Implement minxfmr_close()

---

## Expected Result

A dummy runtime can be called.

Example:

```c
ctx = minxfmr_open();

minxfmr_generate();

minxfmr_reset();

minxfmr_close();
```

---

# Phase 2: Core Data Structures

## Goal

Create the internal foundation.

---

## Implement

### Tensor

Responsibilities:

* Data pointer
* Shape
* Size calculation

---

### Tensor Shape

Responsibilities:

* Dimension management
* Element count

---

### Context

Responsibilities:

* Runtime state
* Memory ownership

---

### KV Cache

Responsibilities:

* Key storage
* Value storage
* Reset

---

## Expected Result

All core structures exist without inference.

---

# Phase 3: GGUF Reader

## Goal

Load model information.

---

## Tasks

* [ ] GGUF header parser
* [ ] Metadata reader
* [ ] Tensor table reader
* [ ] Vocabulary reader

---

## Expected Result

The runtime can inspect a GGUF file.

Example output:

```
Model:
Qwen

Layers:
32

Embedding:
4096
```

---

# Phase 4: Tokenizer

## Goal

Convert text into tokens.

---

## Tasks

* [ ] Load vocabulary
* [ ] Implement encode()
* [ ] Implement decode()

---

## Expected Result

Input:

```
Hello
```

Output:

```
[9707]
```

---

# Phase 5: CPU Backend

## Goal

Create basic tensor operations.

---

## Implement

* [ ] Matrix multiplication
* [ ] Vector operations
* [ ] Addition
* [ ] Multiplication
* [ ] RMSNorm support

---

## Rules

Do not optimize yet.

Correctness first.

---

# Phase 6: Transformer Components

## Goal

Implement decoder-only Transformer.

---

## Implement Order

### 1. Embedding

Token ID

↓

Vector

---

### 2. RMSNorm

Normalize hidden state.

---

### 3. RoPE

Rotary positional embedding.

---

### 4. Attention

Implement:

* Query
* Key
* Value
* Softmax

---

### 5. Feed Forward

Implement:

* Gate
* Up
* Down projection

---

### 6. Decoder Block

Combine:

* Attention
* FFN
* Residual connection

---

# Phase 7: Full Transformer Forward

## Goal

Complete one forward pass.

---

## Flow

```
Token

↓

Embedding

↓

Decoder Blocks

↓

Final Norm

↓

Logits
```

---

## Expected Result

Given a token, output logits.

---

# Phase 8: KV Cache

## Goal

Enable efficient generation.

---

## Tasks

* [ ] Store previous keys
* [ ] Store previous values
* [ ] Reuse cache
* [ ] Implement reset()

---

## Expected Result

Conversation can continue efficiently.

---

# Phase 9: Sampler

## Goal

Convert logits into tokens.

---

## MVP

Implement:

* [ ] Greedy sampling

---

## Future

* Temperature
* Top-K
* Top-P

---

# Phase 10: Token Streaming Generation

## Goal

Connect Transformer to public API.

---

## Flow

```
Prompt

↓

Tokenizer

↓

Transformer

↓

Sampler

↓

Token

↓

Callback

↓

Repeat
```

---

## Expected Result

Tokens are streamed to application.

---

# Phase 11: Performance Improvement

## Goal

Optimize only after correctness.

---

## Measure

* Token/sec
* Memory usage
* Binary size

---

## Possible Optimization

* SIMD
* ARM NEON
* Memory reuse

---

# Phase 12: Android Integration

## Goal

Run minXfmr.cpp on Android.

---

## Tasks

* [ ] Android NDK build
* [ ] JNI wrapper
* [ ] Kotlin sample app
* [ ] ARM64 testing

---

## Architecture

```
Kotlin

↓

JNI

↓

C API

↓

minXfmr.cpp

↓

CPU Backend
```

---

# Phase 13: Jetson Nano Validation

## Goal

Validate architecture on Linux ARM.

---

## Conditions

* CPU only
* CUDA disabled

---

## Purpose

Confirm portability before Android integration.

---

# Testing Strategy

Each phase requires tests.

---

## Unit Tests

Examples:

* Tensor calculation
* Tokenizer
* GGUF parser
* Matrix multiplication

---

## Integration Tests

Examples:

* Load model
* Generate tokens
* Reset session

---

# Implementation Rules

## Rule 1

Never skip phases.

---

## Rule 2

Do not optimize before correctness.

---

## Rule 3

Do not add features without design review.

---

## Rule 4

Keep public API stable.

---

# MVP Completion Criteria

minXfmr.cpp MVP is complete when:

* [ ] GGUF model loads
* [ ] Tokenizer works
* [ ] Transformer forward works
* [ ] Text generation works
* [ ] Streaming callback works
* [ ] reset works
* [ ] Android ARM64 build succeeds

---

# Final Goal

Create the smallest understandable Transformer runtime.

Not the biggest.

Not the most feature-rich.

The simplest runtime that works.
