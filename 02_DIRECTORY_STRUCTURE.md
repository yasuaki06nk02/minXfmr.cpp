# 02_DIRECTORY_STRUCTURE.md

# minXfmr.cpp Directory Structure

> **Minimal and Maintainable Project Organization**

---

# Design Principle

The directory structure follows one rule:

> **One folder has one responsibility.**

Avoid placing code where its purpose is unclear.

Avoid unnecessary layers.

The project should remain understandable by simply reading the directory tree.

---

# Root Structure

```text
minxfmr-cpp/

├── CMakeLists.txt
├── README.md
├── LICENSE
│
├── docs/
│
├── include/
│
├── src/
│
├── tests/
│
└── examples/
```

---

# docs/

Purpose:

Project documentation.

Contains:

* Architecture documents
* API specifications
* Design decisions
* Development roadmap

Structure:

```text
docs/

├── README.md
├── DESIGN_PHILOSOPHY.md
├── ARCHITECTURE.md
├── API.md
├── 91_DEVELOPMENT_ORDER.md
└── 02_DIRECTORY_STRUCTURE.md
```

---

# include/

Purpose:

Public C API headers.

Only files required by external applications belong here.

Structure:

```text
include/

└── minxfmr.h
```

---

# include/minxfmr.h

Responsibility:

Public API definition.

Contains:

* Context declaration
* Public functions
* Callback definition
* Error codes

Example:

```c
typedef struct minxfmr_context minxfmr_context;


typedef void (*minxfmr_callback)(
    const char* token
);


minxfmr_context* minxfmr_open(
    const char* model_path);


int minxfmr_generate(
    minxfmr_context* ctx,
    const char* prompt,
    minxfmr_callback callback);


void minxfmr_reset(
    minxfmr_context* ctx);


void minxfmr_close(
    minxfmr_context* ctx);
```

---

# src/

Purpose:

Internal implementation.

External applications must never depend directly on this directory.

---

# src/runtime/

Responsibility:

Runtime lifecycle management.

Structure:

```text
runtime/

├── context.cpp
└── context.h
```

---

## Context

Class:

```cpp
class Context;
```

Responsibilities:

* Own model
* Own KV cache
* Own tokenizer
* Manage generation state

Does not:

* Perform tensor operations
* Load files directly

---

# src/model/

Responsibility:

Model representation.

Structure:

```text
model/

├── model.cpp
├── model.h
│
└── gguf/
    ├── gguf_reader.cpp
    └── gguf_reader.h
```

---

## Model

Class:

```cpp
class Model;
```

Responsibilities:

* Store model metadata
* Store tensors
* Provide model access

Does not:

* Run inference
* Tokenize text

---

## GGUFReader

Class:

```cpp
class GGUFReader;
```

Responsibilities:

* Read GGUF header
* Read metadata
* Read tensor information
* Read vocabulary

Does not:

* Execute tensors

---

# src/tokenizer/

Responsibility:

Text and token conversion.

Structure:

```text
tokenizer/

├── tokenizer.cpp
└── tokenizer.h
```

---

## Tokenizer

Class:

```cpp
class Tokenizer;
```

Responsibilities:

* Encode text to tokens
* Decode tokens to text

Example:

```text
Text

↓

Token IDs
```

Does not:

* Know Transformer internals

---

# src/tensor/

Responsibility:

Basic tensor representation.

Structure:

```text
tensor/

├── tensor.cpp
└── tensor.h
```

---

## Tensor

Class:

```cpp
struct Tensor;
```

Responsibilities:

* Data pointer
* Shape
* Type
* Size

Does not:

* Know model architecture

---

# src/transformer/

Responsibility:

Transformer computation.

Structure:

```text
transformer/

├── transformer.cpp
├── transformer.h
│
├── attention.cpp
├── attention.h
│
├── feed_forward.cpp
├── feed_forward.h
│
├── rmsnorm.cpp
├── rmsnorm.h
│
└── rope.cpp
    └── rope.h
```

---

# Transformer

Class:

```cpp
class Transformer;
```

Responsibilities:

Execute decoder-only Transformer.

Flow:

```text
Token

↓

Embedding

↓

Decoder Blocks

↓

Logits
```

---

# Attention

Class:

```cpp
class Attention;
```

Responsibilities:

* Query calculation
* Key calculation
* Value calculation
* Attention score

---

# FeedForward

Class:

```cpp
class FeedForward;
```

Responsibilities:

MLP computation.

---

# RMSNorm

Class:

```cpp
class RMSNorm;
```

Responsibilities:

Normalization.

---

# RoPE

Class:

```cpp
class RoPE;
```

Responsibilities:

Rotary positional embedding.

---

# src/backend/

Responsibility:

Numerical computation backend.

Structure:

```text
backend/

└── cpu/

    ├── cpu_backend.cpp
    └── cpu_backend.h
```

---

# CPUBackend

Class:

```cpp
class CPUBackend;
```

Responsibilities:

* Matrix multiplication
* Vector operations
* Element operations

Does not:

* Know LLM concepts

---

# src/cache/

Responsibility:

KV Cache management.

Structure:

```text
cache/

├── kv_cache.cpp
└── kv_cache.h
```

---

# KVCache

Class:

```cpp
class KVCache;
```

Responsibilities:

* Store keys
* Store values
* Clear cache

Does not:

* Perform attention calculation

---

# src/sampler/

Responsibility:

Token selection.

Structure:

```text
sampler/

├── sampler.cpp
└── sampler.h
```

---

# Sampler

Class:

```cpp
class Sampler;
```

Responsibilities:

Convert logits into token IDs.

MVP:

* Greedy sampling

Future:

* Temperature
* Top-K
* Top-P

---

# src/platform/

Responsibility:

Platform-specific implementation.

Structure:

```text
platform/

├── linux/
│
└── android/
```

---

## Linux

Purpose:

Development and testing.

---

## Android

Purpose:

JNI and Android-specific integration.

Contains:

* JNI wrapper
* Android build support

Does not contain:

* Transformer logic

---

# src/util/

Responsibility:

Small general utilities.

Structure:

```text
util/

├── file.cpp
├── memory.cpp
└── logger.cpp
```

---

# tests/

Purpose:

Automated testing.

Structure:

```text
tests/

├── tensor_test.cpp
├── gguf_test.cpp
├── tokenizer_test.cpp
└── runtime_test.cpp
```

---

# examples/

Purpose:

Minimal usage examples.

Structure:

```text
examples/

└── simple_chat.cpp
```

Example:

```cpp
auto ctx = minxfmr_open(
    "model.gguf"
);

minxfmr_generate(
    ctx,
    "Hello",
    callback
);

minxfmr_close(ctx);
```

---

# Dependency Direction

Allowed:

```text
runtime

↓

model

↓

transformer

↓

tensor

↓

backend
```

---

Forbidden:

```text
backend

↓

runtime
```

```text
tokenizer

↓

transformer
```

```text
tensor

↓

model
```

---

# Class Count Target

MVP target:

Approximately 10-15 main classes.

Expected core classes:

```text
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

# Final Rule

Before creating a new file or class, ask:

> **Does this have exactly one responsibility?**

If the answer is unclear, the design should be simplified.
