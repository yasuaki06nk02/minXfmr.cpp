# API.md

# minXfmr.cpp Public API Specification

> **Minimal C API for Transformer Runtime**

---

# Philosophy

minXfmr.cpp exposes a minimal C API.

The API is designed for:

* Android applications
* Embedded systems
* Local AI assistants
* Real-time streaming interfaces

The runtime does not return a completed text response.

Instead, generated tokens are streamed through a callback.

---

# Public API

The MVP provides four functions.

```c
typedef struct minxfmr_context minxfmr_context;


minxfmr_context* minxfmr_open(
    const char* model_path);


int minxfmr_generate(
    minxfmr_context* ctx,
    const char* prompt,
    void (*callback)(const char* token));


void minxfmr_reset(
    minxfmr_context* ctx);


void minxfmr_close(
    minxfmr_context* ctx);
```

---

# Runtime Lifecycle

The basic lifecycle is:

```text
open()

↓

generate()

↓

generate()

↓

reset()

↓

generate()

↓

close()
```

---

# Data Flow

```text
Application

↓

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

Application UI
```

---

# minxfmr_open()

## Purpose

Loads a Transformer model and creates an inference session.

---

## Prototype

```c
minxfmr_context* minxfmr_open(
    const char* model_path);
```

---

## Parameters

### model_path

Path to GGUF model file.

Example:

```text
models/qwen3.gguf
```

---

## Return Value

Success:

```c
minxfmr_context*
```

Failure:

```c
NULL
```

---

## Internal Operations

The function performs:

* GGUF file loading
* Model metadata loading
* Tensor initialization
* Vocabulary loading
* Tokenizer initialization
* KV Cache allocation
* Runtime creation

---

# minxfmr_generate()

## Purpose

Generates tokens from a prompt.

Generated tokens are delivered immediately through a callback.

---

## Prototype

```c
int minxfmr_generate(
    minxfmr_context* ctx,
    const char* prompt,
    void (*callback)(const char* token));
```

---

## Parameters

### ctx

Runtime context created by:

```c
minxfmr_open()
```

---

### prompt

UTF-8 encoded input text.

Example:

```text
こんにちは
```

---

### callback

Function called every time a token is generated.

Example:

```c
void on_token(const char* token)
{
    printf("%s", token);
}
```

---

## Return Value

Success:

```c
0
```

Failure:

```c
non-zero
```

---

# Streaming Behavior

Generation occurs token by token.

Example:

Input:

```text
今日は
```

Output events:

```text
今
日
は
良
い
天
気
で
す
```

The application can display or process tokens immediately.

---

# Callback Rules

The callback:

* Runs synchronously
* Runs on the generation thread
* Must return quickly
* Must not destroy the context

Long processing inside callback reduces generation speed.

---

# minxfmr_reset()

## Purpose

Clears the current conversation state.

---

## Prototype

```c
void minxfmr_reset(
    minxfmr_context* ctx);
```

---

## Behavior

Reset performs:

* KV Cache clearing
* Sequence reset
* Conversation history removal

Reset does NOT:

* Reload model
* Reload tokenizer
* Free memory

---

# minxfmr_close()

## Purpose

Releases all runtime resources.

---

## Prototype

```c
void minxfmr_close(
    minxfmr_context* ctx);
```

---

## Behavior

Releases:

* KV Cache
* Context
* Model tensors
* Tokenizer
* Runtime memory

After close(), the context is invalid.

---

# Memory Management

The application owns only the context handle.

Example:

```c
minxfmr_context* ctx;

ctx = minxfmr_open("model.gguf");

...

minxfmr_close(ctx);
```

The application does not manage:

* Model memory
* Tensor memory
* KV Cache memory
* Token buffers

---

# Thread Safety

MVP limitation:

A context is not thread-safe.

One context must be accessed from one thread.

Example:

Allowed:

```text
Thread A
 |
 Context A


Thread B
 |
 Context B
```

Not allowed:

```text
Thread A
 |
 Context A
 |
Thread B
```

---

# Error Handling

The API does not use exceptions.

Errors are reported by:

* NULL return
* Integer error code

---

# Streaming Cancellation

Future versions may support:

```c
minxfmr_stop(ctx);
```

for:

* User cancellation
* UI stop button
* Timeout control

This is intentionally not part of MVP.

---

# ABI Stability

The public interface uses C ABI.

The internal implementation may use:

* C++
* Zig
* Rust

without changing application code.

---

# Android Integration

Recommended architecture:

```text
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

The Android application never depends directly on C++ classes.

---

# Future Extensions

Possible future additions:

* Streaming metadata callback
* Token ID callback
* Generation parameters
* Stop sequence
* Temperature control
* Top-K
* Top-P

These should be added without breaking the core API.

---

# Design Rule

The user should only need to understand:

```c
open()

generate()

reset()

close()
```

Everything else should remain internal.

> **Small API. Complete Runtime.**
