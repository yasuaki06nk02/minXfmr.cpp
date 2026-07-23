# minXfmr.cpp

**SQLite for local LLM inference**

A Minimal Transformer Runtime

minXfmr.cpp is an open-source Transformer inference runtime designed with one primary goal

Keep it minimal.

The project focuses on building a small, readable, and maintainable runtime for local Transformer models across multiple platforms, including Linux, Windows, Android, Raspberry Pi, and Jetson devices.

Unlike feature-rich inference frameworks, minXfmr.cpp intentionally limits its scope to provide a clean architecture that is easy to understand, maintain, and extend.

---

# Vision

The long-term vision of minXfmr.cpp is

SQLite for Local Transformer Runtime

Just as SQLite provides a small, stable, and dependable SQL engine, minXfmr.cpp aims to provide a compact and reliable Transformer runtime that can be embedded into applications and deployed on edge devices without requiring a large runtime environment.

---

# Project Goals

* Portable architecture
* Native C API
* C++17 implementation
* Minimal public API
* High readability
* High maintainability
* GGUF native support
* CPU inference
* Small binary size
* Minimal external dependencies

The runtime follows a **CPU-first, minimal-dependency design** similar to SQLite: small enough to understand, easy to embed into applications, and portable across different operating systems and hardware platforms.

---

# Non Goals (MVP)

The following features are intentionally excluded from the first release

* GPU acceleration
* CUDA
* Metal
* Vulkan
* OpenCL
* LoRA
* Vision models
* Audio models
* Embedding API
* HTTP server
* RPC
* Multi-GPU
* Distributed inference
* Function Calling
* Agent framework

These features may be considered in the future, but simplicity always takes priority.

---

# Public API

The MVP exposes only four public functions.

```c
TinyLLM minxfmr_open(const char* model_path);

char* minxfmr_generate(
    TinyLLM runtime,
    const char* prompt);

void minxfmr_reset(
    TinyLLM runtime);

void minxfmr_close(
    TinyLLM runtime);
```

Everything else is considered an implementation detail.

---

# Example

```c
TinyLLM ai = minxfmr_open("qwen3.gguf");

char* reply;

reply = minxfmr_generate(ai, "Hello");
printf("%s\\n", reply);

reply = minxfmr_generate(ai, "Do you remember my name?");
printf("%s\\n", reply);

minxfmr_reset(ai);

reply = minxfmr_generate(ai, "Do you remember my name?");
printf("%s\\n", reply);

minxfmr_close(ai);
```

---

# Development Environment

Current development platform

* Ubuntu 22.04+
* C++17
* CMake
* GCC / Clang

Verification platforms

* Windows x64
* Linux x64
* Apple Silicon
* Jetson Nano (CPU)
* Android ARM64

The runtime core is developed on Linux first and then validated across the supported platforms. Platform-specific integrations such as Android (JNI) and future Apple Silicon backends are built on top of the shared core runtime.

---

# Supported Model Format (MVP)

* GGUF
* Decoder-only Transformer models
* Q4_K_M quantization (initial target)

Additional quantization methods may be added while maintaining compatibility with GGUF.

---

# Debug Flags

The CLI includes a few debug-oriented switches that help isolate model loading and tokenization issues.

```bash
--try-all-templates
```

Enables the old template sweep behavior in chat mode. This is useful when the model does not provide a reliable `chat_template` and you want to compare several fallback prompts.

Example:

```bash
minxfmr_cli.exe model.gguf --chat --try-all-templates --temp 0.8 --top_k 8 --log-file chat_debug.log
```

This prints each fallback prompt as `assistant(template N)> ...` and logs the tokenized prompt for comparison.

Sample `--try-all-templates` log fragment (from `--log-file chat_debug_try_all_templates.log`):

```text
[INFO] trying template 0 => "<s><|system|>You are a helpful assistant...</|system|><|user|>hello</|user|>"
[DEBUG] tpl 0 token count=9 ids=[1,234,45,67,89, ...]
[INFO] assistant(template 0)> I am a helpful assistant. How can I help you today?

[INFO] trying template 1 => "<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>>\\n#B# hello [/INST]"
[DEBUG] tpl 1 token count=11 ids=[1,230,90, ...]
[INFO] assistant(template 1)> Hello! What would you like to talk about?

[INFO] Selected template=1 (best coherence)
```

```bash
--transpose-square
--no-transpose-square
--transpose-wq
--no-transpose-wq
--transpose-wk
--no-transpose-wk
--transpose-wv
--no-transpose-wv
--transpose-wo
--no-transpose-wo
```

Controls whether square projection matrices are treated as transposed. The per-weight flags can be combined to search for the best orientation.

Recommended defaults: for GGUF models, transpose behavior is selected automatically from `general.architecture` metadata (llama/mistral/mixtral/qwen2/qwen2moe/gemma => square transpose on, gptneox/gpt2/falcon/bloom/mpt/phi* => off). At load time, attention and FFN linear weights are physically normalized to canonical orientation, so runtime transpose branching is minimized. Use `--transpose-*` / `--no-transpose-*` to explicitly override square-weight policy.

Quick automated search

There is a helper script at `scripts/find_best_transpose.py` that searches transpose combinations and writes logs to `scripts/transpose_search_logs`.

Safety note (Windows)

Running all 16 combinations continuously can cause sustained high CPU load. The Python script now defaults to a safer mode (`--max-masks 4`, cooldown between runs, lower priority). Use full search only when your machine is stable under long CPU-bound workloads.

Safer default usage (Python):

```powershell
python .\\scripts\\find_best_transpose.py .\\model.gguf .\\build\\src\\Release\\minxfmr_cli.exe "hello"
```

Full search (explicit high-load opt-in):

```powershell
python .\\scripts\\find_best_transpose.py .\\model.gguf .\\build\\src\\Release\\minxfmr_cli.exe "hello" --max-masks 16 --allow-high-load --cooldown-sec 5 --priority idle
```

Legacy PowerShell helper (no safety guardrails):

```powershell
.\\scripts\\find_best_transpose.ps1 -ModelPath .\\model.gguf -CliPath .\\build\\src\\Release\\minxfmr_cli.exe -Prompt "hello"
```

The script runs each combination, extracts assistant lines from the per-run log, scores them with a simple heuristic (ASCII letter count), and prints the best-scoring combination.

---

# Design Principles

minXfmr.cpp follows several strict design principles.

## Minimalism

Every line of code has a maintenance cost.

If a feature is not necessary, it should not exist.

---

## Stable API

Public APIs should remain stable.

Internal implementation may evolve freely.

---

## Readability

Code should be understandable by reading it from top to bottom.

Complex abstractions are avoided whenever possible.

---

## Single Responsibility

Each module should perform one task only.

Examples include

* GGUF Reader
* Tokenizer
* Tensor
* Attention
* Sampler
* KV Cache

---

## No Over-Engineering

The project intentionally avoids unnecessary complexity.

Examples

* Deep inheritance
* Large class hierarchies
* Excessive design patterns
* Hidden control flow

Simple code is preferred over clever code.

---

# Repository Structure

```text
minxfmr-cpp

├── docs
├── include
├── samples
├── src
│   ├── backend
│   ├── model
│   ├── runtime
│   ├── tensor
│   ├── platform
│   └── util
├── tests
├── CMakeLists.txt
└── README.md
```

---

# Development Roadmap

MVP development proceeds in the following order

1. GGUF Reader
2. Tokenizer
3. Tensor
4. Embedding
5. RoPE
6. RMSNorm
7. Attention
8. Feed Forward Network
9. Decoder Block
10. Transformer
11. KV Cache
12. Sampler
13. Text Generation
14. C API
15. Platform Integrations

Each step should produce a working build.

---

# Why minXfmr.cpp

Many inference frameworks have grown into large, feature-rich codebases.

minXfmr.cpp explores a different direction.

Instead of maximizing features, it maximizes

* Simplicity
* Clarity
* Maintainability
* Educational value
* Embeddability

The project is designed so that developers can understand the complete runtime architecture without reading hundreds of thousands of lines of code.

---

# License

TBD

---

# Contributing

Contributions are welcome.

Before adding a new feature, ask one question

Does this make minXfmr.cpp simpler?

If the answer is No, the design should be reconsidered.
