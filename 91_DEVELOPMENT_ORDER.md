# 91_DEVELOPMENT_ORDER.md

# minXfmr.cpp Development Order

> **Implementation Roadmap (Aligned with 12_IMPLEMENTATION_PHASES.md)**

---

# Development Philosophy

minXfmr.cpp is developed incrementally.

Each phase must:

* Compile successfully
* Have one clear responsibility
* Be independently testable

Do not move to the next phase with a broken foundation.

---

# Phase 0: Project Foundation

## Goal

Create the minimum C++ project structure.

## Tasks

* [ ] Setup CMake + C++17
* [ ] Create `include/` and `src/`
* [ ] Build `minxfmr_cli`

## Expected Result

A clean build runs on host environment.

---

# Phase 1: Tensor System

## Goal

Create fundamental tensor container and memory helpers.

## Tasks

* [ ] Implement tensor allocation/free
* [ ] Implement F32 read/write helpers
* [ ] Add tensor transpose helper

## Expected Result

Tensor create/read/write/free works correctly.

---

# Phase 2: CPU Backend Basic Operations

## Goal

Provide baseline math operations.

## Tasks

* [ ] MatMul
* [ ] Vector-dot helpers
* [ ] Softmax support helpers

## Expected Result

Core math paths pass reference checks.

---

# Phase 3: Transformer Components

## Goal

Implement independent Transformer building blocks.

## Tasks

* [ ] RMSNorm
* [ ] RoPE
* [ ] Attention
* [ ] FeedForward

## Expected Result

Single components run with valid outputs.

---

# Phase 4: Full Transformer Model

## Goal

Execute decoder stack end-to-end for logits.

## Tasks

* [ ] Multi-layer forward path
* [ ] Embedding + final norm + output projection
* [ ] Layer buffer reuse strategy

## Expected Result

Given token inputs, model produces logits.

---

# Phase 5: Tokenizer

## Goal

Convert UTF-8 text to token IDs and back.

## Tasks

* [ ] Vocabulary loading
* [ ] Encode
* [ ] Decode
* [ ] Special token handling

## Expected Result

Round-trip tests and GGUF vocab loading behave correctly.

---

# Phase 6: GGUF Loader

## Goal

Load metadata, tensors, and tokenizer assets from GGUF.

## Tasks

* [ ] Read model config metadata
* [ ] Locate/load per-layer tensors
* [ ] Load token embedding/lm_head/final norm
* [ ] Load chat template/special tokens when available

## Expected Result

Real GGUF model is opened successfully.

---

# Phase 7: Model Runtime

## Goal

Connect loader outputs with runtime execution context.

## Tasks

* [ ] Build `minxfmr_context`
* [ ] Create KV cache
* [ ] Normalize weight orientations at load
* [ ] Preload weights for backend when enabled

## Expected Result

Runtime context can execute model forward calls safely.

---

# Phase 8: Text Generation

## Goal

Generate tokens autoregressively with callback streaming.

## Tasks

* [ ] Prompt prefill
* [ ] Token-by-token generation loop
* [ ] EOS/limit/repetition guards
* [ ] Stream callback outputs

## Expected Result

Prompt produces streamed text output.

---

# Phase 9: Session Management

## Goal

Stabilize lifecycle APIs.

## Tasks

* [ ] `minxfmr_open`
* [ ] `minxfmr_generate`
* [ ] `minxfmr_reset`
* [ ] `minxfmr_close`

## Expected Result

Repeated open/generate/reset/close cycles are stable and leak-free.

---

# Phase 10: Quantization Support

## Goal

Support Q4_K model paths with minimal complexity.

## Tasks

* [ ] Keep Q4_K tensors packed in memory
* [ ] Dequantize on-the-fly in backend matmul
* [ ] Support Q4_K output head path

## Expected Result

F32 vs Q4_K behavior is acceptable for MVP targets.

---

# Phase 11: CPU Optimization

## Goal

Improve performance after correctness is stable.

## Tasks

* [ ] OpenMP/SIMD-friendly loops
* [ ] Workspace/tensor reuse
* [ ] Cache-friendly data access

## Expected Result

Higher token throughput with unchanged functional behavior.

---

# Phase 12: Linux / Jetson Validation

## Goal

Verify portability on Linux ARM CPU environments.

## Tasks

* [ ] Build and run on Jetson/Linux ARM
* [ ] Check memory and tokens/sec
* [ ] Confirm CPU-only fallback behavior

## Expected Result

Stable inference on embedded-style Linux setups.

---

# Phase 13: Android NDK Integration

## Goal

Integrate the same core runtime into Android apps.

## Tasks

* [ ] Build shared library in Android app toolchain
* [ ] JNI wrapper for C API
* [ ] Kotlin bridge and streaming UI path

## Expected Result

Android app can call open/generate/reset/close successfully.

---

# Phase 14: Release Preparation

## Goal

Prepare a usable and maintainable release.

## Tasks

* [ ] Documentation consistency
* [ ] Remove noisy debug defaults
* [ ] Build profile verification
* [ ] Example command validation

## Expected Result

Release artifacts and docs are aligned and reproducible.

---

# MVP Completion Criteria

MVP is complete when:

* [ ] GGUF model loads
* [ ] Transformer forward works
* [ ] Text generation streams via callback
* [ ] Runtime lifecycle APIs are stable
* [ ] F32/Q4_K paths are validated
* [ ] CPU and optional CUDA selection are functional

Android integration is the next milestone after MVP core completion.
