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

* [x] Setup CMake + C++17
* [x] Create `include/` and `src/`
* [x] Build `minxfmr_cli`

## Expected Result

A clean build runs on host environment.

---

# Phase 1: Tensor System

## Goal

Create fundamental tensor container and memory helpers.

## Tasks

* [x] Implement tensor allocation/free
* [x] Implement F32 read/write helpers
* [x] Add tensor transpose helper

## Expected Result

Tensor create/read/write/free works correctly.

---

# Phase 2: CPU Backend Basic Operations

## Goal

Provide baseline math operations.

## Tasks

* [x] MatMul
* [x] Vector-dot helpers
* [x] Softmax support helpers

## Expected Result

Core math paths pass reference checks.

---

# Phase 3: Transformer Components

## Goal

Implement independent Transformer building blocks.

## Tasks

* [x] RMSNorm
* [x] RoPE
* [x] Attention
* [x] FeedForward

## Expected Result

Single components run with valid outputs.

---

# Phase 4: Full Transformer Model

## Goal

Execute decoder stack end-to-end for logits.

## Tasks

* [x] Multi-layer forward path
* [x] Embedding + final norm + output projection
* [x] Layer buffer reuse strategy

## Expected Result

Given token inputs, model produces logits.

---

# Phase 5: Tokenizer

## Goal

Convert UTF-8 text to token IDs and back.

## Tasks

* [x] Vocabulary loading
* [x] Encode
* [x] Decode
* [x] Special token handling

## Expected Result

Round-trip tests and GGUF vocab loading behave correctly.

---

# Phase 6: GGUF Loader

## Goal

Load metadata, tensors, and tokenizer assets from GGUF.

## Tasks

* [x] Read model config metadata
* [x] Locate/load per-layer tensors
* [x] Load token embedding/lm_head/final norm
* [x] Load chat template/special tokens when available

## Expected Result

Real GGUF model is opened successfully.

---

# Phase 7: Model Runtime

## Goal

Connect loader outputs with runtime execution context.

## Tasks

* [x] Build `minxfmr_context`
* [x] Create KV cache
* [x] Normalize weight orientations at load
* [x] Preload weights for backend when enabled

## Expected Result

Runtime context can execute model forward calls safely.

---

# Phase 8: Text Generation

## Goal

Generate tokens autoregressively with callback streaming.

## Tasks

* [x] Prompt prefill
* [x] Token-by-token generation loop
* [x] EOS/limit/repetition guards
* [x] Stream callback outputs

## Expected Result

Prompt produces streamed text output.

---

# Phase 9: Session Management

## Goal

Stabilize lifecycle APIs.

## Tasks

* [x] `minxfmr_open`
* [x] `minxfmr_generate`
* [x] `minxfmr_reset`
* [x] `minxfmr_close`

## Expected Result

Repeated open/generate/reset/close cycles are stable and leak-free.

---

# Phase 10: Quantization Support

## Goal

Support Q4_K model paths with minimal complexity.

## Tasks

* [x] Keep Q4_K tensors packed in memory
* [x] Dequantize on-the-fly in backend matmul
* [x] Support Q4_K output head path

## Expected Result

F32 vs Q4_K behavior is acceptable for MVP targets.

---

# Phase 11: CPU Optimization

## Goal

Improve performance after correctness is stable.

## Tasks

* [x] OpenMP/SIMD-friendly loops
* [x] Workspace/tensor reuse
* [x] Cache-friendly data access

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

* [x] Documentation consistency
* [ ] Remove noisy debug defaults
* [ ] Build profile verification
* [ ] Example command validation

## Expected Result

Release artifacts and docs are aligned and reproducible.

---

# MVP Completion Criteria

MVP is complete when:

* [x] GGUF model loads
* [x] Transformer forward works
* [x] Text generation streams via callback
* [x] Runtime lifecycle APIs are stable
* [x] F32/Q4_K paths are validated
* [x] CPU and optional CUDA selection are functional

Android integration is the next milestone after MVP core completion.
