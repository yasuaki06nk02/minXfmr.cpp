#pragma once

#include "backend_runtime.h"

// Centralized runtime backend context to avoid scattered globals.
struct BackendContext {
    BackendKind backend = BackendKind::CPU;
    bool initialized = false;

    // Weight transpose flags (previously globals in transformer.cpp)
    bool transpose_square_wq = false;
    bool transpose_square_wk = false;
    bool transpose_square_wv = false;
    bool transpose_square_wo = false;
    bool transpose_square_ffn_gate = false;
    bool transpose_square_ffn_up = false;
    bool transpose_square_ffn_down = false;

    // Prefer llama.cpp/ggml-style linear layout resolution where tensors are
    // interpreted as [out x in] after load, so cols==input_dim maps to the
    // canonical ggml_mul_mat path. This is mainly needed for ambiguous square
    // GGUF weights such as Qwen attn_q/attn_output.
    bool prefer_ggml_mul_mat_layout = false;

    // RoPE pairing mode. false = consecutive pairs (0,1)(2,3)...
    // true = NeoX pairs (0,half)(1,half+1)...
    bool rope_neox = false;

    // Cache append logging env state (-1 = unknown, 0 = disabled, 1 = enabled)
    int cache_append_log_enabled = -1;

    // Scoped compare tracing state for transformer internals.
    bool compare_trace_enabled = false;
    int compare_trace_step = 0;
};

// Return the singleton runtime context.
BackendContext& backend_context();
