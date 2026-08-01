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

    // Cache append logging env state (-1 = unknown, 0 = disabled, 1 = enabled)
    int cache_append_log_enabled = -1;
};

// Return the singleton runtime context.
BackendContext& backend_context();
