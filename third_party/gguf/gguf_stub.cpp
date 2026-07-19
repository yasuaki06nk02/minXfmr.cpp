#include "gguf_stub.h"
#include "../../src/io/gguf_loader.h"

bool gguf_load_projections_fallback(const char* path, Tensor*& Wq, Tensor*& Wk, Tensor*& Wv) {
    // For now delegate to the existing heuristic scanner in src/io/gguf_loader.cpp
    return gguf_try_load_projections(path, Wq, Wk, Wv);
}
