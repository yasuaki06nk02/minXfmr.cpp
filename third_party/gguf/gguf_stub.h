#pragma once
#include "../../src/tensor/tensor.h"
// Minimal stub API: attempt to load projections using existing heuristic as a fallback.
// A full GGUF parser will be implemented here (llama.cpp derived).
bool gguf_load_projections_fallback(const char* path, Tensor*& Wq, Tensor*& Wk, Tensor*& Wv);
