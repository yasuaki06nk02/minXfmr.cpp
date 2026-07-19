#pragma once
#include "../../src/tensor/tensor.h"
#include <cstdint>
#include <string>
#include <vector>

// Minimal GGUF reader: parse header and extract tensors by name.
// Exposes routines to open file, list tensors, and read tensor data (including Q4_K_M dequant).

struct GGUF_TensorInfo {
    std::string name;
    uint64_t offset;
    uint64_t nbytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t ggml_type;
    std::string dtype; // e.g., "f32", "f16", "q4_k", "q8_0"
};

struct GGUF_File {
    std::string path;
    std::vector<uint8_t> data;
    std::vector<GGUF_TensorInfo> tensors;
    // Parsed model metadata (best-effort).
    uint64_t n_layer = 0;
    uint64_t n_ctx = 0;
    uint64_t n_embd = 0;
    uint64_t n_head = 0;
    uint64_t n_head_kv = 0;
    std::vector<std::string> vocab_tokens;
    // Optional metadata supplied by model authors
    std::string chat_template;
    std::vector<std::string> special_tokens;
};

struct GGUF_ModelConfig {
    uint64_t n_layer = 0;
    uint64_t n_ctx = 0;
    uint64_t n_embd = 0;
    uint64_t n_head = 0;
    uint64_t n_head_kv = 0;
};

bool gguf_open(const char* path, GGUF_File& out);
void gguf_close(GGUF_File& f);
bool gguf_find_tensor(const GGUF_File& f, const char* name, GGUF_TensorInfo& out);
bool gguf_find_tensor_any(const GGUF_File& f, const char* const* names, size_t count, GGUF_TensorInfo& out);
bool gguf_read_f32_tensor(const GGUF_File& f, const GGUF_TensorInfo& info, Tensor*& out);
bool gguf_dequant_q4_k_m(const GGUF_File& f, const GGUF_TensorInfo& info, Tensor*& out);
bool gguf_dequant_q6_k(const GGUF_File& f, const GGUF_TensorInfo& info, Tensor*& out);
bool gguf_read_model_config(const char* path, GGUF_ModelConfig& out);
bool gguf_read_vocab(const char* path, std::vector<std::string>& out_tokens);
// Utility: compute statistics of a tensor
void gguf_tensor_stats(const Tensor* t, float& minv, float& maxv, double& mean);
