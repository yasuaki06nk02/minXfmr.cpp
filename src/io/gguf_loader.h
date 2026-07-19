#pragma once
#include "../tensor/tensor.h"
#include <cstdint>
#include <vector>
#include <string>

struct GGUFLoaderModelConfig {
	uint64_t n_layer;
	uint64_t n_ctx;
	uint64_t n_embd;
	uint64_t n_head;
	uint64_t n_head_kv;
};

// Heuristic GGUF loader: best-effort search for int dim followed by 3*d*d floats.
bool gguf_try_load_projections(const char* path, Tensor*& outWq, Tensor*& outWk, Tensor*& outWv);
bool gguf_try_load_projections_for_layer(const char* path, int layer, Tensor*& outWq, Tensor*& outWk, Tensor*& outWv);
bool gguf_try_load_attn_out_for_layer(const char* path, int layer, Tensor*& outWo);
bool gguf_try_load_norms_for_layer(const char* path, int layer, Tensor*& outAttnNorm, Tensor*& outFfnNorm);
bool gguf_try_load_ffn_for_layer(const char* path, int layer, Tensor*& outWgate, Tensor*& outWup, Tensor*& outWdown);
bool gguf_try_read_model_config(const char* path, GGUFLoaderModelConfig& out);
bool gguf_try_read_architecture(const char* path, std::string& out_architecture);
bool gguf_try_load_token_embedding(const char* path, Tensor*& outWemb);
bool gguf_try_load_final_norm(const char* path, Tensor*& outWnorm);
bool gguf_try_load_lm_head(const char* path, Tensor*& outWout);
bool gguf_try_read_vocab(const char* path, std::vector<std::string>& out_tokens);

// Read optional chat template and special tokens from GGUF metadata
bool gguf_try_read_chat_template(const char* path, std::string& out_template);
bool gguf_try_read_special_tokens(const char* path, std::vector<std::string>& out_tokens);

// Query last loaded projection tensor info (tag: "Wq","Wk","Wv").
const char* gguf_last_tensor_name(const char* tag);
size_t gguf_last_tensor_rows(const char* tag);
size_t gguf_last_tensor_cols(const char* tag);
