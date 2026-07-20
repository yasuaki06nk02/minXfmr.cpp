#pragma once
#include "../tensor/tensor.h"

// Minimal decoder block: input -> rmsnorm -> self-attention -> ffn.
// Optional Wq/Wk/Wv tensors can be provided. Optional FFN weights (gate/up/down)
// enable LLaMA-style SwiGLU feed-forward.
// Single-layer forward with optional preallocated workspace for attention
// scores. The workspace pointer, if provided, must have room for at least
// `scores_workspace_len` floats and will be used instead of calling the
// allocator repeatedly during the layer forward.
bool transformer_forward_single_layer(
	const Tensor* input,
	Tensor* output,
	struct KVCache* cache = nullptr,
	size_t layer = 0,
	size_t n_head = 0,
	size_t n_head_kv = 0,
	const Tensor* Wq_in = nullptr,
	const Tensor* Wk_in = nullptr,
	const Tensor* Wv_in = nullptr,
	const Tensor* Wo_in = nullptr,
	const Tensor* Wattn_norm_in = nullptr,
	const Tensor* Wffn_norm_in = nullptr,
	const Tensor* Wffn_gate_in = nullptr,
	const Tensor* Wffn_up_in = nullptr,
	const Tensor* Wffn_down_in = nullptr,
	float* scores_workspace = nullptr,
	size_t scores_workspace_len = 0);

// If enabled, square projection matrices (rows == cols == d_in) are treated as transposed.
void transformer_set_transpose_square_weights(bool enabled);

// Fine-grained control for square projection orientation.
// Each flag applies only when the weight matrix is square.
void transformer_set_transpose_square_weights_for_all(bool wq, bool wk, bool wv, bool wo);
