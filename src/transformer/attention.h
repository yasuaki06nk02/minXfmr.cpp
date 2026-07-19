#pragma once
#include "../tensor/tensor.h"

// Very small attention: compute Q*K^T (without softmax) into out (rows x rows)
bool attention_qk(const Tensor* Q, const Tensor* K, Tensor* out);

// Compute attention output using cached K/V for a given layer.
// Q: q_seq x d, uses cache->keys[layer] (k_seq x d) and cache->vals[layer] (k_seq x d)
// out: q_seq x d (result of softmax(Q K^T) * V)
bool attention_apply_with_cache(const Tensor* Q, const struct KVCache* cache, size_t layer, Tensor* out);
// Compute attention output directly from Q,K,V tensors: out = softmax(Q K^T) V
bool attention_apply_direct(const Tensor* Q, const Tensor* K, const Tensor* V, Tensor* out);

// Set to true to emit a single detailed debug line on next attention call.
void attention_set_debug_once(bool v);
