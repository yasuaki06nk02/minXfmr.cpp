#pragma once
#include "../tensor/tensor.h"

// Apply RoPE in-place to a tensor shaped [seq x (n_heads * head_dim)].
// start_pos is the absolute sequence position of row 0.
void rope_apply(Tensor* tensor, size_t start_pos, size_t n_heads, size_t head_dim, float theta = 10000.0f);
