#pragma once
#include "../tensor/tensor.h"

// Simple feed forward: out = input * W (matrix multiply) + b (b broadcast)
bool ffn_forward(const Tensor* input, const Tensor* W, const Tensor* b, Tensor* out);
