#pragma once
#include "../tensor/tensor.h"

// Simple RMSNorm implementation for 1D vectors (row)
bool rmsnorm_forward(const Tensor* input, Tensor* output, float epsilon = 1e-6f);
