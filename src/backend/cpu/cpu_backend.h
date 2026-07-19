#pragma once
#include "../..//tensor/tensor.h"

// Multiply A (m x k) by B (k x n) into out (m x n). All tensors must be F32.
bool cpu_matmul(const Tensor* A, const Tensor* B, Tensor* out);

// Element-wise add: out = a + b
bool cpu_add(const Tensor* a, const Tensor* b, Tensor* out);

// Get a thread-local float workspace of at least `n` elements. Returned pointer
// remains valid until the next call on the same thread.
float* cpu_workspace(size_t n);
