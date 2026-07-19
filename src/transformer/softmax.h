#pragma once
#include <cstddef>

// Softmax over a row in-place. Numerically stable implementation.
void softmax_row(float* row, size_t len);
