#pragma once
#include "../..//tensor/tensor.h"

// Multiply A (m x k) by B (k x n) into out (m x n). All tensors must be F32.
bool cpu_matmul(const Tensor* A, const Tensor* B, Tensor* out);

// Element-wise add: out = a + b
bool cpu_add(const Tensor* a, const Tensor* b, Tensor* out);

// Get a thread-local float workspace of at least `n` elements. Returned pointer
// remains valid until the next call on the same thread.
float* cpu_workspace(size_t n);

// Reset the per-thread workspace allocation offset to zero. Call at logical
// boundaries (e.g., at the start of a generation) to reuse the buffer.
void cpu_workspace_reset(bool shrink = false);

// Multiply vector (length K) by matrix (K x N) into out (length N).
// mat is expected in row-major order with each row of length N: element (k,n) is mat[k*N + n].
bool cpu_matvec(const float* vec, const float* mat, float* out, size_t K, size_t N);

// Like cpu_matvec but allows specifying the row stride in the underlying
// storage for the matrix: element (k,n) is located at mat[k*mat_row_stride + n].
bool cpu_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride);

// Request a thread-local workspace buffer of at least `n` floats. Alias for cpu_workspace.
float* cpu_request_workspace(size_t n);

// Compute out[j] = dot(vec[0..K-1], mat_rows[j*row_stride + 0..K-1]) for j in [0..Nrows)
// mat_rows is an array of Nrows rows, each at least row_stride long.
bool cpu_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride);

// Compute out[col] = sum_{row=0..Nrows-1} vec[row] * mat_rows[row*row_stride + col] for col in [0..Ncols)
// This computes vec^T * Mat where Mat is Nrows x Ncols stored row-major with stride row_stride.
bool cpu_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride);
