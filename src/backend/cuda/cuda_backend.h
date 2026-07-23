#pragma once

#include "../../tensor/tensor.h"

bool cuda_backend_is_available();

bool cuda_backend_matmul(const Tensor* A, const Tensor* B, Tensor* out);
bool cuda_backend_matmul_rhs_transposed(const Tensor* A, const Tensor* B, Tensor* out);
bool cuda_backend_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride);
bool cuda_backend_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride);
bool cuda_backend_vec_dot_rows_ring(const float* vec, const float* ring, size_t head, size_t seq_max, size_t len, size_t K, size_t row_stride, float* out);
bool cuda_backend_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride);
