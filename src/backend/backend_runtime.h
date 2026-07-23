#pragma once

#include "../tensor/tensor.h"

// Runtime backend selector.
enum class BackendKind {
    CPU = 0,
    CUDA = 1,
};

// Initialize backend from environment variable MINXFMR_BACKEND.
// Values: cpu | cuda | auto (default).
void backend_initialize_from_env();

// Explicitly request backend selection at runtime.
// Returns false when CUDA was requested but is unavailable.
bool backend_set_kind(BackendKind kind);

BackendKind backend_get_kind();
const char* backend_get_name();
bool backend_using_cuda();

bool backend_matmul(const Tensor* A, const Tensor* B, Tensor* out);
bool backend_matmul_rhs_transposed(const Tensor* A, const Tensor* B, Tensor* out);
bool backend_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride);
bool backend_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride);
bool backend_vec_dot_rows_ring(const float* vec, const float* ring, size_t head, size_t seq_max, size_t len, size_t K, size_t row_stride, float* out);
bool backend_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride);

float* backend_request_workspace(size_t n);
void backend_workspace_reset(bool shrink = false);
