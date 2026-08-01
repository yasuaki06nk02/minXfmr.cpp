#pragma once

#include "../../tensor/tensor.h"

bool cuda_backend_is_available();
void cuda_backend_release_resources();
bool cuda_backend_preload_tensor(const Tensor* t);

// CUDA quant parity mode control:
//  -1: follow environment variable MINXFMR_CUDA_QUANT_PARITY
//   0: force quantized CUDA kernels path
//   1: force dequant-F32 parity path
void cuda_backend_set_quant_parity_mode(int mode);
int cuda_backend_get_quant_parity_mode();

// When the last CUDA operation failed in a way that prevented preload/upload,
// this returns a human-readable message (owned by the backend). Empty when
// no message is available.
const char* cuda_backend_last_error_msg();

bool cuda_backend_matmul(const Tensor* A, const Tensor* B, Tensor* out);
bool cuda_backend_matmul_rhs_transposed(const Tensor* A, const Tensor* B, Tensor* out);
bool cuda_backend_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride);
bool cuda_backend_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride);
bool cuda_backend_vec_dot_rows_ring(const float* vec, const float* ring, size_t head, size_t seq_max, size_t len, size_t K, size_t row_stride, float* out);
bool cuda_backend_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride);

// Debug helper: decode quant blocks on device and compare to host-side dequant.
// Prints up to `max_mismatches` mismatches and returns true if max abs diff < 1e-3.
bool cuda_backend_compare_dequant(const Tensor* t, size_t max_mismatches);

// Diagnostic: decode a single Q4_K block on the device into host memory.
// out256 must point to a buffer of at least TENSOR_Q4_K_QK_K floats.
// Returns true on success.
bool cuda_backend_dequant_block_device(const Tensor* t, size_t row, size_t block, float* out256);
