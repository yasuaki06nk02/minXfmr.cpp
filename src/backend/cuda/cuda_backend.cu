#include "cuda_backend.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace {

struct CachedDeviceBuffer {
    float* ptr = nullptr;
    size_t elems = 0;
};

struct CudaState {
    cublasHandle_t handle = nullptr;
    bool ready = false;
    std::unordered_map<const float*, CachedDeviceBuffer> persistent;
    std::mutex mu;
};

CudaState& state() {
    static CudaState s;
    return s;
}

bool ensure_ready() {
    CudaState& s = state();
    if (s.ready) return true;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) return false;
    if (cublasCreate(&s.handle) != CUBLAS_STATUS_SUCCESS) return false;

    s.ready = true;
    return true;
}

void clear_persistent_buffers_locked(CudaState& s) {
    for (auto& kv : s.persistent) {
        if (kv.second.ptr) cudaFree(kv.second.ptr);
    }
    s.persistent.clear();
}

__global__ void matmul_rhs_transposed_kernel(
    const float* A,
    const float* B,
    float* C,
    size_t m,
    size_t n,
    size_t k) {
    const size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    const float* arow = A + row * k;
    const float* brow = B + col * k;
    float acc = 0.0f;
    for (size_t i = 0; i < k; ++i) acc += arow[i] * brow[i];
    C[row * n + col] = acc;
}

__global__ void matvec_strided_kernel(
    const float* vec,
    const float* mat,
    float* out,
    size_t K,
    size_t N,
    size_t row_stride) {
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    float acc = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        acc += vec[k] * mat[k * row_stride + col];
    }
    out[col] = acc;
}

__global__ void vec_dot_rows_kernel(
    const float* vec,
    const float* mat_rows,
    float* out,
    size_t K,
    size_t Nrows,
    size_t row_stride) {
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Nrows) return;

    const float* r = mat_rows + row * row_stride;
    float acc = 0.0f;
    for (size_t k = 0; k < K; ++k) acc += vec[k] * r[k];
    out[row] = acc;
}

__global__ void vec_dot_rows_ring_kernel(
    const float* vec,
    const float* ring,
    size_t head,
    size_t seq_max,
    size_t len,
    size_t K,
    size_t row_stride,
    float* out) {
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= len) return;

    const size_t phys = (head + row) % seq_max;
    const float* r = ring + phys * row_stride;
    float acc = 0.0f;
    for (size_t k = 0; k < K; ++k) acc += vec[k] * r[k];
    out[row] = acc;
}

__global__ void vec_mul_rows_cols_kernel(
    const float* vec,
    const float* mat_rows,
    float* out,
    size_t Nrows,
    size_t Ncols,
    size_t row_stride) {
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= Ncols) return;

    float acc = 0.0f;
    for (size_t row = 0; row < Nrows; ++row) {
        acc += vec[row] * mat_rows[row * row_stride + col];
    }
    out[col] = acc;
}

bool alloc_copy_to_device(const float* src, size_t n, float** dst) {
    if (!src || !dst) return false;
    if (cudaMalloc((void**)dst, n * sizeof(float)) != cudaSuccess) return false;
    if (cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*dst);
        *dst = nullptr;
        return false;
    }
    return true;
}

bool get_or_upload_persistent(const float* src, size_t n, float** dst) {
    if (!src || !dst) return false;
    CudaState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);

    auto it = s.persistent.find(src);
    if (it != s.persistent.end()) {
        if (it->second.elems == n && it->second.ptr != nullptr) {
            *dst = it->second.ptr;
            return true;
        }
        if (it->second.ptr) cudaFree(it->second.ptr);
        s.persistent.erase(it);
    }

    float* d = nullptr;
    if (cudaMalloc((void**)&d, n * sizeof(float)) != cudaSuccess) return false;
    if (cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return false;
    }

    s.persistent[src] = CachedDeviceBuffer{d, n};
    *dst = d;
    return true;
}

bool copy_to_host(float* dst, const float* src, size_t n) {
    return cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

} // namespace

bool cuda_backend_is_available() {
    return ensure_ready();
}

void cuda_backend_release_resources() {
    CudaState& s = state();
    {
        std::lock_guard<std::mutex> lock(s.mu);
        clear_persistent_buffers_locked(s);
    }
    if (s.handle) {
        cublasDestroy(s.handle);
        s.handle = nullptr;
    }
    s.ready = false;
}

bool cuda_backend_preload_tensor(const Tensor* t) {
    if (!ensure_ready()) return false;
    if (!t || t->type != DataType::F32 || !t->data) return false;
    float* d = nullptr;
    return get_or_upload_persistent((const float*)t->data, t->rows * t->cols, &d);
}

bool cuda_backend_matmul(const Tensor* A, const Tensor* B, Tensor* out) {
    if (!ensure_ready()) return false;
    if (!A || !B || !out) return false;
    if (A->type != DataType::F32 || B->type != DataType::F32 || out->type != DataType::F32) return false;
    if (A->cols != B->rows) return false;
    if (out->rows != A->rows || out->cols != B->cols) return false;

    const size_t m = A->rows;
    const size_t k = A->cols;
    const size_t n = B->cols;

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    if (!alloc_copy_to_device((const float*)A->data, m * k, &dA)) return false;
    if (!get_or_upload_persistent((const float*)B->data, k * n, &dB)) {
        cudaFree(dA);
        return false;
    }
    if (cudaMalloc((void**)&dC, m * n * sizeof(float)) != cudaSuccess) {
        cudaFree(dA);
        cudaFree(dB);
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major C = A * B via column-major identity: C^T = B^T * A^T.
    cublasStatus_t st = cublasSgemm(
        state().handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)n,
        (int)m,
        (int)k,
        &alpha,
        dB,
        (int)n,
        dA,
        (int)k,
        &beta,
        dC,
        (int)n);

    bool ok = (st == CUBLAS_STATUS_SUCCESS) && copy_to_host((float*)out->data, dC, m * n);

    cudaFree(dA);
    cudaFree(dC);

    return ok;
}

bool cuda_backend_matmul_rhs_transposed(const Tensor* A, const Tensor* B, Tensor* out) {
    if (!ensure_ready()) return false;
    if (!A || !B || !out) return false;
    if (A->type != DataType::F32 || B->type != DataType::F32 || out->type != DataType::F32) return false;
    if (A->cols != B->cols) return false;
    if (out->rows != A->rows || out->cols != B->rows) return false;

    const size_t m = A->rows;
    const size_t k = A->cols;
    const size_t n = B->rows;

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    if (!alloc_copy_to_device((const float*)A->data, m * k, &dA)) return false;
    if (!get_or_upload_persistent((const float*)B->data, n * k, &dB)) {
        cudaFree(dA);
        return false;
    }
    if (cudaMalloc((void**)&dC, m * n * sizeof(float)) != cudaSuccess) {
        cudaFree(dA);
        cudaFree(dB);
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x), (unsigned int)((m + block.y - 1) / block.y));
    matmul_rhs_transposed_kernel<<<grid, block>>>(dA, dB, dC, m, n, k);

    bool ok = (cudaGetLastError() == cudaSuccess) &&
              (cudaDeviceSynchronize() == cudaSuccess) &&
              copy_to_host((float*)out->data, dC, m * n);

    cudaFree(dA);
    cudaFree(dC);

    return ok;
}

bool cuda_backend_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride) {
    if (!ensure_ready()) return false;
    if (!vec || !mat || !out || K == 0 || N == 0) return false;

    float* dVec = nullptr;
    float* dMat = nullptr;
    float* dOut = nullptr;

    if (!alloc_copy_to_device(vec, K, &dVec)) return false;
    if (!get_or_upload_persistent(mat, K * mat_row_stride, &dMat)) {
        cudaFree(dVec);
        return false;
    }
    if (cudaMalloc((void**)&dOut, N * sizeof(float)) != cudaSuccess) {
        cudaFree(dVec);
        cudaFree(dMat);
        return false;
    }

    dim3 block(256);
    dim3 grid((unsigned int)((N + block.x - 1) / block.x));
    matvec_strided_kernel<<<grid, block>>>(dVec, dMat, dOut, K, N, mat_row_stride);

    bool ok = (cudaGetLastError() == cudaSuccess) &&
              (cudaDeviceSynchronize() == cudaSuccess) &&
              copy_to_host(out, dOut, N);

    cudaFree(dVec);
    cudaFree(dOut);

    return ok;
}

bool cuda_backend_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride) {
    if (!ensure_ready()) return false;
    if (!vec || !mat_rows || !out || K == 0 || Nrows == 0) return false;

    // For tiny workloads, CUDA launch/copy overhead dominates; let CPU fallback handle these.
    if (K * Nrows < 16384) return false;

    float* dVec = nullptr;
    float* dMat = nullptr;
    float* dOut = nullptr;
    bool mat_cached = false;

    if (!alloc_copy_to_device(vec, K, &dVec)) return false;
    if (Nrows >= 256) {
        mat_cached = get_or_upload_persistent(mat_rows, Nrows * row_stride, &dMat);
    }
    if (!mat_cached) {
        if (!alloc_copy_to_device(mat_rows, Nrows * row_stride, &dMat)) {
            cudaFree(dVec);
            return false;
        }
    }
    if (cudaMalloc((void**)&dOut, Nrows * sizeof(float)) != cudaSuccess) {
        cudaFree(dVec);
        if (!mat_cached) cudaFree(dMat);
        return false;
    }

    dim3 block(256);
    dim3 grid((unsigned int)((Nrows + block.x - 1) / block.x));
    vec_dot_rows_kernel<<<grid, block>>>(dVec, dMat, dOut, K, Nrows, row_stride);

    bool ok = (cudaGetLastError() == cudaSuccess) &&
              (cudaDeviceSynchronize() == cudaSuccess) &&
              copy_to_host(out, dOut, Nrows);

    cudaFree(dVec);
    if (!mat_cached) cudaFree(dMat);
    cudaFree(dOut);

    return ok;
}

bool cuda_backend_vec_dot_rows_ring(const float* vec, const float* ring, size_t head, size_t seq_max, size_t len, size_t K, size_t row_stride, float* out) {
    if (!ensure_ready()) return false;
    if (!vec || !ring || !out || K == 0 || len == 0 || seq_max == 0) return false;

    // Decode-time ring operations are often tiny; prefer CPU for this range.
    if (K * len < 16384) return false;

    float* dVec = nullptr;
    float* dRing = nullptr;
    float* dOut = nullptr;

    if (!alloc_copy_to_device(vec, K, &dVec)) return false;
    if (!alloc_copy_to_device(ring, seq_max * row_stride, &dRing)) {
        cudaFree(dVec);
        return false;
    }
    if (cudaMalloc((void**)&dOut, len * sizeof(float)) != cudaSuccess) {
        cudaFree(dVec);
        cudaFree(dRing);
        return false;
    }

    dim3 block(256);
    dim3 grid((unsigned int)((len + block.x - 1) / block.x));
    vec_dot_rows_ring_kernel<<<grid, block>>>(dVec, dRing, head, seq_max, len, K, row_stride, dOut);

    bool ok = (cudaGetLastError() == cudaSuccess) &&
              (cudaDeviceSynchronize() == cudaSuccess) &&
              copy_to_host(out, dOut, len);

    cudaFree(dVec);
    cudaFree(dRing);
    cudaFree(dOut);

    return ok;
}

bool cuda_backend_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride) {
    if (!ensure_ready()) return false;
    if (!vec || !mat_rows || !out || Nrows == 0 || Ncols == 0) return false;

    if (Nrows * Ncols < 16384) return false;

    float* dVec = nullptr;
    float* dMat = nullptr;
    float* dOut = nullptr;

    if (!alloc_copy_to_device(vec, Nrows, &dVec)) return false;
    if (!alloc_copy_to_device(mat_rows, Nrows * row_stride, &dMat)) {
        cudaFree(dVec);
        return false;
    }
    if (cudaMalloc((void**)&dOut, Ncols * sizeof(float)) != cudaSuccess) {
        cudaFree(dVec);
        cudaFree(dMat);
        return false;
    }

    dim3 block(256);
    dim3 grid((unsigned int)((Ncols + block.x - 1) / block.x));
    vec_mul_rows_cols_kernel<<<grid, block>>>(dVec, dMat, dOut, Nrows, Ncols, row_stride);

    bool ok = (cudaGetLastError() == cudaSuccess) &&
              (cudaDeviceSynchronize() == cudaSuccess) &&
              copy_to_host(out, dOut, Ncols);

    cudaFree(dVec);
    cudaFree(dMat);
    cudaFree(dOut);

    return ok;
}
