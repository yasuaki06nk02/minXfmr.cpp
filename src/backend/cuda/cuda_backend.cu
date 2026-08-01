#include "cuda_backend.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <vector>
#include <string>

namespace {

struct CachedDeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
};
struct CachedDequantBuffer {
    float* ptr = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t src_bytes = 0;
    DataType src_type = DataType::F32;
};

struct CudaState {
    cublasHandle_t handle = nullptr;
    bool ready = false;
    std::unordered_map<const void*, CachedDeviceBuffer> persistent;
    std::unordered_map<const void*, CachedDequantBuffer> dequant_f32;
    std::mutex mu;
    std::string last_error;
};

CudaState& state() {
    static CudaState s;
    return s;
}

std::atomic<int>& quant_parity_mode_override() {
    static std::atomic<int> mode{-1};
    return mode;
}

bool cuda_quant_kernels_enabled() {
    static int mode = -1;
    if (mode >= 0) return mode == 1;
    const char* v = std::getenv("MINXFMR_CUDA_QUANT");
    if (!v) {
        // llama.cpp-like default: keep quantized matmul on GPU unless explicitly disabled.
        mode = 1;
    } else {
        mode = (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T') ? 1 : 0;
    }
    if (mode == 0) {
        std::fprintf(stderr, "[cuda] quantized matmul kernels disabled by MINXFMR_CUDA_QUANT; falling back to CPU for quantized weights\n");
    }
    return mode == 1;
}

bool cuda_quant_parity_mode_enabled() {
    const int override_mode = quant_parity_mode_override().load(std::memory_order_relaxed);
    if (override_mode >= 0) return override_mode == 1;

    static int env_mode = -1;
    if (env_mode >= 0) return env_mode == 1;

    const char* v = std::getenv("MINXFMR_CUDA_QUANT_PARITY");
    env_mode = (v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T')) ? 1 : 0;
    if (env_mode == 1) {
        std::fprintf(stderr, "[cuda] quantized matmul parity mode enabled; host-side dequantizing weights to F32 device cache for numerical parity\n");
    }
    return env_mode == 1;
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
    for (auto& kv : s.dequant_f32) {
        if (kv.second.ptr) cudaFree(kv.second.ptr);
    }
    s.dequant_f32.clear();
}

static float fp16_to_fp32_host(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t f = h & 0x3ff;
    uint32_t out;
    if (e == 0) {
        if (f == 0) {
            out = s << 31;
        } else {
            e = 1;
            while ((f & 0x400) == 0) { f <<= 1; --e; }
            f &= 0x3ff;
            out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7f800000 | (f << 13);
    } else {
        out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
    }
    float v;
    std::memcpy(&v, &out, sizeof(v));
    return v;
}

static inline void get_scale_min_k4_host(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 4] >> 6) << 4);
    }
}

static void dequant_q4_k_block_host(const uint8_t* blk, float* dst256) {
    uint16_t hd = 0;
    uint16_t hm = 0;
    std::memcpy(&hd, blk + 0, sizeof(hd));
    std::memcpy(&hm, blk + 2, sizeof(hm));
    const float d = fp16_to_fp32_host(hd);
    const float dmin = fp16_to_fp32_host(hm);

    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;

    int is = 0;
    for (int j = 0; j < (int)TENSOR_Q4_K_QK_K; j += 64) {
        uint8_t sc = 0;
        uint8_t m = 0;

        get_scale_min_k4_host(is + 0, scales, sc, m);
        const float d1 = d * sc;
        const float m1 = dmin * m;

        get_scale_min_k4_host(is + 1, scales, sc, m);
        const float d2 = d * sc;
        const float m2 = dmin * m;

        for (int l = 0; l < 32; ++l) dst256[j + l] = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) dst256[j + 32 + l] = d2 * (q[l] >> 4) - m2;

        q += 32;
        is += 2;
    }
}

static void dequant_q5_0_block_host(const uint8_t* blk, float* dst32) {
    uint16_t hd = 0;
    std::memcpy(&hd, blk, sizeof(hd));
    const float d = fp16_to_fp32_host(hd);

    const uint8_t* qh = blk + 2;
    const uint8_t* qs = blk + 6;

    uint32_t hmask = 0;
    hmask |= (uint32_t)qh[0];
    hmask |= (uint32_t)qh[1] << 8;
    hmask |= (uint32_t)qh[2] << 16;
    hmask |= (uint32_t)qh[3] << 24;

    for (int i = 0; i < 16; ++i) {
        const uint8_t ql = qs[i];
        const int low0 = (int)(ql & 0x0F);
        const int low1 = (int)(ql >> 4);
        const int high0 = (int)((hmask >> i) & 1u);
        const int high1 = (int)((hmask >> (i + 16)) & 1u);
        const int q0 = (high0 << 4) | low0;
        const int q1 = (high1 << 4) | low1;
        dst32[i] = d * (float)(q0 - 16);
        dst32[i + 16] = d * (float)(q1 - 16);
    }
}

static void dequant_q8_0_block_host(const uint8_t* blk, float* dst32) {
    uint16_t hd = 0;
    std::memcpy(&hd, blk, sizeof(hd));
    const float d = fp16_to_fp32_host(hd);
    const int8_t* qs = (const int8_t*)(blk + 2);
    for (int i = 0; i < 32; ++i) dst32[i] = d * (float)qs[i];
}

__device__ __forceinline__ uint16_t load_u16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

__device__ __forceinline__ float fp16_to_fp32_device(uint16_t h) {
    uint32_t s = (h >> 15) & 1u;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t f = h & 0x3ffu;
    uint32_t out;
    if (e == 0u) {
        if (f == 0u) {
            out = s << 31;
        } else {
            e = 1u;
            while ((f & 0x400u) == 0u) {
                f <<= 1;
                --e;
            }
            f &= 0x3ffu;
            out = (s << 31) | ((e + (127u - 15u)) << 23) | (f << 13);
        }
    } else if (e == 31u) {
        out = (s << 31) | 0x7f800000u | (f << 13);
    } else {
        out = (s << 31) | ((e + (127u - 15u)) << 23) | (f << 13);
    }
    return __uint_as_float(out);
}

__device__ __forceinline__ void get_scale_min_k4_device(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        d = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 4] >> 6) << 4);
    }
}

__device__ __forceinline__ void dequant_q4_k_block_device(const uint8_t* blk, float* dst256) {
    const float d = fp16_to_fp32_device(load_u16(blk + 0));
    const float dmin = fp16_to_fp32_device(load_u16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;

    int is = 0;
    for (int j = 0; j < (int)TENSOR_Q4_K_QK_K; j += 64) {
        uint8_t sc = 0;
        uint8_t m = 0;

        get_scale_min_k4_device(is + 0, scales, sc, m);
        const float d1 = d * sc;
        const float m1 = dmin * m;

        get_scale_min_k4_device(is + 1, scales, sc, m);
        const float d2 = d * sc;
        const float m2 = dmin * m;

        #pragma unroll
        for (int l = 0; l < 32; ++l) dst256[j + l] = d1 * (float)(q[l] & 0xFu) - m1;
        #pragma unroll
        for (int l = 0; l < 32; ++l) dst256[j + 32 + l] = d2 * (float)(q[l] >> 4) - m2;

        q += 32;
        is += 2;
    }
}

__device__ __forceinline__ void dequant_q5_0_block_device(const uint8_t* blk, float* dst32) {
    const float d = fp16_to_fp32_device(load_u16(blk));
    const uint8_t* qh = blk + 2;
    const uint8_t* qs = blk + 6;

    uint32_t hmask = 0;
    hmask |= (uint32_t)qh[0];
    hmask |= (uint32_t)qh[1] << 8;
    hmask |= (uint32_t)qh[2] << 16;
    hmask |= (uint32_t)qh[3] << 24;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t ql = qs[i];
        const int low0 = (int)(ql & 0x0F);
        const int low1 = (int)(ql >> 4);
        const int high0 = (int)((hmask >> i) & 1u);
        const int high1 = (int)((hmask >> (i + 16)) & 1u);
        const int q0 = (high0 << 4) | low0;
        const int q1 = (high1 << 4) | low1;
        dst32[i] = d * (float)(q0 - 16);
        dst32[i + 16] = d * (float)(q1 - 16);
    }
}

__device__ __forceinline__ void dequant_q8_0_block_device(const uint8_t* blk, float* dst32) {
    const float d = fp16_to_fp32_device(load_u16(blk));
    const int8_t* qs = (const int8_t*)(blk + 2);
    #pragma unroll
    for (int i = 0; i < 32; ++i) dst32[i] = d * (float)qs[i];
}

bool alloc_copy_to_device(const void* src, size_t bytes, void** dst) {
    if (!src || !dst || bytes == 0) return false;
    if (cudaMalloc(dst, bytes) != cudaSuccess) return false;
    if (cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*dst);
        *dst = nullptr;
        return false;
    }
    return true;
}

bool get_or_upload_persistent(const void* src, size_t bytes, void** dst) {
    if (!src || !dst || bytes == 0) return false;
    CudaState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);

    auto it = s.persistent.find(src);
    if (it != s.persistent.end()) {
        if (it->second.bytes == bytes && it->second.ptr != nullptr) {
            *dst = it->second.ptr;
            return true;
        }
        if (it->second.ptr) cudaFree(it->second.ptr);
        s.persistent.erase(it);
    }

    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) {
        // record a helpful error message for diagnostics
        state().last_error = std::string("cudaMalloc failed allocating ") + std::to_string(bytes) + " bytes";
        return false;
    }
    if (cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        state().last_error = std::string("cudaMemcpy (HostToDevice) failed for ") + std::to_string(bytes) + " bytes";
        cudaFree(d);
        return false;
    }

    s.persistent[src] = CachedDeviceBuffer{d, bytes};
    *dst = d;
    return true;
}

bool copy_to_host(void* dst, const void* src, size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

size_t quant_block_elems(DataType type) {
    if (type == DataType::Q4_K) return TENSOR_Q4_K_QK_K;
    if (type == DataType::Q5_0) return TENSOR_Q5_0_QK;
    if (type == DataType::Q8_0) return TENSOR_Q8_0_QK;
    return 0;
}

size_t quant_row_bytes(DataType type, size_t cols) {
    if (type == DataType::Q4_K) return tensor_q4_k_row_bytes(cols);
    if (type == DataType::Q5_0) return tensor_q5_0_row_bytes(cols);
    if (type == DataType::Q8_0) return tensor_q8_0_row_bytes(cols);
    return 0;
}

bool get_or_build_persistent_dequant_f32(const Tensor* t, float** dst) {
    if (!t || !dst || !t->data || t->bytes == 0) return false;
    if (!(t->type == DataType::Q4_K || t->type == DataType::Q5_0 || t->type == DataType::Q8_0)) return false;

    {
        CudaState& s = state();
        std::lock_guard<std::mutex> lock(s.mu);
        auto it = s.dequant_f32.find(t->data);
        if (it != s.dequant_f32.end()) {
            const CachedDequantBuffer& e = it->second;
            if (e.ptr && e.rows == t->rows && e.cols == t->cols && e.src_bytes == t->bytes && e.src_type == t->type) {
                *dst = e.ptr;
                return true;
            }
            if (e.ptr) cudaFree(e.ptr);
            s.dequant_f32.erase(it);
        }
    }

    size_t row_bytes = 0;
    switch (t->type) {
        case DataType::Q4_K:
            row_bytes = tensor_q4_k_row_bytes(t->cols);
            break;
        case DataType::Q5_0:
            row_bytes = tensor_q5_0_row_bytes(t->cols);
            break;
        case DataType::Q8_0:
            row_bytes = tensor_q8_0_row_bytes(t->cols);
            break;
        default:
            return false;
    }
    if (row_bytes == 0 || t->bytes < t->rows * row_bytes) return false;

    std::vector<float> host(t->rows * t->cols);
    const uint8_t* src = (const uint8_t*)t->data;
    if (t->type == DataType::Q4_K) {
        const size_t blocks_per_row = t->cols / TENSOR_Q4_K_QK_K;
        for (size_t r = 0; r < t->rows; ++r) {
            const uint8_t* rowp = src + r * row_bytes;
            float* outp = host.data() + r * t->cols;
            for (size_t b = 0; b < blocks_per_row; ++b) {
                dequant_q4_k_block_host(rowp + b * TENSOR_Q4_K_BLOCK_SIZE, outp + b * TENSOR_Q4_K_QK_K);
            }
        }
    } else if (t->type == DataType::Q5_0) {
        const size_t blocks_per_row = t->cols / TENSOR_Q5_0_QK;
        for (size_t r = 0; r < t->rows; ++r) {
            const uint8_t* rowp = src + r * row_bytes;
            float* outp = host.data() + r * t->cols;
            for (size_t b = 0; b < blocks_per_row; ++b) {
                dequant_q5_0_block_host(rowp + b * TENSOR_Q5_0_BLOCK_SIZE, outp + b * TENSOR_Q5_0_QK);
            }
        }
    } else {
        const size_t blocks_per_row = t->cols / TENSOR_Q8_0_QK;
        for (size_t r = 0; r < t->rows; ++r) {
            const uint8_t* rowp = src + r * row_bytes;
            float* outp = host.data() + r * t->cols;
            for (size_t b = 0; b < blocks_per_row; ++b) {
                dequant_q8_0_block_host(rowp + b * TENSOR_Q8_0_BLOCK_SIZE, outp + b * TENSOR_Q8_0_QK);
            }
        }
    }

    float* dF = nullptr;
    const size_t total = t->rows * t->cols;
    if (cudaMalloc((void**)&dF, total * sizeof(float)) != cudaSuccess) return false;

    if (cudaMemcpy(dF, host.data(), total * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dF);
        return false;
    }

    {
        CudaState& s = state();
        std::lock_guard<std::mutex> lock(s.mu);
        s.dequant_f32[t->data] = CachedDequantBuffer{dF, t->rows, t->cols, t->bytes, t->type};
    }
    *dst = dF;
    return true;
}

template <size_t BlockElems, size_t BlockSize>
__global__ void matmul_quant_kernel(
    const float* A,
    const uint8_t* B,
    float* C,
    size_t m,
    size_t n,
    size_t k,
    size_t row_bytes) {
    const size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (row < m) && (col < n);

    const size_t block_col0 = (size_t)blockIdx.x * blockDim.x;
    const size_t block = block_col0 / BlockElems;
    const size_t offset = col % BlockElems;
    float acc = 0.0f;
    __shared__ float tile[BlockElems];

    for (size_t kk = 0; kk < k; ++kk) {
        const uint8_t* brow = B + kk * row_bytes + block * BlockSize;

        // Decode one quant block per CTA, then reuse it across all threads.
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            if constexpr (BlockElems == TENSOR_Q4_K_QK_K) {
                dequant_q4_k_block_device(brow, tile);
            } else if constexpr (BlockElems == TENSOR_Q5_0_QK) {
                dequant_q5_0_block_device(brow, tile);
            } else {
                dequant_q8_0_block_device(brow, tile);
            }
        }
        __syncthreads();

        if (active) {
            const float* arow = A + row * k;
            acc += arow[kk] * tile[offset];
        }
        __syncthreads();
    }

    if (active) C[row * n + col] = acc;
}

template <size_t BlockElems, size_t BlockSize>
__global__ void matmul_rhs_quant_kernel(
    const float* A,
    const uint8_t* B,
    float* C,
    size_t m,
    size_t n,
    size_t k,
    size_t row_bytes) {
    const size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    const float* arow = A + row * k;
    const size_t blocks_per_row = k / BlockElems;
    float acc = 0.0f;
    float tile[BlockElems];

    for (size_t blk = 0; blk < blocks_per_row; ++blk) {
        const uint8_t* brow = B + col * row_bytes + blk * BlockSize;
        if constexpr (BlockElems == TENSOR_Q4_K_QK_K) {
            dequant_q4_k_block_device(brow, tile);
        } else if constexpr (BlockElems == TENSOR_Q5_0_QK) {
            dequant_q5_0_block_device(brow, tile);
        } else {
            dequant_q8_0_block_device(brow, tile);
        }
        #pragma unroll
        for (size_t i = 0; i < BlockElems; ++i) {
            acc += arow[blk * BlockElems + i] * tile[i];
        }
    }

    C[row * n + col] = acc;
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

} // namespace

void cuda_backend_set_quant_parity_mode(int mode) {
    if (mode < -1 || mode > 1) return;
    quant_parity_mode_override().store(mode, std::memory_order_relaxed);
}

int cuda_backend_get_quant_parity_mode() {
    return quant_parity_mode_override().load(std::memory_order_relaxed);
}

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
    if (!t || !t->data || t->bytes == 0) return false;
    if (t->type == DataType::Q4_K || t->type == DataType::Q5_0 || t->type == DataType::Q8_0) {
        if (cuda_quant_parity_mode_enabled()) {
            float* dF = nullptr;
            return get_or_build_persistent_dequant_f32(t, &dF);
        }
    }
    void* d = nullptr;
    return get_or_upload_persistent(t->data, t->bytes, &d);
}

bool cuda_backend_matmul(const Tensor* A, const Tensor* B, Tensor* out) {
    if (!ensure_ready()) return false;
    if (!A || !B || !out) return false;
    if (A->type != DataType::F32 || out->type != DataType::F32) return false;
    if (A->cols != B->rows) return false;
    if (out->rows != A->rows || out->cols != B->cols) return false;

    const size_t m = A->rows;
    const size_t k = A->cols;
    const size_t n = B->cols;

    if (B->type == DataType::F32) {
        void* dA = nullptr;
        void* dB = nullptr;
        void* dC = nullptr;

        if (!alloc_copy_to_device(A->data, A->bytes, &dA)) return false;
        if (!get_or_upload_persistent(B->data, B->bytes, &dB)) {
            cudaFree(dA);
            return false;
        }
        if (cudaMalloc(&dC, m * n * sizeof(float)) != cudaSuccess) {
            cudaFree(dA);
            return false;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(
            state().handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)n,
            (int)m,
            (int)k,
            &alpha,
            (const float*)dB,
            (int)n,
            (const float*)dA,
            (int)k,
            &beta,
            (float*)dC,
            (int)n);

        bool ok = (st == CUBLAS_STATUS_SUCCESS) && copy_to_host(out->data, dC, m * n * sizeof(float));

        cudaFree(dA);
        cudaFree(dC);
        return ok;
    }

    if (!cuda_quant_kernels_enabled()) return false;

    if (B->type != DataType::Q4_K && B->type != DataType::Q5_0 && B->type != DataType::Q8_0) return false;

    const size_t block_elems = quant_block_elems(B->type);
    if (k == 0 || n == 0 || block_elems == 0 || (k % block_elems) != 0 || (n % block_elems) != 0) return false;

    if (cuda_quant_parity_mode_enabled()) {
        void* dA = nullptr;
        float* dBf = nullptr;
        void* dC = nullptr;
        if (!alloc_copy_to_device(A->data, A->bytes, &dA)) return false;
        if (!get_or_build_persistent_dequant_f32(B, &dBf)) {
            cudaFree(dA);
            return false;
        }
        if (cudaMalloc(&dC, m * n * sizeof(float)) != cudaSuccess) {
            cudaFree(dA);
            return false;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(
            state().handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)n,
            (int)m,
            (int)k,
            &alpha,
            (const float*)dBf,
            (int)n,
            (const float*)dA,
            (int)k,
            &beta,
            (float*)dC,
            (int)n);

        bool ok = (st == CUBLAS_STATUS_SUCCESS) && copy_to_host(out->data, dC, m * n * sizeof(float));

        cudaFree(dA);
        cudaFree(dC);
        return ok;
    }

    void* dA = nullptr;
    uint8_t* dBq = nullptr;
    void* dC = nullptr;
    if (!alloc_copy_to_device(A->data, A->bytes, &dA)) return false;
    if (!get_or_upload_persistent(B->data, B->bytes, (void**)&dBq)) {
        cudaFree(dA);
        return false;
    }
    if (cudaMalloc(&dC, m * n * sizeof(float)) != cudaSuccess) {
        cudaFree(dA);
        return false;
    }

    const dim3 block(16, 16);
    const dim3 grid((unsigned int)((n + block.x - 1) / block.x), (unsigned int)((m + block.y - 1) / block.y));
    const size_t row_bytes = quant_row_bytes(B->type, B->cols);
    if (row_bytes == 0) {
        cudaFree(dA);
        cudaFree(dC);
        return false;
    }
    if (B->type == DataType::Q4_K) {
        matmul_quant_kernel<TENSOR_Q4_K_QK_K, TENSOR_Q4_K_BLOCK_SIZE><<<grid, block>>>(
            (const float*)dA,
            dBq,
            (float*)dC,
            m,
            n,
            k,
            row_bytes);
    } else if (B->type == DataType::Q5_0) {
        matmul_quant_kernel<TENSOR_Q5_0_QK, TENSOR_Q5_0_BLOCK_SIZE><<<grid, block>>>(
            (const float*)dA,
            dBq,
            (float*)dC,
            m,
            n,
            k,
            row_bytes);
    } else {
        matmul_quant_kernel<TENSOR_Q8_0_QK, TENSOR_Q8_0_BLOCK_SIZE><<<grid, block>>>(
            (const float*)dA,
            dBq,
            (float*)dC,
            m,
            n,
            k,
            row_bytes);
    }

    bool ok = (cudaGetLastError() == cudaSuccess) &&
              (cudaDeviceSynchronize() == cudaSuccess) &&
              copy_to_host(out->data, dC, m * n * sizeof(float));

    cudaFree(dA);
    cudaFree(dC);
    return ok;
}

bool cuda_backend_matmul_rhs_transposed(const Tensor* A, const Tensor* B, Tensor* out) {
    if (!ensure_ready()) return false;
    if (!A || !B || !out) return false;
    if (A->type != DataType::F32 || out->type != DataType::F32) return false;
    if (A->cols != B->cols) return false;
    if (out->rows != A->rows || out->cols != B->rows) return false;

    const size_t m = A->rows;
    const size_t k = A->cols;
    const size_t n = B->rows;

    if (B->type == DataType::F32) {
        void* dA = nullptr;
        void* dB = nullptr;
        void* dC = nullptr;

        if (!alloc_copy_to_device(A->data, A->bytes, &dA)) return false;
        if (!get_or_upload_persistent(B->data, B->bytes, &dB)) {
            cudaFree(dA);
            return false;
        }
        if (cudaMalloc(&dC, m * n * sizeof(float)) != cudaSuccess) {
            cudaFree(dA);
            return false;
        }

        dim3 block(16, 16);
        dim3 grid((unsigned int)((n + block.x - 1) / block.x), (unsigned int)((m + block.y - 1) / block.y));
        matmul_rhs_transposed_kernel<<<grid, block>>>(
            (const float*)dA,
            (const float*)dB,
            (float*)dC,
            m,
            n,
            k);

        bool ok = (cudaGetLastError() == cudaSuccess) &&
                  (cudaDeviceSynchronize() == cudaSuccess) &&
                  copy_to_host(out->data, dC, m * n * sizeof(float));

        cudaFree(dA);
        cudaFree(dC);
        return ok;
    }

    if (!cuda_quant_kernels_enabled()) return false;

    if (B->type != DataType::Q4_K && B->type != DataType::Q5_0 && B->type != DataType::Q8_0) return false;

    const size_t block_elems = quant_block_elems(B->type);
    if (k == 0 || n == 0 || block_elems == 0 || (k % block_elems) != 0) return false;

    if (cuda_quant_parity_mode_enabled()) {
        void* dA = nullptr;
        float* dBf = nullptr;
        void* dC = nullptr;
        if (!alloc_copy_to_device(A->data, A->bytes, &dA)) return false;
        if (!get_or_build_persistent_dequant_f32(B, &dBf)) {
            cudaFree(dA);
            return false;
        }
        if (cudaMalloc(&dC, m * n * sizeof(float)) != cudaSuccess) {
            cudaFree(dA);
            return false;
        }

        dim3 block(16, 16);
        dim3 grid((unsigned int)((n + block.x - 1) / block.x), (unsigned int)((m + block.y - 1) / block.y));

        matmul_rhs_transposed_kernel<<<grid, block>>>(
            (const float*)dA,
            (const float*)dBf,
            (float*)dC,
            m,
            n,
            k);

        bool ok = (cudaGetLastError() == cudaSuccess) &&
                  (cudaDeviceSynchronize() == cudaSuccess) &&
                  copy_to_host(out->data, dC, m * n * sizeof(float));

        cudaFree(dA);
        cudaFree(dC);
        return ok;
    }

    void* dA = nullptr;
    uint8_t* dBq = nullptr;
    void* dC = nullptr;
    if (!alloc_copy_to_device(A->data, A->bytes, &dA)) return false;
    if (!get_or_upload_persistent(B->data, B->bytes, (void**)&dBq)) {
        cudaFree(dA);
        return false;
    }
    if (cudaMalloc(&dC, m * n * sizeof(float)) != cudaSuccess) {
        cudaFree(dA);
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x), (unsigned int)((m + block.y - 1) / block.y));

    const size_t row_bytes = quant_row_bytes(B->type, B->cols);
    if (row_bytes == 0) {
        cudaFree(dA);
        cudaFree(dC);
        return false;
    }
    if (B->type == DataType::Q4_K) {
        matmul_rhs_quant_kernel<TENSOR_Q4_K_QK_K, TENSOR_Q4_K_BLOCK_SIZE><<<grid, block>>>(
            (const float*)dA,
            dBq,
            (float*)dC,
            m,
            n,
            k,
            row_bytes);
    } else if (B->type == DataType::Q5_0) {
        matmul_rhs_quant_kernel<TENSOR_Q5_0_QK, TENSOR_Q5_0_BLOCK_SIZE><<<grid, block>>>(
            (const float*)dA,
            dBq,
            (float*)dC,
            m,
            n,
            k,
            row_bytes);
    } else {
        matmul_rhs_quant_kernel<TENSOR_Q8_0_QK, TENSOR_Q8_0_BLOCK_SIZE><<<grid, block>>>(
            (const float*)dA,
            dBq,
            (float*)dC,
            m,
            n,
            k,
            row_bytes);
    }

    bool ok = (cudaGetLastError() == cudaSuccess) &&
              (cudaDeviceSynchronize() == cudaSuccess) &&
              copy_to_host(out->data, dC, m * n * sizeof(float));

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

    if (!alloc_copy_to_device(vec, K * sizeof(float), (void**)&dVec)) return false;
    if (!get_or_upload_persistent(mat, K * mat_row_stride * sizeof(float), (void**)&dMat)) {
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
              copy_to_host(out, dOut, N * sizeof(float));

    cudaFree(dVec);
    cudaFree(dOut);
    return ok;
}

bool cuda_backend_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride) {
    if (!ensure_ready()) return false;
    if (!vec || !mat_rows || !out || K == 0 || Nrows == 0) return false;

    float* dVec = nullptr;
    float* dMat = nullptr;
    float* dOut = nullptr;
    bool mat_cached = false;

    if (!alloc_copy_to_device(vec, K * sizeof(float), (void**)&dVec)) return false;
    if (Nrows >= 256) {
        mat_cached = get_or_upload_persistent(mat_rows, Nrows * row_stride * sizeof(float), (void**)&dMat);
    }
    if (!mat_cached) {
        if (!alloc_copy_to_device(mat_rows, Nrows * row_stride * sizeof(float), (void**)&dMat)) {
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
              copy_to_host(out, dOut, Nrows * sizeof(float));

    cudaFree(dVec);
    if (!mat_cached) cudaFree(dMat);
    cudaFree(dOut);
    return ok;
}

bool cuda_backend_vec_dot_rows_ring(const float* vec, const float* ring, size_t head, size_t seq_max, size_t len, size_t K, size_t row_stride, float* out) {
    if (!ensure_ready()) return false;
    if (!vec || !ring || !out || K == 0 || len == 0 || seq_max == 0) return false;

    float* dVec = nullptr;
    float* dRing = nullptr;
    float* dOut = nullptr;

    if (!alloc_copy_to_device(vec, K * sizeof(float), (void**)&dVec)) return false;
    if (!alloc_copy_to_device(ring, seq_max * row_stride * sizeof(float), (void**)&dRing)) {
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
              copy_to_host(out, dOut, len * sizeof(float));

    cudaFree(dVec);
    cudaFree(dRing);
    cudaFree(dOut);
    return ok;
}

bool cuda_backend_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride) {
    if (!ensure_ready()) return false;
    if (!vec || !mat_rows || !out || Nrows == 0 || Ncols == 0) return false;

    float* dVec = nullptr;
    float* dMat = nullptr;
    float* dOut = nullptr;

    if (!alloc_copy_to_device(vec, Nrows * sizeof(float), (void**)&dVec)) return false;
    if (!alloc_copy_to_device(mat_rows, Nrows * row_stride * sizeof(float), (void**)&dMat)) {
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
              copy_to_host(out, dOut, Ncols * sizeof(float));

    cudaFree(dVec);
    cudaFree(dMat);
    cudaFree(dOut);
    return ok;
}

const char* cuda_backend_last_error_msg() {
    CudaState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.last_error.empty()) return "";
    return s.last_error.c_str();
}