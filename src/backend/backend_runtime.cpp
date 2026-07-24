#include "backend_runtime.h"

#include "cpu/cpu_backend.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(MINXFMR_ENABLE_CUDA)
#include "cuda/cuda_backend.h"
#endif

static BackendKind g_backend_kind = BackendKind::CPU;
static bool g_backend_initialized = false;

static bool ieq(const char* a, const char* b) {
    if (!a || !b) return false;
    while (*a && *b) {
        if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return false;
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

static float fp16_to_fp32_local(uint16_t h) {
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

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static float dot_q4_k_block(const uint8_t* blk, const float* x256) {
    // Decode one GGUF Q4_K block on the fly and compute dot with 256 F32 values.
    uint16_t hd = 0;
    uint16_t hm = 0;
    std::memcpy(&hd, blk + 0, sizeof(hd));
    std::memcpy(&hm, blk + 2, sizeof(hm));
    const float d = fp16_to_fp32_local(hd);
    const float dmin = fp16_to_fp32_local(hm);

    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;

    float acc = 0.0f;
    int is = 0;
    for (int j = 0; j < (int)TENSOR_Q4_K_QK_K; j += 64) {
        uint8_t sc = 0;
        uint8_t m = 0;

        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc;
        const float m1 = dmin * m;

        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc;
        const float m2 = dmin * m;

        for (int l = 0; l < 32; ++l) acc += x256[j + l] * (d1 * (q[l] & 0xF) - m1);
        for (int l = 0; l < 32; ++l) acc += x256[j + 32 + l] * (d2 * (q[l] >> 4) - m2);

        q += 32;
        is += 2;
    }
    return acc;
}

static float dot_q5_0_block(const uint8_t* blk, const float* x32) {
    uint16_t hd = 0;
    std::memcpy(&hd, blk, sizeof(hd));
    const float d = fp16_to_fp32_local(hd);

    const uint8_t* qh = blk + 2;
    const uint8_t* qs = blk + 6;

    uint32_t hmask = 0;
    hmask |= (uint32_t)qh[0];
    hmask |= (uint32_t)qh[1] << 8;
    hmask |= (uint32_t)qh[2] << 16;
    hmask |= (uint32_t)qh[3] << 24;

    float acc = 0.0f;
    for (int i = 0; i < 16; ++i) {
        const uint8_t ql = qs[i];
        const int low0 = (int)(ql & 0x0F);
        const int low1 = (int)(ql >> 4);
        const int high0 = (int)((hmask >> i) & 1u);
        const int high1 = (int)((hmask >> (i + 16)) & 1u);
        const int q0 = (high0 << 4) | low0;
        const int q1 = (high1 << 4) | low1;
        acc += x32[i] * (d * (float)(q0 - 16));
        acc += x32[i + 16] * (d * (float)(q1 - 16));
    }
    return acc;
}

static float dot_q8_0_block(const uint8_t* blk, const float* x32) {
    uint16_t hd = 0;
    std::memcpy(&hd, blk, sizeof(hd));
    const float d = fp16_to_fp32_local(hd);
    const int8_t* qs = (const int8_t*)(blk + 2);

    float acc = 0.0f;
    for (int i = 0; i < 32; ++i) {
        acc += x32[i] * (d * (float)qs[i]);
    }
    return acc;
}

static bool try_enable_cuda() {
#if defined(MINXFMR_ENABLE_CUDA)
    if (!cuda_backend_is_available()) return false;
    g_backend_kind = BackendKind::CUDA;
    return true;
#else
    return false;
#endif
}

void backend_initialize_from_env() {
    if (g_backend_initialized) return;

    const char* env = std::getenv("MINXFMR_BACKEND");
    bool want_cuda = false;
    bool force_cpu = false;

    // auto: try CUDA first, otherwise keep CPU reference path.
    if (!env || env[0] == '\0' || ieq(env, "auto")) {
        want_cuda = true;
    } else if (ieq(env, "cuda")) {
        want_cuda = true;
    } else if (ieq(env, "cpu")) {
        force_cpu = true;
    }

    if (!force_cpu && want_cuda && try_enable_cuda()) {
        std::fprintf(stderr, "[backend] selected CUDA backend\n");
    } else {
        g_backend_kind = BackendKind::CPU;
        if (env && ieq(env, "cuda")) {
            std::fprintf(stderr, "[backend] CUDA requested but unavailable, falling back to CPU\n");
        } else {
            std::fprintf(stderr, "[backend] selected CPU backend\n");
        }
    }

    g_backend_initialized = true;
}

bool backend_set_kind(BackendKind kind) {
    if (kind == BackendKind::CPU) {
        g_backend_kind = BackendKind::CPU;
        g_backend_initialized = true;
        return true;
    }

    if (kind == BackendKind::CUDA) {
        g_backend_initialized = true;
        if (try_enable_cuda()) return true;
        g_backend_kind = BackendKind::CPU;
        return false;
    }

    return false;
}

BackendKind backend_get_kind() {
    backend_initialize_from_env();
    return g_backend_kind;
}

const char* backend_get_name() {
    return backend_get_kind() == BackendKind::CUDA ? "cuda" : "cpu";
}

bool backend_using_cuda() {
    return backend_get_kind() == BackendKind::CUDA;
}

bool backend_matmul(const Tensor* A, const Tensor* B, Tensor* out) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    // Prefer CUDA when selected and available; fallback is always CPU.
    if (g_backend_kind == BackendKind::CUDA && cuda_backend_matmul(A, B, out)) return true;
#endif
    return cpu_matmul(A, B, out);
}

bool backend_matmul_rhs_transposed(const Tensor* A, const Tensor* B, Tensor* out) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (g_backend_kind == BackendKind::CUDA && cuda_backend_matmul_rhs_transposed(A, B, out)) return true;
#endif

    if (!A || !B || !out) return false;
    if (A->type != DataType::F32 || out->type != DataType::F32) return false;
    if (B->type != DataType::F32 && B->type != DataType::Q4_K && B->type != DataType::Q5_0 && B->type != DataType::Q8_0) return false;
    if (A->cols != B->cols) return false;
    if (out->rows != A->rows || out->cols != B->rows) return false;

    const size_t m = A->rows;
    const size_t k = A->cols;
    const size_t n = B->rows;
    const float* ad = (const float*)A->data;
    float* od = (float*)out->data;

    // Q4_K path: B rows are packed quantized blocks interpreted as transposed RHS.
    if (B->type == DataType::Q4_K) {
        const size_t row_bytes = tensor_q4_k_row_bytes(k);
        if (row_bytes == 0) return false;
        if (B->bytes < B->rows * row_bytes) return false;

        const uint8_t* bdq = (const uint8_t*)B->data;
        const size_t blocks_per_row = k / TENSOR_Q4_K_QK_K;

        const long long work = (long long)(m * n);
        #if defined(_OPENMP)
            #pragma omp parallel for
        #endif
        for (long long idx = 0; idx < work; ++idx) {
            const size_t iu = (size_t)(idx / (long long)n);
            const size_t ju = (size_t)(idx % (long long)n);
            const float* arow = ad + iu * k;
            const uint8_t* brow = bdq + ju * row_bytes;
            float s = 0.0f;
            for (size_t blk = 0; blk < blocks_per_row; ++blk) {
                s += dot_q4_k_block(
                    brow + blk * TENSOR_Q4_K_BLOCK_SIZE,
                    arow + blk * TENSOR_Q4_K_QK_K);
            }
            od[iu * n + ju] = s;
        }
        return true;
    }

    if (B->type == DataType::Q5_0) {
        const size_t row_bytes = tensor_q5_0_row_bytes(k);
        if (row_bytes == 0) return false;
        if (B->bytes < B->rows * row_bytes) return false;

        const uint8_t* bdq = (const uint8_t*)B->data;
        const size_t blocks_per_row = k / TENSOR_Q5_0_QK;

        const long long work = (long long)(m * n);
        #if defined(_OPENMP)
            #pragma omp parallel for
        #endif
        for (long long idx = 0; idx < work; ++idx) {
            const size_t iu = (size_t)(idx / (long long)n);
            const size_t ju = (size_t)(idx % (long long)n);
            const float* arow = ad + iu * k;
            const uint8_t* brow = bdq + ju * row_bytes;
            float s = 0.0f;
            for (size_t blk = 0; blk < blocks_per_row; ++blk) {
                s += dot_q5_0_block(
                    brow + blk * TENSOR_Q5_0_BLOCK_SIZE,
                    arow + blk * TENSOR_Q5_0_QK);
            }
            od[iu * n + ju] = s;
        }
        return true;
    }

    if (B->type == DataType::Q8_0) {
        const size_t row_bytes = tensor_q8_0_row_bytes(k);
        if (row_bytes == 0) return false;
        if (B->bytes < B->rows * row_bytes) return false;

        const uint8_t* bdq = (const uint8_t*)B->data;
        const size_t blocks_per_row = k / TENSOR_Q8_0_QK;

        const long long work = (long long)(m * n);
        #if defined(_OPENMP)
            #pragma omp parallel for
        #endif
        for (long long idx = 0; idx < work; ++idx) {
            const size_t iu = (size_t)(idx / (long long)n);
            const size_t ju = (size_t)(idx % (long long)n);
            const float* arow = ad + iu * k;
            const uint8_t* brow = bdq + ju * row_bytes;
            float s = 0.0f;
            for (size_t blk = 0; blk < blocks_per_row; ++blk) {
                s += dot_q8_0_block(
                    brow + blk * TENSOR_Q8_0_BLOCK_SIZE,
                    arow + blk * TENSOR_Q8_0_QK);
            }
            od[iu * n + ju] = s;
        }
        return true;
    }

    const float* bd = (const float*)B->data;

    const long long work = (long long)(m * n);
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (long long idx = 0; idx < work; ++idx) {
            const size_t iu = (size_t)(idx / (long long)n);
            const size_t ju = (size_t)(idx % (long long)n);
            const float* arow = ad + iu * k;
            const float* brow = bd + ju * k;
            float s = 0.0f;
#if defined(_OPENMP) && !defined(_MSC_VER)
            #pragma omp simd reduction(+:s)
#endif
            for (size_t kk = 0; kk < k; ++kk) s += arow[kk] * brow[kk];
            od[iu * n + ju] = s;
    }
    return true;
}

bool backend_preload_tensor(const Tensor* t) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (g_backend_kind == BackendKind::CUDA) return cuda_backend_preload_tensor(t);
#endif
    (void)t;
    return false;
}

void backend_release_resources() {
#if defined(MINXFMR_ENABLE_CUDA)
    cuda_backend_release_resources();
#endif
}

bool backend_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (g_backend_kind == BackendKind::CUDA && cuda_backend_matvec_strided(vec, mat, out, K, N, mat_row_stride)) return true;
#endif
    return cpu_matvec_strided(vec, mat, out, K, N, mat_row_stride);
}

bool backend_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (g_backend_kind == BackendKind::CUDA && cuda_backend_vec_dot_rows(vec, mat_rows, out, K, Nrows, row_stride)) return true;
#endif
    return cpu_vec_dot_rows(vec, mat_rows, out, K, Nrows, row_stride);
}

bool backend_vec_dot_rows_ring(const float* vec, const float* ring, size_t head, size_t seq_max, size_t len, size_t K, size_t row_stride, float* out) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (g_backend_kind == BackendKind::CUDA && cuda_backend_vec_dot_rows_ring(vec, ring, head, seq_max, len, K, row_stride, out)) return true;
#endif
    return cpu_vec_dot_rows_ring(vec, ring, head, seq_max, len, K, row_stride, out);
}

bool backend_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (g_backend_kind == BackendKind::CUDA && cuda_backend_vec_mul_rows_cols(vec, mat_rows, out, Nrows, Ncols, row_stride)) return true;
#endif
    return cpu_vec_mul_rows_cols(vec, mat_rows, out, Nrows, Ncols, row_stride);
}

float* backend_request_workspace(size_t n) {
    return cpu_request_workspace(n);
}

void backend_workspace_reset(bool shrink) {
    cpu_workspace_reset(shrink);
}
