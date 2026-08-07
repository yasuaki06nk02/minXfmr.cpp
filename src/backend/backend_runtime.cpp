#include "backend_runtime.h"
#include "backend_context.h"

#include "cpu/cpu_backend.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "runtime_config.h"

#if defined(MINXFMR_ENABLE_CUDA)
#include "cuda/cuda_backend.h"
#endif

namespace {
bool should_log_cuda_fallback_once(const char* site, const char* reason) {
    static std::string last_matmul_reason;
    static std::string last_rhs_reason;
    if (!site) site = "";
    if (!reason) reason = "";
    std::string r(reason);
    if (std::strcmp(site, "matmul") == 0) {
        if (r == last_matmul_reason) return false;
        last_matmul_reason = r;
        return true;
    }
    if (std::strcmp(site, "rhs") == 0) {
        if (r == last_rhs_reason) return false;
        last_rhs_reason = r;
        return true;
    }
    return true;
}
}

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
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    int exp = (h >> 10) & 0x1f;
    uint32_t frac = (uint32_t)h & 0x3ffu;

    uint32_t out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            // Normalize subnormal half values with a signed exponent to avoid underflow.
            exp = -14;
            while ((frac & 0x400u) == 0) {
                frac <<= 1;
                --exp;
            }
            frac &= 0x3ffu;
            out = sign | ((uint32_t)(exp + 127) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        out = sign | ((uint32_t)(exp + 112) << 23) | (frac << 13);
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
    float tmp[TENSOR_Q4_K_QK_K];
    tensor_dequant_q4_k_block(blk, tmp);

    double acc = 0.0;
    for (size_t i = 0; i < TENSOR_Q4_K_QK_K; ++i) {
        acc += (double)tmp[i] * (double)x256[i];
    }
    return (float)acc;
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

#include <iostream>

static bool try_enable_cuda() {
#if defined(MINXFMR_ENABLE_CUDA)
    if (!cuda_backend_is_available()) return false;
    backend_context().backend = BackendKind::CUDA;
    return true;
#else
    return false;
#endif
}

void backend_initialize_from_env() {
    if (backend_context().initialized) return;

    std::string env = RuntimeConfig::Instance().getString("MINXFMR_BACKEND");
    bool want_cuda = false;
    bool force_cpu = false;

    // auto: try CUDA first, otherwise keep CPU reference path.
    if (env.empty() || ieq(env.c_str(), "auto")) {
        want_cuda = true;
    } else if (ieq(env.c_str(), "cuda")) {
        want_cuda = true;
    } else if (ieq(env.c_str(), "cpu")) {
        force_cpu = true;
    }

    if (!force_cpu && want_cuda && try_enable_cuda()) {
        std::fprintf(stderr, "[backend] selected CUDA backend\n");
    } else {
        backend_context().backend = BackendKind::CPU;
        if (!env.empty() && ieq(env.c_str(), "cuda")) {
            std::fprintf(stderr, "[backend] CUDA requested but unavailable, falling back to CPU\n");
        } else {
            std::fprintf(stderr, "[backend] selected CPU backend\n");
        }
    }

    backend_context().initialized = true;
}

bool backend_set_kind(BackendKind kind) {
    if (kind == BackendKind::CPU) {
        backend_context().backend = BackendKind::CPU;
        backend_context().initialized = true;
        return true;
    }

    if (kind == BackendKind::CUDA) {
        backend_context().initialized = true;
        if (try_enable_cuda()) return true;
        backend_context().backend = BackendKind::CPU;
        return false;
    }

    return false;
}

BackendKind backend_get_kind() {
    backend_initialize_from_env();
    return backend_context().backend;
}

const char* backend_get_name() {
    return backend_get_kind() == BackendKind::CUDA ? "cuda" : "cpu";
}

bool backend_using_cuda() {
    return backend_get_kind() == BackendKind::CUDA;
}

void backend_set_cuda_quant_parity_mode(int mode) {
#if defined(MINXFMR_ENABLE_CUDA)
    cuda_backend_set_quant_parity_mode(mode);
#else
    (void)mode;
#endif
}

int backend_get_cuda_quant_parity_mode() {
#if defined(MINXFMR_ENABLE_CUDA)
    return cuda_backend_get_quant_parity_mode();
#else
    return -1;
#endif
}

bool backend_cuda_quant_kernels_enabled() {
    // Reconstruct the CUDA kernel enable decision using the same rules
    // as the CUDA backend: honor explicit MINXFMR_CUDA_QUANT if set,
    // otherwise enable when MINXFMR_BACKEND=cuda.
    auto &cfg = RuntimeConfig::Instance();
    if (cfg.has("MINXFMR_CUDA_QUANT")) return cfg.getBool("MINXFMR_CUDA_QUANT");
    std::string bks = cfg.getString("MINXFMR_BACKEND");
    for (char &c : bks) c = (char)std::tolower((unsigned char)c);
    return bks == "cuda";
}

bool backend_cuda_quant_parity_enabled() {
    // Match llama.cpp semantics: explicit environment/CLI flags win first,
    // then any internal override set by model-load code. Default is false.
    auto &cfg = RuntimeConfig::Instance();
    if (cfg.has("MINXFMR_CUDA_QUANT_PARITY")) return cfg.getBool("MINXFMR_CUDA_QUANT_PARITY");

    int override_mode = backend_get_cuda_quant_parity_mode();
    if (override_mode >= 0) return override_mode == 1;

    return false;
}

bool backend_matmul(const Tensor* A, const Tensor* B, Tensor* out) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (backend_context().backend == BackendKind::CUDA) {
        // Try CUDA path first; if it fails (for example CUDA quant kernels
        // are disabled) fall back to the CPU implementation for parity.
        if (cuda_backend_matmul(A, B, out)) return true;
        const char* why = backend_last_preload_error();
        if (should_log_cuda_fallback_once("matmul", why)) {
            std::fprintf(stderr, "[backend] cuda_backend_matmul failed, falling back to CPU: %s\n", why);
        }
    }
#endif
    return cpu_matmul(A, B, out);
}

bool backend_matmul_rhs_transposed(const Tensor* A, const Tensor* B, Tensor* out) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (backend_context().backend == BackendKind::CUDA) {
        // Try CUDA path first; if it fails (e.g. quant kernels disabled),
        // fall back to the CPU implementation which supports staged host
        // dequantization and parity paths.
        if (cuda_backend_matmul_rhs_transposed(A, B, out)) return true;
        const char* why = backend_last_preload_error();
        if (should_log_cuda_fallback_once("rhs", why)) {
            std::fprintf(stderr, "[backend] cuda_backend_matmul_rhs_transposed failed, falling back to CPU: %s\n", why);
        }
    }
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

    if (RuntimeConfig::Instance().getBool("MINXFMR_CHAT_DEBUG")) {
        static bool logged_f32_rhs_probe = false;
        if (!logged_f32_rhs_probe && m > 0 && n > 0) {
            logged_f32_rhs_probe = true;
            const float* a0 = ad;
            const size_t ju = 0;
            double row_major = 0.0;
            double col_major = 0.0;
            double sum_abs_a = 0.0;
            double max_abs_b_row = 0.0;
            for (size_t kk = 0; kk < k; ++kk) {
                const double av = (double)a0[kk];
                const double b_row = (double)bd[ju * k + kk];
                const double b_col = (double)bd[kk * n + ju];
                row_major += av * b_row;
                col_major += av * b_col;
                sum_abs_a += std::fabs(av);
                max_abs_b_row = std::max(max_abs_b_row, std::fabs(b_row));
            }
            std::fprintf(
                stderr,
                "[backend][f32-rhs-probe] m=%zu n=%zu k=%zu row_major_col0=%g col_major_col0=%g sum_abs_a=%g max_abs_b_row0=%g bound_row0=%g\n",
                m,
                n,
                k,
                row_major,
                col_major,
                sum_abs_a,
                max_abs_b_row,
                sum_abs_a * max_abs_b_row);
        }
    }

    const long long work = (long long)(m * n);
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (long long idx = 0; idx < work; ++idx) {
            const size_t iu = (size_t)(idx / (long long)n);
            const size_t ju = (size_t)(idx % (long long)n);
            const float* arow = ad + iu * k;
            float s = 0.0f;
#if defined(_OPENMP) && !defined(_MSC_VER)
            #pragma omp simd reduction(+:s)
#endif
            for (size_t kk = 0; kk < k; ++kk) s += arow[kk] * bd[kk * n + ju];
            od[iu * n + ju] = s;
    }

    if (RuntimeConfig::Instance().getBool("MINXFMR_CHAT_DEBUG")) {
            static int logged_f32_rhs_out = 0;
            if (m == 1 && n == 256 && logged_f32_rhs_out < 64) {
                ++logged_f32_rhs_out;
            float mn = od[0];
            float mx = od[0];
            double sum = 0.0;
            double sum_abs_a = 0.0;
            double max_abs_b_row0 = 0.0;
            const float* a0 = ad;
            for (size_t kk = 0; kk < k; ++kk) {
                const double av = std::fabs((double)a0[kk]);
                const double bv = std::fabs((double)bd[kk]);
                sum_abs_a += av;
                if (bv > max_abs_b_row0) max_abs_b_row0 = bv;
            }
            for (size_t j = 0; j < n; ++j) {
                const float v = od[j];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += (double)v;
            }
                std::fprintf(
                    stderr,
                "[backend][f32-rhs-out] call=%d out=%p n=%zu min=%g max=%g mean=%g sum_abs_a=%g max_abs_b_row0=%g bound0=%g\n",
                    logged_f32_rhs_out,
                    (const void*)od,
                    n,
                    mn,
                    mx,
                sum / (double)n,
                sum_abs_a,
                max_abs_b_row0,
                sum_abs_a * max_abs_b_row0);
        }
    }
    return true;
}

bool backend_preload_tensor(const Tensor* t) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (backend_context().backend == BackendKind::CUDA) return cuda_backend_preload_tensor(t);
#endif
    (void)t;
    return false;
}

const char* backend_last_preload_error() {
#if defined(MINXFMR_ENABLE_CUDA)
    if (backend_context().backend == BackendKind::CUDA) return cuda_backend_last_error_msg();
#endif
    return "";
}

void backend_release_resources() {
#if defined(MINXFMR_ENABLE_CUDA)
    cuda_backend_release_resources();
#endif
}

bool backend_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (backend_context().backend == BackendKind::CUDA) {
        return cuda_backend_matvec_strided(vec, mat, out, K, N, mat_row_stride);
    }
#endif
    return cpu_matvec_strided(vec, mat, out, K, N, mat_row_stride);
}

bool backend_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (backend_context().backend == BackendKind::CUDA) {
        return cuda_backend_vec_dot_rows(vec, mat_rows, out, K, Nrows, row_stride);
    }
#endif
    return cpu_vec_dot_rows(vec, mat_rows, out, K, Nrows, row_stride);
}

bool backend_vec_dot_rows_ring(const float* vec, const float* ring, size_t head, size_t seq_max, size_t len, size_t K, size_t row_stride, float* out) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (backend_context().backend == BackendKind::CUDA) {
        return cuda_backend_vec_dot_rows_ring(vec, ring, head, seq_max, len, K, row_stride, out);
    }
#endif
    return cpu_vec_dot_rows_ring(vec, ring, head, seq_max, len, K, row_stride, out);
}

bool backend_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride) {
    backend_initialize_from_env();
#if defined(MINXFMR_ENABLE_CUDA)
    if (backend_context().backend == BackendKind::CUDA) {
        return cuda_backend_vec_mul_rows_cols(vec, mat_rows, out, Nrows, Ncols, row_stride);
    }
#endif
    return cpu_vec_mul_rows_cols(vec, mat_rows, out, Nrows, Ncols, row_stride);
}

float* backend_request_workspace(size_t n) {
    return cpu_request_workspace(n);
}

void backend_workspace_reset(bool shrink) {
    cpu_workspace_reset(shrink);
}
