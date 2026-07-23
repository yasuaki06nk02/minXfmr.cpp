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
    if (A->type != DataType::F32 || B->type != DataType::F32 || out->type != DataType::F32) return false;
    if (A->cols != B->cols) return false;
    if (out->rows != A->rows || out->cols != B->rows) return false;

    const size_t m = A->rows;
    const size_t k = A->cols;
    const size_t n = B->rows;
    const float* ad = (const float*)A->data;
    const float* bd = (const float*)B->data;
    float* od = (float*)out->data;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float* arow = ad + i * k;
            const float* brow = bd + j * k;
            float s = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) s += arow[kk] * brow[kk];
            od[i * n + j] = s;
        }
    }
    return true;
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
