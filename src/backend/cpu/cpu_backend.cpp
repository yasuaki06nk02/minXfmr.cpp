#include "cpu_backend.h"
#include <cstring>
#include <cmath>
#include <thread>
#include <vector>
#include <cstdlib>

// Simple, safe threaded matmul. Controlled by env MINXFMR_CPU_THREADS (if set),
// otherwise uses hardware_concurrency(). Splits work by output rows.
bool cpu_matmul(const Tensor* A, const Tensor* B, Tensor* out) {
    if (!A || !B || !out) return false;
    if (A->type != DataType::F32 || B->type != DataType::F32 || out->type != DataType::F32) return false;
    size_t m = A->rows;
    size_t k = A->cols;
    size_t kb = B->rows;
    size_t n = B->cols;
    if (k != kb) return false;
    if (out->rows != m || out->cols != n) return false;

    const float* a = (const float*)A->data;
    const float* b = (const float*)B->data;
    float* o = (float*)out->data;

    // zero out output
    std::memset(o, 0, sizeof(float) * m * n);

    unsigned int nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 1;
    const char* env_th = std::getenv("MINXFMR_CPU_THREADS");
    if (env_th) {
        int v = std::atoi(env_th);
        if (v > 0) nthreads = (unsigned int)v;
    }

    // clamp to at most m threads
    if ((size_t)nthreads > m) nthreads = (unsigned int)std::max<size_t>(1, m);

    if (nthreads <= 1) {
        for (size_t i = 0; i < m; ++i) {
            float* orow = o + i * n;
            const float* arow = a + i * k;
            for (size_t kk = 0; kk < k; ++kk) {
                float av = arow[kk];
                const float* brow = b + kk * n;
                for (size_t j = 0; j < n; ++j) {
                    orow[j] += av * brow[j];
                }
            }
        }
        return true;
    }

    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    auto worker = [&](size_t row_start, size_t row_end) {
        for (size_t i = row_start; i < row_end; ++i) {
            float* orow = o + i * n;
            const float* arow = a + i * k;
            for (size_t kk = 0; kk < k; ++kk) {
                float av = arow[kk];
                const float* brow = b + kk * n;
                for (size_t j = 0; j < n; ++j) {
                    orow[j] += av * brow[j];
                }
            }
        }
    };

    size_t rows_per = m / nthreads;
    size_t rem = m % nthreads;
    size_t cur = 0;
    for (unsigned int t = 0; t < nthreads; ++t) {
        size_t rs = cur;
        size_t re = rs + rows_per + (t < rem ? 1 : 0);
        cur = re;
        if (rs >= re) break;
        threads.emplace_back(worker, rs, re);
    }

    for (auto& th : threads) th.join();
    return true;
}

bool cpu_add(const Tensor* a, const Tensor* b, Tensor* out) {
    if (!a || !b || !out) return false;
    if (a->type != DataType::F32 || b->type != DataType::F32 || out->type != DataType::F32) return false;
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (out->rows != a->rows || out->cols != a->cols) return false;
    size_t elems = a->rows * a->cols;
    const float* ad = (const float*)a->data;
    const float* bd = (const float*)b->data;
    float* od = (float*)out->data;
    for (size_t i = 0; i < elems; ++i) od[i] = ad[i] + bd[i];
    return true;
}

// Per-thread workspace with offset-based allocation.
struct Workspace { std::vector<float> buf; size_t offset; };
static thread_local Workspace g_workspace{std::vector<float>(), 0};

float* cpu_workspace(size_t n) {
    if (n == 0) return nullptr;
    size_t need = g_workspace.offset + n;
    if (g_workspace.buf.size() < need) g_workspace.buf.resize(need);
    float* p = g_workspace.buf.data() + g_workspace.offset;
    g_workspace.offset += n;
    return p;
}

void cpu_workspace_reset() {
    g_workspace.offset = 0;
}

bool cpu_matvec(const float* vec, const float* mat, float* out, size_t K, size_t N) {
    if (!vec || !mat || !out) return false;
    // mat is K x N with row-major stride N
    return cpu_matvec_strided(vec, mat, out, K, N, N);
}

bool cpu_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride) {
    if (!vec || !mat || !out) return false;
    if (K == 0 || N == 0) return false;
    for (size_t n = 0; n < N; ++n) {
        double acc = 0.0;
        const float* col = mat + n; // start at offset n, step by mat_row_stride
        for (size_t k = 0; k < K; ++k) {
            acc += (double)vec[k] * (double)col[k * mat_row_stride];
        }
        out[n] = (float)acc;
    }
    return true;
}

float* cpu_request_workspace(size_t n) {
    return cpu_workspace(n);
}

bool cpu_vec_dot_rows(const float* vec, const float* mat_rows, float* out, size_t K, size_t Nrows, size_t row_stride) {
    if (!vec || !mat_rows || !out) return false;
    if (K == 0 || Nrows == 0) return false;
    for (size_t j = 0; j < Nrows; ++j) {
        const float* row = mat_rows + j * row_stride;
        double acc = 0.0;
        for (size_t k = 0; k < K; ++k) acc += (double)vec[k] * (double)row[k];
        out[j] = (float)acc;
    }
    return true;
}

bool cpu_vec_mul_rows_cols(const float* vec, const float* mat_rows, float* out, size_t Nrows, size_t Ncols, size_t row_stride) {
    if (!vec || !mat_rows || !out) return false;
    if (Nrows == 0 || Ncols == 0) return false;
    for (size_t col = 0; col < Ncols; ++col) {
        double acc = 0.0;
        for (size_t row = 0; row < Nrows; ++row) {
            acc += (double)vec[row] * (double)mat_rows[row * row_stride + col];
        }
        out[col] = (float)acc;
    }
    return true;
}
