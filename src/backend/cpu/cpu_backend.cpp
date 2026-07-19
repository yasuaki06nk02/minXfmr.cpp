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
