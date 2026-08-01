#include "cpu_backend.h"
#include "../../tensor/tensor.h"
#include <cstring>
#include <cmath>
#include <thread>
#include <vector>
#include <cstdlib>


struct CpuMatmulThreadConfig {
    unsigned int nthreads;
    bool split_cols;
};

static CpuMatmulThreadConfig cpu_matmul_thread_config(size_t m, size_t n) {
    CpuMatmulThreadConfig cfg{};
    cfg.nthreads = std::thread::hardware_concurrency();
    if (cfg.nthreads == 0) cfg.nthreads = 1;

    const char* env_th = std::getenv("MINXFMR_CPU_THREADS");
    if (env_th) {
        int v = std::atoi(env_th);
        if (v > 0) cfg.nthreads = (unsigned int)v;
    }

    cfg.split_cols = (m < n);
    if (cfg.split_cols) {
        if ((size_t)cfg.nthreads > n) cfg.nthreads = (unsigned int)std::max<size_t>(1, n);
    } else {
        if ((size_t)cfg.nthreads > m) cfg.nthreads = (unsigned int)std::max<size_t>(1, m);
    }
    return cfg;
}

template <size_t BlockElems, size_t BlockSize, typename DequantFn>
static void cpu_matmul_quantized_weight_stationary(
    const float* a,
    const uint8_t* bq,
    float* o,
    size_t m,
    size_t k,
    size_t n,
    size_t row_bytes,
    size_t blocks_per_row,
    DequantFn dequant_block) {
    std::memset(o, 0, sizeof(float) * m * n);
    CpuMatmulThreadConfig cfg = cpu_matmul_thread_config(m, n);

    auto row_worker = [&](size_t row_start, size_t row_end) {
        float tmp[BlockElems];
        for (size_t kk = 0; kk < k; ++kk) {
            const uint8_t* brow = bq + kk * row_bytes;
            for (size_t blk = 0; blk < blocks_per_row; ++blk) {
                dequant_block(brow + blk * BlockSize, tmp);
                for (size_t i = row_start; i < row_end; ++i) {
                    const float av = a[i * k + kk];
                    float* out_blk = o + i * n + blk * BlockElems;
                    for (size_t t = 0; t < BlockElems; ++t) out_blk[t] += av * tmp[t];
                }
            }
        }
    };

    auto col_worker = [&](size_t block_start, size_t block_end) {
        float tmp[BlockElems];
        for (size_t kk = 0; kk < k; ++kk) {
            const uint8_t* brow = bq + kk * row_bytes;
            for (size_t blk = block_start; blk < block_end; ++blk) {
                dequant_block(brow + blk * BlockSize, tmp);
                for (size_t i = 0; i < m; ++i) {
                    const float av = a[i * k + kk];
                    float* out_blk = o + i * n + blk * BlockElems;
                    for (size_t t = 0; t < BlockElems; ++t) out_blk[t] += av * tmp[t];
                }
            }
        }
    };

    if (cfg.nthreads <= 1) {
        row_worker(0, m);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(cfg.nthreads);

    if (!cfg.split_cols) {
        size_t rows_per = m / cfg.nthreads;
        size_t rem = m % cfg.nthreads;
        size_t cur = 0;
        for (unsigned int t = 0; t < cfg.nthreads; ++t) {
            size_t rs = cur;
            size_t re = rs + rows_per + (t < rem ? 1 : 0);
            cur = re;
            if (rs >= re) break;
            threads.emplace_back(row_worker, rs, re);
        }
    } else {
        unsigned int block_threads = cfg.nthreads;
        if ((size_t)block_threads > blocks_per_row) {
            block_threads = (unsigned int)std::max<size_t>(1, blocks_per_row);
        }
        size_t blocks_per = blocks_per_row / block_threads;
        size_t rem = blocks_per_row % block_threads;
        size_t cur_blk = 0;
        for (unsigned int t = 0; t < block_threads; ++t) {
            size_t bs = cur_blk;
            size_t be = bs + blocks_per + (t < rem ? 1 : 0);
            cur_blk = be;
            if (bs >= be) break;
            threads.emplace_back(col_worker, bs, be);
        }
    }

    for (auto& th : threads) th.join();
}

// Simple, safe threaded matmul. Controlled by env MINXFMR_CPU_THREADS (if set),
// otherwise uses hardware_concurrency(). Splits work by output rows.
bool cpu_matmul(const Tensor* A, const Tensor* B, Tensor* out) {
    if (!A || !B || !out) return false;
    if (A->type != DataType::F32 || out->type != DataType::F32) return false;
    if (B->type != DataType::F32 && B->type != DataType::Q4_K && B->type != DataType::Q5_0 && B->type != DataType::Q8_0) return false;
    size_t m = A->rows;
    size_t k = A->cols;
    size_t kb = B->rows;
    size_t n = B->cols;
    if (k != kb) return false;
    if (out->rows != m || out->cols != n) return false;

    const float* a = (const float*)A->data;
    float* o = (float*)out->data;

    if (B->type == DataType::Q4_K) {
        const size_t row_bytes = tensor_q4_k_row_bytes(n);
        if (row_bytes == 0) return false;
        if (B->bytes < B->rows * row_bytes) return false;

        const uint8_t* bq = (const uint8_t*)B->data;
        const size_t blocks_per_row = n / TENSOR_Q4_K_QK_K;

        cpu_matmul_quantized_weight_stationary<TENSOR_Q4_K_QK_K, TENSOR_Q4_K_BLOCK_SIZE>(
            a,
            bq,
            o,
            m,
            k,
            n,
            row_bytes,
            blocks_per_row,
            tensor_dequant_q4_k_block);
        return true;
    }

    if (B->type == DataType::Q5_0) {
        const size_t row_bytes = tensor_q5_0_row_bytes(n);
        if (row_bytes == 0) return false;
        if (B->bytes < B->rows * row_bytes) return false;

        const uint8_t* bq = (const uint8_t*)B->data;
        const size_t blocks_per_row = n / TENSOR_Q5_0_QK;

        cpu_matmul_quantized_weight_stationary<TENSOR_Q5_0_QK, TENSOR_Q5_0_BLOCK_SIZE>(
            a,
            bq,
            o,
            m,
            k,
            n,
            row_bytes,
            blocks_per_row,
            tensor_dequant_q5_0_block);
        return true;
    }

    if (B->type == DataType::Q8_0) {
        const size_t row_bytes = tensor_q8_0_row_bytes(n);
        if (row_bytes == 0) return false;
        if (B->bytes < B->rows * row_bytes) return false;

        const uint8_t* bq = (const uint8_t*)B->data;
        const size_t blocks_per_row = n / TENSOR_Q8_0_QK;

        cpu_matmul_quantized_weight_stationary<TENSOR_Q8_0_QK, TENSOR_Q8_0_BLOCK_SIZE>(
            a,
            bq,
            o,
            m,
            k,
            n,
            row_bytes,
            blocks_per_row,
            tensor_dequant_q8_0_block);
        return true;
    }

    const float* b = (const float*)B->data;

    // zero out output
    std::memset(o, 0, sizeof(float) * m * n);

    CpuMatmulThreadConfig cfg = cpu_matmul_thread_config(m, n);
    unsigned int nthreads = cfg.nthreads;
    bool split_cols = cfg.split_cols;

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

    if (!split_cols) {
        // split by rows (existing behavior)
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
    } else {
        // split by columns (useful when m is small, e.g. m==1)
        auto worker_cols = [&](size_t col_start, size_t col_end) {
            for (size_t i = 0; i < m; ++i) {
                float* orow = o + i * n;
                const float* arow = a + i * k;
                for (size_t kk = 0; kk < k; ++kk) {
                    float av = arow[kk];
                    const float* brow = b + kk * n;
                    for (size_t j = col_start; j < col_end; ++j) {
                        orow[j] += av * brow[j];
                    }
                }
            }
        };

        size_t cols_per = n / nthreads;
        size_t rem = n % nthreads;
        size_t curc = 0;
        for (unsigned int t = 0; t < nthreads; ++t) {
            size_t cs = curc;
            size_t ce = cs + cols_per + (t < rem ? 1 : 0);
            curc = ce;
            if (cs >= ce) break;
            threads.emplace_back(worker_cols, cs, ce);
        }
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

void cpu_workspace_reset(bool shrink) {
    g_workspace.offset = 0;
    if (shrink && g_workspace.buf.size() > (1u << 20)) {
        std::vector<float>().swap(g_workspace.buf);
    }
}

bool cpu_matvec(const float* vec, const float* mat, float* out, size_t K, size_t N) {
    if (!vec || !mat || !out) return false;
    // mat is K x N with row-major stride N
    return cpu_matvec_strided(vec, mat, out, K, N, N);
}

bool cpu_matvec_strided(const float* vec, const float* mat, float* out, size_t K, size_t N, size_t mat_row_stride) {
    if (!vec || !mat || !out) return false;
    if (K == 0 || N == 0) return false;
    // Compute y = x^T * W by streaming each row segment of W contiguously.
    // This avoids cache-thrashing caused by column-stride walks on row-major data.
    std::memset(out, 0, sizeof(float) * N);
    for (size_t k = 0; k < K; ++k) {
        const double scale = (double)vec[k];
        const float* row = mat + k * mat_row_stride;
#if defined(_OPENMP) && !defined(_MSC_VER)
    #pragma omp simd
#endif
        for (size_t n = 0; n < N; ++n) {
            out[n] += (float)(scale * (double)row[n]);
        }
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
#if defined(_OPENMP) && !defined(_MSC_VER)
    #pragma omp simd reduction(+:acc)
#endif
        for (size_t k = 0; k < K; ++k) acc += (double)vec[k] * (double)row[k];
        out[j] = (float)acc;
    }
    return true;
}

bool cpu_vec_dot_rows_ring(const float* vec, const float* ring, size_t head, size_t seq_max, size_t len, size_t K, size_t row_stride, float* out) {
    if (!vec || !ring || !out) return false;
    if (K == 0 || len == 0 || seq_max == 0) return false;
    for (size_t j = 0; j < len; ++j) {
        size_t phys = (head + j) % seq_max;
        const float* row = ring + phys * row_stride;
        double acc = 0.0;
#if defined(_OPENMP) && !defined(_MSC_VER)
    #pragma omp simd reduction(+:acc)
#endif
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
#if defined(_OPENMP) && !defined(_MSC_VER)
    #pragma omp simd reduction(+:acc)
#endif
        for (size_t row = 0; row < Nrows; ++row) {
            acc += (double)vec[row] * (double)mat_rows[row * row_stride + col];
        }
        out[col] = (float)acc;
    }
    return true;
}
