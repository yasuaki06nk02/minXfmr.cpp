#include "cpu_backend.h"
#include <cstring>
#include <cmath>
#include <thread>
#include <vector>
#include <cstdlib>

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

static void dequant_q4_k_block(const uint8_t* blk, float* dst256) {
    uint16_t hd = 0;
    uint16_t hm = 0;
    std::memcpy(&hd, blk + 0, sizeof(hd));
    std::memcpy(&hm, blk + 2, sizeof(hm));
    const float d = fp16_to_fp32_local(hd);
    const float dmin = fp16_to_fp32_local(hm);

    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;

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

        for (int l = 0; l < 32; ++l) dst256[j + l] = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) dst256[j + 32 + l] = d2 * (q[l] >> 4) - m2;

        q += 32;
        is += 2;
    }
}

static void dequant_q5_0_block(const uint8_t* blk, float* dst32) {
    uint16_t hd = 0;
    std::memcpy(&hd, blk, sizeof(hd));
    const float d = fp16_to_fp32_local(hd);

    const uint8_t* qh = blk + 2;  // 32 high bits
    const uint8_t* qs = blk + 6;  // 32 low 4-bit quants (packed into 16 bytes)

    uint32_t hmask = 0;
    hmask |= (uint32_t)qh[0];
    hmask |= (uint32_t)qh[1] << 8;
    hmask |= (uint32_t)qh[2] << 16;
    hmask |= (uint32_t)qh[3] << 24;

    for (int i = 0; i < 32; ++i) {
        const uint8_t ql = qs[i >> 1];
        const int low = (i & 1) ? (int)(ql >> 4) : (int)(ql & 0x0F);
        const int high = (int)((hmask >> i) & 1u);
        const int q = (high << 4) | low; // [0..31]
        dst32[i] = d * (float)(q - 16);
    }
}

static void dequant_q8_0_block(const uint8_t* blk, float* dst32) {
    uint16_t hd = 0;
    std::memcpy(&hd, blk, sizeof(hd));
    const float d = fp16_to_fp32_local(hd);
    const int8_t* qs = (const int8_t*)(blk + 2);
    for (int i = 0; i < 32; ++i) dst32[i] = d * (float)qs[i];
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

        std::memset(o, 0, sizeof(float) * m * n);
        float tmp[TENSOR_Q4_K_QK_K];

        for (size_t i = 0; i < m; ++i) {
            float* orow = o + i * n;
            const float* arow = a + i * k;

            for (size_t kk = 0; kk < k; ++kk) {
                const float av = arow[kk];
                const uint8_t* brow = bq + kk * row_bytes;

                for (size_t blk = 0; blk < blocks_per_row; ++blk) {
                    dequant_q4_k_block(brow + blk * TENSOR_Q4_K_BLOCK_SIZE, tmp);
                    float* out_blk = orow + blk * TENSOR_Q4_K_QK_K;
                    for (size_t t = 0; t < TENSOR_Q4_K_QK_K; ++t) out_blk[t] += av * tmp[t];
                }
            }
        }
        return true;
    }

    if (B->type == DataType::Q5_0) {
        const size_t row_bytes = tensor_q5_0_row_bytes(n);
        if (row_bytes == 0) return false;
        if (B->bytes < B->rows * row_bytes) return false;

        const uint8_t* bq = (const uint8_t*)B->data;
        const size_t blocks_per_row = n / TENSOR_Q5_0_QK;

        std::memset(o, 0, sizeof(float) * m * n);
        float tmp[TENSOR_Q5_0_QK];

        for (size_t i = 0; i < m; ++i) {
            float* orow = o + i * n;
            const float* arow = a + i * k;

            for (size_t kk = 0; kk < k; ++kk) {
                const float av = arow[kk];
                const uint8_t* brow = bq + kk * row_bytes;

                for (size_t blk = 0; blk < blocks_per_row; ++blk) {
                    dequant_q5_0_block(brow + blk * TENSOR_Q5_0_BLOCK_SIZE, tmp);
                    float* out_blk = orow + blk * TENSOR_Q5_0_QK;
                    for (size_t t = 0; t < TENSOR_Q5_0_QK; ++t) out_blk[t] += av * tmp[t];
                }
            }
        }
        return true;
    }

    if (B->type == DataType::Q8_0) {
        const size_t row_bytes = tensor_q8_0_row_bytes(n);
        if (row_bytes == 0) return false;
        if (B->bytes < B->rows * row_bytes) return false;

        const uint8_t* bq = (const uint8_t*)B->data;
        const size_t blocks_per_row = n / TENSOR_Q8_0_QK;

        std::memset(o, 0, sizeof(float) * m * n);
        float tmp[TENSOR_Q8_0_QK];

        for (size_t i = 0; i < m; ++i) {
            float* orow = o + i * n;
            const float* arow = a + i * k;

            for (size_t kk = 0; kk < k; ++kk) {
                const float av = arow[kk];
                const uint8_t* brow = bq + kk * row_bytes;

                for (size_t blk = 0; blk < blocks_per_row; ++blk) {
                    dequant_q8_0_block(brow + blk * TENSOR_Q8_0_BLOCK_SIZE, tmp);
                    float* out_blk = orow + blk * TENSOR_Q8_0_QK;
                    for (size_t t = 0; t < TENSOR_Q8_0_QK; ++t) out_blk[t] += av * tmp[t];
                }
            }
        }
        return true;
    }

    const float* b = (const float*)B->data;

    // zero out output
    std::memset(o, 0, sizeof(float) * m * n);

    unsigned int nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 1;
    const char* env_th = std::getenv("MINXFMR_CPU_THREADS");
    if (env_th) {
        int v = std::atoi(env_th);
        if (v > 0) nthreads = (unsigned int)v;
    }

    // Decide whether to split work by rows (output rows) or columns (output cols).
    // Splitting by rows is efficient when m (rows) is large. When m is small
    // (e.g. m==1 during decode), split by columns so we can still use threads.
    bool split_cols = (m < n);

    if (split_cols) {
        // clamp to at most n threads when splitting columns
        if ((size_t)nthreads > n) nthreads = (unsigned int)std::max<size_t>(1, n);
    } else {
        // clamp to at most m threads when splitting rows
        if ((size_t)nthreads > m) nthreads = (unsigned int)std::max<size_t>(1, m);
    }

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
