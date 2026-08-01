#include "cpu_backend.h"
#include <algorithm>
#include <cstring>

// Simple tiled matmul implementation to improve cache locality.
// This is an optional optimized path enabled via CMake option MINXFMR_TILED_MATMUL.

bool cpu_matmul_tiled_raw(const float* A, const float* B, float* C, size_t m, size_t n, size_t k) {
    if (!A || !B || !C) return false;
    if (m == 0 || n == 0 || k == 0) return false;

    constexpr size_t TILE_M = 16;
    constexpr size_t TILE_N = 64;
    constexpr size_t TILE_K = 256;

    std::memset(C, 0, sizeof(float) * m * n);

    for (size_t kk = 0; kk < k; kk += TILE_K) {
        size_t k_end = std::min(k, kk + TILE_K);
        for (size_t ii = 0; ii < m; ii += TILE_M) {
            size_t i_end = std::min(m, ii + TILE_M);
            for (size_t jj = 0; jj < n; jj += TILE_N) {
                size_t j_end = std::min(n, jj + TILE_N);
                for (size_t i = ii; i < i_end; ++i) {
                    const float* arow = A + i * k;
                    float* orow_base = C + i * n;
                    for (size_t kk2 = kk; kk2 < k_end; ++kk2) {
                        float av = arow[kk2];
                        const float* brow = B + kk2 * n + jj;
                        float* out = orow_base + jj;
                        size_t len = j_end - jj;
                        for (size_t j = 0; j < len; ++j) {
                            out[j] += av * brow[j];
                        }
                    }
                }
            }
        }
    }

    return true;
}
