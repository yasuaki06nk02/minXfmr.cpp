#include "rope.h"
#include "../backend/backend_context.h"
#include <cmath>
#include <vector>

void rope_apply(Tensor* tensor, size_t start_pos, size_t n_heads, size_t head_dim, size_t n_rot, float theta) {
    if (!tensor || tensor->type != DataType::F32) return;
    if (n_heads == 0 || head_dim < 2) return;
    if (tensor->cols != n_heads * head_dim) return;

    float* data = (float*)tensor->data;
    const size_t seq = tensor->rows;
    size_t used_rot = (n_rot > 0) ? n_rot : head_dim;
    // Match llama.cpp-style rotary bounds: rotate at most head_dim and use even dimensions.
    if (used_rot > head_dim) used_rot = head_dim;
    used_rot &= ~((size_t)1);
    if (used_rot < 2) return;
    const size_t half = used_rot / 2;
    const bool rope_neox = backend_context().rope_neox;

    for (size_t r = 0; r < seq; ++r) {
        const double pos = (double)(start_pos + r);
        // ヘッドに依存しない部分を先に計算（本来のコスト）
        std::vector<double> cos_tab(half), sin_tab(half);
        
        for (size_t i = 0; i < half; ++i) {
            const double inv_freq = 1.0 / std::pow((double)theta, (2.0 * (double)i) / (double)used_rot);
            const double ang = pos * inv_freq;
            cos_tab[i] = std::cos(ang);
            sin_tab[i] = std::sin(ang);
        }
        for (size_t h = 0; h < n_heads; ++h) {
            const size_t base = h * head_dim;
            for (size_t i = 0; i < half; ++i) {
                const double c = cos_tab[i];
                const double s = sin_tab[i];
                size_t a;
                size_t b;
                if (rope_neox) {
                    a = base + i;
                    b = base + half + i;
                } else {
                    a = base + (2 * i);
                    b = base + (2 * i + 1);
                }
                // ensure indices are within head_dim
                if (a >= head_dim + base || b >= head_dim + base) continue;
                const size_t aidx = r * tensor->cols + a;
                const size_t bidx = r * tensor->cols + b;
                const double x0 = data[aidx];
                const double x1 = data[bidx];
                data[aidx] = (float)(x0 * c - x1 * s);
                data[bidx] = (float)(x0 * s + x1 * c);
            }
        }
    }
}

void apply_inplace(float* q, float* k, int head_dim, int pos, int n_rot, float theta) {
    if (!q && !k) return;
    if (head_dim < 2) return;
    int used_rot = (n_rot > 0) ? n_rot : head_dim;
    if (used_rot > head_dim) used_rot = head_dim;
    used_rot &= ~1;
    if (used_rot < 2) return;
    const int half = used_rot / 2;
    const double dtheta = (double)theta;
    const bool rope_neox = backend_context().rope_neox;

    for (int i = 0; i < half; ++i) {
        const double inv_freq = 1.0 / std::pow(dtheta, (2.0 * (double)i) / (double)used_rot);
        const double ang = (double)pos * inv_freq;
        const double c = std::cos(ang);
        const double s = std::sin(ang);
        int a, b;
        if (rope_neox) {
            a = i;
            b = half + i;
        } else {
            a = 2 * i;
            b = 2 * i + 1;
        }
        if (a >= head_dim || b >= head_dim) continue;
        if (q) {
            const double x0 = (double)q[a];
            const double x1 = (double)q[b];
            q[a] = (float)(x0 * c - x1 * s);
            q[b] = (float)(x0 * s + x1 * c);
        }
        if (k) {
            const double y0 = (double)k[a];
            const double y1 = (double)k[b];
            k[a] = (float)(y0 * c - y1 * s);
            k[b] = (float)(y0 * s + y1 * c);
        }
    }
}
