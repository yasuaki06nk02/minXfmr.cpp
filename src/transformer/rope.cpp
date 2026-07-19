#include "rope.h"
#include <cmath>

void rope_apply(Tensor* tensor, size_t start_pos, size_t n_heads, size_t head_dim, float theta) {
    if (!tensor || tensor->type != DataType::F32) return;
    if (n_heads == 0 || head_dim < 2) return;
    if (tensor->cols != n_heads * head_dim) return;

    float* data = (float*)tensor->data;
    const size_t seq = tensor->rows;
    const size_t half = head_dim / 2;

    for (size_t r = 0; r < seq; ++r) {
        const double pos = (double)(start_pos + r);
        for (size_t h = 0; h < n_heads; ++h) {
            const size_t base = h * head_dim;
            for (size_t i = 0; i < half; ++i) {
                const double inv_freq = 1.0 / std::pow((double)theta, (2.0 * (double)i) / (double)head_dim);
                const double ang = pos * inv_freq;
                const double c = std::cos(ang);
                const double s = std::sin(ang);
                const size_t a = r * tensor->cols + base + (2 * i);
                const size_t b = a + 1;
                const double x0 = data[a];
                const double x1 = data[b];
                data[a] = (float)(x0 * c - x1 * s);
                data[b] = (float)(x0 * s + x1 * c);
            }
        }
    }
}

void apply_inplace(float* q, float* k, int head_dim, int pos, float theta) {
    if (!q && !k) return;
    if (head_dim < 2) return;
    const int half = head_dim / 2;
    const double dtheta = (double)theta;

    for (int i = 0; i < half; ++i) {
        const double inv_freq = 1.0 / std::pow(dtheta, (2.0 * (double)i) / (double)head_dim);
        const double ang = (double)pos * inv_freq;
        const double c = std::cos(ang);
        const double s = std::sin(ang);
        int a = 2 * i;
        int b = a + 1;
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
