#include "softmax.h"
#include <cmath>

void softmax_row(float* row, size_t len) {
    if (!row || len == 0) return;
    float maxv = row[0];
    for (size_t i = 1; i < len; ++i) {
        if (row[i] > maxv) maxv = row[i];
    }
    double sum = 0.0;
    for (size_t i = 0; i < len; ++i) {
        row[i] = expf(row[i] - maxv);
        sum += row[i];
    }
    if (sum <= 0.0) return;
    const float inv = (float)(1.0 / sum);
    for (size_t i = 0; i < len; ++i) row[i] *= inv;
}
