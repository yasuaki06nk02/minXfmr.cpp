#include "rmsnorm.h"
#include <cmath>
#include <cstring>

bool rmsnorm_forward(const Tensor* input, Tensor* output, float epsilon) {
    if (!input || !output) return false;
    if (input->type != DataType::F32 || output->type != DataType::F32) return false;
    if (input->rows != output->rows || input->cols != output->cols) return false;
    size_t rows = input->rows;
    size_t cols = input->cols;
    const float* in = (const float*)input->data;
    float* out = (float*)output->data;
    for (size_t r = 0; r < rows; ++r) {
        double sumsq = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            double v = in[r*cols + c];
            sumsq += v*v;
        }
        double mean = sumsq / (double)cols;
        double denom = 1.0 / std::sqrt(mean + epsilon);
        for (size_t c = 0; c < cols; ++c) {
            out[r*cols + c] = (float)(in[r*cols + c] * denom);
        }
    }
    return true;
}
