#include "feed_forward.h"
#include "../backend/cpu/cpu_backend.h"

bool ffn_forward(const Tensor* input, const Tensor* W, const Tensor* b, Tensor* out) {
    // input: seq x d, W: d x d_ff, out: seq x d_ff, b: 1 x d_ff
    if (!input || !W || !b || !out) return false;
    if (input->type != DataType::F32 || W->type != DataType::F32 || out->type != DataType::F32) return false;
    // use cpu_matmul for input * W
    if (!cpu_matmul(input, W, out)) return false;
    // add bias
    if (b->rows != 1 || b->cols != out->cols) return false;
    size_t seq = out->rows;
    size_t cols = out->cols;
    float* od = (float*)out->data;
    const float* bd = (const float*)b->data;
    for (size_t i=0;i<seq;++i) for (size_t j=0;j<cols;++j) od[i*cols + j] += bd[j];
    return true;
}
