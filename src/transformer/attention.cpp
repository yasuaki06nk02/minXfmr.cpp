#include "attention.h"
#include <cstring>
#include <cmath>
#include <vector>
#include "../cache/kv_cache.h"
#include "../io/gguf_loader.h"
#include "../backend/cpu/cpu_backend.h"
#include "softmax.h"

static bool attention_debug_once = false;

void attention_set_debug_once(bool v) { attention_debug_once = v; }

bool attention_qk(const Tensor* Q, const Tensor* K, Tensor* out) {
    if (!Q || !K || !out) return false;
    if (Q->type != DataType::F32 || K->type != DataType::F32 || out->type != DataType::F32) return false;
    // Q: seq x d, K: seq x d -> out: seq x seq (scores)
    if (Q->rows != K->rows || Q->cols != K->cols) return false;
    size_t seq = Q->rows;
    size_t d = Q->cols;
    if (out->rows != seq || out->cols != seq) return false;
    const float* q = (const float*)Q->data;
    const float* k = (const float*)K->data;
    float* o = (float*)out->data;
    // zero
    std::memset(o, 0, sizeof(float)*seq*seq);
    for (size_t i=0;i<seq;++i){
        for (size_t j=0;j<seq;++j){
            double s = 0.0;
            for (size_t t=0;t<d;++t) s += (double)q[i*d + t] * (double)k[j*d + t];
            o[i*seq + j] = (float)s;
        }
    }
    return true;
}

// softmax_row provided by src/transformer/softmax.cpp

bool attention_apply_with_cache(const Tensor* Q, const struct KVCache* cache, size_t layer, Tensor* out) {
    if (!Q || !cache || !out) return false;
    if (layer >= cache->layers) return false;
    Tensor* K = cache->keys[layer];
    Tensor* V = cache->vals[layer];
    size_t k_rows = cache->lengths[layer];
    if (!K || !V || k_rows==0) return false;
    size_t q_rows = Q->rows;
    size_t d = Q->cols;
    if (out->rows != q_rows || out->cols != d) return false;
    const float* q = (const float*)Q->data;
    const float* k = (const float*)K->data;
    const float* v = (const float*)V->data;
    float* o = (float*)out->data;
    // temp scores per q row
    float* scores = cpu_workspace(k_rows);
    if (attention_debug_once) {
        const char* wkname = gguf_last_tensor_name("Wk");
        const char* wvname = gguf_last_tensor_name("Wv");
        size_t wk_r = gguf_last_tensor_rows("Wk");
        size_t wk_c = gguf_last_tensor_cols("Wk");
        size_t wv_r = gguf_last_tensor_rows("Wv");
        size_t wv_c = gguf_last_tensor_cols("Wv");
        fprintf(stderr, "[attention] DEBUG once: layer=%zu k_rows=%zu q_rows=%zu d=%zu -- Wk=%s (%zu x %zu) Wv=%s (%zu x %zu)\n",
            layer, k_rows, q_rows, d,
            wkname?wkname:"(none)", wk_r, wk_c,
            wvname?wvname:"(none)", wv_r, wv_c);
        attention_debug_once = false;
    } else {
        fprintf(stderr, "[attention] apply_with_cache layer=%zu k_rows=%zu q_rows=%zu d=%zu\n", layer, k_rows, q_rows, d);
    }
    for (size_t i=0;i<q_rows;++i) {
        // K is expected row-major k_rows x d
        for (size_t j=0;j<k_rows;++j) {
            double s = 0.0;
            for (size_t t=0;t<d;++t) s += (double)q[i*d + t] * (double)k[j*d + t];
            scores[j] = (float)s;
        }
        softmax_row(scores, k_rows);
        if (k_rows > 0 && q_rows <= 2) {
            fprintf(stderr, "[attention] scores for q=%zu:", i);
            for (size_t jj=0;jj<std::min((size_t)4,k_rows);++jj) fprintf(stderr, " %0.3f", scores[jj]);
            fprintf(stderr, "\n");
        }
        // compute weighted sum into out row
        for (size_t t=0;t<d;++t) {
            double acc = 0.0;
            for (size_t j=0;j<k_rows;++j) acc += (double)scores[j] * (double)v[j*d + t];
            o[i*d + t] = (float)acc;
        }
    }
    return true;
}

bool attention_apply_direct(const Tensor* Q, const Tensor* K, const Tensor* V, Tensor* out) {
    if (!Q || !K || !V || !out) return false;
    if (Q->cols != K->cols || K->cols != V->cols) return false;
    size_t q_rows = Q->rows;
    size_t k_rows = K->rows;
    size_t d = Q->cols;
    if (out->rows != q_rows || out->cols != d) return false;
    const float* q = (const float*)Q->data;
    const float* k = (const float*)K->data;
    const float* v = (const float*)V->data;
    float* o = (float*)out->data;
    float* scores = cpu_workspace(k_rows);
    for (size_t i=0;i<q_rows;++i) {
        // K is expected row-major k_rows x d
        for (size_t j=0;j<k_rows;++j) {
            double s = 0.0;
            for (size_t t=0;t<d;++t) s += (double)q[i*d + t] * (double)k[j*d + t];
            scores[j] = (float)s;
        }
        softmax_row(scores, k_rows);
        for (size_t t=0;t<d;++t) {
            double acc = 0.0;
            for (size_t j=0;j<k_rows;++j) acc += (double)scores[j] * (double)v[j*d + t];
            o[i*d + t] = (float)acc;
        }
    }
    return true;
}
