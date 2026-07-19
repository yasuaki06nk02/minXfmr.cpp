#include "transformer.h"
#include "rmsnorm.h"
#include "attention.h"
#include "rope.h"
#include "../cache/kv_cache.h"
#include "feed_forward.h"
#include "../backend/cpu/cpu_backend.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

static bool g_transpose_square_wq = false;
static bool g_transpose_square_wk = false;
static bool g_transpose_square_wv = false;
static bool g_transpose_square_wo = false;
static int g_cache_append_log_enabled = -1;

void transformer_set_transpose_square_weights(bool enabled) {
    g_transpose_square_wq = enabled;
    g_transpose_square_wk = enabled;
    g_transpose_square_wv = enabled;
    g_transpose_square_wo = enabled;
}

void transformer_set_transpose_square_weights_for_all(bool wq, bool wk, bool wv, bool wo) {
    g_transpose_square_wq = wq;
    g_transpose_square_wk = wk;
    g_transpose_square_wv = wv;
    g_transpose_square_wo = wo;
}

static Tensor* tensor_clone_f32(const Tensor* in) {
    if (!in || in->type != DataType::F32) return nullptr;
    Tensor* out = tensor_create_f32(in->rows, in->cols);
    if (!out) return nullptr;
    memcpy(out->data, in->data, sizeof(float) * in->rows * in->cols);
    return out;
}

static Tensor* tensor_slice_cols_f32(const Tensor* in, size_t out_cols) {
    if (!in || in->type != DataType::F32 || out_cols == 0 || out_cols > in->cols) return nullptr;
    if (out_cols == in->cols) return tensor_clone_f32(in);
    Tensor* out = tensor_create_f32(in->rows, out_cols);
    if (!out) return nullptr;
    const float* src = (const float*)in->data;
    float* dst = (float*)out->data;
    for (size_t r = 0; r < in->rows; ++r) {
        memcpy(dst + r * out_cols, src + r * in->cols, sizeof(float) * out_cols);
    }
    return out;
}

// Compute out = A * B^T without materializing B^T.
// A: [m x k], B: [n x k], out: [m x n]
static bool cpu_matmul_rhs_transposed(const Tensor* A, const Tensor* B, Tensor* out) {
    if (!A || !B || !out) return false;
    if (A->type != DataType::F32 || B->type != DataType::F32 || out->type != DataType::F32) return false;
    if (A->cols != B->cols) return false;
    if (out->rows != A->rows || out->cols != B->rows) return false;

    const size_t m = A->rows;
    const size_t k = A->cols;
    const size_t n = B->rows;
    const float* ad = (const float*)A->data;
    const float* bd = (const float*)B->data;
    float* od = (float*)out->data;

    for (size_t i = 0; i < m; ++i) {
        const float* arow = ad + i * k;
        for (size_t j = 0; j < n; ++j) {
            const float* brow = bd + j * k;
            float s = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                s += arow[kk] * brow[kk];
            }
            od[i * n + j] = s;
        }
    }
    return true;
}

static bool cache_append_log_enabled() {
    if (g_cache_append_log_enabled < 0) {
        const char* v = std::getenv("MINXFMR_VERBOSE_CACHE");
        g_cache_append_log_enabled = (v && v[0] == '1') ? 1 : 0;
    }
    return g_cache_append_log_enabled == 1;
}

// Project input (seq x d_in) with a weight matrix that may be stored as
// [d_in x d_out] or [d_out x d_in].
static bool project_with_weight(const Tensor* in, const Tensor* W, Tensor*& out, bool transpose_square) {
    out = nullptr;
    if (!in || !W || in->type != DataType::F32 || W->type != DataType::F32) return false;
    const size_t d_in = in->cols;

    if (transpose_square && W->rows == d_in && W->cols == d_in) {
        out = tensor_create_f32(in->rows, d_in);
        if (!out || !cpu_matmul_rhs_transposed(in, W, out)) {
            tensor_free(out);
            out = nullptr;
            return false;
        }
        return true;
    }

    if (W->rows == d_in) {
        out = tensor_create_f32(in->rows, W->cols);
        if (!out) return false;
        if (!cpu_matmul(in, W, out)) {
            tensor_free(out);
            out = nullptr;
            return false;
        }
        return true;
    }

    if (W->cols == d_in) {
        out = tensor_create_f32(in->rows, W->rows);
        if (!out || !cpu_matmul_rhs_transposed(in, W, out)) {
            tensor_free(out);
            out = nullptr;
            return false;
        }
        return true;
    }

    return false;
}

static inline float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

static void softmax_row(float* row, size_t len) {
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

static bool apply_norm_scale(Tensor* x, const Tensor* w) {
    if (!x || !w || x->type != DataType::F32 || w->type != DataType::F32) return false;
    size_t d = x->cols;
    const float* wd = (const float*)w->data;
    float* xd = (float*)x->data;

    if (w->rows == 1 && w->cols == d) {
        for (size_t r = 0; r < x->rows; ++r) {
            for (size_t c = 0; c < d; ++c) xd[r * d + c] *= wd[c];
        }
        return true;
    }
    if (w->cols == 1 && w->rows == d) {
        for (size_t r = 0; r < x->rows; ++r) {
            for (size_t c = 0; c < d; ++c) xd[r * d + c] *= wd[c];
        }
        return true;
    }
    if (w->rows == d && w->cols == d) {
        for (size_t r = 0; r < x->rows; ++r) {
            for (size_t c = 0; c < d; ++c) xd[r * d + c] *= wd[c * d + c];
        }
        return true;
    }
    return false;
}

bool transformer_forward_single_layer(
    const Tensor* input,
    Tensor* output,
    struct KVCache* cache,
    size_t layer,
    size_t n_head,
    size_t n_head_kv,
    const Tensor* Wq_in,
    const Tensor* Wk_in,
    const Tensor* Wv_in,
    const Tensor* Wo_in,
    const Tensor* Wattn_norm_in,
    const Tensor* Wffn_norm_in,
    const Tensor* Wffn_gate_in,
    const Tensor* Wffn_up_in,
    const Tensor* Wffn_down_in) {
    if (!input || !output) return false;
    // Shapes: input seq x d
    size_t seq = input->rows;
    size_t d = input->cols;
    if (output->rows != seq || output->cols != d) return false;

    // Allocate temporaries
    Tensor* norm = tensor_create_f32(seq, d);
    if (!norm) return false;

    if (!rmsnorm_forward(input, norm)) { tensor_free(norm); return false; }
    if (Wattn_norm_in) apply_norm_scale(norm, Wattn_norm_in);

    // Project norm -> Q,K,V. We accept either [din x dout] or [dout x din] weight layout.
    Tensor* Qraw = nullptr;
    Tensor* Kraw = nullptr;
    Tensor* Vraw = nullptr;

    bool q_ok = Wq_in ? project_with_weight(norm, Wq_in, Qraw, g_transpose_square_wq) : false;
    bool k_ok = Wk_in ? project_with_weight(norm, Wk_in, Kraw, g_transpose_square_wk) : false;
    bool v_ok = Wv_in ? project_with_weight(norm, Wv_in, Vraw, g_transpose_square_wv) : false;

    if (!q_ok || !k_ok || !v_ok) {
        tensor_free(Qraw); tensor_free(Kraw); tensor_free(Vraw);
        Qraw = tensor_clone_f32(norm);
        Kraw = tensor_clone_f32(norm);
        Vraw = tensor_clone_f32(norm);
    }
    if (!Qraw || !Kraw || !Vraw) {
        tensor_free(Qraw); tensor_free(Kraw); tensor_free(Vraw);
        tensor_free(norm);
        return false;
    }

    const size_t model_dim = Qraw->cols;
    const size_t kv_dim = std::min(Kraw->cols, Vraw->cols);
    if (model_dim == 0 || kv_dim == 0) {
        tensor_free(Qraw); tensor_free(Kraw); tensor_free(Vraw);
        tensor_free(norm);
        return false;
    }

    size_t use_n_head = n_head;
    size_t use_n_head_kv = n_head_kv;
    if (use_n_head == 0 || use_n_head_kv == 0) {
        // best-effort inference; prefer 4 kv heads for 256 kv dim / 64 head_dim layouts.
        if (model_dim % 64 == 0 && kv_dim % 64 == 0) {
            use_n_head = model_dim / 64;
            use_n_head_kv = kv_dim / 64;
        } else {
            use_n_head = 1;
            use_n_head_kv = 1;
        }
    }
    if (use_n_head == 0 || use_n_head_kv == 0 || model_dim % use_n_head != 0 || kv_dim % use_n_head_kv != 0) {
        tensor_free(Qraw); tensor_free(Kraw); tensor_free(Vraw);
        tensor_free(norm);
        return false;
    }
    const size_t head_dim = model_dim / use_n_head;
    const size_t kv_head_dim = kv_dim / use_n_head_kv;
    if (head_dim != kv_head_dim) {
        tensor_free(Qraw); tensor_free(Kraw); tensor_free(Vraw);
        tensor_free(norm);
        return false;
    }

    const size_t start_pos = (cache != nullptr && layer < cache->layers) ? cache->lengths[layer] : 0;
    rope_apply(Qraw, start_pos, use_n_head, head_dim);
    rope_apply(Kraw, start_pos, use_n_head_kv, kv_head_dim);

    Tensor* Q = Qraw;
    Tensor* K = Kraw;
    Tensor* V = Vraw;

    // Build concatenated K/V (cache rows + current K/V)
    Tensor* Kconcat = nullptr;
    Tensor* Vconcat = nullptr;
    size_t cached_rows = 0;
    if (cache != nullptr && cache->keys.size() > layer && cache->keys[layer] != nullptr) {
        cached_rows = cache->lengths[layer];
        size_t cached = cached_rows;
        Kconcat = tensor_create_f32(cached + seq, kv_dim);
        Vconcat = tensor_create_f32(cached + seq, kv_dim);
        if (Kconcat && Vconcat) {
            if (cached > 0) {
                memcpy(Kconcat->data, cache->keys[layer]->data, sizeof(float) * cached * kv_dim);
                memcpy(Vconcat->data, cache->vals[layer]->data, sizeof(float) * cached * kv_dim);
            }
            memcpy((char*)Kconcat->data + sizeof(float) * cached * kv_dim, K->data, sizeof(float) * seq * kv_dim);
            memcpy((char*)Vconcat->data + sizeof(float) * cached * kv_dim, V->data, sizeof(float) * seq * kv_dim);
        }
    } else {
        Kconcat = tensor_clone_f32(K);
        Vconcat = tensor_clone_f32(V);
    }

    if (!Kconcat || !Vconcat) {
        tensor_free(norm); tensor_free(Q); tensor_free(K); tensor_free(V); tensor_free(Kconcat); tensor_free(Vconcat);
        return false;
    }

    Tensor* attn_out = tensor_create_f32(seq, model_dim);
    if (!attn_out) {
        tensor_free(norm); tensor_free(Q); tensor_free(K); tensor_free(V); tensor_free(Kconcat); tensor_free(Vconcat);
        return false;
    }

    const float* qd = (const float*)Q->data;
    const float* kd = (const float*)Kconcat->data;
    const float* vd = (const float*)Vconcat->data;
    float* od = (float*)attn_out->data;
    float* scores = cpu_workspace(cached_rows + seq);
    const size_t group = use_n_head / use_n_head_kv;
    const float score_scale = 1.0f / sqrtf((float)head_dim);

    for (size_t qi = 0; qi < seq; ++qi) {
        for (size_t h = 0; h < use_n_head; ++h) {
            const size_t q_off = h * head_dim;
            const size_t kv_h = std::min(h / std::max<size_t>(1, group), use_n_head_kv - 1);
            const size_t kv_off = kv_h * kv_head_dim;

            for (size_t j = 0; j < cached_rows + seq; ++j) {
                double s = 0.0;
                for (size_t t = 0; t < head_dim; ++t) {
                    s += (double)qd[qi * model_dim + q_off + t] * (double)kd[j * kv_dim + kv_off + t];
                }
                scores[j] = (float)s * score_scale;
            }
            softmax_row(scores, cached_rows + seq);
            for (size_t t = 0; t < head_dim; ++t) {
                double acc = 0.0;
                for (size_t j = 0; j < cached_rows + seq; ++j) {
                    acc += (double)scores[j] * (double)vd[j * kv_dim + kv_off + t];
                }
                od[qi * model_dim + q_off + t] = (float)acc;
            }
        }
    }

    // Attention output projection (Wo)
    Tensor* attn_proj = nullptr;
    if (Wo_in && project_with_weight(attn_out, Wo_in, attn_proj, g_transpose_square_wo) && attn_proj && attn_proj->rows == seq && attn_proj->cols == d) {
        // use projected output
    } else {
        tensor_free(attn_proj);
        attn_proj = tensor_clone_f32(attn_out);
    }
    if (!attn_proj) {
        tensor_free(norm); tensor_free(attn_out); tensor_free(Q); tensor_free(K); tensor_free(V);
        return false;
    }

    // First residual: x + attn_proj
    Tensor* resid1 = tensor_create_f32(seq, d);
    if (!resid1) {
        tensor_free(norm); tensor_free(attn_out); tensor_free(attn_proj); tensor_free(Q); tensor_free(K); tensor_free(V);
        return false;
    }
    {
        const float* in_d = (const float*)input->data;
        const float* ap_d = (const float*)attn_proj->data;
        float* r1_d = (float*)resid1->data;
        for (size_t i = 0; i < seq * d; ++i) r1_d[i] = in_d[i] + ap_d[i];
    }

    bool ok = false;
    Tensor* ffn_out = tensor_create_f32(seq, d);
    if (!ffn_out) {
        tensor_free(norm); tensor_free(attn_out);
        tensor_free(Q); tensor_free(K); tensor_free(V);
        return false;
    }

    if (Wffn_gate_in && Wffn_up_in && Wffn_down_in) {
        Tensor* gate = nullptr;
        Tensor* up = nullptr;
        Tensor* fused = nullptr;
        Tensor* down = nullptr;

        Tensor* ffn_norm = tensor_create_f32(seq, d);
        if (ffn_norm && rmsnorm_forward(resid1, ffn_norm)) {
            if (Wffn_norm_in) apply_norm_scale(ffn_norm, Wffn_norm_in);
            bool g_ok = project_with_weight(ffn_norm, Wffn_gate_in, gate, false);
            bool u_ok = project_with_weight(ffn_norm, Wffn_up_in, up, false);
            if (g_ok && u_ok && gate && up && gate->rows == up->rows && gate->cols == up->cols) {
                fused = tensor_create_f32(gate->rows, gate->cols);
                if (fused) {
                    float* fd = (float*)fused->data;
                    const float* gd = (const float*)gate->data;
                    const float* ud = (const float*)up->data;
                    size_t n = gate->rows * gate->cols;
                    for (size_t i = 0; i < n; ++i) {
                        fd[i] = silu_f32(gd[i]) * ud[i];
                    }
                    if (project_with_weight(fused, Wffn_down_in, down, false) && down && down->rows == seq && down->cols == d) {
                        memcpy(ffn_out->data, down->data, sizeof(float) * seq * d);
                        ok = true;
                    }
                }
            }
        }
        tensor_free(ffn_norm);

        tensor_free(gate);
        tensor_free(up);
        tensor_free(fused);
        tensor_free(down);
    }

    if (!ok) {
        // Fallback to identity FFN for compatibility.
        Tensor* W = tensor_create_f32(d, d);
        Tensor* b = tensor_create_f32(1, d);
        if (!W || !b) {
            tensor_free(W); tensor_free(b);
            tensor_free(norm); tensor_free(attn_out); tensor_free(ffn_out);
            tensor_free(Q); tensor_free(K); tensor_free(V);
            return false;
        }
        for (size_t i = 0; i < d; i++) for (size_t j = 0; j < d; j++) tensor_set_f32(W, i, j, (i == j) ? 1.0f : 0.0f);
        for (size_t j = 0; j < d; j++) tensor_set_f32(b, 0, j, 0.0f);
        ok = ffn_forward(resid1, W, b, ffn_out);
        tensor_free(W);
        tensor_free(b);
    }
    if (ok) {
        float* od = (float*)output->data;
        const float* r1 = (const float*)resid1->data;
        float* fd = (float*)ffn_out->data;
        for (size_t i = 0; i < seq * d; ++i) od[i] = r1[i] + fd[i];
    }
    tensor_free(norm); tensor_free(attn_out); tensor_free(attn_proj); tensor_free(resid1); tensor_free(ffn_out);

    // now append current K/V rows to cache if present
    if (cache != nullptr && cache->keys.size() > layer && cache->keys[layer] != nullptr && cache->dim == kv_dim) {
        const float* kd = (const float*)K->data;
        const float* vd = (const float*)V->data;
        for (size_t i=0;i<seq;++i) kvcache_append(cache, layer, &kd[i*kv_dim], &vd[i*kv_dim]);
        if (cache_append_log_enabled()) {
            fprintf(stderr, "[transformer] appended %zu rows to cache layer=%zu (newlen=%zu)\n", seq, layer, cache->lengths[layer]);
        }
    }
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(Kconcat);
    tensor_free(Vconcat);
    return ok;
}
