#include "transformer.h"
#include "rmsnorm.h"
#include "attention.h"
#include "rope.h"
#include "../cache/kv_cache.h"
#include "feed_forward.h"
#include "../backend/backend_runtime.h"
#include "softmax.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstddef>
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
    Tensor* out = tensor_create_f32_noinit(in->rows, in->cols);
    if (!out) return nullptr;
    memcpy(out->data, in->data, sizeof(float) * in->rows * in->cols);
    return out;
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
    if (!in || !W || in->type != DataType::F32) return false;
    if (W->type != DataType::F32 && W->type != DataType::Q4_K && W->type != DataType::Q5_0 && W->type != DataType::Q8_0) return false;
    const size_t d_in = in->cols;

    // Some checkpoints store square matrices in the opposite orientation.
    if (transpose_square && W->rows == d_in && W->cols == d_in) {
        out = tensor_create_f32_noinit(in->rows, d_in);
        if (!out || !backend_matmul_rhs_transposed(in, W, out)) {
            tensor_free(out);
            out = nullptr;
            return false;
        }
        return true;
    }

    if (W->rows == d_in) {
        out = tensor_create_f32_noinit(in->rows, W->cols);
        if (!out) return false;
        if (!backend_matmul(in, W, out)) {
            tensor_free(out);
            out = nullptr;
            return false;
        }
        return true;
    }

    if (W->cols == d_in) {
        out = tensor_create_f32_noinit(in->rows, W->rows);
        if (!out || !backend_matmul_rhs_transposed(in, W, out)) {
            tensor_free(out);
            out = nullptr;
            return false;
        }
        return true;
    }

    return false;
}

static bool add_bias_inplace(Tensor* x, const Tensor* bias) {
    if (!x || !bias || x->type != DataType::F32 || bias->type != DataType::F32) return false;
    const size_t cols = x->cols;
    const float* bd = (const float*)bias->data;
    float* xd = (float*)x->data;

    if (bias->rows == 1 && bias->cols == cols) {
        for (size_t r = 0; r < x->rows; ++r) {
            for (size_t c = 0; c < cols; ++c) xd[r * cols + c] += bd[c];
        }
        return true;
    }
    if (bias->cols == 1 && bias->rows == cols) {
        for (size_t r = 0; r < x->rows; ++r) {
            for (size_t c = 0; c < cols; ++c) xd[r * cols + c] += bd[c];
        }
        return true;
    }
    return false;
}

static inline float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// softmax_row is provided by src/transformer/softmax.cpp

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
    const Tensor* Bq_in,
    const Tensor* Bk_in,
    const Tensor* Bv_in,
    const Tensor* Wo_in,
    const Tensor* Wattn_norm_in,
    const Tensor* Wffn_norm_in,
    const Tensor* Wffn_gate_in,
    const Tensor* Wffn_up_in,
    const Tensor* Wffn_down_in,
    float* scores_workspace,
    size_t scores_workspace_len,
    float rope_theta,
    float rmsnorm_epsilon) {
    if (!input || !output) return false;
    // Shapes: input seq x d
    size_t seq = input->rows;
    size_t d = input->cols;
    if (output->rows != seq || output->cols != d) return false;

    // If caller provided projection weights, they must be compatible with the
    // input hidden dimension `d` (either rows==d or cols==d depending on layout).
    if (Wq_in && !(Wq_in->rows == d || Wq_in->cols == d)) {
        fprintf(stderr, "[transformer] Wq shape mismatch (%zu x %zu) for d=%zu\n", Wq_in->rows, Wq_in->cols, d);
        return false;
    }
    if (Wk_in && !(Wk_in->rows == d || Wk_in->cols == d)) {
        fprintf(stderr, "[transformer] Wk shape mismatch (%zu x %zu) for d=%zu\n", Wk_in->rows, Wk_in->cols, d);
        return false;
    }
    if (Wv_in && !(Wv_in->rows == d || Wv_in->cols == d)) {
        fprintf(stderr, "[transformer] Wv shape mismatch (%zu x %zu) for d=%zu\n", Wv_in->rows, Wv_in->cols, d);
        return false;
    }
    if (Wo_in && !(Wo_in->rows == d || Wo_in->cols == d)) {
        fprintf(stderr, "[transformer] Wo shape mismatch (%zu x %zu) for d=%zu\n", Wo_in->rows, Wo_in->cols, d);
        return false;
    }

    // Stage 1: attention input normalization.
    Tensor* norm = tensor_create_f32_noinit(seq, d);
    if (!norm) return false;

    if (!rmsnorm_forward(input, norm, rmsnorm_epsilon)) { tensor_free(norm); return false; }
    if (Wattn_norm_in) apply_norm_scale(norm, Wattn_norm_in);

    // Stage 2: linear projections to Q/K/V.
    // We accept either [din x dout] or [dout x din] weight layout.
    Tensor* Qraw = nullptr;
    Tensor* Kraw = nullptr;
    Tensor* Vraw = nullptr;

    bool q_ok = Wq_in ? project_with_weight(norm, Wq_in, Qraw, g_transpose_square_wq) : false;
    bool k_ok = Wk_in ? project_with_weight(norm, Wk_in, Kraw, g_transpose_square_wk) : false;
    bool v_ok = Wv_in ? project_with_weight(norm, Wv_in, Vraw, g_transpose_square_wv) : false;

    if (q_ok && Qraw && Bq_in) add_bias_inplace(Qraw, Bq_in);
    if (k_ok && Kraw && Bk_in) add_bias_inplace(Kraw, Bk_in);
    if (v_ok && Vraw && Bv_in) add_bias_inplace(Vraw, Bv_in);

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

    // Attention head size checks: hidden size must be divisible by number of heads.
    if (n_head > 0 && (model_dim % n_head) != 0) {
        fprintf(stderr, "[transformer] n_head mismatch: model_dim=%zu n_head=%zu\n", model_dim, n_head);
        tensor_free(Qraw); tensor_free(Kraw); tensor_free(Vraw);
        tensor_free(norm);
        return false;
    }
    if (n_head_kv > 0 && (kv_dim % n_head_kv) != 0) {
        fprintf(stderr, "[transformer] n_head_kv mismatch: kv_dim=%zu n_head_kv=%zu\n", kv_dim, n_head_kv);
        tensor_free(Qraw); tensor_free(Kraw); tensor_free(Vraw);
        tensor_free(norm);
        return false;
    }

    size_t use_n_head = n_head;
    size_t use_n_head_kv = n_head_kv;
    if (use_n_head == 0 || use_n_head_kv == 0) {
        fprintf(stderr,
            "[transformer] head metadata missing: n_head=%zu n_head_kv=%zu (model_dim=%zu kv_dim=%zu)\n",
            n_head,
            n_head_kv,
            model_dim,
            kv_dim);
        tensor_free(Qraw); tensor_free(Kraw); tensor_free(Vraw);
        tensor_free(norm);
        return false;
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

    // Stage 3: apply RoPE using logical sequence position before cache append.
    const size_t start_pos = (cache != nullptr && layer < cache->layers) ? cache->lengths[layer] : 0;
    rope_apply(Qraw, start_pos, use_n_head, head_dim, rope_theta);
    rope_apply(Kraw, start_pos, use_n_head_kv, kv_head_dim, rope_theta);

    Tensor* Q = Qraw;
    Tensor* K = Kraw;
    Tensor* V = Vraw;

    size_t cached_rows = 0;
    size_t cache_head = 0;
    const float* cache_kd = nullptr;
    const float* cache_vd = nullptr;
    if (cache != nullptr && cache->keys.size() > layer && cache->keys[layer] != nullptr && cache->vals[layer] != nullptr && cache->dim == kv_dim) {
        cached_rows = cache->lengths[layer];
        cache_head = cache->heads[layer];
        cache_kd = (const float*)cache->keys[layer]->data;
        cache_vd = (const float*)cache->vals[layer]->data;
    }

    // Stage 4: causal self-attention against [cached tokens + current tokens].
    Tensor* attn_out = tensor_create_f32_noinit(seq, model_dim);
    if (!attn_out) {
        tensor_free(norm); tensor_free(Q); tensor_free(K); tensor_free(V);
        return false;
    }

    const float* qd = (const float*)Q->data;
    const float* kd = (const float*)K->data;
    const float* vd = (const float*)V->data;
    float* od = (float*)attn_out->data;
    const size_t J = cached_rows + seq;
    float* scores = nullptr;
    if (scores_workspace && scores_workspace_len >= J) {
        scores = scores_workspace;
    } else {
        scores = backend_request_workspace(J);
    }
    const size_t group = use_n_head / use_n_head_kv;
    const float score_scale = 1.0f / sqrtf((float)head_dim);

    for (size_t qi = 0; qi < seq; ++qi) {
        for (size_t h = 0; h < use_n_head; ++h) {
            const size_t q_off = h * head_dim;
            const size_t kv_h = std::min(h / std::max<size_t>(1, group), use_n_head_kv - 1);
            const size_t kv_off = kv_h * kv_head_dim;

            const float* qptr = qd + qi * model_dim + q_off;

            // scores for cached rows from ring buffer
            if (cached_rows > 0) {
                const float* k_ring_base = cache_kd + kv_off;
                if (!backend_vec_dot_rows_ring(qptr, k_ring_base, cache_head, cache->seq_max, cached_rows, head_dim, kv_dim, scores)) {
                    tensor_free(norm); tensor_free(Q); tensor_free(K); tensor_free(V); tensor_free(attn_out);
                    return false;
                }
            }
            // scores for current rows from contiguous current-K
            if (seq > 0) {
                const float* k_cur_base = kd + kv_off;
                if (!backend_vec_dot_rows(qptr, k_cur_base, scores + cached_rows, head_dim, seq, kv_dim)) {
                    tensor_free(norm); tensor_free(Q); tensor_free(K); tensor_free(V); tensor_free(attn_out);
                    return false;
                }
            }

            for (size_t j = 0; j < J; ++j) scores[j] *= score_scale;

            const size_t q_pos = cached_rows + qi;
            for (size_t j = q_pos + 1; j < J; ++j) scores[j] = -1e30f;

            softmax_row(scores, J);

            // outvec = scores^T * V_submatrix, with cached-ring part + current-contiguous part.
            float* outvec = od + qi * model_dim + q_off;
            std::memset(outvec, 0, sizeof(float) * head_dim);

            if (cached_rows > 0) {
                for (size_t j = 0; j < cached_rows; ++j) {
                    size_t phys = (cache_head + j) % cache->seq_max;
                    const float w = scores[j];
                    const float* vrow = cache_vd + phys * kv_dim + kv_off;
#if defined(_OPENMP) && !defined(_MSC_VER)
                    #pragma omp simd
#endif
                    for (size_t t = 0; t < head_dim; ++t) {
                        outvec[t] += w * vrow[t];
                    }
                }
            }

            if (seq > 0) {
                float* outvec_cur = backend_request_workspace(head_dim);
                if (!outvec_cur) {
                    tensor_free(norm); tensor_free(Q); tensor_free(K); tensor_free(V); tensor_free(attn_out);
                    return false;
                }
                const float* v_cur_base = vd + kv_off;
                if (!backend_vec_mul_rows_cols(scores + cached_rows, v_cur_base, outvec_cur, seq, head_dim, kv_dim)) {
                    tensor_free(norm); tensor_free(Q); tensor_free(K); tensor_free(V); tensor_free(attn_out);
                    return false;
                }
#if defined(_OPENMP) && !defined(_MSC_VER)
                #pragma omp simd
#endif
                for (size_t t = 0; t < head_dim; ++t) {
                    outvec[t] += outvec_cur[t];
                }
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
    Tensor* resid1 = tensor_create_f32_noinit(seq, d);
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
    Tensor* ffn_out = tensor_create_f32_noinit(seq, d);
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

        Tensor* ffn_norm = tensor_create_f32_noinit(seq, d);
        if (ffn_norm && rmsnorm_forward(resid1, ffn_norm, rmsnorm_epsilon)) {
            if (Wffn_norm_in) apply_norm_scale(ffn_norm, Wffn_norm_in);
            bool g_ok = project_with_weight(ffn_norm, Wffn_gate_in, gate, false);
            bool u_ok = project_with_weight(ffn_norm, Wffn_up_in, up, false);
            if (g_ok && u_ok && gate && up && gate->rows == up->rows && gate->cols == up->cols) {
                fused = tensor_create_f32_noinit(gate->rows, gate->cols);
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
        std::memset(ffn_out->data, 0, sizeof(float) * seq * d);
        ok = true;
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
        const float* kd_cache = (const float*)K->data;
        const float* vd_cache = (const float*)V->data;
        for (size_t i=0;i<seq;++i) kvcache_append(cache, layer, &kd_cache[i*kv_dim], &vd_cache[i*kv_dim]);
        if (cache_append_log_enabled()) {
            fprintf(stderr, "[transformer] appended %zu rows to cache layer=%zu (newlen=%zu)\n", seq, layer, cache->lengths[layer]);
        }
    }
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    return ok;
}
