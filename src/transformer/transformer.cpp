#include "transformer.h"
#include "rmsnorm.h"
#include "attention.h"
#include "rope.h"
#include "../cache/kv_cache.h"
#include "feed_forward.h"
#include "../backend/backend_runtime.h"
#include "../backend/backend_context.h"
#include "softmax.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "runtime_config.h"

void transformer_set_transpose_square_weights(bool enabled) {
    backend_context().transpose_square_wq = enabled;
    backend_context().transpose_square_wk = enabled;
    backend_context().transpose_square_wv = enabled;
    backend_context().transpose_square_wo = enabled;
    backend_context().transpose_square_ffn_gate = enabled;
    backend_context().transpose_square_ffn_up = enabled;
    backend_context().transpose_square_ffn_down = enabled;
}

void transformer_set_transpose_square_weights_for_all(
    bool wq,
    bool wk,
    bool wv,
    bool wo,
    bool ffn_gate,
    bool ffn_up,
    bool ffn_down) {
    backend_context().transpose_square_wq = wq;
    backend_context().transpose_square_wk = wk;
    backend_context().transpose_square_wv = wv;
    backend_context().transpose_square_wo = wo;
    backend_context().transpose_square_ffn_gate = ffn_gate;
    backend_context().transpose_square_ffn_up = ffn_up;
    backend_context().transpose_square_ffn_down = ffn_down;
}

static Tensor* tensor_clone_f32(const Tensor* in) {
    if (!in || in->type != DataType::F32) return nullptr;
    Tensor* out = tensor_create_f32_noinit(in->rows, in->cols);
    if (!out) return nullptr;
    memcpy(out->data, in->data, sizeof(float) * in->rows * in->cols);
    return out;
}

static bool cache_append_log_enabled() {
    if (backend_context().cache_append_log_enabled < 0) {
        backend_context().cache_append_log_enabled = RuntimeConfig::Instance().verbose_cache() ? 1 : 0;
    }
    return backend_context().cache_append_log_enabled == 1;
}

// Project input (seq x d_in) with a weight matrix that may be stored as
// [d_in x d_out] or [d_out x d_in].
static bool project_with_weight(const Tensor* in, const Tensor* W, Tensor*& out, bool transpose_square) {
    out = nullptr;
    bool debug = RuntimeConfig::Instance().getBool("MINXFMR_CHAT_DEBUG");
    const bool prefer_ggml_mul_mat_layout = backend_context().prefer_ggml_mul_mat_layout;
    auto dtype_to_str = [](DataType t)->const char* {
        switch (t) {
            case DataType::F32: return "F32";
            case DataType::Q4_K: return "Q4_K";
            case DataType::Q5_0: return "Q5_0";
            case DataType::Q8_0: return "Q8_0";
            default: return "UNKNOWN";
        }
    };
    if (!in || !W) {
        if (debug) fprintf(stderr, "[transformer] project_with_weight: null input or weight (in=%p W=%p)\n", (const void*)in, (const void*)W);
        return false;
    }
    if (in->type != DataType::F32) {
        if (debug) fprintf(stderr, "[transformer] project_with_weight: input not F32 (type=%s)\n", dtype_to_str(in->type));
        return false;
    }
    if (W->type != DataType::F32 && W->type != DataType::Q4_K && W->type != DataType::Q5_0 && W->type != DataType::Q8_0) {
        if (debug) fprintf(stderr, "[transformer] project_with_weight: unsupported weight type=%s\n", dtype_to_str(W->type));
        return false;
    }
    const size_t d_in = in->cols;

    // Some checkpoints store square matrices in the opposite orientation.
    if (transpose_square && W->rows == d_in && W->cols == d_in && W->type != DataType::F32) {
        if (debug) fprintf(stderr, "[transformer] project_with_weight: using square_rhs_transposed path in_cols=%zu W=%zux%zu type=%s\n", d_in, W->rows, W->cols, dtype_to_str(W->type));
        out = tensor_create_f32_noinit(in->rows, d_in);
        if (!out) {
            if (debug) fprintf(stderr, "[transformer] project_with_weight: failed to allocate out tensor\n");
            return false;
        }
        if (!backend_matmul_rhs_transposed(in, W, out)) {
            if (debug) fprintf(stderr, "[transformer] project_with_weight: backend_matmul_rhs_transposed failed: %s\n", backend_last_preload_error());
            tensor_free(out);
            out = nullptr;
            return false;
        }
        return true;
    }

    // llama.cpp / ggml convention: loaded GGUF linear weights are usually
    // interpreted as [out x in], so cols==input_dim is the native multiply
    // direction. Prefer this for architectures that opt into ggml layout
    // compatibility, especially when square weights are ambiguous.
    if (prefer_ggml_mul_mat_layout && W->cols == d_in) {
        if (debug) fprintf(stderr, "[transformer] project_with_weight: using ggml/rhs_transposed path in_cols=%zu W_rows=%zu W_cols=%zu type=%s\n", d_in, W->rows, W->cols, dtype_to_str(W->type));
        out = tensor_create_f32_noinit(in->rows, W->rows);
        if (!out) {
            if (debug) fprintf(stderr, "[transformer] project_with_weight: failed to allocate out tensor\n");
            return false;
        }
        if (!backend_matmul_rhs_transposed(in, W, out)) {
            if (debug) fprintf(stderr, "[transformer] project_with_weight: backend_matmul_rhs_transposed failed: %s\n", backend_last_preload_error());
            tensor_free(out);
            out = nullptr;
            return false;
        }
        return true;
    }

    if (W->rows == d_in) {
        if (debug) fprintf(stderr, "[transformer] project_with_weight: using matmul path in_cols=%zu W_rows=%zu W_cols=%zu type=%s\n", d_in, W->rows, W->cols, dtype_to_str(W->type));
        out = tensor_create_f32_noinit(in->rows, W->cols);
        if (!out) {
            if (debug) fprintf(stderr, "[transformer] project_with_weight: failed to allocate out tensor\n");
            return false;
        }
        if (!backend_matmul(in, W, out)) {
            if (debug) fprintf(stderr, "[transformer] project_with_weight: backend_matmul failed: %s\n", backend_last_preload_error());
            tensor_free(out);
            out = nullptr;
            return false;
        }
        return true;
    }

    if (W->cols == d_in) {
        if (debug) fprintf(stderr, "[transformer] project_with_weight: using rhs_transposed path in_cols=%zu W_rows=%zu W_cols=%zu type=%s\n", d_in, W->rows, W->cols, dtype_to_str(W->type));
        out = tensor_create_f32_noinit(in->rows, W->rows);
        if (!out) {
            if (debug) fprintf(stderr, "[transformer] project_with_weight: failed to allocate out tensor\n");
            return false;
        }
        if (!backend_matmul_rhs_transposed(in, W, out)) {
            if (debug) fprintf(stderr, "[transformer] project_with_weight: backend_matmul_rhs_transposed failed: %s\n", backend_last_preload_error());
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

static void compare_dump_tensor_stats(const char* phase, size_t layer, const Tensor* t) {
    if (!phase || !backend_context().compare_trace_enabled) return;
    if (!t || t->type != DataType::F32 || !t->data) return;
    const size_t n = t->rows * t->cols;
    if (n == 0) return;

    const float* d = (const float*)t->data;
    double mn = (double)d[0];
    double mx = (double)d[0];
    double sum = 0.0;
    double sum2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = (double)d[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sum2 += v * v;
    }
    const double mean = sum / (double)n;
    double var = (sum2 / (double)n) - (mean * mean);
    if (var < 0.0) var = 0.0;
    const double stddev = std::sqrt(var);

    std::ostringstream js;
    js << "{\"phase\":\"" << phase << "\",\"t\":" << backend_context().compare_trace_step
       << ",\"layer\":" << layer
       << ",\"n\":" << n
       << ",\"min\":" << std::fixed << std::setprecision(6) << mn
       << ",\"max\":" << std::fixed << std::setprecision(6) << mx
       << ",\"mean\":" << std::fixed << std::setprecision(6) << mean
       << ",\"std\":" << std::fixed << std::setprecision(6) << stddev
       << "}";
    fprintf(stderr, "%s\n", js.str().c_str());
    fflush(stderr);
}

static void compare_dump_tensor_last_row_stats(const char* phase, size_t layer, const Tensor* t) {
    if (!phase || !backend_context().compare_trace_enabled) return;
    if (!t || t->type != DataType::F32 || !t->data || t->rows == 0 || t->cols == 0) return;
    const size_t row = t->rows - 1;
    const size_t n = t->cols;
    const float* d = ((const float*)t->data) + row * t->cols;

    double mn = (double)d[0];
    double mx = (double)d[0];
    double sum = 0.0;
    double sum2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = (double)d[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sum2 += v * v;
    }
    const double mean = sum / (double)n;
    double var = (sum2 / (double)n) - (mean * mean);
    if (var < 0.0) var = 0.0;
    const double stddev = std::sqrt(var);

    std::ostringstream js;
    js << "{\"phase\":\"" << phase << "\",\"t\":" << backend_context().compare_trace_step
       << ",\"layer\":" << layer
       << ",\"row\":" << row
       << ",\"n\":" << n
       << ",\"min\":" << std::fixed << std::setprecision(6) << mn
       << ",\"max\":" << std::fixed << std::setprecision(6) << mx
       << ",\"mean\":" << std::fixed << std::setprecision(6) << mean
       << ",\"std\":" << std::fixed << std::setprecision(6) << stddev
       << "}";
    fprintf(stderr, "%s\n", js.str().c_str());
    fflush(stderr);
}

static const char* dtype_name(DataType t) {
    switch (t) {
        case DataType::F32: return "F32";
        case DataType::Q4_K: return "Q4_K";
        case DataType::Q5_0: return "Q5_0";
        case DataType::Q8_0: return "Q8_0";
        default: return "UNKNOWN";
    }
}

static void compare_dump_projection_path(const char* phase, size_t layer, const Tensor* in, const Tensor* w, bool transpose_square) {
    if (!backend_context().compare_trace_enabled) return;
    if (!phase || !in || !w) return;

    const size_t d_in = in->cols;
    const char* path = "incompatible";
    if (transpose_square && w->rows == d_in && w->cols == d_in && w->type != DataType::F32) {
        path = "square_rhs_transposed";
    } else if (backend_context().prefer_ggml_mul_mat_layout && w->cols == d_in) {
        path = "ggml_rhs_transposed";
    } else if (w->rows == d_in) {
        path = "matmul";
    } else if (w->cols == d_in) {
        path = "rhs_transposed";
    }

    std::ostringstream os;
    os << std::fixed << std::setprecision(0);
    os << "{\"phase\":\"" << phase << "\",\"t\":" << backend_context().compare_trace_step
       << ",\"layer\":" << layer
       << ",\"in_cols\":" << d_in
       << ",\"w_rows\":" << w->rows
       << ",\"w_cols\":" << w->cols
       << ",\"w_type\":\"" << dtype_name(w->type) << "\""
       << ",\"transpose_square\":" << (transpose_square ? 1 : 0)
       << ",\"path\":\"" << path << "\"}"
       << "\n";
    fputs(os.str().c_str(), stderr);
    fflush(stderr);
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
    uint64_t rope_n_rot,
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

    bool debug_proj = RuntimeConfig::Instance().getBool("MINXFMR_CHAT_DEBUG");
    if (debug_proj) {
        fprintf(stderr, "[transformer] layer=%zu proj input shape=%zux%zu transpose_wq=%d transpose_wk=%d transpose_wv=%d\n",
            layer, norm->rows, norm->cols,
            backend_context().transpose_square_wq ? 1 : 0,
            backend_context().transpose_square_wk ? 1 : 0,
            backend_context().transpose_square_wv ? 1 : 0);
        if (Wq_in) fprintf(stderr, "[transformer] layer=%zu Wq shape=%zux%zu type=%d\n", layer, Wq_in->rows, Wq_in->cols, (int)Wq_in->type);
        if (Wk_in) fprintf(stderr, "[transformer] layer=%zu Wk shape=%zux%zu type=%d\n", layer, Wk_in->rows, Wk_in->cols, (int)Wk_in->type);
        if (Wv_in) fprintf(stderr, "[transformer] layer=%zu Wv shape=%zux%zu type=%d\n", layer, Wv_in->rows, Wv_in->cols, (int)Wv_in->type);
    }

    bool q_ok = Wq_in ? project_with_weight(norm, Wq_in, Qraw, backend_context().transpose_square_wq) : false;
    bool k_ok = Wk_in ? project_with_weight(norm, Wk_in, Kraw, backend_context().transpose_square_wk) : false;
    bool v_ok = Wv_in ? project_with_weight(norm, Wv_in, Vraw, backend_context().transpose_square_wv) : false;

    if (debug_proj) {
        fprintf(stderr, "[transformer] layer=%zu proj results q_ok=%d k_ok=%d v_ok=%d Qraw=%p Kraw=%p Vraw=%p\n",
            layer, q_ok ? 1 : 0, k_ok ? 1 : 0, v_ok ? 1 : 0, (const void*)Qraw, (const void*)Kraw, (const void*)Vraw);
    }

    if (q_ok && Qraw && Bq_in) add_bias_inplace(Qraw, Bq_in);
    if (k_ok && Kraw && Bk_in) add_bias_inplace(Kraw, Bk_in);
    if (v_ok && Vraw && Bv_in) add_bias_inplace(Vraw, Bv_in);

    if (!q_ok || !k_ok || !v_ok) {
        fprintf(stderr,
            "[transformer] missing projection weights at layer=%zu (q_ok=%d k_ok=%d v_ok=%d)\n",
            layer,
            q_ok ? 1 : 0,
            k_ok ? 1 : 0,
            v_ok ? 1 : 0);
        tensor_free(Qraw);
        tensor_free(Kraw);
        tensor_free(Vraw);
        tensor_free(norm);
        return false;
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
    // Clamp/normalize rotary dimensions per projection width to mirror llama.cpp
    // behavior for architectures where metadata can exceed per-head width.
    const size_t start_pos = (cache != nullptr && layer < cache->layers) ? cache->lengths[layer] : 0;
    size_t q_rot = (rope_n_rot > 0) ? (size_t)rope_n_rot : head_dim;
    size_t k_rot = (rope_n_rot > 0) ? (size_t)rope_n_rot : kv_head_dim;
    if (q_rot > head_dim) q_rot = head_dim;
    if (k_rot > kv_head_dim) k_rot = kv_head_dim;
    q_rot &= ~((size_t)1);
    k_rot &= ~((size_t)1);
    rope_apply(Qraw, start_pos, use_n_head, head_dim, q_rot, rope_theta);
    rope_apply(Kraw, start_pos, use_n_head_kv, kv_head_dim, k_rot, rope_theta);

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
    std::vector<float> scores_fallback;
    if (scores_workspace && scores_workspace_len >= J) {
        scores = scores_workspace;
    } else {
        scores_fallback.assign(J, 0.0f);
        scores = scores_fallback.data();
    }
    const size_t group = use_n_head / use_n_head_kv;
    const float score_scale = 1.0f / sqrtf((float)head_dim);
    std::vector<float> outvec_cur_buf(head_dim, 0.0f);

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
                float* outvec_cur = outvec_cur_buf.data();
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
    if (Wo_in && project_with_weight(attn_out, Wo_in, attn_proj, backend_context().transpose_square_wo) && attn_proj && attn_proj->rows == seq && attn_proj->cols == d) {
        // use projected output
    } else {
        tensor_free(attn_proj);
        attn_proj = tensor_clone_f32(attn_out);
    }
    if (!attn_proj) {
        tensor_free(norm); tensor_free(attn_out); tensor_free(Q); tensor_free(K); tensor_free(V);
        return false;
    }
    compare_dump_tensor_stats("layer_attn_proj", layer, attn_proj);
    if (backend_context().compare_trace_step < 0) compare_dump_tensor_last_row_stats("prefill_last_layer_attn_proj", layer, attn_proj);

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
    compare_dump_tensor_stats("layer_resid1", layer, resid1);
    if (backend_context().compare_trace_step < 0) compare_dump_tensor_last_row_stats("prefill_last_layer_resid1", layer, resid1);

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
            compare_dump_tensor_stats("layer_ffn_norm", layer, ffn_norm);
            if (backend_context().compare_trace_step < 0) compare_dump_tensor_last_row_stats("prefill_last_layer_ffn_norm", layer, ffn_norm);
            if (backend_context().compare_trace_step < 0) {
                compare_dump_projection_path("prefill_last_layer_ffn_gate_proj", layer, ffn_norm, Wffn_gate_in, backend_context().transpose_square_ffn_gate);
                compare_dump_projection_path("prefill_last_layer_ffn_up_proj", layer, ffn_norm, Wffn_up_in, backend_context().transpose_square_ffn_up);
            }
            bool g_ok = project_with_weight(ffn_norm, Wffn_gate_in, gate, backend_context().transpose_square_ffn_gate);
            bool u_ok = project_with_weight(ffn_norm, Wffn_up_in, up, backend_context().transpose_square_ffn_up);
            if (g_ok && u_ok && gate && up && gate->rows == up->rows && gate->cols == up->cols) {
                compare_dump_tensor_stats("layer_ffn_gate", layer, gate);
                compare_dump_tensor_stats("layer_ffn_up", layer, up);
                if (backend_context().compare_trace_step < 0) {
                    compare_dump_tensor_last_row_stats("prefill_last_layer_ffn_gate", layer, gate);
                    compare_dump_tensor_last_row_stats("prefill_last_layer_ffn_up", layer, up);
                }
                fused = tensor_create_f32_noinit(gate->rows, gate->cols);
                if (fused) {
                    float* fd = (float*)fused->data;
                    const float* gd = (const float*)gate->data;
                    const float* ud = (const float*)up->data;
                    size_t n = gate->rows * gate->cols;
                    for (size_t i = 0; i < n; ++i) {
                        fd[i] = silu_f32(gd[i]) * ud[i];
                    }
                    compare_dump_tensor_stats("layer_ffn_fused", layer, fused);
                    if (backend_context().compare_trace_step < 0) compare_dump_tensor_last_row_stats("prefill_last_layer_ffn_fused", layer, fused);
                    if (backend_context().compare_trace_step < 0) {
                        compare_dump_projection_path("prefill_last_layer_ffn_down_proj", layer, fused, Wffn_down_in, backend_context().transpose_square_ffn_down);
                    }
                    if (project_with_weight(fused, Wffn_down_in, down, backend_context().transpose_square_ffn_down) && down && down->rows == seq && down->cols == d) {
                        compare_dump_tensor_stats("layer_ffn_down", layer, down);
                        if (backend_context().compare_trace_step < 0) compare_dump_tensor_last_row_stats("prefill_last_layer_ffn_down", layer, down);
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
    compare_dump_tensor_stats("layer_ffn_out", layer, ffn_out);
    if (ok) {
        float* od = (float*)output->data;
        const float* r1 = (const float*)resid1->data;
        float* fd = (float*)ffn_out->data;
        for (size_t i = 0; i < seq * d; ++i) od[i] = r1[i] + fd[i];
    }
    compare_dump_tensor_stats("layer_output", layer, output);
    if (backend_context().compare_trace_step < 0) compare_dump_tensor_last_row_stats("prefill_last_layer_output", layer, output);
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
