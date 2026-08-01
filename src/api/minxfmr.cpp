#include <cstring>
#include <cstdio>
#include <random>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include <cassert>
#include "minxfmr.h"
#include "../tokenizer/tokenizer.h"
#include "../cache/kv_cache.h"
#include "../transformer/rmsnorm.h"
#include "../transformer/transformer.h"
#include "../io/gguf_loader.h"
#include "../../third_party/gguf/gguf_reader.h"
#include <sstream>
#include <iomanip>
#include <cctype>
#include "../backend/backend_runtime.h"

// Runtime context owned by minxfmr_open()/minxfmr_close().
// This struct keeps both model weights and per-request reusable buffers.
struct minxfmr_context {
    KVCache* cache;
    Tensor* Wemb;
    Tensor* Wq;
    Tensor* Wk;
    Tensor* Wv;
    Tensor* Wout;
    Tensor* Wnorm;
    std::vector<Tensor*> Wq_layers;
    std::vector<Tensor*> Wk_layers;
    std::vector<Tensor*> Wv_layers;
    std::vector<Tensor*> Bq_layers;
    std::vector<Tensor*> Bk_layers;
    std::vector<Tensor*> Bv_layers;
    std::vector<Tensor*> Wo_layers;
    std::vector<Tensor*> Wattn_norm_layers;
    std::vector<Tensor*> Wffn_norm_layers;
    std::vector<Tensor*> Wffn_gate_layers;
    std::vector<Tensor*> Wffn_up_layers;
    std::vector<Tensor*> Wffn_down_layers;
    size_t model_dim;
    size_t kv_dim;
    size_t n_layer;
    size_t n_head;
    size_t n_head_kv;
    size_t n_intermediate;
    float rope_theta;
    float rmsnorm_epsilon;
    size_t seq_max;
    std::vector<float> scores_workspace;
    Tensor* layer_buf_a;
    Tensor* layer_buf_b;
    int dummy;
    // optional metadata
    std::string chat_template;
    std::vector<std::string> special_tokens;

    // Reusable per-token workspaces (allocated once, reused every generate step)
    Tensor* embed_buf;    // 1 x model_dim（量子化 embedding 用）
    Tensor* hidden_buf;   // 1 x model_dim（forward 出力）
    std::vector<double> logits_buf;
    std::vector<int>    order_buf;
};

static std::string render_token_piece(const std::string& piece) {
    if (piece.empty()) return piece;
    if (piece.size() == 6 && piece.rfind("<0x", 0) == 0 && piece[5] == '>') return std::string();
    return tokenizer_render_piece(piece);
}

static size_t infer_kv_dim_from_weight(const Tensor* w, size_t model_dim) {
    if (!w) return 0;
    if (model_dim > 0) {
        if (w->rows == model_dim) return w->cols;
        if (w->cols == model_dim) return w->rows;
    }
    return std::min(w->rows, w->cols);
}

static Tensor* tensor_clone_f32_local(const Tensor* in) {
    if (!in || in->type != DataType::F32) return nullptr;
    Tensor* out = tensor_create_f32_noinit(in->rows, in->cols);
    if (!out) return nullptr;
    memcpy(out->data, in->data, sizeof(float) * in->rows * in->cols);
    return out;
}

static bool is_supported_quantized_type(DataType t) {
    return t == DataType::Q4_K || t == DataType::Q5_0 || t == DataType::Q8_0;
}

static bool ensure_f32_tensor_shape(Tensor*& t, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return false;
    if (t && t->type == DataType::F32 && t->rows == rows && t->cols == cols) return true;
    if (t) {
        tensor_free(t);
        t = nullptr;
    }
    t = tensor_create_f32_noinit(rows, cols);
    return t != nullptr;
}

// out_owned が使えるときはそこに書き込む。
// F32 row-major のときは zero-copy view を返す（呼び出し側で in != out_owned なら free する）。
static Tensor* token_embedding_row_into(const minxfmr_context* ctx, int token_id, Tensor* out_owned) {
    if (!ctx || token_id < 0) return nullptr;
    const Tensor* emb = ctx->Wemb;

    // If the model did not include an explicit token embedding matrix, try to
    // synthesize a per-token embedding from the output head `Wout` when
    // available (common when embeddings are tied). This allows generation to
    // proceed even when `tok_embeddings` is absent in the GGUF file.
    if (!emb && ctx->Wout) {
        const Tensor* out = ctx->Wout;
        // Case A: Wout is vocab x dim (rows == vocab, cols == model_dim)
        if (out->type == DataType::F32 && (size_t)token_id < out->rows) {
            if (out->cols > 0) {
                float* ptr = (float*)out->data + (size_t)token_id * out->cols;
                return tensor_create_f32_view(1, out->cols, ptr);
            }
        }
        // Case B: Wout is dim x vocab (rows == model_dim, cols == vocab)
        if (out->type == DataType::F32 && (size_t)token_id < out->cols) {
            const size_t dim = out->rows;
            Tensor* row = out_owned;
            if (!row || row->type != DataType::F32 || row->rows != 1 || row->cols != dim) {
                row = tensor_create_f32_noinit(1, dim);
                if (!row) return nullptr;
            }
            float* dst = (float*)row->data;
            float* src = (float*)out->data;
            for (size_t r = 0; r < dim; ++r) dst[r] = src[r * out->cols + (size_t)token_id];
            return row;
        }
        // Case C: quantized Wout — try dequantizing a row if rows==vocab
        if (is_supported_quantized_type(out->type) && (size_t)token_id < out->rows) {
            const size_t cols = out->cols;
            Tensor* row = out_owned;
            if (!row || row->type != DataType::F32 || row->rows != 1 || row->cols != cols) {
                row = tensor_create_f32_noinit(1, cols);
                if (!row) return nullptr;
            }
            if (!tensor_dequant_row(out, (size_t)token_id, (float*)row->data)) {
                if (row != out_owned) tensor_free(row);
                return nullptr;
            }
            return row;
        }
        // If we couldn't synthesize from Wout fall through and produce a safe zero-vector below.
    }

    // ---- F32, row-major: zero-copy view ----
    if (emb && emb->type == DataType::F32 && emb->rows >= emb->cols) {
        if ((size_t)token_id >= emb->rows) return nullptr;
        float* ptr = (float*)emb->data + (size_t)token_id * emb->cols;
        return tensor_create_f32_view(1, emb->cols, ptr);
    }

    // ---- それ以外: out_owned にコピー / dequant ----
    const size_t cols = (emb->rows >= emb->cols) ? emb->cols : emb->rows;

    // token_id 範囲チェック
    if (emb->rows >= emb->cols) {
        if ((size_t)token_id >= emb->rows) return nullptr;
    } else {
        if ((size_t)token_id >= emb->cols) return nullptr;
    }

    Tensor* row = out_owned;
    if (!row || row->type != DataType::F32 || row->rows != 1 || row->cols != cols) {
        row = tensor_create_f32_noinit(1, cols);
        if (!row) return nullptr;
    }

    if (emb->type == DataType::F32) {
        // column-major: 列 token_id を 1xD に集める
        const float* src = (const float*)emb->data;
        float* dst = (float*)row->data;
        for (size_t r = 0; r < emb->rows; ++r) {
            dst[r] = src[r * emb->cols + (size_t)token_id];
        }
        return row;
    }

    // 量子化: 共通 dequant を使用（1で入れた場合）
    if (!tensor_dequant_row(emb, (size_t)token_id, (float*)row->data)) {
        if (row != out_owned) tensor_free(row);
        return nullptr;
    }
    return row;
}

static bool env_enabled(const char* key) {
    const char* v = std::getenv(key);
    if (!v || v[0] == '\0') return false;
    return v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T';
}

static bool chat_debug_enabled() {
    return env_enabled("MINXFMR_CHAT_DEBUG");
}

// Enable verbose per-token generation logs when MINXFMR_VERBOSE_GEN=1.
static bool gen_verbose_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("MINXFMR_VERBOSE_GEN");
        enabled = (v && v[0] == '1') ? 1 : 0;
    }
    return enabled == 1;
}

static bool transpose_square_inplace(Tensor*& t) {
    if (!t || t->type != DataType::F32) return true;
    if (t->rows != t->cols) return true;
    Tensor* tr = tensor_transpose_f32(t);
    if (!tr) return false;
    tensor_free(t);
    t = tr;
    return true;
}

static bool normalize_linear_inplace(Tensor*& t, size_t in_dim, bool transpose_square, bool& transposed) {
    transposed = false;
    if (!t) return false;
    if (in_dim == 0) return false;

    // Fast path: already F32.
    if (t->type == DataType::F32) {
        if (t->rows == in_dim && t->cols == in_dim) {
            if (!transpose_square) return true;
            Tensor* tr = tensor_transpose_f32(t);
            if (!tr) return false;
            tensor_free(t);
            t = tr;
            transposed = true;
            return true;
        }

        if (t->rows == in_dim) return true;
        if (t->cols == in_dim) {
            Tensor* tr = tensor_transpose_f32(t);
            if (!tr) return false;
            tensor_free(t);
            t = tr;
            transposed = true;
            return true;
        }
        return false;
    }

    // Handle quantized tensors by dequantizing -> physical transpose -> keep as F32.
    // This avoids runtime transposition overhead for quantized square matrices
    // at the cost of keeping an F32 copy in memory for the weight.
    if (t->type == DataType::Q4_K || t->type == DataType::Q5_0 || t->type == DataType::Q8_0) {
        // If already in a compatible orientation and no transpose requested, accept as-is.
        if (t->rows == in_dim && t->cols == in_dim && !transpose_square) return true;
        // If rows==in_dim and no transpose needed, OK.
        if (t->rows == in_dim && t->cols != in_dim) return true;

        // If we need to produce rows==in_dim (either because cols==in_dim or square+transpose),
        // prefer doing a packed-format transpose when possible to avoid creating an F32 copy.
        if (t->cols == in_dim || (t->rows == in_dim && t->cols == in_dim && transpose_square)) {
            // Try packed in-place transpose for supported formats (Q5_0/Q8_0).
            if (tensor_transpose_packed_inplace(t)) {
                transposed = true;
                return true;
            }

            // Fallback: dequantize entire tensor into an F32 buffer and transpose.
            const size_t rows = t->rows;
            const size_t cols = t->cols;
            // Allocate temporary F32 buffer (rows x cols).
            Tensor* tmp = tensor_create_f32_noinit(rows, cols);
            if (!tmp) return false;
            float* tmpd = (float*)tmp->data;
            for (size_t r = 0; r < rows; ++r) {
                if (!tensor_dequant_row(t, r, tmpd + r * cols)) {
                    tensor_free(tmp);
                    return false;
                }
            }
            // Transpose the temporary f32 matrix into final tensor.
            Tensor* tr = tensor_transpose_f32(tmp);
            tensor_free(tmp);
            if (!tr) return false;
            tensor_free(t);
            t = tr;
            transposed = true;
            return true;
        }

        // Otherwise the quantized tensor is incompatible with in_dim.
        return (t->rows == in_dim || t->cols == in_dim);
    }

    // Unknown type.
    return false;
}

static bool quantized_square_needs_runtime_transpose(const Tensor* t, size_t in_dim, bool desired_transpose) {
    if (!desired_transpose || !t) return false;
    if (in_dim == 0) return false;
    if (t->type == DataType::F32) return false;
    return (t->rows == in_dim && t->cols == in_dim);
}

static bool arch_uses_square_transpose(const std::string& arch_lc) {
    // Architectures whose exported square attention matrices are commonly laid out
    // in the opposite orientation for this runtime. This list is heuristic-based
    // and mirrors common exporter conventions (inspired by llama.cpp heuristics).

    // Positive matches: families known to export attention/project weights
    // in the alternate orientation (require square transpose).
    if (arch_lc.rfind("qwen", 0) == 0) return true; // qwen, qwen2, qwen2-* etc.
    if (arch_lc.find("llama") != std::string::npos) return true;
    if (arch_lc.find("mistral") != std::string::npos) return true;
    if (arch_lc.find("mixtral") != std::string::npos) return true;
    if (arch_lc.find("gemma") != std::string::npos) return true;
    if (arch_lc.find("xgen") != std::string::npos) return true;
    if (arch_lc.find("orca") != std::string::npos) return true;

    // Negative matches: families typically using the "standard" layout.
    if (arch_lc.find("gptneox") != std::string::npos) return false;
    if (arch_lc.find("gpt2") != std::string::npos) return false;
    if (arch_lc.find("falcon") != std::string::npos) return false;
    if (arch_lc.find("bloom") != std::string::npos) return false;
    if (arch_lc.find("mpt") != std::string::npos) return false;
    if (arch_lc.find("phi") != std::string::npos) return false;

    // Default conservative behavior: assume no square transpose required.
    return false;
}

static bool arch_prefers_cuda_quant_parity(const std::string& arch_lc) {
    // Some architectures are known to produce quantized blocks that are
    // numerically more stable when host-dequant parity mode is used.
    // Enable parity (host dequant -> device F32) for those families.
    if (arch_lc.empty()) return false;
    if (arch_lc.rfind("qwen", 0) == 0) return true;
    if (arch_lc.find("gemma") != std::string::npos) return true;
    // Keep parity off by default for others; users may force via env var.
    return false;
}

static bool apply_norm_scale_local(Tensor* x, const Tensor* w) {
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

static bool apply_final_norm_inplace(Tensor* x, const Tensor* wnorm, float rmsnorm_epsilon) {
    if (!x) return false;
    Tensor* tmp = tensor_create_f32(x->rows, x->cols);
    if (!tmp) return false;
    bool ok = rmsnorm_forward(x, tmp, rmsnorm_epsilon);
    if (ok && wnorm) ok = apply_norm_scale_local(tmp, wnorm);
    if (ok) memcpy(x->data, tmp->data, sizeof(float) * x->rows * x->cols);
    tensor_free(tmp);
    return ok;
}
static void log_vocab_specials(const std::vector<std::string>& vocab) {
    if (!chat_debug_enabled()) return;
    size_t shown = 0;
    for (size_t i = 0; i < vocab.size() && shown < 40; ++i) {
        const std::string& tok = vocab[i];
        if (tok.find('<') != std::string::npos || tok.find('[') != std::string::npos || tok.find(']') != std::string::npos) {
            fprintf(stderr, "[minxfmr] vocab[%zu]=%s\n", i, tok.c_str());
            ++shown;
        }
    }
}

static bool run_stack_forward(minxfmr_context* ctx, const Tensor* input, Tensor* output) {
    if (!ctx || !input || !output) return false;
    if (input->type != DataType::F32 || output->type != DataType::F32) return false;
    // Transformer hidden size must match the model config when available.
    if (ctx->model_dim > 0) assert(input->cols == ctx->model_dim);

    if (!ensure_f32_tensor_shape(ctx->layer_buf_a, input->rows, input->cols)) return false;
    if (!ensure_f32_tensor_shape(ctx->layer_buf_b, input->rows, input->cols)) return false;
    memcpy(ctx->layer_buf_a->data, input->data, sizeof(float) * input->rows * input->cols);

    Tensor* cur = ctx->layer_buf_a;
    Tensor* nxt = ctx->layer_buf_b;

    size_t layers_to_run = 1;
    if (!ctx->Wq_layers.empty() && ctx->Wq_layers.size() == ctx->Wk_layers.size() && ctx->Wq_layers.size() == ctx->Wv_layers.size()) {
        layers_to_run = ctx->Wq_layers.size();
    }
    if (ctx->cache && layers_to_run > ctx->cache->layers) layers_to_run = ctx->cache->layers;

    // Run decoder blocks sequentially and ping-pong between two reusable buffers.
    for (size_t l = 0; l < layers_to_run; ++l) {
        const Tensor* Wq = ctx->Wq;
        const Tensor* Wk = ctx->Wk;
        const Tensor* Wv = ctx->Wv;
        const Tensor* Bq = nullptr;
        const Tensor* Bk = nullptr;
        const Tensor* Bv = nullptr;
        const Tensor* Wo = nullptr;
        const Tensor* WattnNorm = nullptr;
        const Tensor* WffnNorm = nullptr;
        if (!ctx->Wq_layers.empty()) {
            if (ctx->Wq_layers[l]) Wq = ctx->Wq_layers[l];
            if (ctx->Wk_layers[l]) Wk = ctx->Wk_layers[l];
            if (ctx->Wv_layers[l]) Wv = ctx->Wv_layers[l];
            if (l < ctx->Bq_layers.size() && ctx->Bq_layers[l]) Bq = ctx->Bq_layers[l];
            if (l < ctx->Bk_layers.size() && ctx->Bk_layers[l]) Bk = ctx->Bk_layers[l];
            if (l < ctx->Bv_layers.size() && ctx->Bv_layers[l]) Bv = ctx->Bv_layers[l];
            if (!ctx->Wo_layers.empty()) Wo = ctx->Wo_layers[l];
            if (!ctx->Wattn_norm_layers.empty()) WattnNorm = ctx->Wattn_norm_layers[l];
            if (!ctx->Wffn_norm_layers.empty()) WffnNorm = ctx->Wffn_norm_layers[l];
        }

        const Tensor* Wfg = nullptr;
        const Tensor* Wfu = nullptr;
        const Tensor* Wfd = nullptr;
        if (!ctx->Wffn_gate_layers.empty()) {
            Wfg = ctx->Wffn_gate_layers[l];
            Wfu = ctx->Wffn_up_layers[l];
            Wfd = ctx->Wffn_down_layers[l];
        }

        float* scores_buf = ctx->scores_workspace.data();
        size_t scores_len = ctx->scores_workspace.size();

        bool ok = transformer_forward_single_layer(cur, nxt, ctx->cache, l, ctx->n_head, ctx->n_head_kv, Wq, Wk, Wv, Bq, Bk, Bv, Wo, WattnNorm, WffnNorm, Wfg, Wfu, Wfd, scores_buf, scores_len, ctx->rope_theta, ctx->rmsnorm_epsilon);
        if (!ok) {
            return false;
        }
        Tensor* tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    if (output->rows != cur->rows || output->cols != cur->cols) {
        return false;
    }
    memcpy(output->data, cur->data, sizeof(float) * cur->rows * cur->cols);
    return true;
}

static void free_layer_weights(std::vector<Tensor*>& vec) {
    for (Tensor* t : vec) if (t) tensor_free(t);
    vec.clear();
}

static void preload_context_weights_to_backend(const minxfmr_context* ctx) {
    if (!ctx || !backend_using_cuda()) return;

    auto preload = [](const Tensor* t) {
        if (t) (void)backend_preload_tensor(t);
    };

    preload(ctx->Wemb);
    preload(ctx->Wq);
    preload(ctx->Wk);
    preload(ctx->Wv);
    preload(ctx->Wout);
    preload(ctx->Wnorm);

    for (const Tensor* t : ctx->Wq_layers) preload(t);
    for (const Tensor* t : ctx->Wk_layers) preload(t);
    for (const Tensor* t : ctx->Wv_layers) preload(t);
    for (const Tensor* t : ctx->Wo_layers) preload(t);
    for (const Tensor* t : ctx->Wattn_norm_layers) preload(t);
    for (const Tensor* t : ctx->Wffn_norm_layers) preload(t);
    for (const Tensor* t : ctx->Wffn_gate_layers) preload(t);
    for (const Tensor* t : ctx->Wffn_up_layers) preload(t);
    for (const Tensor* t : ctx->Wffn_down_layers) preload(t);
    for (const Tensor* t : ctx->Bq_layers) preload(t);
    for (const Tensor* t : ctx->Bk_layers) preload(t);
    for (const Tensor* t : ctx->Bv_layers) preload(t);
}

minxfmr_context* minxfmr_open(const char* model_path) {
    return minxfmr_open_with_layer(model_path, 0);
}

minxfmr_context* minxfmr_open_with_layer(const char* model_path, int projection_layer) {
    if (!model_path) return nullptr;

    backend_initialize_from_env();
    fprintf(stderr, "[minxfmr] backend=%s\n", backend_get_name());

    minxfmr_context* ctx = new (std::nothrow) minxfmr_context();
    if (!ctx) return nullptr;

    ctx->dummy = 0;
    ctx->cache = nullptr;
    ctx->Wemb = nullptr;
    ctx->Wq = nullptr;
    ctx->Wk = nullptr;
    ctx->Wv = nullptr;
    ctx->Wout = nullptr;
    ctx->Wnorm = nullptr;
    ctx->Wq_layers.clear();
    ctx->Wk_layers.clear();
    ctx->Wv_layers.clear();
    ctx->Wo_layers.clear();
    ctx->Wattn_norm_layers.clear();
    ctx->Wffn_norm_layers.clear();
    ctx->Wffn_gate_layers.clear();
    ctx->Wffn_up_layers.clear();
    ctx->Wffn_down_layers.clear();
    ctx->model_dim = 0;
    ctx->kv_dim = 0;
    ctx->n_layer = 1;
    ctx->n_head = 0;
    ctx->n_head_kv = 0;
    ctx->n_intermediate = 0;
    ctx->rope_theta = 10000.0f;
    ctx->rmsnorm_epsilon = 1e-6f;
    ctx->seq_max = 128;
    ctx->layer_buf_a = nullptr;
    ctx->layer_buf_b = nullptr;
    ctx->embed_buf = nullptr;
    ctx->hidden_buf = nullptr;
    ctx->logits_buf.clear();
    ctx->order_buf.clear();

    const char* ext = strrchr(model_path, '.');
    const bool looks_gguf = (ext != nullptr) && (_stricmp(ext, ".gguf") == 0);

    bool desired_transpose_wq = false;
    bool desired_transpose_wk = false;
    bool desired_transpose_wv = false;
    bool desired_transpose_wo = false;
    bool desired_transpose_ffn_square = false;

    // Non-GGUF debug path: load simple text weights used by early bring-up.
    if (!looks_gguf) {
        FILE* f = fopen(model_path, "r");
        if (f) {
            int d = 0;
            if (fscanf(f, "%d", &d) == 1 && d > 0) {
                size_t need = (size_t)3 * (size_t)d * (size_t)d;
                std::vector<float> buf;
                buf.reserve(need);
                for (size_t i = 0; i < need; ++i) {
                    float v;
                    if (fscanf(f, "%f", &v) == 1) buf.push_back(v);
                    else break;
                }
                if (buf.size() == need) {
                    Tensor* TWq = tensor_create_f32((size_t)d, (size_t)d);
                    Tensor* TWk = tensor_create_f32((size_t)d, (size_t)d);
                    Tensor* TWv = tensor_create_f32((size_t)d, (size_t)d);
                    if (TWq && TWk && TWv) {
                        size_t off = 0;
                        for (int i = 0; i < d * d; ++i) tensor_set_f32(TWq, (size_t)(i / d), (size_t)(i % d), buf[off++]);
                        for (int i = 0; i < d * d; ++i) tensor_set_f32(TWk, (size_t)(i / d), (size_t)(i % d), buf[off++]);
                        for (int i = 0; i < d * d; ++i) tensor_set_f32(TWv, (size_t)(i / d), (size_t)(i % d), buf[off++]);
                        ctx->Wq = TWq;
                        ctx->Wk = TWk;
                        ctx->Wv = TWv;
                        ctx->model_dim = (size_t)d;
                        ctx->kv_dim = (size_t)d;
                        fprintf(stderr, "[minxfmr] loaded weights from %s dim=%d\n", model_path, d);
                    } else {
                        tensor_free(TWq);
                        tensor_free(TWk);
                        tensor_free(TWv);
                    }
                } else {
                    fprintf(stderr, "[minxfmr] weight file %s malformed: expected %zu floats, got %zu\n", model_path, need, buf.size());
                }
            } else {
                fprintf(stderr, "[minxfmr] could not read dim from %s\n", model_path);
            }
            fclose(f);
        } else {
            fprintf(stderr, "[minxfmr] weight file %s not found, proceeding without projections\n", model_path);
        }
    }

    // Main path: load GGUF metadata + per-layer tensors.
    if (looks_gguf) {
        GGUFLoaderModelConfig cfg{0, 0, 0, 0, 0, 0, 0.0f, 1e-6f};
        if (gguf_try_read_model_config(model_path, cfg)) {
            if (cfg.n_layer > 0) ctx->n_layer = (size_t)cfg.n_layer;
            if (cfg.n_ctx > 0) ctx->seq_max = (size_t)cfg.n_ctx;
            if (cfg.n_embd > 0) ctx->model_dim = (size_t)cfg.n_embd;
            if (cfg.n_head > 0) ctx->n_head = (size_t)cfg.n_head;
            if (cfg.n_head_kv > 0) ctx->n_head_kv = (size_t)cfg.n_head_kv;
            if (cfg.n_intermediate > 0) ctx->n_intermediate = (size_t)cfg.n_intermediate;
            if (cfg.rope_freq_base > 0.0f) ctx->rope_theta = cfg.rope_freq_base;
            if (cfg.rmsnorm_epsilon > 0.0f) ctx->rmsnorm_epsilon = cfg.rmsnorm_epsilon;
            fprintf(stderr, "[minxfmr] gguf meta layers=%zu ctx=%zu embd=%zu head=%zu head_kv=%zu\n",
                ctx->n_layer, ctx->seq_max, ctx->model_dim, (size_t)cfg.n_head, (size_t)cfg.n_head_kv);
            fprintf(stderr, "[minxfmr] ffn intermediate=%zu\n", ctx->n_intermediate);
            fprintf(stderr, "[minxfmr] rope theta=%g\n", (double)ctx->rope_theta);
            fprintf(stderr, "[minxfmr] rmsnorm epsilon=%g\n", (double)ctx->rmsnorm_epsilon);
            if (cfg.n_head == 0 || cfg.n_head_kv == 0) {
                fprintf(stderr,
                    "[minxfmr] warning: gguf head metadata missing (llama.attention.head_count / llama.attention.head_count_kv). "
                    "defaults may be incorrect for non-LLaMA architectures.\n");
            }
        }

        // llama.cpp-like behavior: default layout policy from model architecture metadata.
        // Explicit CLI transpose flags still win (MINXFMR_TRANSPOSE_USER_OVERRIDE).
        {
            const bool user_override = env_enabled("MINXFMR_TRANSPOSE_USER_OVERRIDE");
            if (user_override) {
                desired_transpose_wq = env_enabled("MINXFMR_TRANSPOSE_WQ");
                desired_transpose_wk = env_enabled("MINXFMR_TRANSPOSE_WK");
                desired_transpose_wv = env_enabled("MINXFMR_TRANSPOSE_WV");
                desired_transpose_wo = env_enabled("MINXFMR_TRANSPOSE_WO");
                desired_transpose_ffn_square = desired_transpose_wq || desired_transpose_wk || desired_transpose_wv || desired_transpose_wo;
                fprintf(stderr,
                    "[minxfmr] manual orientation: wq=%s wk=%s wv=%s wo=%s\n",
                    desired_transpose_wq ? "on" : "off",
                    desired_transpose_wk ? "on" : "off",
                    desired_transpose_wv ? "on" : "off",
                    desired_transpose_wo ? "on" : "off");
            } else {
                std::string arch;
                if (gguf_try_read_architecture(model_path, arch)) {
                    std::string arch_lc = arch;
                    for (char& ch : arch_lc) ch = (char)std::tolower((unsigned char)ch);

                    const bool needs_square_transpose = arch_uses_square_transpose(arch_lc);
                    desired_transpose_wq = needs_square_transpose;
                    desired_transpose_wk = needs_square_transpose;
                    desired_transpose_wv = needs_square_transpose;
                    desired_transpose_wo = needs_square_transpose;
                    desired_transpose_ffn_square = needs_square_transpose;
                    fprintf(stderr,
                        "[minxfmr] auto orientation from gguf architecture='%s': square_transpose=%s\n",
                        arch.c_str(),
                        needs_square_transpose ? "on" : "off");

                    if (backend_using_cuda() && !std::getenv("MINXFMR_CUDA_QUANT_PARITY")) {
                        const bool prefer_parity = arch_prefers_cuda_quant_parity(arch_lc);
                        backend_set_cuda_quant_parity_mode(prefer_parity ? 1 : 0);
                        fprintf(stderr,
                            "[minxfmr] auto cuda quant policy from architecture='%s': %s\n",
                            arch.c_str(),
                            prefer_parity ? "parity(dequant_f32)" : "quant-kernel");
                    }
                } else {
                    fprintf(stderr, "[minxfmr] auto orientation: architecture metadata missing, default square_transpose=off\n");
                    if (backend_using_cuda() && !std::getenv("MINXFMR_CUDA_QUANT_PARITY")) {
                        backend_set_cuda_quant_parity_mode(0);
                    }
                }
            }
        }

        Tensor* wemb = nullptr;
        if (gguf_try_load_token_embedding(model_path, wemb)) {
            ctx->Wemb = wemb;
            fprintf(stderr, "[minxfmr] loaded token embedding rows=%zu cols=%zu\n", wemb->rows, wemb->cols);
        }

        ctx->Wq_layers.resize(ctx->n_layer, nullptr);
        ctx->Wk_layers.resize(ctx->n_layer, nullptr);
        ctx->Wv_layers.resize(ctx->n_layer, nullptr);
        ctx->Bq_layers.resize(ctx->n_layer, nullptr);
        ctx->Bk_layers.resize(ctx->n_layer, nullptr);
        ctx->Bv_layers.resize(ctx->n_layer, nullptr);
        ctx->Wo_layers.resize(ctx->n_layer, nullptr);
        ctx->Wattn_norm_layers.resize(ctx->n_layer, nullptr);
        ctx->Wffn_norm_layers.resize(ctx->n_layer, nullptr);
        ctx->Wffn_gate_layers.resize(ctx->n_layer, nullptr);
        ctx->Wffn_up_layers.resize(ctx->n_layer, nullptr);
        ctx->Wffn_down_layers.resize(ctx->n_layer, nullptr);

        size_t loaded_attn_layers = 0;
        size_t loaded_attn_bias_layers = 0;
        size_t loaded_wo_layers = 0;
        size_t loaded_norm_layers = 0;
        size_t loaded_ffn_layers = 0;
        int last_progress_bucket = -1;
        for (size_t l = 0; l < ctx->n_layer; ++l) {
            int percent = (int)(((double)(l) / (double)ctx->n_layer) * 100.0);
            int bucket = percent / 10;
            if (bucket != last_progress_bucket || l == 0 || l + 1 == ctx->n_layer) {
                fprintf(stderr, "[minxfmr] loading layers %zu/%zu (%d%%)\n", l, ctx->n_layer, percent);
                last_progress_bucket = bucket;
            }
            Tensor* lq = nullptr;
            Tensor* lk = nullptr;
            Tensor* lv = nullptr;
            if (gguf_try_load_projections_for_layer(model_path, (int)l, lq, lk, lv)) {
                ctx->Wq_layers[l] = lq;
                ctx->Wk_layers[l] = lk;
                ctx->Wv_layers[l] = lv;
                loaded_attn_layers++;
            }

            Tensor* bq = nullptr;
            Tensor* bk = nullptr;
            Tensor* bv = nullptr;
            if (gguf_try_load_projection_biases_for_layer(model_path, (int)l, bq, bk, bv)) {
                ctx->Bq_layers[l] = bq;
                ctx->Bk_layers[l] = bk;
                ctx->Bv_layers[l] = bv;
                loaded_attn_bias_layers++;
            }

            Tensor* wo = nullptr;
            if (gguf_try_load_attn_out_for_layer(model_path, (int)l, wo)) {
                ctx->Wo_layers[l] = wo;
                loaded_wo_layers++;
            }

            Tensor* an = nullptr;
            Tensor* fn = nullptr;
            if (gguf_try_load_norms_for_layer(model_path, (int)l, an, fn)) {
                ctx->Wattn_norm_layers[l] = an;
                ctx->Wffn_norm_layers[l] = fn;
                loaded_norm_layers++;
            }

            Tensor* fg = nullptr;
            Tensor* fu = nullptr;
            Tensor* fd = nullptr;
            if (gguf_try_load_ffn_for_layer(model_path, (int)l, fg, fu, fd)) {
                ctx->Wffn_gate_layers[l] = fg;
                ctx->Wffn_up_layers[l] = fu;
                ctx->Wffn_down_layers[l] = fd;
                loaded_ffn_layers++;
            }
        }
        fprintf(stderr, "[minxfmr] model load complete (100%%)\n");
        if (loaded_attn_layers > 0) {
            fprintf(stderr, "[minxfmr] loaded per-layer projections: %zu/%zu layers\n", loaded_attn_layers, ctx->n_layer);
            for (size_t i = 0; i < ctx->Wq_layers.size(); ++i) {
                if (ctx->Wq_layers[i] && ctx->Wk_layers[i] && ctx->Wv_layers[i]) {
                    ctx->Wq = tensor_clone_f32_local(ctx->Wq_layers[i]);
                    ctx->Wk = tensor_clone_f32_local(ctx->Wk_layers[i]);
                    ctx->Wv = tensor_clone_f32_local(ctx->Wv_layers[i]);
                    break;
                }
            }
        }
        fprintf(stderr, "[minxfmr] projection load summary: attn=%zu bias=%zu wo=%zu norm=%zu ffn=%zu of %zu layers\n",
            loaded_attn_layers, loaded_attn_bias_layers, loaded_wo_layers, loaded_norm_layers, loaded_ffn_layers, ctx->n_layer);
        if (loaded_attn_bias_layers > 0) {
            fprintf(stderr, "[minxfmr] loaded per-layer projection biases: %zu/%zu layers\n", loaded_attn_bias_layers, ctx->n_layer);
        }
        if (loaded_ffn_layers > 0) {
            fprintf(stderr, "[minxfmr] loaded per-layer ffn weights: %zu/%zu layers\n", loaded_ffn_layers, ctx->n_layer);
        }
        if (loaded_wo_layers > 0) {
            fprintf(stderr, "[minxfmr] loaded per-layer Wo weights: %zu/%zu layers\n", loaded_wo_layers, ctx->n_layer);
        }
        if (loaded_norm_layers > 0) {
            fprintf(stderr, "[minxfmr] loaded per-layer norm weights: %zu/%zu layers\n", loaded_norm_layers, ctx->n_layer);
        }

        if (!ctx->Wq) {
            Tensor* gWq = nullptr;
            Tensor* gWk = nullptr;
            Tensor* gWv = nullptr;
            if (gguf_try_load_projections_for_layer(model_path, projection_layer, gWq, gWk, gWv)) {
                ctx->Wq = gWq;
                ctx->Wk = gWk;
                ctx->Wv = gWv;
                fprintf(stderr, "[minxfmr] loaded projections from gguf %s layer=%d dim=%zux%zu\n", model_path, projection_layer, ctx->Wq->rows, ctx->Wq->cols);
            }
        }

        std::vector<std::string> vocab;
            // Prefer loading tokenizer vocabulary + metadata directly from GGUF when available.
            {
                GGUF_File gf;
                if (gguf_open(model_path, gf)) {
                    if (!gf.vocab_tokens.empty()) {
                        if (tokenizer_load_from_gguf(gf)) {
                            fprintf(stderr, "[minxfmr] loaded tokenizer vocab+meta from gguf size=%zu\n", gf.vocab_tokens.size());
                            log_vocab_specials(gf.vocab_tokens);
                        } else {
                            tokenizer_load_from_list(gf.vocab_tokens);
                            fprintf(stderr, "[minxfmr] loaded tokenizer vocab from gguf size=%zu\n", gf.vocab_tokens.size());
                            log_vocab_specials(gf.vocab_tokens);
                        }
                    }
                    gguf_close(gf);
                }
            }

        Tensor* wout = nullptr;
        if (gguf_try_load_lm_head(model_path, wout)) {
            ctx->Wout = wout;
            fprintf(stderr, "[minxfmr] loaded output head rows=%zu cols=%zu\n", wout->rows, wout->cols);
        }

        Tensor* wnorm = nullptr;
        if (gguf_try_load_final_norm(model_path, wnorm)) {
            ctx->Wnorm = wnorm;
            fprintf(stderr, "[minxfmr] loaded final norm rows=%zu cols=%zu\n", wnorm->rows, wnorm->cols);
        }
    }

    auto first_nonnull = [](const std::vector<Tensor*>& v) -> const Tensor* {
        for (const Tensor* t : v) if (t) return t;
        return nullptr;
    };

    if (looks_gguf) {
        size_t attn_ready = 0;
        size_t wo_ready = 0;
        size_t ffn_ready = 0;
        for (size_t l = 0; l < ctx->n_layer; ++l) {
            if (l < ctx->Wq_layers.size() && l < ctx->Wk_layers.size() && l < ctx->Wv_layers.size() &&
                ctx->Wq_layers[l] && ctx->Wk_layers[l] && ctx->Wv_layers[l]) {
                ++attn_ready;
            }
            if (l < ctx->Wo_layers.size() && ctx->Wo_layers[l]) ++wo_ready;
            if (l < ctx->Wffn_gate_layers.size() && l < ctx->Wffn_up_layers.size() && l < ctx->Wffn_down_layers.size() &&
                ctx->Wffn_gate_layers[l] && ctx->Wffn_up_layers[l] && ctx->Wffn_down_layers[l]) {
                ++ffn_ready;
            }
        }

        // Ensure that essential decoder tensors exist either as per-layer
        // weights or as global projection tensors (ctx->Wq).
        bool attn_present = (attn_ready > 0) || (ctx->Wq != nullptr);
        bool wo_present = (wo_ready > 0);
        bool ffn_present = (ffn_ready > 0);

        if (!attn_present || !wo_present || !ffn_present) {
            fprintf(stderr,
                "[minxfmr] fatal: decoder tensors are missing (attn_present=%d wo_present=%d ffn_present=%d layers=%zu).\n",
                (int)attn_present,
                (int)wo_present,
                (int)ffn_present,
                ctx->n_layer);
            fprintf(stderr,
                "[minxfmr] fatal: GGUF tensor resolution failed (naming mismatch, unsupported layout, or incomplete/truncated file); refusing to run to avoid gibberish output.\n");
            minxfmr_close(ctx);
            return nullptr;
        }
    }

    // Derive missing dimensions from loaded tensors so later code can validate shapes.
    const Tensor* wq_ref = ctx->Wq ? ctx->Wq : first_nonnull(ctx->Wq_layers);
    const Tensor* wk_ref = ctx->Wk ? ctx->Wk : first_nonnull(ctx->Wk_layers);
    const Tensor* wv_ref = ctx->Wv ? ctx->Wv : first_nonnull(ctx->Wv_layers);

    if (ctx->n_head > 0 && ctx->n_head_kv > 0 && ctx->model_dim > 0 && (ctx->model_dim % ctx->n_head) == 0) {
        const size_t head_dim = ctx->model_dim / ctx->n_head;
        const size_t meta_kv_dim = head_dim * ctx->n_head_kv;
        if (meta_kv_dim > 0) {
            ctx->kv_dim = meta_kv_dim;
        }
    }

    if (wq_ref) {
        if (ctx->model_dim == 0) {
            ctx->model_dim = (wq_ref->rows == wq_ref->cols) ? wq_ref->rows : std::max(wq_ref->rows, wq_ref->cols);
        }
        size_t kv = infer_kv_dim_from_weight(wk_ref, ctx->model_dim);
        size_t vv = infer_kv_dim_from_weight(wv_ref, ctx->model_dim);
        if (ctx->kv_dim == 0) {
            if (kv == 0) kv = ctx->model_dim;
            if (vv > 0) kv = std::min(kv, vv);
            ctx->kv_dim = kv;
        }
    }
    if (ctx->Wemb && ctx->model_dim == 0) {
        ctx->model_dim = (ctx->Wemb->rows >= ctx->Wemb->cols) ? ctx->Wemb->cols : ctx->Wemb->rows;
    }

    if (ctx->n_layer == 0) ctx->n_layer = 1;
    if (ctx->seq_max < 16) ctx->seq_max = 16;
    if (ctx->seq_max > 8192) ctx->seq_max = 8192;
    if (ctx->model_dim == 0) ctx->model_dim = 4;
    if (ctx->kv_dim == 0) ctx->kv_dim = ctx->model_dim;
    if (ctx->n_head == 0) ctx->n_head = 1;
    if (ctx->n_head_kv == 0) ctx->n_head_kv = 1;

    // Physical normalization: apply square-matrix transposes once at load time.
    // After this, runtime projection path can use non-transpose behavior.
    {
        size_t n_wq = 0, n_wk = 0, n_wv = 0, n_wo = 0;
        size_t n_ffn_gate = 0, n_ffn_up = 0, n_ffn_down = 0;
        size_t bad_attn = 0, bad_ffn = 0;

        for (size_t l = 0; l < ctx->n_layer; ++l) {
            bool tr = false;
            Tensor*& wq = ctx->Wq_layers[l];
            if (wq) {
                if (normalize_linear_inplace(wq, ctx->model_dim, desired_transpose_wq, tr)) {
                    if (tr) ++n_wq;
                } else {
                    ++bad_attn;
                }
            }

            Tensor*& wk = ctx->Wk_layers[l];
            if (wk) {
                if (normalize_linear_inplace(wk, ctx->model_dim, desired_transpose_wk, tr)) {
                    if (tr) ++n_wk;
                } else {
                    ++bad_attn;
                }
            }

            Tensor*& wv = ctx->Wv_layers[l];
            if (wv) {
                if (normalize_linear_inplace(wv, ctx->model_dim, desired_transpose_wv, tr)) {
                    if (tr) ++n_wv;
                } else {
                    ++bad_attn;
                }
            }

            Tensor*& wo = ctx->Wo_layers[l];
            if (wo) {
                if (normalize_linear_inplace(wo, ctx->model_dim, desired_transpose_wo, tr)) {
                    if (tr) ++n_wo;
                } else {
                    ++bad_attn;
                }
            }

            Tensor*& wfg = ctx->Wffn_gate_layers[l];
            Tensor*& wfu = ctx->Wffn_up_layers[l];
            Tensor*& wfd = ctx->Wffn_down_layers[l];

            size_t ffn_hidden = ctx->n_intermediate;
            if (wfg) {
                if (normalize_linear_inplace(wfg, ctx->model_dim, desired_transpose_ffn_square, tr)) {
                    if (tr) ++n_ffn_gate;
                    if (wfg->rows == ctx->model_dim) ffn_hidden = wfg->cols;
                    else if (wfg->cols == ctx->model_dim) ffn_hidden = wfg->rows;
                } else {
                    ++bad_ffn;
                }
            }

            if (wfu) {
                if (normalize_linear_inplace(wfu, ctx->model_dim, desired_transpose_ffn_square, tr)) {
                    if (tr) ++n_ffn_up;
                    if (ffn_hidden == 0 && wfu->rows == ctx->model_dim) ffn_hidden = wfu->cols;
                    else if (ffn_hidden == 0 && wfu->cols == ctx->model_dim) ffn_hidden = wfu->rows;
                } else {
                    ++bad_ffn;
                }
            }

            if (wfg && ffn_hidden > 0) {
                const bool gate_ok =
                    (wfg->rows == ctx->model_dim && wfg->cols == ffn_hidden) ||
                    (wfg->rows == ffn_hidden && wfg->cols == ctx->model_dim);
                if (!gate_ok) ++bad_ffn;
            }
            if (wfu && ffn_hidden > 0) {
                const bool up_ok =
                    (wfu->rows == ctx->model_dim && wfu->cols == ffn_hidden) ||
                    (wfu->rows == ffn_hidden && wfu->cols == ctx->model_dim);
                if (!up_ok) ++bad_ffn;
            }

            if (wfd) {
                if (ffn_hidden > 0) {
                    if (normalize_linear_inplace(wfd, ffn_hidden, desired_transpose_ffn_square, tr)) {
                        if (tr) ++n_ffn_down;
                        const bool down_ok =
                            (wfd->rows == ffn_hidden && wfd->cols == ctx->model_dim) ||
                            (wfd->rows == ctx->model_dim && wfd->cols == ffn_hidden);
                        if (!down_ok) ++bad_ffn;
                    } else {
                        ++bad_ffn;
                    }
                } else {
                    ++bad_ffn;
                }
            }
        }

        bool tr = false;
        if (ctx->Wq && normalize_linear_inplace(ctx->Wq, ctx->model_dim, desired_transpose_wq, tr) && tr) ++n_wq;
        if (ctx->Wk && normalize_linear_inplace(ctx->Wk, ctx->model_dim, desired_transpose_wk, tr) && tr) ++n_wk;
        if (ctx->Wv && normalize_linear_inplace(ctx->Wv, ctx->model_dim, desired_transpose_wv, tr) && tr) ++n_wv;

        fprintf(stderr,
            "[minxfmr] normalized attention weights at load: Wq=%zu Wk=%zu Wv=%zu Wo=%zu bad=%zu\n",
            n_wq, n_wk, n_wv, n_wo, bad_attn);
        fprintf(stderr,
            "[minxfmr] normalized ffn weights at load: gate=%zu up=%zu down=%zu bad=%zu\n",
            n_ffn_gate, n_ffn_up, n_ffn_down, bad_ffn);

        // Keep runtime square-transpose enabled only for quantized square matrices
        // that could not be physically transposed at load-time.
        bool rt_wq = false;
        bool rt_wk = false;
        bool rt_wv = false;
        bool rt_wo = false;
        for (size_t l = 0; l < ctx->n_layer; ++l) {
            if (quantized_square_needs_runtime_transpose(ctx->Wq_layers[l], ctx->model_dim, desired_transpose_wq)) rt_wq = true;
            if (quantized_square_needs_runtime_transpose(ctx->Wk_layers[l], ctx->model_dim, desired_transpose_wk)) rt_wk = true;
            if (quantized_square_needs_runtime_transpose(ctx->Wv_layers[l], ctx->model_dim, desired_transpose_wv)) rt_wv = true;
            if (quantized_square_needs_runtime_transpose(ctx->Wo_layers[l], ctx->model_dim, desired_transpose_wo)) rt_wo = true;
        }
        if (quantized_square_needs_runtime_transpose(ctx->Wq, ctx->model_dim, desired_transpose_wq)) rt_wq = true;
        if (quantized_square_needs_runtime_transpose(ctx->Wk, ctx->model_dim, desired_transpose_wk)) rt_wk = true;
        if (quantized_square_needs_runtime_transpose(ctx->Wv, ctx->model_dim, desired_transpose_wv)) rt_wv = true;

        transformer_set_transpose_square_weights_for_all(rt_wq, rt_wk, rt_wv, rt_wo);
        fprintf(stderr,
            "[minxfmr] runtime square transpose after normalization: wq=%s wk=%s wv=%s wo=%s\n",
            rt_wq ? "on" : "off",
            rt_wk ? "on" : "off",
            rt_wv ? "on" : "off",
            rt_wo ? "on" : "off");
    }

    // optional chat metadata
    ctx->chat_template = std::string();
    ctx->special_tokens.clear();

    // try to read optional metadata from gguf
    std::string tmp_template;
    if (gguf_try_read_chat_template(model_path, tmp_template)) {
        ctx->chat_template = tmp_template;
        fprintf(stderr, "[minxfmr] loaded chat template from gguf length=%zu\n", ctx->chat_template.size());
    }
    std::vector<std::string> tmp_specials;
    if (gguf_try_read_special_tokens(model_path, tmp_specials)) {
        ctx->special_tokens = tmp_specials;
        fprintf(stderr, "[minxfmr] loaded %zu special tokens from gguf\n", ctx->special_tokens.size());
    }

    ctx->cache = kvcache_create(ctx->n_layer, ctx->seq_max, ctx->kv_dim);
    if (!ctx->cache) {
        fprintf(stderr, "[minxfmr] failed to create kvcache with layers=%zu seq=%zu dim=%zu\n", ctx->n_layer, ctx->seq_max, ctx->kv_dim);
    }

    fprintf(stderr, "[minxfmr] runtime config model_dim=%zu kv_dim=%zu layers=%zu seq_max=%zu\n",
        ctx->model_dim, ctx->kv_dim, ctx->n_layer, ctx->seq_max);

    // Preallocate scores workspace for attention.
    // Length J = cached_rows + seq. Max value is approx seq_max.
    ctx->scores_workspace.assign(ctx->seq_max + 128, 0.0f);
    ctx->layer_buf_a = nullptr;
    ctx->layer_buf_b = nullptr;
    ctx->embed_buf = nullptr;
    ctx->hidden_buf = nullptr;
    ctx->logits_buf.clear();
    ctx->order_buf.clear();

    preload_context_weights_to_backend(ctx);

    fprintf(stderr, "minxfmr: opened model %s\n", model_path);
    return ctx;
}

int minxfmr_generate(minxfmr_context* ctx, const char* prompt, void (*callback)(const char* token), double temperature, int top_k) {
    if (!ctx || !prompt) return -1;

    // Reset per-call backend workspace allocations for this generation call.
    backend_workspace_reset();

    // 1) Tokenize prompt and prefill cache by running prompt tokens.
    std::vector<int> ids = tokenizer_encode(prompt);
    // debug: log prompt token ids and decoded prompt
    if (!ids.empty() && chat_debug_enabled()) {
        fprintf(stderr, "[minxfmr] prompt token count=%zu\n", ids.size());
        fprintf(stderr, "[minxfmr] prompt ids:");
        for (size_t i=0;i<ids.size();++i) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "\n");
        std::string dbg = tokenizer_decode(ids);
        fprintf(stderr, "[minxfmr] prompt decoded: %s\n", dbg.c_str());
    }
    
    size_t vocab_size_base = tokenizer_vocab_size();
    if (vocab_size_base == 0) vocab_size_base = 16;
   

    KVCache* cache = ctx->cache;
    if (!cache) return -2;

    // Each generation call rebuilds the full prompt, so the cache must start clean.
    // Otherwise chat turns duplicate prior context and quickly degrade into garbage.
    kvcache_reset(cache);

    const size_t dim = ctx->model_dim > 0 ? ctx->model_dim : cache->dim;

    // 1トークン分の作業バッファを一度だけ確保（以降のステップで再利用）
    if (!ensure_f32_tensor_shape(ctx->hidden_buf, 1, dim)) return -3;
    if (!ensure_f32_tensor_shape(ctx->embed_buf, 1, dim))  return -3;

    
    if (vocab_size_base == 0) vocab_size_base = 16;
    if (ctx->logits_buf.size() < vocab_size_base) {
        ctx->logits_buf.assign(vocab_size_base, 0.0);
    }
    if (ctx->order_buf.size() < vocab_size_base) {
        ctx->order_buf.resize(vocab_size_base);
    }

    Tensor* last_out_prefill = nullptr;
    int last = ids.empty() ? 0 : ids.back();
    for (int id : ids) {
        Tensor* in = token_embedding_row_into(ctx, id, ctx->embed_buf);
        if (!in) continue;

        // view（F32 row-major）のときだけ free が必要。embed_buf 再利用時は free しない。
        const bool in_is_view = (in != ctx->embed_buf);

        if (!run_stack_forward(ctx, in, ctx->hidden_buf)) {
            fprintf(stderr, "[minxfmr] forward failed during prompt prefill (token id=%d)\n", id);
            fflush(stderr);
            if (in_is_view) tensor_free(in);
            if (last_out_prefill) tensor_free(last_out_prefill);
            return -3;
        }
        if (in_is_view) tensor_free(in);

        if (id == last) {
            if (last_out_prefill) tensor_free(last_out_prefill);
            last_out_prefill = tensor_clone_f32_local(ctx->hidden_buf);
        }
        // hidden_buf は再利用するので free しない
    }
    // When temperature <= 0, use greedy sampling (deterministic argmax).
    bool sampler_greedy = (temperature <= 0.0);

    // Helper predicates for token-level control: treat explicit EOS tokens
    // as generation terminators and skip role/template markers when
    // streaming/pushing history to avoid template leakage.
    auto is_eos_token = [](const std::string &s) {
        return s == "</s>" || s == "<|endoftext|>" || s == "<|pad|>" || s == "<|im_end|>";
    };
    auto is_role_token = [](const std::string &s) {
        return s == "<s>" || s == "</s>" || s == "[INST]" || s == "[/INST]" ||
               s == "<|assistant|>" || s == "<|user|>" || s == "<|im_start|>" || s == "<|im_end|>" ||
               s == "<assistant>" || s == "<user>" ||
               s == "[/ASSISTANT]" || s == "[/USER]" || s == "speaker" || s == "<speaker>" ||
               s == "<<SYS>>" || s == "<</SYS>>" ||
               s == "<tool_call>" || s == "</tool_call>" ||
               s == "<tool_response>" || s == "</tool_response>";
    };
    if (top_k <= 0) top_k = 1;
    static std::mt19937 rng((unsigned)std::random_device{}());

    int max_steps = 24;
    if (const char* env_steps = std::getenv("MINXFMR_MAX_GEN_TOKENS")) {
        int parsed = atoi(env_steps);
        if (parsed >= 1 && parsed <= 256) max_steps = parsed;
    }

    std::vector<int> recent_tokens;
    recent_tokens.reserve(64);

    // If requested via environment, emit a single-line JSON object to stdout with
    // the generated token ids, token strings and base64-encoded decoded text.
    bool emit_json = false;
    if (const char* env_json = std::getenv("MINXFMR_EMIT_JSON")) {
        if (env_json && env_json[0] != '\0') emit_json = true;
    }
    std::vector<int> gen_ids;
    std::vector<std::string> gen_token_strs;
    // Buffer for consecutive byte-fallback tokens like <0xE3><0x81>...
    std::vector<unsigned char> pending_bytes;
    // Track the ids and raw token strings corresponding to pending_bytes
    std::vector<int> pending_bytes_ids;
    std::vector<std::string> pending_bytes_raws;
    // Buffer for recently-seen token fragments that may form a role/template
    // marker split across several token ids (e.g. '[', 'INST', ']'). Each
    // entry holds the token id and its raw token string.
    std::vector<std::pair<int,std::string>> pending_token_buf;
    // Preallocate a reusable workspace for logits chunking to avoid repeated
    // allocations inside the token loop. This buffer is intentionally separate
    // from cpu_workspace because run_stack_forward may grow/reset that workspace.
    const size_t OUT_CHUNK = 4096;
    size_t global_out_vocab = 0;
    if (ctx->Wout && (ctx->Wout->type == DataType::F32 || is_supported_quantized_type(ctx->Wout->type))) {
        if (ctx->Wout->rows == dim) global_out_vocab = ctx->Wout->cols;
        else if (ctx->Wout->cols == dim) global_out_vocab = ctx->Wout->rows;
    }
    size_t global_chunk_size = (global_out_vocab > 0) ? std::min(OUT_CHUNK, global_out_vocab) : 0;
    std::vector<float> logits_chunk_buffer;
    if (global_chunk_size > 0) logits_chunk_buffer.resize(global_chunk_size);

    // 2) Autoregressive loop: predict one token at a time.
    int t = 0;
    std::string gen_break_reason = "none";
    int gen_tokens_emitted = 0;
    std::string last_emitted_raw_tok;
    int repeat_run = 0;
    for (t = 0; t < max_steps; ++t) {
        if (gen_verbose_enabled()) {
            fprintf(stderr, "[minxfmr] gen loop step=%d last=%d emitted=%d\n", t, last, gen_tokens_emitted);
            fflush(stderr);
        }

        Tensor* out = ctx->hidden_buf;
        int next = 0;

        if (t == 0 && last_out_prefill) {
            // prefill 最終 hidden を再利用バッファへコピー（clone テンソルを作らない）
            std::memcpy(out->data, last_out_prefill->data, sizeof(float) * dim);
        } else {
            Tensor* in = token_embedding_row_into(ctx, last, ctx->embed_buf);
            if (!in) {
                fprintf(stderr, "[minxfmr] no input embedding at step=%d last=%d\n", t, last);
                fflush(stderr);
                gen_break_reason = "no_in";
                break;
            }
            const bool in_is_view = (in != ctx->embed_buf);
            if (!run_stack_forward(ctx, in, out)) {
                fprintf(stderr, "[minxfmr] forward failed during generation step=%d last=%d\n", t, last);
                fflush(stderr);
                if (in_is_view) tensor_free(in);
                if (last_out_prefill) tensor_free(last_out_prefill);
                return -3;
            }
            if (in_is_view) tensor_free(in);
        }

        if (ctx->Wnorm) {
            apply_final_norm_inplace(out, ctx->Wnorm, ctx->rmsnorm_epsilon);
        }

        // ---- logits: 再利用バッファ ----
        size_t vocab_size = vocab_size_base;
        std::vector<double>& logits = ctx->logits_buf;
        if (logits.size() < vocab_size) {
            logits.assign(vocab_size, 0.0);
        } else {
            std::fill(logits.begin(), logits.begin() + (std::ptrdiff_t)vocab_size, 0.0);
        }

        const float* od = (const float*)out->data;

        if (ctx->Wout && (ctx->Wout->type == DataType::F32 || is_supported_quantized_type(ctx->Wout->type))) {
            if (is_supported_quantized_type(ctx->Wout->type)) {
                size_t out_vocab = 0;
                bool rhs_transposed = false;
                if (ctx->Wout->rows == dim) {
                    out_vocab = ctx->Wout->cols;
                } else if (ctx->Wout->cols == dim) {
                    out_vocab = ctx->Wout->rows;
                    rhs_transposed = true;
                }

                size_t use_vocab = std::min(vocab_size, out_vocab);
                bool ok = false;
                if (out_vocab > 0) {
                    Tensor* logits_t = tensor_create_f32(1, out_vocab);
                    if (logits_t) {
                        ok = rhs_transposed ?
                            backend_matmul_rhs_transposed(out, ctx->Wout, logits_t) :
                            backend_matmul(out, ctx->Wout, logits_t);
                        if (ok) {
                            const float* ldata = (const float*)logits_t->data;
                            for (size_t j = 0; j < use_vocab; ++j) logits[j] = (double)ldata[j];
                            vocab_size = use_vocab;
                        }
                        tensor_free(logits_t);
                    }
                }

                if (!ok) {
                    for (size_t i = 0; i < vocab_size; ++i) {
                        size_t idx = (dim == 0) ? 0 : (i % dim);
                        logits[i] = (double)od[idx];
                    }
                }
            } else {
            const float* wd = (const float*)ctx->Wout->data;
            // Case A: Wout is [dim x vocab] (rows == dim)
            if (ctx->Wout->rows == dim) {
                size_t out_vocab = ctx->Wout->cols;
                size_t use_vocab = std::min(vocab_size, out_vocab);
                const size_t CHUNK = OUT_CHUNK;
                for (size_t off = 0; off < use_vocab; off += CHUNK) {
                    size_t cur = std::min(CHUNK, use_vocab - off);
                    float* ltmp = logits_chunk_buffer.empty() ? nullptr : logits_chunk_buffer.data();
                    bool ok = false;
                    if (ltmp) ok = backend_matvec_strided(od, wd + off, ltmp, dim, cur, out_vocab);
                    if (ok) {
                        for (size_t j = 0; j < cur; ++j) logits[off + j] = (double)ltmp[j];
                    } else {
                        // Fallback scalar compute for this chunk
                        for (size_t j = 0; j < cur; ++j) {
                            double s = 0.0;
                            size_t gj = off + j;
                            for (size_t i = 0; i < dim; ++i) s += (double)od[i] * (double)wd[i * out_vocab + gj];
                            logits[gj] = s;
                        }
                    }
                }
                vocab_size = use_vocab;
            }
            // Case B: Wout is [vocab x dim] (cols == dim)
            else if (ctx->Wout->cols == dim) {
                size_t out_vocab = ctx->Wout->rows;
                size_t use_vocab = std::min(vocab_size, out_vocab);
                const size_t CHUNK = OUT_CHUNK;
                for (size_t off = 0; off < use_vocab; off += CHUNK) {
                    size_t cur = std::min(CHUNK, use_vocab - off);
                    float* ltmp = logits_chunk_buffer.empty() ? nullptr : logits_chunk_buffer.data();
                    const float* rowptr = wd + off * ctx->Wout->cols; // each row has length dim
                    bool ok = false;
                    if (ltmp) ok = backend_vec_dot_rows(od, rowptr, ltmp, dim, cur, ctx->Wout->cols);
                    if (ok) {
                        for (size_t j = 0; j < cur; ++j) logits[off + j] = (double)ltmp[j];
                    } else {
                        for (size_t j = 0; j < cur; ++j) {
                            double s = 0.0;
                            size_t gj = off + j;
                            for (size_t i = 0; i < dim; ++i) s += (double)od[i] * (double)wd[gj * dim + i];
                            logits[gj] = s;
                        }
                    }
                }
                vocab_size = use_vocab;
            } else {
                for (size_t i = 0; i < vocab_size; ++i) {
                    size_t idx = (dim == 0) ? 0 : (i % dim);
                    logits[i] = (double)od[idx];
                }
            }
            }
        } else {
            for (size_t i = 0; i < vocab_size; ++i) {
                size_t idx = (dim == 0) ? 0 : (i % dim);
                logits[i] = (double)od[idx];
            }
        }

        for (size_t i = 0; i < recent_tokens.size(); ++i) {
            int rid = recent_tokens[i];
            if (rid >= 0 && (size_t)rid < logits.size()) logits[(size_t)rid] -= 0.75;
        }

        int k_use = top_k;
        if (k_use <= 0) k_use = 1;
        if ((size_t)k_use > vocab_size) k_use = (int)vocab_size;

        std::vector<int>& order = ctx->order_buf;
        if (order.size() < vocab_size) order.resize(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) order[i] = (int)i;

        std::partial_sort(order.begin(),
                          order.begin() + k_use,
                          order.begin() + (std::ptrdiff_t)vocab_size,
            [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; });
        
        std::vector<double> probs;
        probs.reserve((size_t)k_use);
        double maxs = logits[(size_t)order[0]];

        if (sampler_greedy) {
            // Deterministic greedy: pick the highest-logit token.
            next = order[0];
        } else {
            double sum = 0.0;
            for (int i = 0; i < k_use; ++i) {
                double p = exp((logits[(size_t)order[(size_t)i]] - maxs) / temperature);
                probs.push_back(p);
                sum += p;
            }
            if (sum <= 0.0) {
                next = order[0];
            } else {
                for (double& p : probs) p /= sum;
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int pick = dist(rng);
                next = order[(size_t)pick];
            }
        }

        std::string raw_tok = tokenizer_id_to_token(next);
        double chosen_logit = 0.0;
        if (next >= 0 && (size_t)next < logits.size()) chosen_logit = logits[(size_t)next];
        std::string preview = render_token_piece(raw_tok);

        auto is_visible_piece = [](const std::string& s) {
            if (s.empty()) return false;
            for (unsigned char c : s) {
                if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '\v' && c != '\f') return true;
            }
            return false;
        };
        // Note: do not apply language-specific leading-piece heuristics here.
        // The user requested not to bias toward English-like tokens; keep
        // only a simple visibility check for first-token handling.

        if (gen_tokens_emitted == 0 && (!is_visible_piece(preview) || is_eos_token(raw_tok) || is_role_token(raw_tok))) {
            for (int i = 1; i < k_use; ++i) {
                int alt = order[(size_t)i];
                std::string alt_raw = tokenizer_id_to_token(alt);
                std::string alt_preview = render_token_piece(alt_raw);
                if (is_visible_piece(alt_preview) && !is_eos_token(alt_raw) && !is_role_token(alt_raw)) {
                    next = alt;
                    raw_tok = alt_raw;
                    if (next >= 0 && (size_t)next < logits.size()) chosen_logit = logits[(size_t)next];
                    preview = alt_preview;
                    if (chat_debug_enabled()) {
                        fprintf(stderr, "[minxfmr] avoided empty first token, switched to id=%d raw='%s'\n", next, raw_tok.c_str());
                    }
                    break;
                }
            }
        }

        if (gen_verbose_enabled()) {
            fprintf(stderr, "[minxfmr] step=%d selected id=%d logit=%f sampler=%s k=%d raw_tok='%s' preview='%s' pending_bytes=%zu pending_fragments=%zu\n",
                t, next, chosen_logit, sampler_greedy ? "greedy" : "sample", k_use, raw_tok.c_str(), preview.c_str(), pending_bytes.size(), pending_token_buf.size());
            fflush(stderr);
        }

        if (!raw_tok.empty() && raw_tok == last_emitted_raw_tok) {
            ++repeat_run;
        } else {
            last_emitted_raw_tok = raw_tok;
            repeat_run = 1;
        }
        if (repeat_run >= 16) {
            if (chat_debug_enabled()) {
                fprintf(stderr, "[minxfmr] stopping on repetition token='%s' run=%d\n", raw_tok.c_str(), repeat_run);
                fflush(stderr);
            }
            gen_break_reason = "repeat";
            //if (in) tensor_free(in);
            //if (out) tensor_free(out);
            break;
        }

        // If model emits an explicit EOS token, stop generation immediately.
        if (is_eos_token(raw_tok)) {
            if (chat_debug_enabled()) {
                fprintf(stderr, "[minxfmr] stopping on EOS token id=%d raw='%s'\n", next, raw_tok.c_str());
                fflush(stderr);
            }
            gen_break_reason = "eos";
            //if (in) tensor_free(in);
            //if (out) tensor_free(out);
            break;
        }

        // If this is a role/template marker, skip emitting it to the user
        // and do not add it to recent token penalties or history. Still
        // continue the generation loop normally so resources are freed.
        bool skip_role = is_role_token(raw_tok);

        // Helper: detect byte-fallback token like <0xE3>
        auto is_byte_fallback = [](const std::string &s) {
            return s.size() == 6 && s.rfind("<0x", 0) == 0 && s[5] == '>';
        };

        if (!skip_role) {
            if (is_byte_fallback(raw_tok)) {
                // decode hex and accumulate
                auto hex_to_nibble = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                    return -1;
                };
                int hi = hex_to_nibble(raw_tok[3]);
                int lo = hex_to_nibble(raw_tok[4]);
                if (hi >= 0 && lo >= 0) {
                    unsigned char b = (unsigned char)((hi << 4) | lo);
                    pending_bytes.push_back(b);
                }
                // Track the id and raw token string for later emission/logging
                pending_bytes_ids.push_back(next);
                pending_bytes_raws.push_back(raw_tok);
                ++gen_tokens_emitted;
            } else {
                // Append token fragment to the pending fragment buffer
                pending_token_buf.emplace_back(next, raw_tok);

                // Build concatenated fragment and check against known role markers
                std::string concat;
                concat.reserve(pending_token_buf.size() * 8);
                for (auto &pp : pending_token_buf) concat += pp.second;

                static const std::vector<std::string> role_markers = {
                    "<s>", "[INST]", "[/INST]", "<|assistant|>", "<|user|>",
                    "<|im_start|>", "<|im_end|>", "<<SYS>>", "<</SYS>>",
                    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"
                };

                bool is_prefix = false;
                bool is_exact = false;
                for (const std::string &m : role_markers) {
                    if (m.rfind(concat, 0) == 0) {
                        is_prefix = true;
                        if (m == concat) { is_exact = true; break; }
                    }
                }

                if (is_exact) {
                    // Suppress the complete marker sequence
                    if (chat_debug_enabled()) {
                        fprintf(stderr, "[minxfmr] suppressed role/template marker seq='%s'\n", concat.c_str());
                        fflush(stderr);
                    }
                    pending_token_buf.clear();
                } else if (is_prefix) {
                    // Wait for more fragments before deciding; do nothing now.
                } else {
                    // Not a role marker: flush any pending raw bytes first
                    if (!pending_bytes.empty()) {
                        std::string pb;
                        pb.reserve(pending_bytes.size());
                        for (unsigned char c : pending_bytes) pb.push_back((char)c);

                        // Record ids/raws for JSON/debug and log the emission
                        for (size_t ii = 0; ii < pending_bytes_ids.size(); ++ii) {
                            gen_ids.push_back(pending_bytes_ids[ii]);
                            gen_token_strs.push_back(pending_bytes_raws[ii]);
                        }
                        if (chat_debug_enabled()) {
                            std::ostringstream oh;
                            oh << std::hex << std::setfill('0');
                            for (unsigned char c : pb) oh << "\\x" << std::setw(2) << (int)((unsigned char)c);
                            std::string hexs = oh.str();
                            std::string cumul = tokenizer_decode(gen_ids);
                            std::ostringstream oi;
                            for (size_t ii = 0; ii < pending_bytes_ids.size(); ++ii) {
                                if (ii) oi << ',';
                                oi << pending_bytes_ids[ii];
                            }
                            fprintf(stderr, "[minxfmr] emit bytes ids=[%s] hex=%s cumulative='%s'\n", oi.str().c_str(), hexs.c_str(), cumul.c_str());
                            fflush(stderr);
                        }
                        if (callback) callback(pb.c_str());
                        pending_bytes.clear();
                        pending_bytes_ids.clear();
                        pending_bytes_raws.clear();
                    }

                    // Flush accumulated fragments as normal tokens
                    for (auto &pp : pending_token_buf) {
                        int id_flush = pp.first;
                        const std::string &raw_flush = pp.second;
                        std::string tok = render_token_piece(raw_flush);
                        if (tok.empty()) tok = " ";

                        // Record id/raw, then emit and log
                        gen_ids.push_back(id_flush);
                        gen_token_strs.push_back(raw_flush);
                        if (chat_debug_enabled()) {
                            std::ostringstream oh;
                            oh << std::hex << std::setfill('0');
                            for (unsigned char c : tok) oh << "\\x" << std::setw(2) << (int)((unsigned char)c);
                            std::string hexs = oh.str();
                            std::string cumul = tokenizer_decode(gen_ids);
                            fprintf(stderr, "[minxfmr] emit id=%d raw='%s' rendered='%s' hex=%s cumulative='%s'\n",
                                id_flush, raw_flush.c_str(), tok.c_str(), hexs.c_str(), cumul.c_str());
                            fflush(stderr);
                        }
                        if (callback) callback(tok.c_str());
                        ++gen_tokens_emitted;
                        recent_tokens.push_back(id_flush);
                        if (recent_tokens.size() > 48) recent_tokens.erase(recent_tokens.begin());
                    }
                    pending_token_buf.clear();
                }
            }
        } else {
            // If the token itself is already a complete role marker string,
            // just log and skip it. This path handles cases where tokenizer
            // returns the full marker in one token.
            if (chat_debug_enabled()) {
                fprintf(stderr, "[minxfmr] suppressed role/template token id=%d raw='%s'\n", next, raw_tok.c_str());
                fflush(stderr);
            }
        }

        if (!skip_role) recent_tokens.push_back(next);
        if (recent_tokens.size() > 48) recent_tokens.erase(recent_tokens.begin());

        // Removed heuristic: previously we stopped generation early when a
        // punctuation token ('.','!','?') appeared after a few tokens.
        // This caused premature truncation of replies; rely on explicit EOS
        // tokens and max token limits instead.

        //if (in) tensor_free(in);
        //if (out) tensor_free(out);
        last = next;
    }

    if (last_out_prefill) tensor_free(last_out_prefill);

    // 3) Flush buffered fragments and finalize logs.
    // Flush any pending byte-fallbacks accumulated during streaming.
    if (!pending_bytes.empty()) {
        std::string pb;
        pb.reserve(pending_bytes.size());
        for (unsigned char c : pending_bytes) pb.push_back((char)c);
        for (size_t ii = 0; ii < pending_bytes_ids.size(); ++ii) {
            gen_ids.push_back(pending_bytes_ids[ii]);
            gen_token_strs.push_back(pending_bytes_raws[ii]);
        }
        if (chat_debug_enabled()) {
            std::ostringstream oh;
            oh << std::hex << std::setfill('0');
            for (unsigned char c : pb) oh << "\\x" << std::setw(2) << (int)((unsigned char)c);
            std::string hexs = oh.str();
            std::string cumul = tokenizer_decode(gen_ids);
            std::ostringstream oi;
            for (size_t ii = 0; ii < pending_bytes_ids.size(); ++ii) {
                if (ii) oi << ',';
                oi << pending_bytes_ids[ii];
            }
            fprintf(stderr, "[minxfmr] emit bytes ids=[%s] hex=%s cumulative='%s'\n", oi.str().c_str(), hexs.c_str(), cumul.c_str());
            fflush(stderr);
        }
        if (callback) callback(pb.c_str());
        pending_bytes.clear();
        pending_bytes_ids.clear();
        pending_bytes_raws.clear();
    }

    // Flush any pending token fragments that were not part of suppressed
    // role/template markers (e.g. a lone '[' that didn't become '[INST]').
    if (!pending_token_buf.empty()) {
        for (auto &pp : pending_token_buf) {
            int id_flush = pp.first;
            const std::string &raw_flush = pp.second;
            std::string tok = render_token_piece(raw_flush);
            if (tok.empty()) tok = " ";
            gen_ids.push_back(id_flush);
            gen_token_strs.push_back(raw_flush);
            if (chat_debug_enabled()) {
                std::ostringstream oh;
                oh << std::hex << std::setfill('0');
                for (unsigned char c : tok) oh << "\\x" << std::setw(2) << (int)((unsigned char)c);
                std::string hexs = oh.str();
                std::string cumul = tokenizer_decode(gen_ids);
                fprintf(stderr, "[minxfmr] emit id=%d raw='%s' rendered='%s' hex=%s cumulative='%s'\n",
                    id_flush, raw_flush.c_str(), tok.c_str(), hexs.c_str(), cumul.c_str());
                fflush(stderr);
            }
            if (callback) callback(tok.c_str());
            ++gen_tokens_emitted;
            recent_tokens.push_back(id_flush);
            if (recent_tokens.size() > 48) recent_tokens.erase(recent_tokens.begin());
        }
        pending_token_buf.clear();
    }

    if (gen_break_reason == "none") gen_break_reason = "max_steps";
    if (chat_debug_enabled()) {
        fprintf(stderr, "[minxfmr] generation finished: emitted=%d last=%d reason=%s steps=%d max=%d\n",
            gen_tokens_emitted, last, gen_break_reason.c_str(), t, max_steps);
        fflush(stderr);
    }

    // If emitting JSON, build object and write to stdout.
    if (emit_json) {
        // helper: base64 encode decoded text
        auto base64_encode = [](const std::string& in) {
            static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::string out;
            int val=0, valb=-6;
            for (unsigned char c : in) {
                val = (val<<8) + c;
                valb += 8;
                while (valb>=0) {
                    out.push_back(b64[(val>>valb)&0x3F]);
                    valb-=6;
                }
            }
            if (valb>-6) out.push_back(b64[((val<<8)>>(valb+8))&0x3F]);
            while (out.size()%4) out.push_back('=');
            return out;
        };

        auto json_escape = [](const std::string& s) {
            std::string o;
            o.reserve(s.size()*2);
            for (unsigned char c : s) {
                if (c == '"') { o += "\\\""; }
                else if (c == '\\') { o += "\\\\"; }
                else if (c >= 0x20 && c <= 0x7E) { o.push_back((char)c); }
                else {
                    // non-printable / non-ascii -> emit as \u00XX
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04X", (unsigned int)c);
                    o += buf;
                }
            }
            return o;
        };

        std::string decoded = tokenizer_decode(gen_ids);
        std::string b64 = base64_encode(decoded);

        std::ostringstream js;
        js << "{";
        js << "\"token_ids\": [";
        for (size_t i = 0; i < gen_ids.size(); ++i) {
            if (i) js << ", ";
            js << gen_ids[i];
        }
        js << "], ";
        js << "\"tokens\": [";
        for (size_t i = 0; i < gen_token_strs.size(); ++i) {
            if (i) js << ", ";
            js << '"' << json_escape(gen_token_strs[i]) << '"';
        }
        js << "], ";
        js << "\"text_b64\": \"" << b64 << "\"";
        js << "}";
        std::string out = js.str();
        // print single-line JSON to stdout
        fprintf(stdout, "%s\n", out.c_str());
        fflush(stdout);
    }

    return 0;
}

void minxfmr_reset(minxfmr_context* ctx) {
    if (!ctx) return;
    if (ctx->cache) kvcache_reset(ctx->cache);
    ctx->dummy = 0;
}

void minxfmr_close(minxfmr_context* ctx) {
    if (!ctx) return;
    backend_release_resources();
    if (ctx->cache) kvcache_free(ctx->cache);
    if (ctx->Wemb) tensor_free(ctx->Wemb);
    if (ctx->Wnorm) tensor_free(ctx->Wnorm);
    if (ctx->Wq) tensor_free(ctx->Wq);
    if (ctx->Wk) tensor_free(ctx->Wk);
    if (ctx->Wv) tensor_free(ctx->Wv);
    if (ctx->Wout) tensor_free(ctx->Wout);
    if (ctx->layer_buf_a) tensor_free(ctx->layer_buf_a);
    if (ctx->layer_buf_b) tensor_free(ctx->layer_buf_b);
    if (ctx->embed_buf) tensor_free(ctx->embed_buf);
    if (ctx->hidden_buf) tensor_free(ctx->hidden_buf);
    free_layer_weights(ctx->Wq_layers);
    free_layer_weights(ctx->Wk_layers);
    free_layer_weights(ctx->Wv_layers);
    free_layer_weights(ctx->Bq_layers);
    free_layer_weights(ctx->Bk_layers);
    free_layer_weights(ctx->Bv_layers);
    free_layer_weights(ctx->Wo_layers);
    free_layer_weights(ctx->Wattn_norm_layers);
    free_layer_weights(ctx->Wffn_norm_layers);
    free_layer_weights(ctx->Wffn_gate_layers);
    free_layer_weights(ctx->Wffn_up_layers);
    free_layer_weights(ctx->Wffn_down_layers);
    delete ctx;
}

void minxfmr_print_weights(minxfmr_context* ctx) {
    if (!ctx) {
        fprintf(stderr, "[minxfmr] no context\n");
        return;
    }
    if (ctx->Wq && ctx->Wk && ctx->Wv) {
        fprintf(stderr, "[minxfmr] Wq dim=%zux%zu sample00=%f\n", ctx->Wq->rows, ctx->Wq->cols, tensor_get_f32(ctx->Wq, 0, 0));
        fprintf(stderr, "[minxfmr] Wk dim=%zux%zu sample00=%f\n", ctx->Wk->rows, ctx->Wk->cols, tensor_get_f32(ctx->Wk, 0, 0));
        fprintf(stderr, "[minxfmr] Wv dim=%zux%zu sample00=%f\n", ctx->Wv->rows, ctx->Wv->cols, tensor_get_f32(ctx->Wv, 0, 0));
    } else {
        fprintf(stderr, "[minxfmr] no projection weights loaded\n");
    }
    if (!ctx->Wq_layers.empty()) {
        fprintf(stderr, "[minxfmr] per-layer projections loaded for %zu layers\n", ctx->Wq_layers.size());
    }
    if (!ctx->Wffn_gate_layers.empty()) {
        fprintf(stderr, "[minxfmr] per-layer ffn loaded for %zu layers\n", ctx->Wffn_gate_layers.size());
    }
    if (!ctx->Wo_layers.empty()) {
        fprintf(stderr, "[minxfmr] per-layer Wo loaded for %zu layers\n", ctx->Wo_layers.size());
    }
}

const char* minxfmr_get_chat_template(minxfmr_context* ctx) {
    if (!ctx) return nullptr;
    if (ctx->chat_template.empty()) return nullptr;
    return ctx->chat_template.c_str();
}

size_t minxfmr_get_special_tokens_count(minxfmr_context* ctx) {
    if (!ctx) return 0;
    return ctx->special_tokens.size();
}

const char* minxfmr_get_special_token(minxfmr_context* ctx, size_t idx) {
    if (!ctx) return nullptr;
    if (idx >= ctx->special_tokens.size()) return nullptr;
    return ctx->special_tokens[idx].c_str();
}
