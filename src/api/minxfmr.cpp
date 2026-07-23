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
    int dummy;
    // optional metadata
    std::string chat_template;
    std::vector<std::string> special_tokens;
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
    Tensor* out = tensor_create_f32(in->rows, in->cols);
    if (!out) return nullptr;
    memcpy(out->data, in->data, sizeof(float) * in->rows * in->cols);
    return out;
}

static Tensor* token_embedding_row(const minxfmr_context* ctx, int token_id) {
    if (!ctx || !ctx->Wemb || ctx->Wemb->type != DataType::F32 || token_id < 0) return nullptr;
    const Tensor* emb = ctx->Wemb;
    const size_t dim = ctx->model_dim > 0 ? ctx->model_dim : ((emb->rows > emb->cols) ? emb->cols : emb->rows);

    // If embeddings are stored as rows (rows >= cols) we can return a lightweight
    // view into the backing storage instead of copying the row.
    if (emb->rows >= emb->cols) {
        if ((size_t)token_id >= emb->rows) token_id = (int)((size_t)token_id % emb->rows);
        float* ptr = (float*)emb->data + (size_t)token_id * emb->cols;
        Tensor* view = tensor_create_f32_view(1, emb->cols, ptr);
        return view;
    }

    // Otherwise the token vectors are laid out in columns; fall back to copying.
    if ((size_t)token_id >= emb->cols) token_id = (int)((size_t)token_id % emb->cols);
    Tensor* row = tensor_create_f32(1, emb->rows);
    if (!row) return nullptr;
    float* dst = (float*)row->data;
    const float* src = (const float*)emb->data;
    for (size_t r = 0; r < emb->rows; ++r) dst[r] = src[r * emb->cols + (size_t)token_id];
    (void)dim;
    return row;
}

static bool env_enabled(const char* key) {
    const char* v = std::getenv(key);
    if (!v || v[0] == '\0') return false;
    return v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T';
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
    if (!t || t->type != DataType::F32) return false;
    if (in_dim == 0) return false;

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

static bool arch_uses_square_transpose(const std::string& arch_lc) {
    // Architectures whose exported square attention matrices are commonly laid out
    // in the opposite orientation for this runtime.
    if (arch_lc == "llama") return true;
    if (arch_lc == "mistral") return true;
    if (arch_lc == "mixtral") return true;
    if (arch_lc == "qwen2") return true;
    if (arch_lc == "qwen2moe") return true;
    if (arch_lc == "gemma") return true;
    // Explicitly-known non-LLAMA families (keep square transpose off by default).
    if (arch_lc == "gptneox") return false;
    if (arch_lc == "gpt2") return false;
    if (arch_lc == "falcon") return false;
    if (arch_lc == "bloom") return false;
    if (arch_lc == "mpt") return false;
    if (arch_lc == "phi") return false;
    if (arch_lc == "phi2") return false;
    if (arch_lc == "phi3") return false;
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

    Tensor* cur = tensor_clone_f32_local(input);
    if (!cur) return false;

    size_t layers_to_run = 1;
    if (!ctx->Wq_layers.empty() && ctx->Wq_layers.size() == ctx->Wk_layers.size() && ctx->Wq_layers.size() == ctx->Wv_layers.size()) {
        layers_to_run = ctx->Wq_layers.size();
    }
    if (ctx->cache && layers_to_run > ctx->cache->layers) layers_to_run = ctx->cache->layers;

    for (size_t l = 0; l < layers_to_run; ++l) {
        Tensor* nxt = tensor_create_f32(cur->rows, cur->cols);
        if (!nxt) {
            tensor_free(cur);
            return false;
        }

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
        tensor_free(cur);
        if (!ok) {
            tensor_free(nxt);
            return false;
        }
        cur = nxt;
    }

    if (output->rows != cur->rows || output->cols != cur->cols) {
        tensor_free(cur);
        return false;
    }
    memcpy(output->data, cur->data, sizeof(float) * cur->rows * cur->cols);
    tensor_free(cur);
    return true;
}

static void free_layer_weights(std::vector<Tensor*>& vec) {
    for (Tensor* t : vec) if (t) tensor_free(t);
    vec.clear();
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

    const char* ext = strrchr(model_path, '.');
    const bool looks_gguf = (ext != nullptr) && (_stricmp(ext, ".gguf") == 0);

    bool desired_transpose_wq = false;
    bool desired_transpose_wk = false;
    bool desired_transpose_wv = false;
    bool desired_transpose_wo = false;
    bool desired_transpose_ffn_square = false;

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
                } else {
                    fprintf(stderr, "[minxfmr] auto orientation: architecture metadata missing, default square_transpose=off\n");
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
        for (size_t l = 0; l < ctx->n_layer; ++l) {
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

    if (ctx->Wq) {
        if (ctx->model_dim == 0) {
            ctx->model_dim = (ctx->Wq->rows == ctx->Wq->cols) ? ctx->Wq->rows : std::max(ctx->Wq->rows, ctx->Wq->cols);
        }
        size_t kv = infer_kv_dim_from_weight(ctx->Wk, ctx->model_dim);
        size_t vv = infer_kv_dim_from_weight(ctx->Wv, ctx->model_dim);
        if (kv == 0) kv = ctx->model_dim;
        if (vv > 0) kv = std::min(kv, vv);
        ctx->kv_dim = kv;
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

            if (wfg && ffn_hidden > 0 && wfg->rows != ctx->model_dim && wfg->cols != ffn_hidden) ++bad_ffn;
            if (wfu && ffn_hidden > 0 && wfu->rows != ctx->model_dim && wfu->cols != ffn_hidden) ++bad_ffn;

            if (wfd) {
                if (ffn_hidden > 0) {
                    if (normalize_linear_inplace(wfd, ffn_hidden, desired_transpose_ffn_square, tr)) {
                        if (tr) ++n_ffn_down;
                        if (wfd->rows != ffn_hidden && wfd->cols != ctx->model_dim) ++bad_ffn;
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

        transformer_set_transpose_square_weights_for_all(false, false, false, false);
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

    fprintf(stderr, "minxfmr: opened model %s\n", model_path);
    return ctx;
}

int minxfmr_generate(minxfmr_context* ctx, const char* prompt, void (*callback)(const char* token), double temperature, int top_k) {
    if (!ctx || !prompt) return -1;

    // Reset per-call backend workspace allocations for this generation call.
    backend_workspace_reset();

    std::vector<int> ids = tokenizer_encode(prompt);
    // debug: log prompt token ids and decoded prompt
    if (!ids.empty()) {
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

    Tensor* last_out_prefill = nullptr;
    int last = ids.empty() ? 0 : ids.back();
    for (int id : ids) {
        Tensor* in = token_embedding_row(ctx, id);
        Tensor* out = tensor_create_f32(1, dim);
        if (!in || !out) {
            tensor_free(in);
            tensor_free(out);
            continue;
        }
        run_stack_forward(ctx, in, out);
        tensor_free(in);
        if (id == last) {
            last_out_prefill = tensor_clone_f32_local(out);
        }
        tensor_free(out);
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
    // Buffer for recently-seen token fragments that may form a role/template
    // marker split across several token ids (e.g. '[', 'INST', ']'). Each
    // entry holds the token id and its raw token string.
    std::vector<std::pair<int,std::string>> pending_token_buf;
    // Preallocate a reusable workspace for logits chunking to avoid repeated
    // allocations inside the token loop. This buffer is intentionally separate
    // from cpu_workspace because run_stack_forward may grow/reset that workspace.
    const size_t OUT_CHUNK = 4096;
    size_t global_out_vocab = 0;
    if (ctx->Wout && ctx->Wout->type == DataType::F32) {
        if (ctx->Wout->rows == dim) global_out_vocab = ctx->Wout->cols;
        else if (ctx->Wout->cols == dim) global_out_vocab = ctx->Wout->rows;
    }
    size_t global_chunk_size = (global_out_vocab > 0) ? std::min(OUT_CHUNK, global_out_vocab) : 0;
    std::vector<float> logits_chunk_buffer;
    if (global_chunk_size > 0) logits_chunk_buffer.resize(global_chunk_size);

    int t = 0;
    std::string gen_break_reason = "none";
    int gen_tokens_emitted = 0;
    std::string last_emitted_raw_tok;
    int repeat_run = 0;
    for (t = 0; t < max_steps; ++t) {
        fprintf(stderr, "[minxfmr] gen loop step=%d last=%d emitted=%d\n", t, last, gen_tokens_emitted);
        fflush(stderr);
        Tensor* in = nullptr;
        Tensor* out = nullptr;
        int next = 0;

        if (t == 0 && last_out_prefill) {
            out = tensor_clone_f32_local(last_out_prefill);
        } else {
            in = token_embedding_row(ctx, last);
            out = tensor_create_f32(1, dim);
            if (in && out) run_stack_forward(ctx, in, out);
        }

        if (!out) { fprintf(stderr, "[minxfmr] no output tensor at step=%d last=%d\n", t, last); fflush(stderr); gen_break_reason = "no_out"; break; }

        if (ctx->Wnorm) apply_final_norm_inplace(out, ctx->Wnorm, ctx->rmsnorm_epsilon);

        size_t vocab_size = vocab_size_base;
        std::vector<double> logits(vocab_size, 0.0);
        const float* od = (const float*)out->data;

        if (ctx->Wout && ctx->Wout->type == DataType::F32) {
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

        std::vector<int> order(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) order[i] = (int)i;
        std::partial_sort(order.begin(), order.begin() + k_use, order.end(),
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

        auto is_bad_leading_piece = [](const std::string& s) {
            if (s.empty()) return true;
            size_t i = 0;
            while (i < s.size()) {
                unsigned char c = (unsigned char)s[i];
                if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f') {
                    ++i;
                    continue;
                }
                if (c == '|' || c == '<' || c == '>' || c == '[' || c == ']' || c == '{' || c == '}') return true;
                if (c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?' || c == '\'' || c == '"') return true;
                if (c == ':' && i + 1 < s.size()) {
                    unsigned char n = (unsigned char)s[i + 1];
                    if ((n >= 'A' && n <= 'Z') || (n >= 'a' && n <= 'z')) return true;
                }
                if (c == 'Q' && i + 1 < s.size() && s[i + 1] == ':') return true;
                return false;
            }
            return true;
        };

        if (gen_tokens_emitted == 0 && (!is_visible_piece(preview) || is_bad_leading_piece(preview) || is_eos_token(raw_tok) || is_role_token(raw_tok))) {
            for (int i = 1; i < k_use; ++i) {
                int alt = order[(size_t)i];
                std::string alt_raw = tokenizer_id_to_token(alt);
                std::string alt_preview = render_token_piece(alt_raw);
                if (is_visible_piece(alt_preview) && !is_bad_leading_piece(alt_preview) && !is_eos_token(alt_raw) && !is_role_token(alt_raw)) {
                    next = alt;
                    raw_tok = alt_raw;
                    if (next >= 0 && (size_t)next < logits.size()) chosen_logit = logits[(size_t)next];
                    preview = alt_preview;
                    fprintf(stderr, "[minxfmr] avoided empty first token, switched to id=%d raw='%s'\n", next, raw_tok.c_str());
                    break;
                }
            }
        }

        fprintf(stderr, "[minxfmr] step=%d selected id=%d logit=%f sampler=%s k=%d raw_tok='%s' preview='%s' pending_bytes=%zu pending_fragments=%zu\n",
            t, next, chosen_logit, sampler_greedy ? "greedy" : "sample", k_use, raw_tok.c_str(), preview.c_str(), pending_bytes.size(), pending_token_buf.size());
        fflush(stderr);

        if (!raw_tok.empty() && raw_tok == last_emitted_raw_tok) {
            ++repeat_run;
        } else {
            last_emitted_raw_tok = raw_tok;
            repeat_run = 1;
        }
        if (repeat_run >= 16) {
            fprintf(stderr, "[minxfmr] stopping on repetition token='%s' run=%d\n", raw_tok.c_str(), repeat_run);
            fflush(stderr);
            gen_break_reason = "repeat";
            if (in) tensor_free(in);
            if (out) tensor_free(out);
            break;
        }

        // If model emits an explicit EOS token, stop generation immediately.
        if (is_eos_token(raw_tok)) {
            fprintf(stderr, "[minxfmr] stopping on EOS token id=%d raw='%s'\n", next, raw_tok.c_str());
            fflush(stderr);
            gen_break_reason = "eos";
            if (in) tensor_free(in);
            if (out) tensor_free(out);
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
                if (emit_json) {
                    gen_ids.push_back(next);
                    gen_token_strs.push_back(raw_tok);
                }
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
                    fprintf(stderr, "[minxfmr] suppressed role/template marker seq='%s'\n", concat.c_str());
                    fflush(stderr);
                    pending_token_buf.clear();
                } else if (is_prefix) {
                    // Wait for more fragments before deciding; do nothing now.
                } else {
                    // Not a role marker: flush any pending raw bytes first
                    if (!pending_bytes.empty()) {
                        std::string pb;
                        pb.reserve(pending_bytes.size());
                        for (unsigned char c : pending_bytes) pb.push_back((char)c);
                        if (callback) callback(pb.c_str());
                        pending_bytes.clear();
                    }

                    // Flush accumulated fragments as normal tokens
                    for (auto &pp : pending_token_buf) {
                        int id_flush = pp.first;
                        const std::string &raw_flush = pp.second;
                        std::string tok = render_token_piece(raw_flush);
                        if (tok.empty()) tok = " ";
                        if (emit_json) {
                            gen_ids.push_back(id_flush);
                            gen_token_strs.push_back(raw_flush);
                        } else {
                            if (callback) callback(tok.c_str());
                        }
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
            fprintf(stderr, "[minxfmr] suppressed role/template token id=%d raw='%s'\n", next, raw_tok.c_str());
            fflush(stderr);
        }

        if (!skip_role) recent_tokens.push_back(next);
        if (recent_tokens.size() > 48) recent_tokens.erase(recent_tokens.begin());

        // Removed heuristic: previously we stopped generation early when a
        // punctuation token ('.','!','?') appeared after a few tokens.
        // This caused premature truncation of replies; rely on explicit EOS
        // tokens and max token limits instead.

        if (in) tensor_free(in);
        if (out) tensor_free(out);
        last = next;
    }

    if (last_out_prefill) tensor_free(last_out_prefill);

    // Flush any pending byte-fallbacks accumulated during streaming.
    if (!pending_bytes.empty()) {
        std::string pb;
        pb.reserve(pending_bytes.size());
        for (unsigned char c : pending_bytes) pb.push_back((char)c);
        if (callback) callback(pb.c_str());
        pending_bytes.clear();
    }

    // Flush any pending token fragments that were not part of suppressed
    // role/template markers (e.g. a lone '[' that didn't become '[INST]').
    if (!pending_token_buf.empty()) {
        for (auto &pp : pending_token_buf) {
            int id_flush = pp.first;
            const std::string &raw_flush = pp.second;
            std::string tok = render_token_piece(raw_flush);
            if (tok.empty()) tok = " ";
            if (emit_json) {
                gen_ids.push_back(id_flush);
                gen_token_strs.push_back(raw_flush);
            } else {
                if (callback) callback(tok.c_str());
            }
            ++gen_tokens_emitted;
            recent_tokens.push_back(id_flush);
            if (recent_tokens.size() > 48) recent_tokens.erase(recent_tokens.begin());
        }
        pending_token_buf.clear();
    }

    if (gen_break_reason == "none") gen_break_reason = "max_steps";
    fprintf(stderr, "[minxfmr] generation finished: emitted=%d last=%d reason=%s steps=%d max=%d\n",
        gen_tokens_emitted, last, gen_break_reason.c_str(), t, max_steps);
    fflush(stderr);

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
    if (ctx->cache) kvcache_free(ctx->cache);
    if (ctx->Wemb) tensor_free(ctx->Wemb);
    if (ctx->Wnorm) tensor_free(ctx->Wnorm);
    if (ctx->Wq) tensor_free(ctx->Wq);
    if (ctx->Wk) tensor_free(ctx->Wk);
    if (ctx->Wv) tensor_free(ctx->Wv);
    if (ctx->Wout) tensor_free(ctx->Wout);
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
