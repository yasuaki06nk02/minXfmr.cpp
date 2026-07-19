#include "gguf_loader.h"
#include "../../third_party/gguf/gguf_reader.h"
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

// backing storage for accessors
static std::string gguf_last_name_Wq;
static size_t gguf_last_rows_Wq=0, gguf_last_cols_Wq=0;
static std::string gguf_last_name_Wk;
static size_t gguf_last_rows_Wk=0, gguf_last_cols_Wk=0;
static std::string gguf_last_name_Wv;
static size_t gguf_last_rows_Wv=0, gguf_last_cols_Wv=0;

bool gguf_try_load_projections(const char* path, Tensor*& outWq, Tensor*& outWk, Tensor*& outWv) {
    return gguf_try_load_projections_for_layer(path, 0, outWq, outWk, outWv);
}

bool gguf_try_load_projections_for_layer(const char* path, int layer, Tensor*& outWq, Tensor*& outWk, Tensor*& outWv) {
    outWq = outWk = outWv = nullptr;
    GGUF_File gf;
    if (!gguf_open(path, gf)) return false;
    (void)0;

    auto make_candidates = [&](char proj) {
        std::vector<std::string> names;
        if (layer >= 0) {
            char b0[64], b1[96], b2[80];
            const char* suf = (proj == 'q') ? "q" : ((proj == 'k') ? "k" : "v");
            std::snprintf(b0, sizeof(b0), "blk.%d.attn_%c.weight", layer, proj);
            std::snprintf(b1, sizeof(b1), "model.layers.%d.attention.w%c.weight", layer, proj);
            std::snprintf(b2, sizeof(b2), "layers.%d.attention.w%c.weight", layer, proj);
            char b3[96], b4[96];
            std::snprintf(b3, sizeof(b3), "model.blocks.%d.attention.w%c.weight", layer, proj);
            std::snprintf(b4, sizeof(b4), "layers.%d.attn.w%c.weight", layer, proj);
            names.emplace_back(b0);
            names.emplace_back(b1);
            names.emplace_back(b2);
            names.emplace_back(b3);
            names.emplace_back(b4);
            names.emplace_back(std::string(suf) + "_proj.weight");
            names.emplace_back(std::string("attn_") + suf + ".weight");
            names.emplace_back(std::string("w") + suf + ".weight");
            names.emplace_back(std::string("w") + suf);
        } else {
            names.emplace_back(std::string((proj == 'q') ? "q_proj.weight" : ((proj == 'k') ? "k_proj.weight" : "v_proj.weight")));
            names.emplace_back(std::string((proj == 'q') ? "attn_q.weight" : ((proj == 'k') ? "attn_k.weight" : "attn_v.weight")));
            names.emplace_back(std::string((proj == 'q') ? "wq.weight" : ((proj == 'k') ? "wk.weight" : "wv.weight")));
            names.emplace_back(std::string((proj == 'q') ? "wq" : ((proj == 'k') ? "wk" : "wv")));
        }
        return names;
    };

    auto load_one = [&](const std::vector<std::string>& names, const char* tag, Tensor*& out) {
        std::vector<const char*> cands;
        cands.reserve(names.size());
        for (const std::string& s : names) cands.push_back(s.c_str());
        GGUF_TensorInfo info;
        if (!gguf_find_tensor_any(gf, cands.data(), cands.size(), info)) return;
        Tensor* t = nullptr;
        if (gguf_read_tensor(gf, info, t)) {
            if (strcmp(tag, "Wq")==0) { gguf_last_name_Wq = info.name; gguf_last_rows_Wq = info.rows; gguf_last_cols_Wq = info.cols; }
            if (strcmp(tag, "Wk")==0) { gguf_last_name_Wk = info.name; gguf_last_rows_Wk = info.rows; gguf_last_cols_Wk = info.cols; }
            if (strcmp(tag, "Wv")==0) { gguf_last_name_Wv = info.name; gguf_last_rows_Wv = info.rows; gguf_last_cols_Wv = info.cols; }
            float minv,maxv; double mean; gguf_tensor_stats(t,minv,maxv,mean);
            fprintf(stderr, "[gguf] loaded %s from '%s' rows=%u cols=%u type=%s min=%f max=%f mean=%f\n", tag, info.name.c_str(), info.rows, info.cols, info.dtype.c_str(), minv, maxv, mean);
            out = t;
            return;
        }
        fprintf(stderr, "[gguf] found %s tensor '%s' but type '%s' is not yet supported\n", tag, info.name.c_str(), info.dtype.c_str());
    };

    const auto q_names = make_candidates('q');
    const auto k_names = make_candidates('k');
    const auto v_names = make_candidates('v');

    load_one(q_names, "Wq", outWq);
    load_one(k_names, "Wk", outWk);
    load_one(v_names, "Wv", outWv);

    // Normalize Wk/Wv layout: if they appear transposed (rows == model_dim, cols == model_dim/k),
    // transpose them so they become k_rows x d for downstream code.
    auto normalize_kv = [&](Tensor*& T, const char* tag) {
        if (!T) return;
        // heuristics: if rows > cols treat as transposed source (d x k_rows) and transpose to k_rows x d
        if (T->rows > T->cols) {
            Tensor* t2 = tensor_transpose_f32(T);
            if (t2) {
                tensor_free(T);
                T = t2;
                fprintf(stderr, "[gguf] normalized %s by transposing to rows=%zu cols=%zu\n", tag, T->rows, T->cols);
            }
        }
    };
    normalize_kv(outWk, "Wk");
    normalize_kv(outWv, "Wv");

    (void)0;

    gguf_close(gf);
    return (outWq != nullptr && outWk != nullptr && outWv != nullptr);
}

bool gguf_try_load_attn_out_for_layer(const char* path, int layer, Tensor*& outWo) {
    outWo = nullptr;
    GGUF_File gf;
    if (!gguf_open(path, gf)) return false;

    std::vector<std::string> names;
    char b0[96], b1[96], b2[96], b3[96];
    std::snprintf(b0, sizeof(b0), "blk.%d.attn_output.weight", layer);
    std::snprintf(b1, sizeof(b1), "model.layers.%d.attention.wo.weight", layer);
    std::snprintf(b2, sizeof(b2), "layers.%d.attention.wo.weight", layer);
    std::snprintf(b3, sizeof(b3), "model.blocks.%d.attention.wo.weight", layer);
    names.emplace_back(b0);
    names.emplace_back(b1);
    names.emplace_back(b2);
    names.emplace_back(b3);
    names.emplace_back("o_proj.weight");
    names.emplace_back("attn_output.weight");
    names.emplace_back("wo.weight");
    names.emplace_back("wo");

    std::vector<const char*> cands;
    cands.reserve(names.size());
    for (const std::string& s : names) cands.push_back(s.c_str());

    GGUF_TensorInfo info;
    if (!gguf_find_tensor_any(gf, cands.data(), cands.size(), info)) {
        gguf_close(gf);
        return false;
    }

    Tensor* t = nullptr;
    if (gguf_read_tensor(gf, info, t)) {
        float minv, maxv;
        double mean;
        gguf_tensor_stats(t, minv, maxv,mean);
        fprintf(stderr, "[gguf] loaded Wo from '%s' rows=%u cols=%u type=%s min=%f max=%f mean=%f\n",
            info.name.c_str(), info.rows, info.cols, info.dtype.c_str(), minv, maxv, mean);
        outWo = t;
    }
    gguf_close(gf);
    return (outWo != nullptr);
}

bool gguf_try_load_norms_for_layer(const char* path, int layer, Tensor*& outAttnNorm, Tensor*& outFfnNorm) {
    outAttnNorm = nullptr;
    outFfnNorm = nullptr;
    GGUF_File gf;
    if (!gguf_open(path, gf)) return false;

    auto load_one = [&](const std::vector<std::string>& names, const char* tag, Tensor*& out) {
        std::vector<const char*> cands;
        cands.reserve(names.size());
        for (const std::string& s : names) cands.push_back(s.c_str());
        GGUF_TensorInfo info;
        if (!gguf_find_tensor_any(gf, cands.data(), cands.size(), info)) return;

        Tensor* t = nullptr;
        if (gguf_read_tensor(gf, info, t)) {
            float minv, maxv;
            double mean;
            gguf_tensor_stats(t, minv, maxv, mean);
            fprintf(stderr, "[gguf] loaded %s from '%s' rows=%u cols=%u type=%s min=%f max=%f mean=%f\n",
                tag, info.name.c_str(), info.rows, info.cols, info.dtype.c_str(), minv, maxv, mean);
            out = t;
        }
    };

    std::vector<std::string> attn_names;
    std::vector<std::string> ffn_names;
    char a0[96], a1[96], a2[96], f0[96], f1[96], f2[96];
    std::snprintf(a0, sizeof(a0), "blk.%d.attn_norm.weight", layer);
    std::snprintf(a1, sizeof(a1), "model.layers.%d.input_layernorm.weight", layer);
    std::snprintf(a2, sizeof(a2), "layers.%d.attention_norm.weight", layer);
    attn_names.emplace_back(a0);
    attn_names.emplace_back(a1);
    attn_names.emplace_back(a2);

    std::snprintf(f0, sizeof(f0), "blk.%d.ffn_norm.weight", layer);
    std::snprintf(f1, sizeof(f1), "model.layers.%d.post_attention_layernorm.weight", layer);
    std::snprintf(f2, sizeof(f2), "layers.%d.ffn_norm.weight", layer);
    ffn_names.emplace_back(f0);
    ffn_names.emplace_back(f1);
    ffn_names.emplace_back(f2);

    load_one(attn_names, "attn_norm", outAttnNorm);
    load_one(ffn_names, "ffn_norm", outFfnNorm);

    gguf_close(gf);
    return (outAttnNorm != nullptr && outFfnNorm != nullptr);
}

bool gguf_try_load_ffn_for_layer(const char* path, int layer, Tensor*& outWgate, Tensor*& outWup, Tensor*& outWdown) {
    outWgate = outWup = outWdown = nullptr;
    GGUF_File gf;
    if (!gguf_open(path, gf)) return false;

    auto load_one = [&](const std::vector<std::string>& names, const char* tag, Tensor*& out) {
        std::vector<const char*> cands;
        cands.reserve(names.size());
        for (const std::string& s : names) cands.push_back(s.c_str());
        GGUF_TensorInfo info;
        if (!gguf_find_tensor_any(gf, cands.data(), cands.size(), info)) return;

        Tensor* t = nullptr;
        if (gguf_read_tensor(gf, info, t)) {
            float minv, maxv;
            double mean;
            gguf_tensor_stats(t, minv, maxv, mean);
            fprintf(stderr, "[gguf] loaded %s from '%s' rows=%u cols=%u type=%s min=%f max=%f mean=%f\n",
                tag, info.name.c_str(), info.rows, info.cols, info.dtype.c_str(), minv, maxv, mean);
            out = t;
        }
    };

    std::vector<std::string> gate_names;
    std::vector<std::string> up_names;
    std::vector<std::string> down_names;
    {
        char b0[96], b1[96], b2[96], b3[96], b4[96], b5[96];
        std::snprintf(b0, sizeof(b0), "blk.%d.ffn_gate.weight", layer);
        std::snprintf(b1, sizeof(b1), "blk.%d.ffn_up.weight", layer);
        std::snprintf(b2, sizeof(b2), "blk.%d.ffn_down.weight", layer);
        gate_names.emplace_back(b0);
        up_names.emplace_back(b1);
        down_names.emplace_back(b2);

        std::snprintf(b3, sizeof(b3), "model.layers.%d.mlp.gate_proj.weight", layer);
        std::snprintf(b4, sizeof(b4), "model.layers.%d.mlp.up_proj.weight", layer);
        std::snprintf(b5, sizeof(b5), "model.layers.%d.mlp.down_proj.weight", layer);
        gate_names.emplace_back(b3);
        up_names.emplace_back(b4);
        down_names.emplace_back(b5);

        std::snprintf(b3, sizeof(b3), "layers.%d.feed_forward.w1.weight", layer);
        std::snprintf(b4, sizeof(b4), "layers.%d.feed_forward.w3.weight", layer);
        std::snprintf(b5, sizeof(b5), "layers.%d.feed_forward.w2.weight", layer);
        gate_names.emplace_back(b3);
        up_names.emplace_back(b4);
        down_names.emplace_back(b5);

        std::snprintf(b3, sizeof(b3), "model.layers.%d.feed_forward.w1.weight", layer);
        std::snprintf(b4, sizeof(b4), "model.layers.%d.feed_forward.w3.weight", layer);
        std::snprintf(b5, sizeof(b5), "model.layers.%d.feed_forward.w2.weight", layer);
        gate_names.emplace_back(b3);
        up_names.emplace_back(b4);
        down_names.emplace_back(b5);
    }

    load_one(gate_names, "Wffn_gate", outWgate);
    load_one(up_names, "Wffn_up", outWup);
    load_one(down_names, "Wffn_down", outWdown);

    gguf_close(gf);
    return (outWgate != nullptr && outWup != nullptr && outWdown != nullptr);
}

bool gguf_try_read_model_config(const char* path, GGUFLoaderModelConfig& out) {
    out = GGUFLoaderModelConfig{0,0,0,0,0};
    GGUF_ModelConfig cfg;
    if (!gguf_read_model_config(path, cfg)) return false;
    out.n_layer = cfg.n_layer;
    out.n_ctx = cfg.n_ctx;
    out.n_embd = cfg.n_embd;
    out.n_head = cfg.n_head;
    out.n_head_kv = cfg.n_head_kv;
    return true;
}

bool gguf_try_read_architecture(const char* path, std::string& out_architecture) {
    out_architecture.clear();
    return gguf_read_architecture(path, out_architecture);
}

bool gguf_try_load_token_embedding(const char* path, Tensor*& outWemb) {
    outWemb = nullptr;
    GGUF_File gf;
    if (!gguf_open(path, gf)) return false;

    const char* candidates[] = {
        "tok_embeddings.weight",
        "token_embd.weight",
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "tok_embeddings",
        "token_embd"
    };
    GGUF_TensorInfo info;
    if (!gguf_find_tensor_any(gf, candidates, sizeof(candidates)/sizeof(candidates[0]), info)) {
        gguf_close(gf);
        return false;
    }

    Tensor* t = nullptr;
    if (gguf_read_tensor(gf, info, t)) {
        float minv, maxv;
        double mean;
        gguf_tensor_stats(t, minv, maxv, mean);
        fprintf(stderr, "[gguf] loaded token embedding from '%s' rows=%u cols=%u type=%s min=%f max=%f mean=%f\n",
            info.name.c_str(), info.rows, info.cols, info.dtype.c_str(), minv, maxv, mean);
        outWemb = t;
    }
    gguf_close(gf);
    return (outWemb != nullptr);
}

bool gguf_try_read_vocab(const char* path, std::vector<std::string>& out_tokens) {
    return gguf_read_vocab(path, out_tokens);
}

bool gguf_try_read_chat_template(const char* path, std::string& out_template) {
    out_template.clear();
    GGUF_File gf;
    if (!gguf_open(path, gf)) return false;
    if (!gf.chat_template.empty()) out_template = gf.chat_template;
    gguf_close(gf);
    return !out_template.empty();
}

bool gguf_try_read_special_tokens(const char* path, std::vector<std::string>& out_tokens) {
    out_tokens.clear();
    GGUF_File gf;
    if (!gguf_open(path, gf)) return false;
    out_tokens = gf.special_tokens;
    gguf_close(gf);
    return !out_tokens.empty();
}

bool gguf_try_load_final_norm(const char* path, Tensor*& outWnorm) {
    outWnorm = nullptr;
    GGUF_File gf;
    if (!gguf_open(path, gf)) return false;

    const char* candidates[] = {
        "output_norm.weight",
        "model.norm.weight",
        "norm.weight",
        "ln_f.weight",
        "final_norm.weight",
        "model.final_layernorm.weight"
    };
    GGUF_TensorInfo info;
    if (!gguf_find_tensor_any(gf, candidates, sizeof(candidates)/sizeof(candidates[0]), info)) {
        gguf_close(gf);
        return false;
    }

    Tensor* t = nullptr;
    if (gguf_read_tensor(gf, info, t)) {
        float minv, maxv;
        double mean;
        gguf_tensor_stats(t, minv, maxv, mean);
        fprintf(stderr, "[gguf] loaded final norm from '%s' rows=%u cols=%u type=%s min=%f max=%f mean=%f\n",
            info.name.c_str(), info.rows, info.cols, info.dtype.c_str(), minv, maxv, mean);
        outWnorm = t;
    }
    gguf_close(gf);
    return (outWnorm != nullptr);
}

bool gguf_try_load_lm_head(const char* path, Tensor*& outWout) {
    outWout = nullptr;
    GGUF_File gf;
    if (!gguf_open(path, gf)) return false;

    const char* candidates[] = {
        "output.weight",
        "lm_head.weight",
        "model.output.weight",
        "tok_embeddings.weight",
        "token_embd.weight"
    };
    GGUF_TensorInfo info;
    if (!gguf_find_tensor_any(gf, candidates, sizeof(candidates)/sizeof(candidates[0]), info)) {
        gguf_close(gf);
        return false;
    }

    Tensor* t = nullptr;
    if (gguf_read_tensor(gf, info, t)) {
        float minv, maxv;
        double mean;
        gguf_tensor_stats(t, minv, maxv, mean);
        fprintf(stderr, "[gguf] loaded lm_head from '%s' rows=%u cols=%u type=%s min=%f max=%f mean=%f\n",
            info.name.c_str(), info.rows, info.cols, info.dtype.c_str(), minv, maxv, mean);
        outWout = t;
    }
    gguf_close(gf);
    return (outWout != nullptr);
}

const char* gguf_last_tensor_name(const char* tag) {
    if (!tag) return nullptr;
    if (strcmp(tag, "Wq") == 0) return gguf_last_name_Wq.empty() ? nullptr : gguf_last_name_Wq.c_str();
    if (strcmp(tag, "Wk") == 0) return gguf_last_name_Wk.empty() ? nullptr : gguf_last_name_Wk.c_str();
    if (strcmp(tag, "Wv") == 0) return gguf_last_name_Wv.empty() ? nullptr : gguf_last_name_Wv.c_str();
    return nullptr;
}
size_t gguf_last_tensor_rows(const char* tag) {
    if (!tag) return 0;
    if (strcmp(tag, "Wq") == 0) return gguf_last_rows_Wq;
    if (strcmp(tag, "Wk") == 0) return gguf_last_rows_Wk;
    if (strcmp(tag, "Wv") == 0) return gguf_last_rows_Wv;
    return 0;
}
size_t gguf_last_tensor_cols(const char* tag) {
    if (!tag) return 0;
    if (strcmp(tag, "Wq") == 0) return gguf_last_cols_Wq;
    if (strcmp(tag, "Wk") == 0) return gguf_last_cols_Wk;
    if (strcmp(tag, "Wv") == 0) return gguf_last_cols_Wv;
    return 0;
}
