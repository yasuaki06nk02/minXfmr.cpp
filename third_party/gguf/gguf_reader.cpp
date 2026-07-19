#include "gguf_reader.h"
#include <cstdio>
#include <cstring>
#include <iostream>

static bool rd_u32(const std::vector<uint8_t>& d, size_t& p, uint32_t& out) {
    if (p + 4 > d.size()) return false;
    out = (uint32_t)d[p] | ((uint32_t)d[p+1] << 8) | ((uint32_t)d[p+2] << 16) | ((uint32_t)d[p+3] << 24);
    p += 4;
    return true;
}

static bool rd_u64(const std::vector<uint8_t>& d, size_t& p, uint64_t& out) {
    if (p + 8 > d.size()) return false;
    out = 0;
    for (int i = 0; i < 8; ++i) out |= ((uint64_t)d[p+i]) << (8*i);
    p += 8;
    return true;
}

static bool rd_str(const std::vector<uint8_t>& d, size_t& p, std::string& s) {
    uint64_t n = 0;
    if (!rd_u64(d, p, n)) return false;
    if (p + n > d.size()) return false;
    s.assign((const char*)d.data() + p, (size_t)n);
    p += (size_t)n;
    return true;
}

static bool skip_gguf_value(const std::vector<uint8_t>& d, size_t& p, uint32_t t) {
    // gguf metadata scalar/array value types
    // 0:u8 1:i8 2:u16 3:i16 4:u32 5:i32 6:f32 7:bool 8:string 9:array 10:u64 11:i64 12:f64
    auto skip_n = [&](size_t n)->bool { if (p + n > d.size()) return false; p += n; return true; };
    switch (t) {
        case 0: case 1: case 7: return skip_n(1);
        case 2: case 3: return skip_n(2);
        case 4: case 5: case 6: return skip_n(4);
        case 10: case 11: case 12: return skip_n(8);
        case 8: {
            std::string tmp;
            return rd_str(d, p, tmp);
        }
        case 9: {
            uint32_t elem_t = 0;
            uint64_t n = 0;
            if (!rd_u32(d, p, elem_t)) return false;
            if (!rd_u64(d, p, n)) return false;
            for (uint64_t i = 0; i < n; ++i) {
                if (!skip_gguf_value(d, p, elem_t)) return false;
            }
            return true;
        }
        default:
            return false;
    }
}

static const char* type_name(uint32_t t) {
    switch (t) {
        case 0: return "f32";
        case 1: return "f16";
        case 2: return "q4_0";
        case 3: return "q4_1";
        case 6: return "q5_0";
        case 7: return "q5_1";
        case 8: return "q8_0";
        case 9: return "q8_1";
        case 10: return "q2_k";
        case 11: return "q3_k";
        case 12: return "q4_k";
        case 13: return "q5_k";
        case 14: return "q6_k";
        case 15: return "q8_k";
        case 16: return "iq2_xxs";
        case 17: return "iq2_xs";
        case 18: return "iq3_xxs";
        case 19: return "iq1_s";
        case 20: return "iq4_nl";
        case 21: return "iq3_s";
        case 22: return "iq2_s";
        case 23: return "iq4_xs";
        case 24: return "i8";
        case 25: return "i16";
        case 26: return "i32";
        case 27: return "i64";
        case 28: return "f64";
        case 29: return "iq1_m";
        case 30: return "bf16";
        case 34: return "tq1_0";
        case 35: return "tq2_0";
        case 39: return "mxfp4";
        case 40: return "nvfp4";
        case 41: return "q1_0";
        case 42: return "q2_0";
        default: return "unknown";
    }
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t f = h & 0x3ff;
    uint32_t out;
    if (e == 0) {
        if (f == 0) {
            out = s << 31;
        } else {
            e = 1;
            while ((f & 0x400) == 0) { f <<= 1; --e; }
            f &= 0x3ff;
            out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7f800000 | (f << 13);
    } else {
        out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
    }
    float v;
    memcpy(&v, &out, sizeof(v));
    return v;
}

// NOTE: This is a simplified GGUF reader. It does not fully implement GGUF.
// It heuristically scans for tensor entries and supports reading float tensors
// and a basic Q4_K_M dequant path (adapted from open-source implementations).

bool gguf_open(const char* path, GGUF_File& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return false; }
    rewind(f);
    out.data.resize(sz);
    if ((long)fread(out.data.data(), 1, sz, f) != sz) { fclose(f); return false; }
    fclose(f);
    out.path = path;
    out.tensors.clear();
    out.vocab_tokens.clear();
    out.n_layer = 0;
    out.n_ctx = 0;
    out.n_embd = 0;
    out.n_head = 0;
    out.n_head_kv = 0;
    out.architecture.clear();

    size_t p = 0;
    if (out.data.size() < 24) return false;
    if (!(out.data[0] == 'G' && out.data[1] == 'G' && out.data[2] == 'U' && out.data[3] == 'F')) return false;
    p = 4;

    uint32_t version = 0;
    if (!rd_u32(out.data, p, version)) return false;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!rd_u64(out.data, p, n_tensors)) return false;
    if (!rd_u64(out.data, p, n_kv)) return false;

    uint32_t alignment = 32;
    auto read_meta_u64 = [&](uint32_t t, size_t at, uint64_t& v)->bool {
        if (t == 4) { // u32
            if (at + 4 > out.data.size()) return false;
            v = (uint64_t)out.data[at] | ((uint64_t)out.data[at+1] << 8) | ((uint64_t)out.data[at+2] << 16) | ((uint64_t)out.data[at+3] << 24);
            return true;
        }
        if (t == 5) { // i32
            if (at + 4 > out.data.size()) return false;
            int32_t sv = (int32_t)((uint32_t)out.data[at] | ((uint32_t)out.data[at+1] << 8) | ((uint32_t)out.data[at+2] << 16) | ((uint32_t)out.data[at+3] << 24));
            if (sv < 0) return false;
            v = (uint64_t)sv;
            return true;
        }
        if (t == 10) { // u64
            if (at + 8 > out.data.size()) return false;
            uint64_t vv = 0;
            for (int b = 0; b < 8; ++b) vv |= ((uint64_t)out.data[at+b]) << (8*b);
            v = vv;
            return true;
        }
        if (t == 11) { // i64
            if (at + 8 > out.data.size()) return false;
            int64_t sv = 0;
            for (int b = 0; b < 8; ++b) sv |= ((int64_t)out.data[at+b]) << (8*b);
            if (sv < 0) return false;
            v = (uint64_t)sv;
            return true;
        }
        return false;
    };

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t t = 0;
        if (!rd_str(out.data, p, key)) return false;
        if (!rd_u32(out.data, p, t)) return false;
        size_t before = p;

        // Special-case tokenizer vocabulary array and related metadata.
        if (t == 9) {
            uint32_t elem_t = 0;
            uint64_t n = 0;
            if (!rd_u32(out.data, p, elem_t)) return false;
            if (!rd_u64(out.data, p, n)) return false;
            if (key == "tokenizer.ggml.tokens" && elem_t == 8) {
                out.vocab_tokens.clear();
                out.vocab_tokens.reserve((size_t)n);
                for (uint64_t k = 0; k < n; ++k) {
                    std::string tok;
                    if (!rd_str(out.data, p, tok)) return false;
                    out.vocab_tokens.push_back(tok);
                }
            } else if (key == "tokenizer.ggml.scores" && (elem_t == 6 || elem_t == 12)) {
                out.vocab_scores.clear();
                out.vocab_scores.reserve((size_t)n);
                for (uint64_t k = 0; k < n; ++k) {
                    if (elem_t == 6) {
                        if (p + 4 > out.data.size()) return false;
                        float v;
                        memcpy(&v, out.data.data() + p, 4);
                        p += 4;
                        out.vocab_scores.push_back(v);
                    } else {
                        if (p + 8 > out.data.size()) return false;
                        double dv;
                        memcpy(&dv, out.data.data() + p, 8);
                        p += 8;
                        out.vocab_scores.push_back((float)dv);
                    }
                }
            } else if (key == "tokenizer.ggml.token_type" && (elem_t == 0 || elem_t == 1 || elem_t == 2 || elem_t == 3 || elem_t == 4 || elem_t == 5 || elem_t == 10 || elem_t == 11)) {
                out.vocab_types.clear();
                out.vocab_types.reserve((size_t)n);
                for (uint64_t k = 0; k < n; ++k) {
                    if (elem_t == 0) {
                        if (p + 1 > out.data.size()) return false;
                        uint8_t v = out.data[p];
                        p += 1;
                        out.vocab_types.push_back((int)v);
                    } else if (elem_t == 1) {
                        if (p + 1 > out.data.size()) return false;
                        int8_t sv = (int8_t)out.data[p];
                        p += 1;
                        out.vocab_types.push_back((int)sv);
                    } else if (elem_t == 2) {
                        if (p + 2 > out.data.size()) return false;
                        uint16_t v = (uint16_t)out.data[p] | ((uint16_t)out.data[p+1] << 8);
                        p += 2;
                        out.vocab_types.push_back((int)v);
                    } else if (elem_t == 3) {
                        if (p + 2 > out.data.size()) return false;
                        int16_t sv = (int16_t)((uint16_t)out.data[p] | ((uint16_t)out.data[p+1] << 8));
                        p += 2;
                        out.vocab_types.push_back((int)sv);
                    } else if (elem_t == 4) {
                        if (p + 4 > out.data.size()) return false;
                        uint32_t v = (uint32_t)out.data[p] | ((uint32_t)out.data[p+1] << 8) | ((uint32_t)out.data[p+2] << 16) | ((uint32_t)out.data[p+3] << 24);
                        p += 4;
                        out.vocab_types.push_back((int)v);
                    } else if (elem_t == 5) {
                        if (p + 4 > out.data.size()) return false;
                        int32_t sv = (int32_t)((uint32_t)out.data[p] | ((uint32_t)out.data[p+1] << 8) | ((uint32_t)out.data[p+2] << 16) | ((uint32_t)out.data[p+3] << 24));
                        p += 4;
                        out.vocab_types.push_back((int)sv);
                    } else if (elem_t == 10) {
                        if (p + 8 > out.data.size()) return false;
                        uint64_t v = 0;
                        for (int b = 0; b < 8; ++b) v |= ((uint64_t)out.data[p+b]) << (8*b);
                        p += 8;
                        out.vocab_types.push_back((int)v);
                    } else if (elem_t == 11) {
                        if (p + 8 > out.data.size()) return false;
                        int64_t sv = 0;
                        for (int b = 0; b < 8; ++b) sv |= ((int64_t)out.data[p+b]) << (8*b);
                        p += 8;
                        out.vocab_types.push_back((int)sv);
                    }
                }
            } else if ((key == "special_tokens" || key == "tokenizer.special_tokens") && elem_t == 8) {
                out.special_tokens.clear();
                out.special_tokens.reserve((size_t)n);
                for (uint64_t k = 0; k < n; ++k) {
                    std::string tok;
                    if (!rd_str(out.data, p, tok)) return false;
                    out.special_tokens.push_back(tok);
                }
            } else {
                for (uint64_t k = 0; k < n; ++k) {
                    if (!skip_gguf_value(out.data, p, elem_t)) return false;
                }
            }
        } else {
            if (!skip_gguf_value(out.data, p, t)) return false;
        }

        uint64_t mv = 0;
        if (read_meta_u64(t, before, mv)) {
            // support multiple common metadata key names for portability
            if (key == "llama.block_count" || key == "num_layers" || key == "model.num_layers") {
                out.n_layer = mv;
            } else if (key == "llama.context_length" || key == "context_length") {
                out.n_ctx = mv;
            } else if (key == "llama.embedding_length" || key == "hidden_size") {
                out.n_embd = mv;
            } else if (key == "llama.attention.head_count" || key == "num_heads") {
                out.n_head = mv;
            } else if (key == "llama.attention.head_count_kv" || key == "num_heads_kv") {
                out.n_head_kv = mv;
            } else if (key == "llama.feed_forward_length" || key == "intermediate_size" || key == "feed_forward_length" || key == "ffn_size") {
                out.n_intermediate = mv;
            } else if (key == "vocab_size" || key == "tokenizer.vocab_size") {
                out.vocab_size = mv;
            }
        }

        if (key == "general.alignment") {
            // best effort: read scalar values from bytes we skipped
            if (t == 4 && before + 4 <= out.data.size()) {
                uint32_t v = (uint32_t)out.data[before] | ((uint32_t)out.data[before+1] << 8) | ((uint32_t)out.data[before+2] << 16) | ((uint32_t)out.data[before+3] << 24);
                if (v > 0 && v <= 4096) alignment = v;
            } else if (t == 10 && before + 8 <= out.data.size()) {
                uint64_t v = 0;
                for (int b = 0; b < 8; ++b) v |= ((uint64_t)out.data[before+b]) << (8*b);
                if (v > 0 && v <= 4096) alignment = (uint32_t)v;
            }
        }
        // Try to capture chat template strings if present
        if ((key == "chat_template" || key == "general.chat_template" || key == "model.chat_template") && t == 8) {
            // we already skipped/consumed the string at 'before', re-parse it
            size_t tmp_p = before;
            std::string tmp;
            if (rd_str(out.data, tmp_p, tmp)) {
                out.chat_template = tmp;
            }
        }
        if ((key == "general.architecture" || key == "architecture" || key == "model.architecture") && t == 8) {
            size_t tmp_p = before;
            std::string tmp;
            if (rd_str(out.data, tmp_p, tmp)) {
                out.architecture = tmp;
            }
        }
    }

    struct TmpT {
        std::string name;
        uint64_t ne[4] = {1,1,1,1};
        uint32_t n_dims = 0;
        uint32_t type = 0;
        uint64_t rel_off = 0;
        uint64_t nbytes = 0;
    };
    std::vector<TmpT> tmp;
    tmp.reserve((size_t)n_tensors);

    for (uint64_t i = 0; i < n_tensors; ++i) {
        TmpT t;
        if (!rd_str(out.data, p, t.name)) return false;
        if (!rd_u32(out.data, p, t.n_dims)) return false;
        if (t.n_dims == 0 || t.n_dims > 4) return false;
        for (uint32_t j = 0; j < t.n_dims; ++j) {
            if (!rd_u64(out.data, p, t.ne[j])) return false;
        }
        if (!rd_u32(out.data, p, t.type)) return false;
        if (!rd_u64(out.data, p, t.rel_off)) return false;

        uint64_t n = 1;
        for (uint32_t j = 0; j < t.n_dims; ++j) n *= t.ne[j];
        // minimal byte size estimate per type
        uint64_t bpe = 0;
        switch (t.type) {
            case 0: bpe = 4; break; // f32
            case 1: bpe = 2; break; // f16
            case 8: bpe = 1; break; // q8_0 packed (under-estimate check is handled later)
            case 2: bpe = 1; break; // q4_0 packed
            case 12: bpe = 1; break; // q4_k packed
            default: bpe = 1; break;
        }
        t.nbytes = n * bpe;
        tmp.push_back(t);
    }

    size_t tensor_data = p;
    if (alignment > 1) {
        size_t rem = tensor_data % alignment;
        if (rem) tensor_data += (alignment - rem);
    }
    if (tensor_data > out.data.size()) return false;

    for (const auto& t : tmp) {
        GGUF_TensorInfo ti;
        ti.name = t.name;
        ti.ggml_type = t.type;
        ti.dtype = type_name(t.type);
        ti.cols = (uint32_t)t.ne[0];
        ti.rows = (uint32_t)((t.n_dims >= 2) ? t.ne[1] : 1);
        ti.offset = tensor_data + (size_t)t.rel_off;
        ti.nbytes = t.nbytes;
        if (ti.offset < out.data.size()) out.tensors.push_back(ti);
    }

    // if vocab array was present, prefer its length as vocab_size
    if (out.vocab_size == 0 && !out.vocab_tokens.empty()) {
        out.vocab_size = out.vocab_tokens.size();
    }

    return true;
}

void gguf_close(GGUF_File& f) {
    f.data.clear(); f.tensors.clear(); f.path.clear();
    f.n_layer = f.n_ctx = f.n_embd = f.n_head = f.n_head_kv = 0;
    f.architecture.clear();
    f.n_intermediate = 0;
    f.vocab_size = 0;
    f.vocab_tokens.clear();
    f.vocab_scores.clear();
    f.vocab_types.clear();
    f.chat_template.clear();
    f.special_tokens.clear();
}

bool gguf_read_model_config(const char* path, GGUF_ModelConfig& out) {
    out = GGUF_ModelConfig{};
    GGUF_File f;
    if (!gguf_open(path, f)) return false;
    // canonical fields
    out.n_layer = f.n_layer;
    out.n_ctx = f.n_ctx;
    out.n_embd = f.n_embd;
    out.n_head = f.n_head;
    out.n_head_kv = f.n_head_kv;

    // populate synonym fields for downstream tooling
    out.context_length = f.n_ctx;
    out.hidden_size = f.n_embd;
    out.num_layers = f.n_layer;
    out.intermediate_size = f.n_intermediate;
    out.num_heads = f.n_head;
    out.vocab_size = f.vocab_size;
    gguf_close(f);
    return true;
}

bool gguf_read_vocab(const char* path, std::vector<std::string>& out_tokens) {
    out_tokens.clear();
    GGUF_File f;
    if (!gguf_open(path, f)) return false;
    out_tokens = f.vocab_tokens;
    gguf_close(f);
    return !out_tokens.empty();
}

bool gguf_read_architecture(const char* path, std::string& out_architecture) {
    out_architecture.clear();
    GGUF_File f;
    if (!gguf_open(path, f)) return false;
    out_architecture = f.architecture;
    gguf_close(f);
    return !out_architecture.empty();
}

bool gguf_find_tensor(const GGUF_File& f, const char* name, GGUF_TensorInfo& out) {
    const std::string target(name);
    for (const auto& t : f.tensors) {
        if (t.name == target || t.name.find(target) != std::string::npos) {
            out = t;
            return true;
        }
    }
    return false;
}

bool gguf_find_tensor_any(const GGUF_File& f, const char* const* names, size_t count, GGUF_TensorInfo& out) {
    for (size_t i = 0; i < count; ++i) {
        if (gguf_find_tensor(f, names[i], out)) return true;
    }
    return false;
}

bool gguf_read_f32_tensor(const GGUF_File& f, const GGUF_TensorInfo& info, Tensor*& out) {
    out = nullptr;
    if (info.offset >= f.data.size()) return false;
    const size_t n = (size_t)info.rows * info.cols;
    out = tensor_create_f32(info.rows, info.cols);
    if (!out) return false;

    if (info.ggml_type == 0) {
        const size_t need = n * sizeof(float);
        if (info.offset + need > f.data.size()) { tensor_free(out); out = nullptr; return false; }
        memcpy(out->data, f.data.data() + info.offset, need);
        return true;
    }
    if (info.ggml_type == 1) {
        const size_t need = n * sizeof(uint16_t);
        if (info.offset + need > f.data.size()) { tensor_free(out); out = nullptr; return false; }
        const uint16_t* src = (const uint16_t*)(f.data.data() + info.offset);
        float* dst = (float*)out->data;
        for (size_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(src[i]);
        return true;
    }

    tensor_free(out);
    out = nullptr;
    return false;
}

bool gguf_dequant_q4_k_m(const GGUF_File& f, const GGUF_TensorInfo& info, Tensor*& out) {
    out = nullptr;

    // Q4_K_M tensors are encoded as GGML_TYPE_Q4_K blocks.
    if (info.ggml_type != 12) return false;

    const uint32_t rows = info.rows;
    const uint32_t cols = info.cols;

    // q4_k block operates on 256 values.
    static const uint32_t QK_K = 256;
    static const size_t BLOCK_SIZE = 2 + 2 + 12 + (QK_K / 2); // fp16 d, fp16 dmin, scales[12], qs[128]

    if (rows == 0 || cols == 0 || (cols % QK_K) != 0) {
        return false;
    }

    const size_t blocks_per_row = cols / QK_K;
    const size_t total_blocks = (size_t)rows * blocks_per_row;
    const size_t need = total_blocks * BLOCK_SIZE;
    if (info.offset + need > f.data.size()) {
        return false;
    }

    out = tensor_create_f32(rows, cols);
    if (!out) return false;

    auto rd_f16_at = [&](const uint8_t* p) -> float {
        uint16_t h;
        memcpy(&h, p, sizeof(h));
        return fp16_to_fp32(h);
    };

    auto get_scale_min_k4 = [](int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
        if (j < 4) {
            d = q[j] & 63;
            m = q[j + 4] & 63;
        } else {
            d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
        }
    };

    const uint8_t* src = f.data.data() + info.offset;
    float* dst = (float*)out->data;

    for (uint32_t r = 0; r < rows; ++r) {
        float* row_dst = dst + (size_t)r * cols;
        const uint8_t* row_src = src + (size_t)r * blocks_per_row * BLOCK_SIZE;

        for (size_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_src + b * BLOCK_SIZE;

            const float d = rd_f16_at(blk + 0);
            const float dmin = rd_f16_at(blk + 2);
            const uint8_t* scales = blk + 4;
            const uint8_t* q = blk + 16;

            float* block_dst = row_dst + b * QK_K;

            int is = 0;
            for (int j = 0; j < (int)QK_K; j += 64) {
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, sc, m);
                const float d1 = d * sc;
                const float m1 = dmin * m;

                get_scale_min_k4(is + 1, scales, sc, m);
                const float d2 = d * sc;
                const float m2 = dmin * m;

                for (int l = 0; l < 32; ++l) {
                    block_dst[j + l] = d1 * (q[l] & 0xF) - m1;
                }
                for (int l = 0; l < 32; ++l) {
                    block_dst[j + 32 + l] = d2 * (q[l] >> 4) - m2;
                }

                q += 32;
                is += 2;
            }
        }
    }

    return true;
}

bool gguf_dequant_q6_k(const GGUF_File& f, const GGUF_TensorInfo& info, Tensor*& out) {
    out = nullptr;

    // GGML_TYPE_Q6_K
    if (info.ggml_type != 14) return false;

    const uint32_t rows = info.rows;
    const uint32_t cols = info.cols;

    static const uint32_t QK_K = 256;
    // ql[128], qh[64], scales[16], d(fp16)
    static const size_t BLOCK_SIZE = (QK_K / 2) + (QK_K / 4) + (QK_K / 16) + 2;

    if (rows == 0 || cols == 0 || (cols % QK_K) != 0) {
        return false;
    }

    const size_t blocks_per_row = cols / QK_K;
    const size_t total_blocks = (size_t)rows * blocks_per_row;
    const size_t need = total_blocks * BLOCK_SIZE;
    if (info.offset + need > f.data.size()) {
        return false;
    }

    out = tensor_create_f32(rows, cols);
    if (!out) return false;

    auto rd_f16_at = [&](const uint8_t* p) -> float {
        uint16_t h;
        memcpy(&h, p, sizeof(h));
        return fp16_to_fp32(h);
    };

    const uint8_t* src = f.data.data() + info.offset;
    float* dst = (float*)out->data;

    for (uint32_t r = 0; r < rows; ++r) {
        float* row_dst = dst + (size_t)r * cols;
        const uint8_t* row_src = src + (size_t)r * blocks_per_row * BLOCK_SIZE;

        for (size_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_src + b * BLOCK_SIZE;

            const uint8_t* ql = blk;
            const uint8_t* qh = blk + (QK_K / 2);
            const int8_t* scales = (const int8_t*)(blk + (QK_K / 2) + (QK_K / 4));
            const float d = rd_f16_at(blk + (QK_K / 2) + (QK_K / 4) + (QK_K / 16));

            float* block_dst = row_dst + b * QK_K;

            const uint8_t* ql_it = ql;
            const uint8_t* qh_it = qh;
            const int8_t* sc_it = scales;
            float* y = block_dst;

            for (int n = 0; n < (int)QK_K; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int8_t q1 = (int8_t)((ql_it[l + 0] & 0xF) | (((qh_it[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t)((ql_it[l + 32] & 0xF) | (((qh_it[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t)((ql_it[l + 0] >> 4) | (((qh_it[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t)((ql_it[l + 32] >> 4) | (((qh_it[l] >> 6) & 3) << 4)) - 32;

                    y[l + 0] = d * sc_it[is + 0] * q1;
                    y[l + 32] = d * sc_it[is + 2] * q2;
                    y[l + 64] = d * sc_it[is + 4] * q3;
                    y[l + 96] = d * sc_it[is + 6] * q4;
                }

                y += 128;
                ql_it += 64;
                qh_it += 32;
                sc_it += 8;
            }
        }
    }

    return true;
}

bool gguf_read_tensor(const GGUF_File& f, const GGUF_TensorInfo& info, Tensor*& out) {
    out = nullptr;
    // Try direct f32/f16 read
    if (gguf_read_f32_tensor(f, info, out)) return true;
    // Try known dequant paths
    if (gguf_dequant_q4_k_m(f, info, out)) return true;
    if (gguf_dequant_q6_k(f, info, out)) return true;
    // Unsupported type
    out = nullptr;
    return false;
}

void gguf_tensor_stats(const Tensor* t, float& minv, float& maxv, double& mean) {
    minv = 1e30f; maxv = -1e30f; mean = 0.0;
    size_t n = t->rows * t->cols;
    const float* d = (const float*)t->data;
    for (size_t i=0;i<n;i++) {
        float v = d[i]; if (v < minv) minv = v; if (v > maxv) maxv = v; mean += v;
    }
    if (n>0) mean /= (double)n;
}
