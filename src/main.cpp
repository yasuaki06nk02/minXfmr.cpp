#include <cstdio>
#include "minxfmr.h"
#include "tensor/tensor.h"
#include "backend/cpu/cpu_backend.h"
#include "transformer/rmsnorm.h"
#include "transformer/rope.h"
#include "transformer/attention.h"
#include "transformer/feed_forward.h"
#include "transformer/transformer.h"
#include "tokenizer/tokenizer.h"
#include "cache/kv_cache.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "transformer/attention.h"
#ifdef _WIN32
#include <windows.h>
#endif

// model token output should go to stdout only (clean for chat). Logs go to stderr.
static void print_callback(const char* token) {
    if (!token) return;
    printf("%s", token);
    fflush(stdout);
}

// temporary buffer used to collect generated tokens during chat turn
static std::string gen_outbuf_global;
static void gen_collect_callback(const char* token) {
    if (!token) return;
    gen_outbuf_global.append(token);
    // also print to stdout so user sees streaming
    printf("%s", token);
    fflush(stdout);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    const char* model = "dummy.gguf";
    const char* prompt = "Hello world";
    bool test_weights = false;
    bool debug_attn_once = false;
    int projection_layer = 0;
    bool chat_mode = false;
    const char* system_prompt = nullptr;
    int max_history = 6;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 8;
    int max_gen_tokens = 24;
    const char* stop_token = nullptr;
    const char* log_file = nullptr;
    bool run_selftest = false;
    // Defaults updated to the verified best mask (mask=0): no transposes
    bool transpose_square = false;
    bool transpose_wq = false;
    bool transpose_wk = false;
    bool transpose_wv = false;
    bool transpose_wo = false;
    bool transpose_user_override = false;
    bool try_all_templates = false;
    bool emit_vocab = false;
    if (argc > 1) model = argv[1];
    if (argc > 2) prompt = argv[2];
        bool run_once = false;
        for (int i=1;i<argc;i++) {
            if (strcmp(argv[i], "--test-weights")==0) test_weights = true;
            if (strcmp(argv[i], "--prompt")==0 && i+1<argc) { prompt = argv[i+1]; run_once = true; }
            if (strcmp(argv[i], "--layer")==0 && i+1<argc) {
                projection_layer = (int)strtol(argv[i+1], nullptr, 10);
            }
            if (strcmp(argv[i], "--debug-attn")==0) {
                debug_attn_once = true;
            }
            if (strcmp(argv[i], "--chat")==0) {
                chat_mode = true;
            }
            if (strcmp(argv[i], "--system")==0 && i+1<argc) {
                system_prompt = argv[i+1];
            }
            if (strcmp(argv[i], "--max-history")==0 && i+1<argc) {
                max_history = (int)strtol(argv[i+1], nullptr, 10);
            }
            if (strcmp(argv[i], "--temp")==0 && i+1<argc) {
                temperature = (float)strtod(argv[i+1], nullptr);
            }
            if (strcmp(argv[i], "--top_p")==0 && i+1<argc) {
                top_p = (float)strtod(argv[i+1], nullptr);
            }
            if (strcmp(argv[i], "--top_k")==0 && i+1<argc) {
                top_k = (int)strtol(argv[i+1], nullptr, 10);
            }
            if (strcmp(argv[i], "--max-gen-tokens")==0 && i+1<argc) {
                max_gen_tokens = (int)strtol(argv[i+1], nullptr, 10);
            }
            if (strcmp(argv[i], "--stop")==0 && i+1<argc) {
                stop_token = argv[i+1];
            }
            if (strcmp(argv[i], "--log-file")==0 && i+1<argc) {
                log_file = argv[i+1];
            }
            if (strcmp(argv[i], "--selftest")==0) {
                run_selftest = true;
            }
            if (strcmp(argv[i], "--transpose-square")==0) {
                transpose_square = true;
                transpose_wq = transpose_wk = transpose_wv = transpose_wo = true;
                transpose_user_override = true;
            }
            if (strcmp(argv[i], "--no-transpose-square")==0) {
                transpose_square = false;
                transpose_wq = transpose_wk = transpose_wv = transpose_wo = false;
                transpose_user_override = true;
            }
            if (strcmp(argv[i], "--transpose-wq")==0) { transpose_wq = true; transpose_user_override = true; }
            if (strcmp(argv[i], "--no-transpose-wq")==0) { transpose_wq = false; transpose_user_override = true; }
            if (strcmp(argv[i], "--transpose-wk")==0) { transpose_wk = true; transpose_user_override = true; }
            if (strcmp(argv[i], "--no-transpose-wk")==0) { transpose_wk = false; transpose_user_override = true; }
            if (strcmp(argv[i], "--transpose-wv")==0) { transpose_wv = true; transpose_user_override = true; }
            if (strcmp(argv[i], "--no-transpose-wv")==0) { transpose_wv = false; transpose_user_override = true; }
            if (strcmp(argv[i], "--transpose-wo")==0) { transpose_wo = true; transpose_user_override = true; }
            if (strcmp(argv[i], "--no-transpose-wo")==0) { transpose_wo = false; transpose_user_override = true; }
            if (strcmp(argv[i], "--transpose-all")==0) {
                transpose_square = true;
                transpose_wq = transpose_wk = transpose_wv = transpose_wo = true;
                transpose_user_override = true;
            }
            if (strcmp(argv[i], "--no-transpose-all")==0) {
                transpose_square = false;
                transpose_wq = transpose_wk = transpose_wv = transpose_wo = false;
                transpose_user_override = true;
            }
            if (strcmp(argv[i], "--try-all-templates")==0) {
                try_all_templates = true;
            }
            if (strcmp(argv[i], "--emit-vocab")==0) {
                emit_vocab = true;
            }
        }

    if (log_file) {
        FILE* lf = freopen(log_file, "w", stderr);
        if (!lf) {
            fprintf(stderr, "failed to open log file: %s\n", log_file);
        }
    }

    // Ensure tokenizer has a baseline vocab even when selftest is off.
    tokenizer_load_from_list({"Hello","world","I","am","fine","today","how","are","you","?",".","<unk>"});

    if (transpose_user_override) {
        transformer_set_transpose_square_weights_for_all(transpose_wq, transpose_wk, transpose_wv, transpose_wo);
    }
    if (max_gen_tokens < 1) max_gen_tokens = 1;
    if (max_gen_tokens > 256) max_gen_tokens = 256;
#ifdef _WIN32
    {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", max_gen_tokens);
        _putenv_s("MINXFMR_MAX_GEN_TOKENS", buf);
        _putenv_s("MINXFMR_TRANSPOSE_USER_OVERRIDE", transpose_user_override ? "1" : "");
        _putenv_s("MINXFMR_TRANSPOSE_WQ", transpose_user_override ? (transpose_wq ? "1" : "0") : "");
        _putenv_s("MINXFMR_TRANSPOSE_WK", transpose_user_override ? (transpose_wk ? "1" : "0") : "");
        _putenv_s("MINXFMR_TRANSPOSE_WV", transpose_user_override ? (transpose_wv ? "1" : "0") : "");
        _putenv_s("MINXFMR_TRANSPOSE_WO", transpose_user_override ? (transpose_wo ? "1" : "0") : "");
    }
#else
    {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", max_gen_tokens);
        setenv("MINXFMR_MAX_GEN_TOKENS", buf, 1);
        if (transpose_user_override) {
            setenv("MINXFMR_TRANSPOSE_USER_OVERRIDE", "1", 1);
            setenv("MINXFMR_TRANSPOSE_WQ", transpose_wq ? "1" : "0", 1);
            setenv("MINXFMR_TRANSPOSE_WK", transpose_wk ? "1" : "0", 1);
            setenv("MINXFMR_TRANSPOSE_WV", transpose_wv ? "1" : "0", 1);
            setenv("MINXFMR_TRANSPOSE_WO", transpose_wo ? "1" : "0", 1);
        } else {
            unsetenv("MINXFMR_TRANSPOSE_USER_OVERRIDE");
            unsetenv("MINXFMR_TRANSPOSE_WQ");
            unsetenv("MINXFMR_TRANSPOSE_WK");
            unsetenv("MINXFMR_TRANSPOSE_WV");
            unsetenv("MINXFMR_TRANSPOSE_WO");
        }
    }
#endif
    if (transpose_user_override) {
        fprintf(stderr, "[main] square-weight transpose mode (user): wq=%s wk=%s wv=%s wo=%s\n",
            transpose_wq ? "on" : "off",
            transpose_wk ? "on" : "off",
            transpose_wv ? "on" : "off",
            transpose_wo ? "on" : "off");
    } else {
        fprintf(stderr, "[main] square-weight transpose mode: auto (by model architecture)\n");
    }

    minxfmr_context* ctx = minxfmr_open_with_layer(model, projection_layer);
    if (!ctx) return 1;

    // retrieve optional chat template and special tokens from model
    const char* model_chat_template = minxfmr_get_chat_template(ctx);
    std::vector<std::string> model_specials;
    size_t scnt = minxfmr_get_special_tokens_count(ctx);
    for (size_t i = 0; i < scnt; ++i) {
        const char* s = minxfmr_get_special_token(ctx, i);
        if (s) model_specials.emplace_back(s);
    }
    if (!model_specials.empty()) {
        tokenizer_add_special_tokens(model_specials);
        fprintf(stderr, "[main] registered %zu model special tokens into tokenizer\n", model_specials.size());
    }

    if (test_weights) {
        minxfmr_print_weights(ctx);
    }

    if (emit_vocab) {
        size_t n = tokenizer_vocab_size();
        // simple JSON escape
        auto json_escape = [&](const std::string &s) {
            std::string o;
            o.reserve(s.size()*2);
            for (unsigned char c : s) {
                if (c == '"') { o += "\\\""; }
                else if (c == '\\') { o += "\\\\"; }
                else if (c >= 0x20 && c <= 0x7E) { o.push_back((char)c); }
                else {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04X", (unsigned int)c);
                    o += buf;
                }
            }
            return o;
        };
        printf("[");
        for (size_t i = 0; i < n; ++i) {
            if (i) printf(",");
            std::string tok = tokenizer_id_to_token((int)i);
            std::string esc = json_escape(tok);
            printf("\"%s\"", esc.c_str());
        }
        printf("]\n");
        return 0;
    }

    if (run_selftest) {
    // Phase1: simple tensor test
    Tensor* t = tensor_create_f32(2, 3);
    if (t) {
        tensor_set_f32(t, 0, 0, 1.5f);
        tensor_set_f32(t, 1, 2, 3.25f);
        printf("tensor[0,0]=%f\n", tensor_get_f32(t,0,0));
        printf("tensor[1,2]=%f\n", tensor_get_f32(t,1,2));
        tensor_free(t);
    }

    // Phase2: CPU backend test (matrix multiply)
    Tensor* A = tensor_create_f32(2,2);
    Tensor* B = tensor_create_f32(2,2);
    Tensor* C = tensor_create_f32(2,2);
    if (A && B && C) {
        tensor_set_f32(A,0,0,1.0f); tensor_set_f32(A,0,1,2.0f);
        tensor_set_f32(A,1,0,3.0f); tensor_set_f32(A,1,1,4.0f);

        tensor_set_f32(B,0,0,5.0f); tensor_set_f32(B,0,1,6.0f);
        tensor_set_f32(B,1,0,7.0f); tensor_set_f32(B,1,1,8.0f);

        if (cpu_matmul(A,B,C)) {
            printf("matmul result:\n");
            for (size_t i=0;i<2;++i) {
                for (size_t j=0;j<2;++j) {
                    printf(" %f", tensor_get_f32(C,i,j));
                }
                printf("\n");
            }
        }
        tensor_free(A); tensor_free(B); tensor_free(C);
    }

    // Phase3: Transformer component smoke test
    // create a small seq=2, dim=4 token hidden matrix
    Tensor* hidden = tensor_create_f32(2,4);
    Tensor* hidden_out = tensor_create_f32(2,4);
    if (hidden && hidden_out) {
        // fill
        for (size_t i=0;i<2;i++) for (size_t j=0;j<4;j++) tensor_set_f32(hidden,i,j, (float)(i*4 + j + 1));
        // RMSNorm
        if (rmsnorm_forward(hidden, hidden_out)) {
            printf("rmsnorm ok\n");
        }
        // Attention QK scores
        Tensor* scores = tensor_create_f32(2,2);
        if (scores && attention_qk(hidden, hidden, scores)) {
            printf("attention qk ok\n");
        }
        // FFN: use W = identity 4x4, b = zeros 1x4, out = seq x 4
        Tensor* W = tensor_create_f32(4,4);
        Tensor* b = tensor_create_f32(1,4);
        Tensor* ffn_out = tensor_create_f32(2,4);
        if (W && b && ffn_out) {
            for (size_t i=0;i<4;i++) for (size_t j=0;j<4;j++) tensor_set_f32(W,i,j, (i==j)?1.0f:0.0f);
            for (size_t j=0;j<4;j++) tensor_set_f32(b,0,j,0.0f);
            if (ffn_forward(hidden, W, b, ffn_out)) printf("ffn ok\n");
        }
        tensor_free(scores);
        tensor_free(W); tensor_free(b); tensor_free(ffn_out);
        tensor_free(hidden); tensor_free(hidden_out);
    }

    // Phase4: single-layer transformer forward
    Tensor* in = tensor_create_f32(2,4);
    Tensor* out = tensor_create_f32(2,4);
    if (in && out) {
        for (size_t i=0;i<2;i++) for (size_t j=0;j<4;j++) tensor_set_f32(in,i,j,(float)(i*4 + j + 1));
        if (transformer_forward_single_layer(in,out, nullptr, 0, 1, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)) {
            printf("transformer single layer ok\n");
        }
        tensor_free(in); tensor_free(out);
    }

    // Phase5: Tokenizer test
    tokenizer_load_from_list({"Hello","world","Do","you","remember","my","name","<unk>"});
    std::vector<int> ids = tokenizer_encode("Hello world unknown_token");
    printf("token ids:");
    for (int id: ids) printf(" %d", id);
    printf("\n");

    printf("decoded: %s\n", tokenizer_decode(ids).c_str());

    // KV Cache test
    printf("-- before kvcache create\n");
    KVCache* cache = kvcache_create(2, 8, 4); // 2 layers, seq_max=8, dim=4
    printf("-- after kvcache create\n");
    if (!cache) {
        printf("kvcache create failed\n");
    }
    if (cache) {
        float krow[4] = {0.1f,0.2f,0.3f,0.4f};
        float vrow[4] = {1.1f,1.2f,1.3f,1.4f};
        kvcache_append(cache, 0, krow, vrow);
        kvcache_append(cache, 0, krow, vrow);
        printf("kvcache layer0 length=%zu\n", cache->lengths[0]);
        kvcache_reset(cache);
        printf("kvcache layer0 after reset=%zu\n", cache->lengths[0]);
        kvcache_free(cache);
    }
    }

            if (run_once && !chat_mode) {
                if (debug_attn_once) attention_set_debug_once(true);
                minxfmr_generate(ctx, prompt, print_callback, temperature, top_k);
                printf("\n");
        } else if (chat_mode) {
            printf("Entering chat mode. Type 'reset' to clear history, 'exit' to quit.\n");
            std::vector<std::string> history; // alternating user/assistant entries
            char line[1024];
            while (true) {
                printf("you> ");
                fflush(stdout);
                if (!fgets(line, sizeof(line), stdin)) break;
                size_t L = strlen(line); if (L>0 && line[L-1]=='\n') line[L-1]='\0';
                if (strcmp(line, "exit") == 0) break;
                if (strcmp(line, "reset") == 0) { history.clear(); minxfmr_reset(ctx); printf("history reset\n"); continue; }
                if (strlen(line) == 0) continue;
                std::string assembled;
                if (model_chat_template && model_chat_template[0] != '\0') {
                    // build history blob
                    std::string history_blob;
                    size_t begin = history.size() > (size_t)max_history ? history.size() - (size_t)max_history : 0;
                    for (size_t i = begin; i + 1 < history.size(); i += 2) {
                        history_blob += history[i];
                        history_blob += "\n";
                        history_blob += history[i+1];
                        history_blob += "\n";
                    }
                    // simple placeholder replacement for {{SYSTEM}},{{HISTORY}},{{USER}}
                    assembled = model_chat_template;
                    auto replace_all = [&](const std::string& key, const std::string& val) {
                        size_t pos = 0;
                        while ((pos = assembled.find(key, pos)) != std::string::npos) {
                            assembled.replace(pos, key.size(), val);
                            pos += val.size();
                        }
                    };
                    replace_all("{{SYSTEM}}", system_prompt ? system_prompt : "");
                    replace_all("{{HISTORY}}", history_blob);
                    replace_all("{{USER}}", line);
                    // debug: log assembled prompt tokens
                    std::vector<int> dbg_ids = tokenizer_encode(assembled);
                    fprintf(stderr, "[main] assembled prompt token count=%zu\n", dbg_ids.size());
                    if (!dbg_ids.empty()) {
                        fprintf(stderr, "[main] token ids:");
                        for (size_t ii=0; ii<dbg_ids.size(); ++ii) fprintf(stderr, " %d", dbg_ids[ii]);
                        fprintf(stderr, "\n");
                        std::string dbg_dec = tokenizer_decode(dbg_ids);
                        fprintf(stderr, "[main] decoded assembled prompt: %s\n", dbg_dec.c_str());
                    }
                } else {
                    assembled += "<s>[INST] ";
                    if (system_prompt && system_prompt[0] != '\0') {
                        assembled += "<<SYS>>\n";
                        assembled += system_prompt;
                        assembled += "\n<</SYS>>\n\n";
                    }
                    size_t begin = history.size() > (size_t)max_history ? history.size() - (size_t)max_history : 0;
                    for (size_t i = begin; i + 1 < history.size(); i += 2) {
                        assembled += history[i];
                        assembled += " [/INST] ";
                        assembled += history[i + 1];
                        assembled += " </s><s>[INST] ";
                    }
                    assembled += line;
                    assembled += " [/INST]";
                }
                // If model doesn't provide a template, default to single deterministic template.
                // Optional debug mode can still run all candidate templates.
                if (!(model_chat_template && model_chat_template[0] != '\0') && try_all_templates) {
                    std::vector<std::string> candidates = {
                        "<s>[INST] {{SYSTEM}}\n{{HISTORY}}{{USER}} [/INST]",
                        "[INST]{{SYSTEM}}\n{{HISTORY}}{{USER}}[/INST]",
                        "<s>{{SYSTEM}}\n{{HISTORY}}User: {{USER}}\nAssistant:\n",
                        "<s>[INST] {{USER}} [/INST]",
                    };
                    for (size_t ci=0; ci<candidates.size(); ++ci) {
                        std::string tpl = candidates[ci];
                        auto replace_all_local = [&](const std::string& key, const std::string& val) {
                            size_t pos = 0; while ((pos = tpl.find(key, pos)) != std::string::npos) { tpl.replace(pos, key.size(), val); pos += val.size(); }
                        };
                        size_t begin = history.size() > (size_t)max_history ? history.size() - (size_t)max_history : 0;
                        std::string history_blob;
                        for (size_t i = begin; i + 1 < history.size(); i += 2) { history_blob += history[i]; history_blob += "\n"; history_blob += history[i+1]; history_blob += "\n"; }
                        replace_all_local("{{SYSTEM}}", system_prompt ? system_prompt : "");
                        replace_all_local("{{HISTORY}}", history_blob);
                        replace_all_local("{{USER}}", line);
                        fprintf(stderr, "[main] trying template %zu => '%s'\n", ci, tpl.c_str());
                        // log tokens
                        std::vector<int> dbg_ids = tokenizer_encode(tpl);
                        fprintf(stderr, "[main] tpl %zu token count=%zu\n", ci, dbg_ids.size());
                        fprintf(stderr, "[main] tpl %zu ids:", ci);
                        for (size_t ii=0; ii<dbg_ids.size(); ++ii) fprintf(stderr, " %d", dbg_ids[ii]);
                        fprintf(stderr, "\n");
                        // run generation for this template
                        gen_outbuf_global.clear();
                        printf("assistant(template %zu)> ", ci);
                        minxfmr_generate(ctx, tpl.c_str(), gen_collect_callback, temperature, top_k);
                        printf("\n");
                        fprintf(stderr, "[main] tpl %zu output: %s\n", ci, gen_outbuf_global.c_str());
                    }
                    // push only the raw line into history as before
                    history.push_back(line);
                    history.push_back(gen_outbuf_global);
                    continue;
                }
                gen_outbuf_global.clear();
                printf("assistant> ");
                if (debug_attn_once) attention_set_debug_once(true);
                minxfmr_generate(ctx, assembled.c_str(), gen_collect_callback, temperature, top_k);
                printf("\n");
                history.push_back(line);
                history.push_back(gen_outbuf_global);
                
            }
        } else {
            printf("Entering interactive mode. Type 'reset' to clear cache, 'exit' to quit.\n");
            char line[256];
            while (true) {
                printf("prompt> ");
                fflush(stdout);
                if (!fgets(line, sizeof(line), stdin)) break;
                // trim newline
                size_t L = strlen(line); if (L>0 && line[L-1]=='\n') line[L-1]='\0';
                if (strcmp(line, "exit") == 0) break;
                if (strcmp(line, "reset") == 0) { minxfmr_reset(ctx); printf("context reset\n"); continue; }
                if (strlen(line) == 0) continue;
                minxfmr_generate(ctx, line, print_callback, 1.0, 3);
                printf("\n");
            }
        }

    minxfmr_close(ctx);
    return 0;
}
