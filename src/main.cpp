#include <cstdio>
#include "minxfmr.h"
#include "tensor/tensor.h"
#include "backend/backend_runtime.h"
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
#include <cctype>
#include "transformer/attention.h"
#ifdef _WIN32
#include <windows.h>
#endif
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

static bool read_chat_line_utf8(std::string& out) {
    out.clear();
#ifdef _WIN32
    HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    if (hIn != INVALID_HANDLE_VALUE && GetConsoleMode(hIn, &mode)) {
        wchar_t wbuf[1024];
        DWORD nread = 0;
        if (!ReadConsoleW(hIn, wbuf, (DWORD)(sizeof(wbuf) / sizeof(wbuf[0]) - 1), &nread, nullptr)) {
            return false;
        }
        if (nread == 0) return false;
        std::wstring ws(wbuf, wbuf + nread);
        while (!ws.empty() && (ws.back() == L'\r' || ws.back() == L'\n')) ws.pop_back();
        int bytes = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
        if (bytes <= 0) {
            out.clear();
            return true;
        }
        out.resize((size_t)bytes);
        WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), &out[0], bytes, nullptr, nullptr);
        return true;
    }
#endif
    char line[1024];
    if (!fgets(line, sizeof(line), stdin)) return false;
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[--len] = '\0';
    }
    out.assign(line, len);
    return true;
}

// model token output should go to stdout only (clean for chat). Logs go to stderr.
static void write_stdout_bytes(const char* token) {
    if (!token) return;
#ifdef _WIN32
    // If stdout is attached to a Windows console, write wide chars so
    // UTF-8 is displayed correctly. Otherwise fall back to fwrite.
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (hOut != INVALID_HANDLE_VALUE && GetConsoleMode(hOut, &mode)) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, token, -1, NULL, 0);
        if (wlen > 0) {
            std::wstring wbuf;
            wbuf.resize((size_t)wlen - 1);
            MultiByteToWideChar(CP_UTF8, 0, token, -1, &wbuf[0], wlen);
            DWORD written = 0;
            WriteConsoleW(hOut, wbuf.c_str(), (DWORD)wbuf.size(), &written, NULL);
            return;
        }
    }
    // Fallback to byte-wise write (works for pipes/redirection).
    std::fwrite(token, 1, std::strlen(token), stdout);
    std::fflush(stdout);
#else
    std::fwrite(token, 1, std::strlen(token), stdout);
    std::fflush(stdout);
#endif
}

static void print_callback(const char* token) {
    write_stdout_bytes(token);
}

// temporary buffer used to collect generated tokens during chat turn
static std::string gen_outbuf_global;
static void gen_collect_callback(const char* token) {
    if (!token) return;
    gen_outbuf_global.append(token);
    // also print to stdout so user sees streaming
    write_stdout_bytes(token);
}

static bool chat_debug_enabled() {
    const char* v = std::getenv("MINXFMR_CHAT_DEBUG");
    if (!v || !v[0]) return false;
    const char c = v[0];
    return c == '1' || c == 'y' || c == 'Y' || c == 't' || c == 'T';
}

static std::string utf8_truncate_for_history(const std::string& text, size_t max_bytes) {
    if (text.size() <= max_bytes) return text;
    size_t end = max_bytes;
    while (end > 0 && (((unsigned char)text[end]) & 0xC0) == 0x80) --end;
    if (end == 0) return std::string();
    return text.substr(0, end);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    std::string model = "dummy.gguf";
    std::string prompt = "Hello world";
    bool test_weights = false;
    bool debug_attn_once = false;
    int projection_layer = 0;
    bool chat_mode = false;
    const char* system_prompt = nullptr;
    int max_history = 12;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 8;
    int max_gen_tokens = 128;
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
    bool emit_vocab = false;
    bool temp_set_by_user = false;
    bool topk_set_by_user = false;
    bool run_once = false;
    bool show_help = false;
        int argi = 1;
        while (argi < argc) {
            const char* a = argv[argi];
            if (strcmp(a, "--test-weights") == 0) { test_weights = true; argi++; continue; }
            if (strcmp(a, "--prompt") == 0 && argi+1 < argc) { prompt = argv[argi+1]; run_once = true; argi += 2; continue; }
            if (strcmp(a, "--model") == 0 && argi+1 < argc) { model = argv[argi+1]; argi += 2; continue; }
            if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) { show_help = true; argi++; continue; }
            // positional args: model and prompt
            if (a[0] != '-') {
                if (model == "dummy.gguf") { model = a; argi++; continue; }
                if (prompt == "Hello world") { prompt = a; argi++; continue; }
            }
            if (strcmp(a, "--layer") == 0 && argi+1 < argc) { projection_layer = (int)strtol(argv[argi+1], nullptr, 10); argi += 2; continue; }
            if (strcmp(a, "--debug-attn") == 0) { debug_attn_once = true; argi++; continue; }
            if (strcmp(a, "--chat") == 0) { chat_mode = true; argi++; continue; }
            if (strcmp(a, "--system") == 0 && argi+1 < argc) { system_prompt = argv[argi+1]; argi += 2; continue; }
            if (strcmp(a, "--max-history") == 0 && argi+1 < argc) { max_history = (int)strtol(argv[argi+1], nullptr, 10); argi += 2; continue; }
            if (strcmp(a, "--temp") == 0 && argi+1 < argc) { temperature = (float)strtod(argv[argi+1], nullptr); temp_set_by_user = true; argi += 2; continue; }
            if (strcmp(a, "--top_p") == 0 && argi+1 < argc) { top_p = (float)strtod(argv[argi+1], nullptr); argi += 2; continue; }
            if (strcmp(a, "--top_k") == 0 && argi+1 < argc) { top_k = (int)strtol(argv[argi+1], nullptr, 10); topk_set_by_user = true; argi += 2; continue; }
            if (strcmp(a, "--max-gen-tokens") == 0 && argi+1 < argc) { max_gen_tokens = (int)strtol(argv[argi+1], nullptr, 10); argi += 2; continue; }
            if (strcmp(a, "--stop") == 0 && argi+1 < argc) { stop_token = argv[argi+1]; argi += 2; continue; }
            if (strcmp(a, "--log-file") == 0 && argi+1 < argc) { log_file = argv[argi+1]; argi += 2; continue; }
            if (strcmp(a, "--selftest") == 0) { run_selftest = true; argi++; continue; }
            if (strcmp(a, "--transpose-square") == 0) { transpose_square = true; transpose_wq = transpose_wk = transpose_wv = transpose_wo = true; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--no-transpose-square") == 0) { transpose_square = false; transpose_wq = transpose_wk = transpose_wv = transpose_wo = false; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--transpose-wq") == 0) { transpose_wq = true; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--no-transpose-wq") == 0) { transpose_wq = false; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--transpose-wk") == 0) { transpose_wk = true; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--no-transpose-wk") == 0) { transpose_wk = false; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--transpose-wv") == 0) { transpose_wv = true; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--no-transpose-wv") == 0) { transpose_wv = false; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--transpose-wo") == 0) { transpose_wo = true; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--no-transpose-wo") == 0) { transpose_wo = false; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--transpose-all") == 0) { transpose_square = true; transpose_wq = transpose_wk = transpose_wv = transpose_wo = true; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--no-transpose-all") == 0) { transpose_square = false; transpose_wq = transpose_wk = transpose_wv = transpose_wo = false; transpose_user_override = true; argi++; continue; }
            if (strcmp(a, "--emit-vocab") == 0) { emit_vocab = true; argi++; continue; }
            // Unknown or unsupported option: skip it
            argi++;
        }

    if (show_help) {
        printf("minxfmr_cli - simple CLI for minXfmr\n");
        printf("Usage: minxfmr_cli [MODEL] [PROMPT] [OPTIONS]\n");
        printf("\nOptions:\n");
        printf("  --model <path>        Specify path to model (.gguf). Overrides positional MODEL.\n");
        printf("  --prompt <text>       Run single prompt and exit (non-interactive).\n");
        printf("  --chat                Enter chat mode (interactive).\n");
        printf("  --temp <n>            Sampling temperature (default 0.7 in chat).\n");
        printf("  --top_k <n>           Top-k sampling parameter.\n");
        printf("  --max-gen-tokens <n>  Maximum tokens to generate (1-256).\n");
        printf("  --help, -h            Show this help message.\n");
        printf("\nYou can also set environment variables such as MINXFMR_BACKEND, MINXFMR_GGUF_VERBOSE.\n");
        return 0;
    }

    if (chat_mode) {
        // Chat defaults should remain stable but not fully deterministic;
        // pure greedy often gets trapped into short/repetitive fragments.
        if (!temp_set_by_user) temperature = 0.7f;
        if (!topk_set_by_user) top_k = 40;
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

    const bool env_transpose_override = []() {
        const char* v = std::getenv("MINXFMR_TRANSPOSE_USER_OVERRIDE");
        return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
    }();
    if (max_gen_tokens < 1) max_gen_tokens = 1;
    if (max_gen_tokens > 256) max_gen_tokens = 256;
#ifdef _WIN32
    {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", max_gen_tokens);
        _putenv_s("MINXFMR_MAX_GEN_TOKENS", buf);
        if (transpose_user_override) {
            _putenv_s("MINXFMR_TRANSPOSE_USER_OVERRIDE", "1");
            _putenv_s("MINXFMR_TRANSPOSE_WQ", transpose_wq ? "1" : "0");
            _putenv_s("MINXFMR_TRANSPOSE_WK", transpose_wk ? "1" : "0");
            _putenv_s("MINXFMR_TRANSPOSE_WV", transpose_wv ? "1" : "0");
            _putenv_s("MINXFMR_TRANSPOSE_WO", transpose_wo ? "1" : "0");
        }
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
        }
    }
#endif
    if (transpose_user_override) {
        fprintf(stderr, "[main] square-weight transpose mode (user): wq=%s wk=%s wv=%s wo=%s\n",
            transpose_wq ? "on" : "off",
            transpose_wk ? "on" : "off",
            transpose_wv ? "on" : "off",
            transpose_wo ? "on" : "off");
    } else if (env_transpose_override) {
        fprintf(stderr, "[main] square-weight transpose mode (user env): using MINXFMR_TRANSPOSE_* overrides\n");
    } else {
        fprintf(stderr, "[main] square-weight transpose mode: auto (by model architecture)\n");
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

            backend_initialize_from_env();
            if (backend_matmul(A,B,C)) {
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

        // Phase2b: rhs-transposed matmul test (used by square-weight projection paths)
        Tensor* AT = tensor_create_f32(2,3);
        Tensor* BT = tensor_create_f32(2,3);
        Tensor* CT = tensor_create_f32(2,2);
        if (AT && BT && CT) {
            tensor_set_f32(AT,0,0,1.0f); tensor_set_f32(AT,0,1,2.0f); tensor_set_f32(AT,0,2,3.0f);
            tensor_set_f32(AT,1,0,4.0f); tensor_set_f32(AT,1,1,5.0f); tensor_set_f32(AT,1,2,6.0f);
            tensor_set_f32(BT,0,0,7.0f); tensor_set_f32(BT,0,1,8.0f); tensor_set_f32(BT,0,2,9.0f);
            tensor_set_f32(BT,1,0,10.0f); tensor_set_f32(BT,1,1,11.0f); tensor_set_f32(BT,1,2,12.0f);
            if (backend_matmul_rhs_transposed(AT, BT, CT)) {
                printf("rhs_transposed result:\n");
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        printf(" %f", tensor_get_f32(CT, i, j));
                    }
                    printf("\n");
                }
            }
            tensor_free(AT); tensor_free(BT); tensor_free(CT);
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
        KVCache* cache = kvcache_create(2, 8, 4); // 2 layers, seq_max=8, dim=4
        if (cache) {
            float krow[4] = {0.1f,0.2f,0.3f,0.4f};
            float vrow[4] = {1.1f,1.2f,1.3f,1.4f};
            kvcache_append(cache, 0, krow, vrow);
            kvcache_append(cache, 0, krow, vrow);
            if (cache->lengths[0] == 2) printf("kvcache ok\n");
            kvcache_reset(cache);
            kvcache_free(cache);
        }
        return 0;
    }

    minxfmr_context* ctx = minxfmr_open_with_layer(model.c_str(), projection_layer);
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

    auto sanitize_assistant_text = [](const std::string& text) {
        static const std::vector<std::string> markers = {
            "<s>", "</s>", "[INST]", "[/INST]", "<|assistant|>", "<|user|>",
            "<assistant>", "<user>", "[/ASSISTANT]", "[/USER]", "<<SYS>>", "<</SYS>>", "speaker",
            "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"
        };
        std::string out;
        out.reserve(text.size());
        for (size_t i = 0; i < text.size(); ) {
            bool matched = false;
            for (const std::string& marker : markers) {
                if (text.compare(i, marker.size(), marker) == 0) {
                    i += marker.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) out.push_back(text[i++]);
        }
        return out;
    };

    auto looks_like_qwen_jinja_template = [](const char* tpl) {
        if (!tpl || tpl[0] == '\0') return false;
        return std::strstr(tpl, "<|im_start|>") != nullptr &&
               (std::strstr(tpl, "{%-") != nullptr || std::strstr(tpl, "{{-") != nullptr || std::strstr(tpl, "messages[") != nullptr);
    };

    auto build_history_blob = [&](const std::vector<std::string>& history) {
        std::string history_blob;
        size_t begin = history.size() > (size_t)max_history ? history.size() - (size_t)max_history : 0;
        for (size_t i = begin; i + 1 < history.size(); i += 2) {
            history_blob += history[i];
            history_blob += "\n";
            history_blob += history[i + 1];
            history_blob += "\n";
        }
        return history_blob;
    };

    auto render_template_auto = [&](const std::vector<std::string>& history, const char* template_text, const char* system_text, const char* user_text, bool* used_auto_qwen) {
        if (used_auto_qwen) *used_auto_qwen = false;

        auto append_qwen_message = [&](std::string& prompt_text, const char* role, const std::string& content) {
            prompt_text += "<|im_start|>";
            prompt_text += role;
            prompt_text += '\n';
            prompt_text += content;
            prompt_text += "<|im_end|>\n";
        };

        if (template_text && template_text[0] != '\0') {
            std::string assembled = template_text;
            auto replace_all = [&](const std::string& key, const std::string& val) {
                size_t pos = 0;
                while ((pos = assembled.find(key, pos)) != std::string::npos) {
                    assembled.replace(pos, key.size(), val);
                    pos += val.size();
                }
            };

            if (assembled.find("{{SYSTEM}}") != std::string::npos || assembled.find("{{HISTORY}}") != std::string::npos || assembled.find("{{USER}}") != std::string::npos) {
                replace_all("{{SYSTEM}}", system_text ? system_text : "");
                replace_all("{{HISTORY}}", build_history_blob(history));
                replace_all("{{USER}}", user_text ? user_text : "");
                return assembled;
            }

            if (looks_like_qwen_jinja_template(template_text)) {
                if (used_auto_qwen) *used_auto_qwen = true;
                std::string prompt_text;
                const char* sys = (system_text && system_text[0] != '\0')
                    ? system_text
                    : "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Reply in the same language as the user.";
                append_qwen_message(prompt_text, "system", sys);
                size_t begin = history.size() > (size_t)max_history ? history.size() - (size_t)max_history : 0;
                for (size_t i = begin; i + 1 < history.size(); i += 2) {
                    append_qwen_message(prompt_text, "user", history[i]);
                    append_qwen_message(prompt_text, "assistant", history[i + 1]);
                }
                prompt_text += "<|im_start|>user\n";
                prompt_text += (user_text ? user_text : "");
                prompt_text += "<|im_end|>\n";
                prompt_text += "<|im_start|>assistant\n";
                return prompt_text;
            }

            return assembled;
        }

        std::string assembled;
        assembled += "<s>[INST] ";
        if (system_text && system_text[0] != '\0') {
            assembled += "<<SYS>>\n";
            assembled += system_text;
            assembled += "\n<</SYS>>\n\n";
        }
        size_t begin = history.size() > (size_t)max_history ? history.size() - (size_t)max_history : 0;
        for (size_t i = begin; i + 1 < history.size(); i += 2) {
            assembled += history[i];
            assembled += " [/INST] ";
            assembled += history[i + 1];
            assembled += " </s><s>[INST] ";
        }
        assembled += (user_text ? user_text : "");
        assembled += " [/INST]";
        return assembled;
    };

            if (run_once && !chat_mode) {
                if (debug_attn_once) attention_set_debug_once(true);
                minxfmr_generate(ctx, prompt.c_str(), print_callback, temperature, top_k);
                printf("\n");
        } else if (chat_mode) {
            printf("Entering chat mode. Type 'reset' to clear history, 'exit' to quit.\n");
            std::vector<std::string> history; // alternating user/assistant entries
            while (true) {
                printf("you> ");
                fflush(stdout);
                std::string line;
                if (!read_chat_line_utf8(line)) break;
                if (line == "exit") break;
                if (line == "reset") { history.clear(); minxfmr_reset(ctx); printf("history reset\n"); continue; }
                if (line.empty()) continue;
                if ((history.size() % 2) != 0) {
                    if (chat_debug_enabled()) {
                        fprintf(stderr, "[main] dropping stray history entry to preserve user/assistant pairing\n");
                    }
                    history.pop_back();
                }
                bool used_auto_qwen = false;
                std::string assembled = render_template_auto(history, model_chat_template, system_prompt, line.c_str(), &used_auto_qwen);
                if (used_auto_qwen && chat_debug_enabled()) {
                    fprintf(stderr, "[main] auto-detected qwen-style chat template\n");
                }
                if (chat_debug_enabled()) {
                    std::vector<int> dbg_ids = tokenizer_encode(assembled);
                    fprintf(stderr, "[main] assembled prompt token count=%zu\n", dbg_ids.size());
                    if (!dbg_ids.empty()) {
                        fprintf(stderr, "[main] token ids:");
                        for (size_t ii=0; ii<dbg_ids.size(); ++ii) fprintf(stderr, " %d", dbg_ids[ii]);
                        fprintf(stderr, "\n");
                        std::string dbg_dec = tokenizer_decode(dbg_ids);
                        fprintf(stderr, "[main] decoded assembled prompt: %s\n", dbg_dec.c_str());
                    }
                }
                gen_outbuf_global.clear();
                printf("assistant> ");
                if (debug_attn_once) attention_set_debug_once(true);
                minxfmr_reset(ctx);
                minxfmr_generate(ctx, assembled.c_str(), gen_collect_callback, temperature, top_k);
                printf("\n");

                std::string sanitized = sanitize_assistant_text(gen_outbuf_global);
                if (!sanitized.empty()) {
                    sanitized = utf8_truncate_for_history(sanitized, 320);
                    history.push_back(line);
                    history.push_back(sanitized);
                } else {
                    if (chat_debug_enabled()) {
                        fprintf(stderr, "[main] skipping history store due to empty assistant text\n");
                    }
                }
                
            }
        } else {
            printf("Entering interactive mode. Type 'reset' to clear cache, 'exit' to quit.\n");
            std::string line;
            while (true) {
                printf("prompt> ");
                fflush(stdout);
                if (!read_chat_line_utf8(line)) break;
                if (line == "exit") break;
                if (line == "reset") { minxfmr_reset(ctx); printf("context reset\n"); continue; }
                if (line.empty()) continue;
                minxfmr_generate(ctx, line.c_str(), print_callback, temperature, top_k);
                printf("\n");
            }
        }

    minxfmr_close(ctx);
    return 0;
}
