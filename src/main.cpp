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
#include "io/gguf_loader.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cctype>
#include "transformer/attention.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include "runtime_config.h"
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
    // Use byte-wise UTF-8 writes even when attached to the Windows console.
    // Mixing WriteConsoleW (wide API) with byte-based stderr/fprintf can
    // produce garbled/duplicated output on some consoles. The program sets
    // the console CP to UTF-8 earlier, so fwrite of UTF-8 bytes is reliable
    // and avoids the wide/byte interop issues.
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
    return RuntimeConfig::Instance().chat_debug();
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
    float min_p = 0.0f;
    int top_k = 8;
    int repeat_last_n = 64;
    float repeat_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
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
    bool topp_set_by_user = false;
    bool minp_set_by_user = false;
    bool topk_set_by_user = false;
    bool compare_logits = false;
    int compare_top_n = 10;
    int compare_steps = 3;
    bool run_once = false;
    bool show_help = false;
    bool dump_chat_template = false;
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
            if (strcmp(a, "--top_p") == 0 && argi+1 < argc) { top_p = (float)strtod(argv[argi+1], nullptr); topp_set_by_user = true; argi += 2; continue; }
            if ((strcmp(a, "--min_p") == 0 || strcmp(a, "--min-p") == 0) && argi+1 < argc) { min_p = (float)strtod(argv[argi+1], nullptr); minp_set_by_user = true; argi += 2; continue; }
            if (strcmp(a, "--top_k") == 0 && argi+1 < argc) { top_k = (int)strtol(argv[argi+1], nullptr, 10); topk_set_by_user = true; argi += 2; continue; }
            if (strcmp(a, "--repeat-last-n") == 0 && argi+1 < argc) { repeat_last_n = (int)strtol(argv[argi+1], nullptr, 10); argi += 2; continue; }
            if (strcmp(a, "--repeat-penalty") == 0 && argi+1 < argc) { repeat_penalty = (float)strtod(argv[argi+1], nullptr); argi += 2; continue; }
            if (strcmp(a, "--presence-penalty") == 0 && argi+1 < argc) { presence_penalty = (float)strtod(argv[argi+1], nullptr); argi += 2; continue; }
            if (strcmp(a, "--frequency-penalty") == 0 && argi+1 < argc) { frequency_penalty = (float)strtod(argv[argi+1], nullptr); argi += 2; continue; }
            if (strcmp(a, "--compare-logits") == 0) { compare_logits = true; argi++; continue; }
            if (strcmp(a, "--compare-top-n") == 0 && argi+1 < argc) { compare_top_n = (int)strtol(argv[argi+1], nullptr, 10); argi += 2; continue; }
            if (strcmp(a, "--compare-steps") == 0 && argi+1 < argc) { compare_steps = (int)strtol(argv[argi+1], nullptr, 10); argi += 2; continue; }
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
            if (strcmp(a, "--dump-chat-template") == 0) { dump_chat_template = true; argi++; continue; }
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
        printf("  --top_p <n>           Top-p nucleus sampling (default 0.95 in chat).\n");
        printf("  --min_p <n>           Min-p sampling (alias: --min-p, default 0.05 in chat).\n");
        printf("  --repeat-last-n <n>   Last n tokens used for repetition penalties (default 64, 0=off).\n");
        printf("  --repeat-penalty <n>  Repeat penalty (>1.0 stronger, 1.0=off).\n");
        printf("  --presence-penalty <n> Presence penalty (default 0.0).\n");
        printf("  --frequency-penalty <n> Frequency penalty (default 0.0).\n");
        printf("  --compare-logits      Emit JSONL compare traces for prompt and first steps.\n");
        printf("  --compare-top-n <n>   Number of top logits to dump per step (default 10).\n");
        printf("  --compare-steps <n>   Number of initial steps to dump (default 3).\n");
        printf("  --max-gen-tokens <n>  Maximum tokens to generate (1-256).\n");
        printf("  --dump-chat-template  Print tokenizer.chat_template loaded from GGUF and exit.\n");
        printf("  --help, -h            Show this help message.\n");
        printf("\nYou can also set environment variables such as MINXFMR_BACKEND, MINXFMR_GGUF_VERBOSE.\n");
        return 0;
    }

    if (chat_mode) {
        // Chat defaults should remain stable but not fully deterministic;
        // pure greedy often gets trapped into short/repetitive fragments.
        if (!temp_set_by_user) temperature = 0.7f;
        if (!topk_set_by_user) top_k = 40;
        if (!topp_set_by_user) top_p = 0.95f;
        // Keep min-p disabled by default to match common llama.cpp behavior.
        if (!minp_set_by_user) min_p = 0.0f;
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

    const bool env_transpose_override = RuntimeConfig::Instance().transpose_user_override();
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
    if (chat_debug_enabled()) {
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
    bool qwen_chat_fallback = false;
    {
        std::string arch;
        if (gguf_try_read_architecture(model.c_str(), arch)) {
            std::string arch_lc = arch;
            for (char& ch : arch_lc) ch = (char)std::tolower((unsigned char)ch);
            qwen_chat_fallback = (arch_lc.rfind("qwen", 0) == 0);
        }
        if (!qwen_chat_fallback) {
            std::string model_lc = model;
            for (char& ch : model_lc) ch = (char)std::tolower((unsigned char)ch);
            qwen_chat_fallback = (model_lc.find("qwen") != std::string::npos);
        }
        if ((model_chat_template == nullptr || model_chat_template[0] == '\0') && qwen_chat_fallback) {
            fprintf(stderr, "[main] chat_template missing in GGUF; using built-in Qwen fallback format\n");
        }
    }
    std::vector<std::string> model_specials;
    size_t scnt = minxfmr_get_special_tokens_count(ctx);
    for (size_t i = 0; i < scnt; ++i) {
        const char* s = minxfmr_get_special_token(ctx, i);
        if (s) model_specials.emplace_back(s);
    }
    if (!model_specials.empty()) {
        tokenizer_add_special_tokens(model_specials);
        if (chat_debug_enabled()) fprintf(stderr, "[main] registered %zu model special tokens into tokenizer\n", model_specials.size());
    }

    if (test_weights) {
        minxfmr_print_weights(ctx);
    }

    if (dump_chat_template) {
        if (model_chat_template && model_chat_template[0] != '\0') {
            const size_t n = std::strlen(model_chat_template);
            printf("# tokenizer.chat_template (source=gguf metadata, length=%zu)\n", n);
            printf("%s\n", model_chat_template);
        } else {
            printf("# tokenizer.chat_template not found in gguf metadata\n");
        }
        minxfmr_close(ctx);
        return 0;
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

    auto json_escape = [](const std::string& s) {
        std::string out;
        out.reserve(s.size() * 2);
        for (unsigned char c : s) {
            if (c == '"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else if (c >= 0x20 && c <= 0x7E) out.push_back((char)c);
            else {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04X", (unsigned int)c);
                out += buf;
            }
        }
        return out;
    };

    struct ChatTemplateMessage {
        std::string role;
        std::string content;
    };

    auto render_template_auto = [&](const std::vector<std::string>& history, const char* template_text, const char* system_text, const char* user_text, std::string* generation_prompt_out) {
        if (generation_prompt_out) generation_prompt_out->clear();

        auto trim_ws = [](const std::string& s) {
            size_t b = 0;
            while (b < s.size() && std::isspace((unsigned char)s[b])) ++b;
            size_t e = s.size();
            while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
            return s.substr(b, e - b);
        };

        auto build_messages = [&](const std::vector<std::string>& hist, const char* sys, const char* usr) {
            std::vector<ChatTemplateMessage> msgs;
            if (sys && sys[0] != '\0') {
                msgs.push_back(ChatTemplateMessage{"system", std::string(sys)});
            }
            size_t begin = hist.size() > (size_t)max_history ? hist.size() - (size_t)max_history : 0;
            for (size_t i = begin; i + 1 < hist.size(); i += 2) {
                msgs.push_back(ChatTemplateMessage{"user", hist[i]});
                msgs.push_back(ChatTemplateMessage{"assistant", hist[i + 1]});
            }
            msgs.push_back(ChatTemplateMessage{"user", usr ? std::string(usr) : std::string()});
            return msgs;
        };

        struct ListView {
            const std::vector<ChatTemplateMessage>* data = nullptr;
            size_t begin = 0;
            size_t end = 0;
        };

        struct Value {
            enum class Kind { None, Str, Bool, Int, Message, List } kind = Kind::None;
            std::string s;
            bool b = false;
            long long i = 0;
            ChatTemplateMessage msg;
            ListView list;

            static Value from_str(const std::string& v) { Value x; x.kind = Kind::Str; x.s = v; return x; }
            static Value from_bool(bool v) { Value x; x.kind = Kind::Bool; x.b = v; return x; }
            static Value from_int(long long v) { Value x; x.kind = Kind::Int; x.i = v; return x; }
            static Value from_msg(const ChatTemplateMessage& v) { Value x; x.kind = Kind::Message; x.msg = v; return x; }
            static Value from_list(const std::vector<ChatTemplateMessage>* p, size_t begin, size_t end) {
                Value x;
                x.kind = Kind::List;
                x.list.data = p;
                x.list.begin = begin;
                x.list.end = end;
                return x;
            }

            bool truthy() const {
                if (kind == Kind::Bool) return b;
                if (kind == Kind::Int) return i != 0;
                if (kind == Kind::Str) return !s.empty();
                if (kind == Kind::Message) return !msg.role.empty() || !msg.content.empty();
                if (kind == Kind::List) return list.data && list.begin < list.end;
                return false;
            }

            std::string to_string() const {
                if (kind == Kind::Str) return s;
                if (kind == Kind::Bool) return b ? "true" : "false";
                if (kind == Kind::Int) return std::to_string(i);
                if (kind == Kind::Message) return msg.content;
                return std::string();
            }
        };

        auto split_top_level = [&](const std::string& s, const std::string& sep) {
            std::vector<std::string> out;
            size_t start = 0;
            int bracket_depth = 0;
            bool in_single = false;
            bool in_double = false;
            for (size_t i = 0; i < s.size(); ++i) {
                char c = s[i];
                if (c == '\\' && (in_single || in_double) && i + 1 < s.size()) {
                    ++i;
                    continue;
                }
                if (!in_double && c == '\'') in_single = !in_single;
                else if (!in_single && c == '"') in_double = !in_double;
                else if (!in_single && !in_double) {
                    if (c == '[' || c == '(') ++bracket_depth;
                    else if ((c == ']' || c == ')') && bracket_depth > 0) --bracket_depth;
                    if (bracket_depth == 0 && i + sep.size() <= s.size() && s.compare(i, sep.size(), sep) == 0) {
                        out.push_back(s.substr(start, i - start));
                        i += sep.size() - 1;
                        start = i + 1;
                    }
                }
            }
            out.push_back(s.substr(start));
            return out;
        };

        auto parse_string_literal = [&](const std::string& expr, std::string* out) {
            if (expr.size() < 2) return false;
            char q = expr.front();
            if ((q != '\'' && q != '"') || expr.back() != q) return false;
            std::string v;
            v.reserve(expr.size());
            for (size_t i = 1; i + 1 < expr.size(); ++i) {
                char c = expr[i];
                if (c == '\\' && i + 1 < expr.size() - 1) {
                    char n = expr[++i];
                    if (n == 'n') v.push_back('\n');
                    else if (n == 't') v.push_back('\t');
                    else if (n == 'r') v.push_back('\r');
                    else v.push_back(n);
                } else {
                    v.push_back(c);
                }
            }
            *out = v;
            return true;
        };

        auto unwrap_outer_parens = [&](const std::string& raw) {
            std::string s = trim_ws(raw);
            while (s.size() >= 2 && s.front() == '(' && s.back() == ')') {
                int depth = 0;
                bool in_single = false;
                bool in_double = false;
                bool wraps_all = true;
                for (size_t i = 0; i < s.size(); ++i) {
                    char c = s[i];
                    if (c == '\\' && (in_single || in_double) && i + 1 < s.size()) {
                        ++i;
                        continue;
                    }
                    if (!in_double && c == '\'') in_single = !in_single;
                    else if (!in_single && c == '"') in_double = !in_double;
                    else if (!in_single && !in_double) {
                        if (c == '(') ++depth;
                        else if (c == ')') {
                            --depth;
                            if (depth == 0 && i + 1 < s.size()) {
                                wraps_all = false;
                                break;
                            }
                        }
                    }
                }
                if (!wraps_all) break;
                s = trim_ws(s.substr(1, s.size() - 2));
            }
            return s;
        };

        auto value_eq = [&](const Value& a, const Value& b) {
            if (a.kind == Value::Kind::Bool || b.kind == Value::Kind::Bool) return a.truthy() == b.truthy();
            return a.to_string() == b.to_string();
        };

        std::vector<ChatTemplateMessage> messages = build_messages(history, system_text, user_text);
        std::unordered_map<std::string, Value> vars;
        vars["messages"] = Value::from_list(&messages, 0, messages.size());
        vars["loop_messages"] = Value::from_list(&messages, 0, messages.size());
        vars["tools"] = Value::from_bool(false);
        vars["add_generation_prompt"] = Value::from_bool(true);
        vars["loop.first"] = Value::from_bool(false);
        vars["loop.last"] = Value::from_bool(false);
        vars["loop.index0"] = Value::from_int(0);

        std::function<Value(const std::string&)> eval_expr;
        std::function<bool(const std::string&)> eval_cond;

        auto parse_var_path = [&](const std::string& raw) -> Value {
            std::string s = unwrap_outer_parens(raw);
            if (s.empty()) return Value();

            if (s.rfind("loop.", 0) == 0) {
                auto lit = vars.find(s);
                return (lit != vars.end()) ? lit->second : Value();
            }

            size_t pos = 0;
            auto read_ident = [&](size_t p, size_t* next) {
                if (p >= s.size() || !(std::isalpha((unsigned char)s[p]) || s[p] == '_')) return std::string();
                size_t i = p + 1;
                while (i < s.size() && (std::isalnum((unsigned char)s[i]) || s[i] == '_')) ++i;
                *next = i;
                return s.substr(p, i - p);
            };

            size_t next = 0;
            std::string ident = read_ident(pos, &next);
            if (ident.empty()) return Value();
            pos = next;

            auto it = vars.find(ident);
            Value cur = (it != vars.end()) ? it->second : Value();

            while (pos < s.size()) {
                if (s[pos] == '.') {
                    ++pos;
                    std::string field = read_ident(pos, &next);
                    if (field.empty()) return Value();
                    pos = next;
                    if (cur.kind == Value::Kind::Message) {
                        if (field == "role") cur = Value::from_str(cur.msg.role);
                        else if (field == "content") cur = Value::from_str(cur.msg.content);
                        else if (field == "tool_calls") cur = Value();
                        else cur = Value();
                    } else {
                        cur = Value();
                    }
                    continue;
                }

                if (s[pos] == '[') {
                    size_t close = s.find(']', pos + 1);
                    if (close == std::string::npos) return Value();
                    std::string inner = trim_ws(s.substr(pos + 1, close - (pos + 1)));

                    if (cur.kind == Value::Kind::List && cur.list.data) {
                        size_t list_size = cur.list.end > cur.list.begin ? (cur.list.end - cur.list.begin) : 0;
                        size_t col = inner.find(':');

                        auto parse_index_expr = [&](const std::string& idx_expr, size_t* out_idx) {
                            std::string e = trim_ws(idx_expr);
                            if (e.empty()) return false;

                            auto parse_digits = [&](const std::string& d, long long* v) {
                                if (d.empty()) return false;
                                for (char ch : d) if (!std::isdigit((unsigned char)ch)) return false;
                                *v = std::strtoll(d.c_str(), nullptr, 10);
                                return true;
                            };

                            long long base = 0;
                            size_t op_pos = std::string::npos;
                            char op = 0;
                            for (size_t ii = 0; ii < e.size(); ++ii) {
                                if ((e[ii] == '+' || e[ii] == '-') && ii > 0) {
                                    op_pos = ii;
                                    op = e[ii];
                                    break;
                                }
                            }

                            if (op_pos != std::string::npos) {
                                std::string left = trim_ws(e.substr(0, op_pos));
                                std::string right = trim_ws(e.substr(op_pos + 1));
                                long long lv = 0;
                                long long rv = 0;
                                if (left == "loop.index0") {
                                    auto itl = vars.find("loop.index0");
                                    if (itl == vars.end() || itl->second.kind != Value::Kind::Int) return false;
                                    lv = itl->second.i;
                                } else if (!parse_digits(left, &lv)) {
                                    return false;
                                }
                                if (!parse_digits(right, &rv)) return false;
                                base = (op == '+') ? (lv + rv) : (lv - rv);
                            } else {
                                if (e == "loop.index0") {
                                    auto itl = vars.find("loop.index0");
                                    if (itl == vars.end() || itl->second.kind != Value::Kind::Int) return false;
                                    base = itl->second.i;
                                } else if (!parse_digits(e, &base)) {
                                    return false;
                                }
                            }

                            if (base < 0) return false;
                            *out_idx = (size_t)base;
                            return true;
                        };

                        if (col != std::string::npos) {
                            std::string a = trim_ws(inner.substr(0, col));
                            std::string b = trim_ws(inner.substr(col + 1));
                            size_t sb = 0;
                            size_t se = list_size;
                            if (!a.empty() && !parse_index_expr(a, &sb)) return Value();
                            if (!b.empty() && !parse_index_expr(b, &se)) return Value();
                            if (sb > list_size) sb = list_size;
                            if (se > list_size) se = list_size;
                            if (sb > se) sb = se;
                            cur = Value::from_list(cur.list.data, cur.list.begin + sb, cur.list.begin + se);
                        } else {
                            size_t idx = 0;
                            if (!parse_index_expr(inner, &idx)) return Value();
                            if (idx >= list_size) return Value();
                            cur = Value::from_msg((*cur.list.data)[cur.list.begin + idx]);
                        }
                    } else if (cur.kind == Value::Kind::Message) {
                        std::string key;
                        if (!parse_string_literal(inner, &key)) return Value();
                        if (key == "role") cur = Value::from_str(cur.msg.role);
                        else if (key == "content") cur = Value::from_str(cur.msg.content);
                        else cur = Value();
                    } else {
                        return Value();
                    }

                    pos = close + 1;
                    continue;
                }

                break;
            }

            return cur;
        };

        eval_expr = [&](const std::string& raw) -> Value {
            std::string expr = unwrap_outer_parens(raw);
            if (expr.empty()) return Value();

            if (expr == "true") return Value::from_bool(true);
            if (expr == "false") return Value::from_bool(false);

            auto pipes = split_top_level(expr, "|");
            Value v;
            if (pipes.size() > 1) {
                v = eval_expr(pipes[0]);
                for (size_t i = 1; i < pipes.size(); ++i) {
                    std::string f = trim_ws(pipes[i]);
                    if (f == "trim") {
                        std::string t = trim_ws(v.to_string());
                        v = Value::from_str(t);
                    } else if (f == "tojson") {
                        if (v.kind == Value::Kind::Str) {
                            v = Value::from_str(std::string("\"") + json_escape(v.s) + "\"");
                        } else if (v.kind == Value::Kind::Message) {
                            std::string obj = std::string("{\"role\":\"") + json_escape(v.msg.role) +
                                              "\",\"content\":\"" + json_escape(v.msg.content) + "\"}";
                            v = Value::from_str(obj);
                        } else {
                            v = Value::from_str(v.to_string());
                        }
                    }
                }
                return v;
            }

            auto plus = split_top_level(expr, "+");
            if (plus.size() > 1) {
                std::string out;
                for (const std::string& part : plus) out += eval_expr(part).to_string();
                return Value::from_str(out);
            }

            std::string lit;
            if (parse_string_literal(expr, &lit)) return Value::from_str(lit);

            return parse_var_path(expr);
        };

        eval_cond = [&](const std::string& raw) -> bool {
            std::string expr = unwrap_outer_parens(raw);
            if (expr.empty()) return false;

            auto ors = split_top_level(expr, " or ");
            if (ors.size() > 1) {
                for (const std::string& p : ors) if (eval_cond(p)) return true;
                return false;
            }

            auto ands = split_top_level(expr, " and ");
            if (ands.size() > 1) {
                for (const std::string& p : ands) if (!eval_cond(p)) return false;
                return true;
            }

            auto pos_defined = expr.find(" is defined");
            if (pos_defined != std::string::npos && trim_ws(expr.substr(pos_defined)) == "is defined") {
                Value v = eval_expr(expr.substr(0, pos_defined));
                return v.kind != Value::Kind::None;
            }

            size_t pos_ne = expr.find("!=");
            if (pos_ne != std::string::npos) {
                Value a = eval_expr(expr.substr(0, pos_ne));
                Value b = eval_expr(expr.substr(pos_ne + 2));
                return !value_eq(a, b);
            }

            size_t pos_eq = expr.find("==");
            if (pos_eq != std::string::npos) {
                Value a = eval_expr(expr.substr(0, pos_eq));
                Value b = eval_expr(expr.substr(pos_eq + 2));
                return value_eq(a, b);
            }

            if (expr.rfind("not ", 0) == 0) return !eval_cond(expr.substr(4));
            return eval_expr(expr).truthy();
        };

        struct Segment {
            enum class Kind { Text, Expr, Stmt } kind = Kind::Text;
            std::string text;
        };

        std::vector<Segment> segs;
        auto rstrip_ws = [](std::string& s) {
            while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
        };

        if (template_text && template_text[0] != '\0') {
            std::string tpl(template_text);
            size_t pos = 0;
            while (pos < tpl.size()) {
                size_t n_expr = tpl.find("{{", pos);
                size_t n_stmt = tpl.find("{%", pos);
                size_t n_cmt = tpl.find("{#", pos);
                size_t n = std::string::npos;
                if (n_expr != std::string::npos) n = n_expr;
                if (n_stmt != std::string::npos && (n == std::string::npos || n_stmt < n)) n = n_stmt;
                if (n_cmt != std::string::npos && (n == std::string::npos || n_cmt < n)) n = n_cmt;
                if (n == std::string::npos) {
                    segs.push_back(Segment{Segment::Kind::Text, tpl.substr(pos)});
                    break;
                }

                if (n > pos) segs.push_back(Segment{Segment::Kind::Text, tpl.substr(pos, n - pos)});

                if (tpl.compare(n, 2, "{#") == 0) {
                    size_t end = tpl.find("#}", n + 2);
                    if (end == std::string::npos) break;
                    pos = end + 2;
                    continue;
                }

                bool is_expr = tpl.compare(n, 2, "{{") == 0;
                bool ltrim = (n + 2 < tpl.size() && tpl[n + 2] == '-');
                size_t start_inner = n + (ltrim ? 3 : 2);
                std::string close = is_expr ? "}}" : "%}";
                size_t end = tpl.find(close, start_inner);
                if (end == std::string::npos) break;
                bool rtrim = (end > start_inner && tpl[end - 1] == '-');
                size_t end_inner = rtrim ? end - 1 : end;
                if (ltrim && !segs.empty() && segs.back().kind == Segment::Kind::Text) rstrip_ws(segs.back().text);

                std::string inner = trim_ws(tpl.substr(start_inner, end_inner - start_inner));
                segs.push_back(Segment{is_expr ? Segment::Kind::Expr : Segment::Kind::Stmt, inner});

                pos = end + 2;
                if (rtrim) {
                    while (pos < tpl.size() && std::isspace((unsigned char)tpl[pos])) ++pos;
                }
            }
        }

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

            if (!segs.empty() && (assembled.find("{{") != std::string::npos || assembled.find("{%") != std::string::npos)) {
                auto stmt_head = [&](const std::string& stmt) {
                    size_t p = 0;
                    while (p < stmt.size() && !std::isspace((unsigned char)stmt[p])) ++p;
                    return stmt.substr(0, p);
                };

                auto starts_with = [&](const std::string& s, const char* pfx) {
                    size_t n = std::strlen(pfx);
                    return s.size() >= n && s.compare(0, n, pfx) == 0;
                };

                std::function<void(size_t&, const std::vector<std::string>&)> skip_until;
                std::function<std::string(size_t&, const std::vector<std::string>&)> render_until;
                std::function<std::string(size_t&, const std::string&)> render_if;

                auto is_end_tag = [&](const std::string& kw, const std::vector<std::string>& ends) {
                    for (const std::string& e : ends) if (kw == e) return true;
                    return false;
                };

                skip_until = [&](size_t& idx, const std::vector<std::string>& ends) {
                    while (idx < segs.size()) {
                        const Segment& seg = segs[idx];
                        if (seg.kind != Segment::Kind::Stmt) {
                            ++idx;
                            continue;
                        }
                        std::string kw = stmt_head(seg.text);
                        if (is_end_tag(kw, ends)) return;
                        if (kw == "if") {
                            ++idx;
                            skip_until(idx, {"endif"});
                            if (idx < segs.size() && stmt_head(segs[idx].text) == "endif") ++idx;
                            continue;
                        }
                        if (kw == "for") {
                            ++idx;
                            skip_until(idx, {"endfor"});
                            if (idx < segs.size() && stmt_head(segs[idx].text) == "endfor") ++idx;
                            continue;
                        }
                        ++idx;
                    }
                };

                render_if = [&](size_t& idx, const std::string& first_cond) {
                    bool taken = false;
                    std::string out;
                    std::string cond = first_cond;

                    while (true) {
                        bool use_branch = !taken && eval_cond(cond);
                        if (use_branch) {
                            out += render_until(idx, {"elif", "else", "endif"});
                            taken = true;
                        } else {
                            skip_until(idx, {"elif", "else", "endif"});
                        }

                        if (idx >= segs.size()) break;
                        const std::string stmt = segs[idx].text;
                        const std::string kw = stmt_head(stmt);
                        if (kw == "elif") {
                            cond = trim_ws(stmt.substr(4));
                            ++idx;
                            continue;
                        }
                        if (kw == "else") {
                            ++idx;
                            if (!taken) out += render_until(idx, {"endif"});
                            else skip_until(idx, {"endif"});
                            if (idx < segs.size() && stmt_head(segs[idx].text) == "endif") ++idx;
                            break;
                        }
                        if (kw == "endif") {
                            ++idx;
                            break;
                        }
                        break;
                    }

                    return out;
                };

                render_until = [&](size_t& idx, const std::vector<std::string>& ends) {
                    std::string out;
                    while (idx < segs.size()) {
                        const Segment& seg = segs[idx];
                        if (seg.kind == Segment::Kind::Text) {
                            out += seg.text;
                            ++idx;
                            continue;
                        }
                        if (seg.kind == Segment::Kind::Expr) {
                            out += eval_expr(seg.text).to_string();
                            ++idx;
                            continue;
                        }

                        const std::string kw = stmt_head(seg.text);
                        if (is_end_tag(kw, ends)) break;

                        if (kw == "set") {
                            std::string body = trim_ws(seg.text.substr(3));
                            size_t eq = body.find('=');
                            if (eq != std::string::npos) {
                                std::string name = trim_ws(body.substr(0, eq));
                                std::string expr = trim_ws(body.substr(eq + 1));
                                vars[name] = eval_expr(expr);
                            }
                            ++idx;
                            continue;
                        }

                        if (kw == "if") {
                            std::string cond = trim_ws(seg.text.substr(2));
                            ++idx;
                            out += render_if(idx, cond);
                            continue;
                        }

                        if (kw == "for") {
                            std::string body = trim_ws(seg.text.substr(3));
                            size_t in_pos = body.find(" in ");
                            if (in_pos == std::string::npos) {
                                ++idx;
                                continue;
                            }
                            std::string var_name = trim_ws(body.substr(0, in_pos));
                            std::string list_expr = trim_ws(body.substr(in_pos + 4));
                            Value list_val = eval_expr(list_expr);
                            ++idx;
                            size_t block_start = idx;

                            if (list_val.kind == Value::Kind::List && list_val.list.data) {
                                const size_t total = list_val.list.end - list_val.list.begin;
                                for (size_t i = list_val.list.begin; i < list_val.list.end; ++i) {
                                    vars[var_name] = Value::from_msg((*list_val.list.data)[i]);
                                    size_t rel = i - list_val.list.begin;
                                    vars["loop.first"] = Value::from_bool(rel == 0);
                                    vars["loop.last"] = Value::from_bool(rel + 1 == total);
                                    vars["loop.index0"] = Value::from_int((long long)rel);
                                    size_t local = block_start;
                                    out += render_until(local, {"endfor"});
                                }
                            }

                            skip_until(idx, {"endfor"});
                            if (idx < segs.size() && stmt_head(segs[idx].text) == "endfor") ++idx;
                            continue;
                        }

                        ++idx;
                    }
                    return out;
                };

                size_t idx = 0;
                std::string rendered = render_until(idx, {});
                if (generation_prompt_out && rendered.size() >= 22) {
                    const std::string tail = "<|im_start|>assistant\n";
                    if (rendered.compare(rendered.size() - tail.size(), tail.size(), tail) == 0) {
                        *generation_prompt_out = tail;
                    }
                }
                return rendered;
            }

            return assembled;
        }

        if (qwen_chat_fallback) {
            std::string assembled;
            const char* sys = (system_text && system_text[0] != '\0') ? system_text : "You are a helpful assistant.";
            assembled += "<|im_start|>system\n";
            assembled += sys;
            assembled += "<|im_end|>\n";

            size_t begin = history.size() > (size_t)max_history ? history.size() - (size_t)max_history : 0;
            for (size_t i = begin; i + 1 < history.size(); i += 2) {
                assembled += "<|im_start|>user\n";
                assembled += history[i];
                assembled += "<|im_end|>\n";
                assembled += "<|im_start|>assistant\n";
                assembled += history[i + 1];
                assembled += "<|im_end|>\n";
            }

            assembled += "<|im_start|>user\n";
            assembled += (user_text ? user_text : "");
            assembled += "<|im_end|>\n";
            assembled += "<|im_start|>assistant\n";
            if (generation_prompt_out) *generation_prompt_out = "<|im_start|>assistant\n";
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
                std::string assembled = prompt;
                std::string generation_prompt;
                if (model_chat_template && model_chat_template[0] != '\0') {
                    assembled = render_template_auto({}, model_chat_template, system_prompt, prompt.c_str(), &generation_prompt);
                }
                if (compare_logits && !generation_prompt.empty()) {
                    fprintf(stderr, "{\"phase\":\"chat_prompt\",\"prompt\":\"%s\",\"generation_prompt\":\"%s\"}\n",
                            json_escape(assembled).c_str(),
                            json_escape(generation_prompt).c_str());
                    fflush(stderr);
                }
                if (chat_debug_enabled()) {
                    std::vector<int> dbg_ids = tokenizer_encode(assembled);
                    fprintf(stderr, "[main] assembled prompt token count=%zu\n", dbg_ids.size());
                    if (!dbg_ids.empty()) {
                        fprintf(stderr, "[main] token ids:");
                        for (size_t ii = 0; ii < dbg_ids.size(); ++ii) fprintf(stderr, " %d", dbg_ids[ii]);
                        fprintf(stderr, "\n");
                        std::string dbg_dec = tokenizer_decode(dbg_ids);
                        fprintf(stderr, "[main] decoded assembled prompt: %s\n", dbg_dec.c_str());
                    }
                }
                if (debug_attn_once) attention_set_debug_once(true);
                minxfmr_generate(ctx, assembled.c_str(), print_callback, temperature, top_k, top_p, min_p,
                         repeat_last_n, repeat_penalty, frequency_penalty, presence_penalty,
                         compare_logits, compare_top_n, compare_steps);
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
                std::string generation_prompt;
                std::string assembled = render_template_auto(history, model_chat_template, system_prompt, line.c_str(), &generation_prompt);
                if (compare_logits) {
                    fprintf(stderr, "{\"phase\":\"chat_prompt\",\"prompt\":\"%s\",\"generation_prompt\":\"%s\"}\n",
                            json_escape(assembled).c_str(),
                            json_escape(generation_prompt).c_str());
                    fflush(stderr);
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
                minxfmr_generate(ctx, assembled.c_str(), gen_collect_callback, temperature, top_k, top_p, min_p,
                                 repeat_last_n, repeat_penalty, frequency_penalty, presence_penalty,
                                 compare_logits, compare_top_n, compare_steps);
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
                minxfmr_generate(ctx, line.c_str(), print_callback, temperature, top_k, top_p, min_p,
                                 repeat_last_n, repeat_penalty, frequency_penalty, presence_penalty,
                                 compare_logits, compare_top_n, compare_steps);
                printf("\n");
            }
        }

    minxfmr_close(ctx);
    return 0;
}
