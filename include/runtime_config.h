#pragma once

#include <string>

class RuntimeConfig {
public:
    static RuntimeConfig& Instance();

    // Generic accessors (read from environment on each call)
    bool has(const char* key) const;
    bool getBool(const char* key) const;
    int getInt(const char* key, int defaultValue) const;
    std::string getString(const char* key) const;

    // Convenience helpers for commonly used settings
    int cpu_threads() const { return getInt("MINXFMR_CPU_THREADS", 0); }
    bool gguf_verbose() const { return getBool("MINXFMR_GGUF_VERBOSE"); }
    bool verbose_cache() const { return getBool("MINXFMR_VERBOSE_CACHE"); }
    bool chat_debug() const { return getBool("MINXFMR_CHAT_DEBUG"); }
    bool gen_verbose() const { return getBool("MINXFMR_VERBOSE_GEN"); }
    bool cuda_quant_parity_set() const { return has("MINXFMR_CUDA_QUANT_PARITY"); }
    bool cuda_quant_parity() const { return getBool("MINXFMR_CUDA_QUANT_PARITY"); }
    bool cuda_quant_atomic_safe() const { return getBool("MINXFMR_CUDA_QUANT_ATOMIC_SAFE"); }
    bool cuda_quant_enabled() const;
    std::string backend_str() const { return getString("MINXFMR_BACKEND"); }
    int max_gen_tokens() const { return getInt("MINXFMR_MAX_GEN_TOKENS", -1); }
    bool emit_json() const { return getBool("MINXFMR_EMIT_JSON"); }
    bool transpose_user_override() const { return getBool("MINXFMR_TRANSPOSE_USER_OVERRIDE"); }
    bool transpose_wq() const { return getBool("MINXFMR_TRANSPOSE_WQ"); }
    bool transpose_wk() const { return getBool("MINXFMR_TRANSPOSE_WK"); }
    bool transpose_wv() const { return getBool("MINXFMR_TRANSPOSE_WV"); }
    bool transpose_wo() const { return getBool("MINXFMR_TRANSPOSE_WO"); }

private:
    RuntimeConfig() = default;
    RuntimeConfig(const RuntimeConfig&) = delete;
    RuntimeConfig& operator=(const RuntimeConfig&) = delete;
};
