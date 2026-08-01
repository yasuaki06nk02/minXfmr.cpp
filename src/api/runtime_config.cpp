#include "runtime_config.h"
#include <cstdlib>
#include <algorithm>

RuntimeConfig& RuntimeConfig::Instance() {
    static RuntimeConfig s;
    return s;
}

bool RuntimeConfig::has(const char* key) const {
    const char* v = std::getenv(key);
    return v && v[0] != '\0';
}

bool RuntimeConfig::getBool(const char* key) const {
    const char* v = std::getenv(key);
    if (!v || v[0] == '\0') return false;
    return v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T';
}

int RuntimeConfig::getInt(const char* key, int defaultValue) const {
    const char* v = std::getenv(key);
    if (!v || v[0] == '\0') return defaultValue;
    return std::atoi(v);
}

std::string RuntimeConfig::getString(const char* key) const {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string();
}

bool RuntimeConfig::cuda_quant_enabled() const {
    // If explicitly set, honor it. Otherwise default to enabling when
    // the backend is explicitly requested as CUDA.
    if (has("MINXFMR_CUDA_QUANT")) return getBool("MINXFMR_CUDA_QUANT");
    std::string bk = getString("MINXFMR_BACKEND");
    if (bk.empty()) return false;
    for (char &c : bk) c = (char)std::tolower((unsigned char)c);
    return bk == "cuda";
}
