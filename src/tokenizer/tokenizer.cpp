#include "tokenizer/tokenizer.h"
#include "../../third_party/gguf/gguf_reader.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#if defined(__has_include)
#if __has_include(<normaliz.h>)
#include <normaliz.h>
#define MINXFMR_HAS_NORMALIZ 1
#pragma comment(lib, "Normaliz.lib")
#endif
#endif
#endif

static std::vector<std::string> g_vocab;
static std::unordered_map<std::string,int> g_vid;
static std::vector<float> g_vocab_scores;
static std::vector<int>   g_vocab_types;
static std::unordered_map<uint8_t, std::string> g_byte_encoder;
static std::unordered_map<std::string, uint8_t> g_byte_decoder;
static std::string g_tokenizer_model;
static std::string g_tokenizer_pre;
static std::vector<std::string> g_tokenizer_merges;
static bool g_use_bpe = false;
enum class BpePretokenizerMode {
    Generic,
    Qwen2,
    Qwen35,
};
static BpePretokenizerMode g_bpe_pretokenizer = BpePretokenizerMode::Generic;
static std::unordered_map<std::string, int> g_bpe_ranks;
static size_t g_max_token_len = 0;
static const std::string kSpmMarker("\xE2\x96\x81");
static const std::string kGptSpaceMarker("\xC3\x84\xC2\xA0");
static const std::string kGptNewlineMarker("\xC3\x84\xC2\x8A");
static std::string g_word_marker = kSpmMarker;
static std::string g_newline_token = "\n";

// Byte trie for greedy longest-match tokenization.
struct TrieNode {
    int token_id;
    std::unordered_map<unsigned char,int> children;
    TrieNode() : token_id(-1), children() {}
};

static std::vector<TrieNode> g_trie;

static void trie_clear() {
    g_trie.clear();
    g_trie.emplace_back(); // root
}

static void trie_insert(const std::string &tok, int id) {
    if (tok.empty()) return;
    int node = 0;
    for (unsigned char uc : tok) {
        auto it = g_trie[node].children.find(uc);
        if (it == g_trie[node].children.end()) {
            int nxt = (int)g_trie.size();
            g_trie[node].children[uc] = nxt;
            g_trie.emplace_back();
            node = nxt;
        } else {
            node = it->second;
        }
    }
    g_trie[node].token_id = id;
}

static void trie_build_from_vocab() {
    // Rebuild once after vocabulary is loaded.
    trie_clear();
    for (size_t i = 0; i < g_vocab.size(); ++i) {
        trie_insert(g_vocab[i], (int)i);
    }
}

static void init_byte_level_maps() {
    if (!g_byte_encoder.empty()) return;

    // GPT-2 style byte-to-unicode reversible mapping.
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        std::string u;
        char32_t cp = (char32_t)cs[i];
        if (cp <= 0x7F) {
            u.push_back((char)cp);
        } else if (cp <= 0x7FF) {
            u.push_back((char)(0xC0 | (cp >> 6)));
            u.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            u.push_back((char)(0xE0 | (cp >> 12)));
            u.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            u.push_back((char)(0x80 | (cp & 0x3F)));
        }
        g_byte_encoder[(uint8_t)bs[i]] = u;
        g_byte_decoder[u] = (uint8_t)bs[i];
    }
}

static std::string byte_level_encode(const std::string& text) {
    init_byte_level_maps();

    std::string out;
    out.reserve(text.size() * 2);
    for (unsigned char b : text) {
        out += g_byte_encoder[(uint8_t)b];
    }
    return out;
}

static std::string byte_level_decode(const std::string& text) {
    init_byte_level_maps();

    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        bool matched = false;
        for (int len = 3; len >= 1; --len) {
            if (i + (size_t)len > text.size()) continue;
            std::string sub = text.substr(i, (size_t)len);
            auto it = g_byte_decoder.find(sub);
            if (it != g_byte_decoder.end()) {
                out.push_back((char)it->second);
                i += (size_t)len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            out.push_back(text[i++]);
        }
    }
    return out;
}

static void detect_tokenizer_markers_from_vocab() {
    size_t spm_count = 0;
    size_t gpt_count = 0;
    for (const std::string& tok : g_vocab) {
        if (tok.rfind(kSpmMarker, 0) == 0) ++spm_count;
        if (tok.rfind(kGptSpaceMarker, 0) == 0) ++gpt_count;
    }
    g_word_marker = (gpt_count > spm_count) ? kGptSpaceMarker : kSpmMarker;

    if (g_vid.find(kGptNewlineMarker) != g_vid.end()) {
        g_newline_token = kGptNewlineMarker;
    } else if (g_vid.find("\n") != g_vid.end()) {
        g_newline_token = "\n";
    } else {
        g_newline_token = "\n";
    }
}

static std::string bpe_pair_key(const std::string& a, const std::string& b) {
    std::string k;
    k.reserve(a.size() + b.size() + 1);
    k += a;
    k.push_back('\x1f');
    k += b;
    return k;
}

static std::vector<std::string> split_byte_level_symbols(const std::string& encoded) {
    std::vector<std::string> syms;
    syms.reserve(encoded.size());
    for (size_t i = 0; i < encoded.size();) {
        bool matched = false;
        for (int len = 3; len >= 1; --len) {
            if (i + (size_t)len > encoded.size()) continue;
            std::string sub = encoded.substr(i, (size_t)len);
            if (g_byte_decoder.find(sub) == g_byte_decoder.end()) continue;
            syms.push_back(sub);
            i += (size_t)len;
            matched = true;
            break;
        }
        if (!matched) {
            syms.push_back(encoded.substr(i, 1));
            ++i;
        }
    }
    return syms;
}

static std::vector<std::string> apply_bpe_merges(std::vector<std::string> syms) {
    if (!g_use_bpe || g_bpe_ranks.empty()) return syms;
    if (syms.size() < 2) return syms;

    while (syms.size() >= 2) {
        int best_rank = (std::numeric_limits<int>::max)();
        size_t best_i = syms.size();

        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = g_bpe_ranks.find(bpe_pair_key(syms[i], syms[i + 1]));
            if (it == g_bpe_ranks.end()) continue;
            if (it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }

        if (best_i == syms.size()) break;

        std::vector<std::string> next;
        next.reserve(syms.size());
        size_t i = 0;
        while (i < syms.size()) {
            if (i + 1 < syms.size()) {
                auto it = g_bpe_ranks.find(bpe_pair_key(syms[i], syms[i + 1]));
                if (it != g_bpe_ranks.end() && it->second == best_rank) {
                    next.push_back(syms[i] + syms[i + 1]);
                    i += 2;
                    continue;
                }
            }
            next.push_back(syms[i]);
            ++i;
        }
        syms.swap(next);
    }

    return syms;
}

static bool utf8_decode_one(const std::string& s, size_t pos, uint32_t& cp, size_t& len) {
    if (pos >= s.size()) return false;
    unsigned char c0 = (unsigned char)s[pos];
    if (c0 < 0x80) {
        cp = c0;
        len = 1;
        return true;
    }
    if ((c0 >> 5) == 0x6 && pos + 1 < s.size()) {
        cp = ((uint32_t)(c0 & 0x1F) << 6) | ((uint32_t)s[pos + 1] & 0x3F);
        len = 2;
        return true;
    }
    if ((c0 >> 4) == 0xE && pos + 2 < s.size()) {
        cp = ((uint32_t)(c0 & 0x0F) << 12) |
             (((uint32_t)s[pos + 1] & 0x3F) << 6) |
             ((uint32_t)s[pos + 2] & 0x3F);
        len = 3;
        return true;
    }
    if ((c0 >> 3) == 0x1E && pos + 3 < s.size()) {
        cp = ((uint32_t)(c0 & 0x07) << 18) |
             (((uint32_t)s[pos + 1] & 0x3F) << 12) |
             (((uint32_t)s[pos + 2] & 0x3F) << 6) |
             ((uint32_t)s[pos + 3] & 0x3F);
        len = 4;
        return true;
    }
    cp = c0;
    len = 1;
    return true;
}

static bool is_unicode_newline(uint32_t cp) {
    return cp == '\n';
}

static bool is_unicode_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\v' || cp == '\f' || cp == 0x00A0 || cp == 0x3000;
}

static bool is_unicode_letter(uint32_t cp);

static bool is_unicode_digit(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    return false;
}

static bool is_unicode_mark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x1AB0 && cp <= 0x1AFF) ||
           (cp >= 0x1DC0 && cp <= 0x1DFF) ||
           (cp >= 0x20D0 && cp <= 0x20FF) ||
           (cp >= 0xFE20 && cp <= 0xFE2F);
}

static bool is_qwen_word_char(uint32_t cp, BpePretokenizerMode mode) {
    if (is_unicode_letter(cp) || is_unicode_digit(cp)) {
        return true;
    }
    if (mode == BpePretokenizerMode::Qwen35 && is_unicode_mark(cp)) {
        return true;
    }
    return false;
}

static bool is_unicode_letter(uint32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
    if ((cp >= 0x00C0 && cp <= 0x02AF) || (cp >= 0x0370 && cp <= 0x052F)) return true;
    if ((cp >= 0x0590 && cp <= 0x08FF) || (cp >= 0x0900 && cp <= 0x0E7F)) return true;
    if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF)) return true;
    if ((cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF)) return true;
    if ((cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF)) return true;
    if ((cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x20000 && cp <= 0x2FA1F)) return true;
    return false;
}

static bool match_ascii_contraction(const std::string& s, size_t pos, size_t& matched_len) {
    matched_len = 0;
    if (pos >= s.size() || s[pos] != '\'') return false;
    auto ieq = [](char a, char b) {
        return std::tolower((unsigned char)a) == std::tolower((unsigned char)b);
    };
    if (pos + 2 <= s.size()) {
        char c1 = s[pos + 1];
        if (ieq(c1, 's') || ieq(c1, 't') || ieq(c1, 'm') || ieq(c1, 'd')) {
            matched_len = 2;
            return true;
        }
    }
    if (pos + 3 <= s.size()) {
        char c1 = s[pos + 1];
        char c2 = s[pos + 2];
        if ((ieq(c1, 'r') && ieq(c2, 'e')) ||
            (ieq(c1, 'v') && ieq(c2, 'e')) ||
            (ieq(c1, 'l') && ieq(c2, 'l'))) {
            matched_len = 3;
            return true;
        }
    }
    return false;
}

static std::vector<std::string> pretokenize_bpe_text(const std::string& text, BpePretokenizerMode mode) {
    std::vector<std::string> parts;
    for (size_t i = 0; i < text.size();) {
        size_t contraction_len = 0;
        if (match_ascii_contraction(text, i, contraction_len)) {
            parts.push_back(text.substr(i, contraction_len));
            i += contraction_len;
            continue;
        }

        uint32_t cp = 0;
        size_t len = 0;
        if (!utf8_decode_one(text, i, cp, len)) break;

        if (cp == ' ' && i + len < text.size()) {
            uint32_t next_cp = 0;
            size_t next_len = 0;
            if (utf8_decode_one(text, i + len, next_cp, next_len) && !is_unicode_space(next_cp) && !is_unicode_newline(next_cp) && !is_qwen_word_char(next_cp, mode)) {
                size_t j = i + len + next_len;
                while (j < text.size()) {
                    uint32_t cur_cp = 0;
                    size_t cur_len = 0;
                    if (!utf8_decode_one(text, j, cur_cp, cur_len)) break;
                    if (is_unicode_space(cur_cp) || is_qwen_word_char(cur_cp, mode)) break;
                    if (is_unicode_newline(cur_cp)) break;
                    j += cur_len;
                }
                while (j < text.size()) {
                    uint32_t cur_cp = 0;
                    size_t cur_len = 0;
                    if (!utf8_decode_one(text, j, cur_cp, cur_len) || !is_unicode_newline(cur_cp)) break;
                    j += cur_len;
                }
                parts.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }

        if (!is_unicode_newline(cp) && !is_qwen_word_char(cp, mode) && i + len < text.size()) {
            uint32_t next_cp = 0;
            size_t next_len = 0;
            if (utf8_decode_one(text, i + len, next_cp, next_len) && is_qwen_word_char(next_cp, mode)) {
                size_t j = i + len + next_len;
                while (j < text.size()) {
                    uint32_t cur_cp = 0;
                    size_t cur_len = 0;
                    if (!utf8_decode_one(text, j, cur_cp, cur_len) || !is_qwen_word_char(cur_cp, mode)) break;
                    j += cur_len;
                }
                parts.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }

        if (is_qwen_word_char(cp, mode)) {
            size_t j = i + len;
            while (j < text.size()) {
                uint32_t cur_cp = 0;
                size_t cur_len = 0;
                if (!utf8_decode_one(text, j, cur_cp, cur_len) || !is_qwen_word_char(cur_cp, mode)) break;
                j += cur_len;
            }
            parts.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        if (is_unicode_digit(cp)) {
            parts.push_back(text.substr(i, len));
            i += len;
            continue;
        }

        if (is_unicode_space(cp) || is_unicode_newline(cp)) {
            size_t j = i + len;
            if (is_unicode_space(cp)) {
                while (j < text.size()) {
                    uint32_t cur_cp = 0;
                    size_t cur_len = 0;
                    if (!utf8_decode_one(text, j, cur_cp, cur_len) || !is_unicode_space(cur_cp)) break;
                    j += cur_len;
                }
                size_t k = j;
                while (k < text.size()) {
                    uint32_t cur_cp = 0;
                    size_t cur_len = 0;
                    if (!utf8_decode_one(text, k, cur_cp, cur_len) || !is_unicode_newline(cur_cp)) break;
                    k += cur_len;
                }
                if (k > j) {
                    parts.push_back(text.substr(i, k - i));
                    i = k;
                    continue;
                }
            } else {
                while (j < text.size()) {
                    uint32_t cur_cp = 0;
                    size_t cur_len = 0;
                    if (!utf8_decode_one(text, j, cur_cp, cur_len) || !is_unicode_newline(cur_cp)) break;
                    j += cur_len;
                }
            }
            parts.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        size_t j = i + len;
        while (j < text.size()) {
            uint32_t cur_cp = 0;
            size_t cur_len = 0;
            if (!utf8_decode_one(text, j, cur_cp, cur_len)) break;
            if (is_unicode_space(cur_cp) || is_unicode_newline(cur_cp) || is_unicode_letter(cur_cp) || is_unicode_digit(cur_cp)) break;
            j += cur_len;
        }
        while (j < text.size()) {
            uint32_t cur_cp = 0;
            size_t cur_len = 0;
            if (!utf8_decode_one(text, j, cur_cp, cur_len) || !is_unicode_newline(cur_cp)) break;
            j += cur_len;
        }
        parts.push_back(text.substr(i, j - i));
        i = j;
    }
    return parts;
}

static bool is_special_like_token_text(const std::string& tok) {
    if (tok.empty()) return false;
    if (tok == "<unk>" || tok == "</s>" || tok == "<s>" || tok == "<pad>") return true;
    if (tok == "[INST]" || tok == "[/INST]" || tok == "[SYS]" || tok == "[/SYS]") return true;
    if (tok.rfind("<|", 0) == 0 && tok.size() >= 4 && tok.back() == '>') return true;
    if (tok.front() == '<' && tok.back() == '>' && (tok.find('|') != std::string::npos || tok.find("assistant") != std::string::npos || tok.find("user") != std::string::npos || tok.find("tool") != std::string::npos)) return true;
    if (tok.front() == '[' && tok.back() == ']' && (tok.find("INST") != std::string::npos || tok.find("SYS") != std::string::npos || tok.find("ASSISTANT") != std::string::npos || tok.find("USER") != std::string::npos)) return true;
    return false;
}

static std::string normalize_text_for_tokenizer(const std::string& s_in) {
#if defined(_WIN32) && defined(MINXFMR_HAS_NORMALIZ)
    // Use Windows Normaliz API to normalize to NFKC where available
    int wlen = MultiByteToWideChar(CP_UTF8, 0, s_in.c_str(), (int)s_in.size(), NULL, 0);
    if (wlen > 0) {
        std::wstring wbuf(wlen, 0);
        MultiByteToWideChar(CP_UTF8, 0, s_in.c_str(), (int)s_in.size(), &wbuf[0], wlen);
        int nlen = NormalizeString(NormalizationKC, wbuf.c_str(), wlen, NULL, 0);
        if (nlen > 0) {
            std::wstring nbuf(nlen, 0);
            NormalizeString(NormalizationKC, wbuf.c_str(), wlen, &nbuf[0], nlen);
            int outlen = WideCharToMultiByte(CP_UTF8, 0, nbuf.c_str(), nlen, NULL, 0, NULL, NULL);
            if (outlen > 0) {
                std::string out(outlen, 0);
                WideCharToMultiByte(CP_UTF8, 0, nbuf.c_str(), nlen, &out[0], outlen, NULL, NULL);
                // Keep normalization conservative for multilingual text:
                // preserve spacing and punctuation exactly except for CRLF/NBSP normalization.
                size_t pos = 0;
                while ((pos = out.find("\xC2\xA0", pos)) != std::string::npos) out.replace(pos, 2, " ");
                pos = 0;
                while ((pos = out.find("\xE3\x80\x80", pos)) != std::string::npos) out.replace(pos, 3, " ");
                std::string norm;
                norm.reserve(out.size());
                for (unsigned char c : out) {
                    if (c == '\r') continue;
                    norm.push_back((char)c);
                }
                return norm;
            }
        }
    }
#endif
    // Cross-platform fallback with minimal transformation.
    std::string s = s_in;
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char ch = (unsigned char)s[i];
        if (i == 0 && s.size() >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF) {
            i = 2;
            continue;
        }
        if (ch == 0xC2 && i + 1 < s.size() && (unsigned char)s[i + 1] == 0xA0) {
            ch = ' ';
            ++i;
        } else if (ch == 0xE3 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x80 && (unsigned char)s[i + 2] == 0x80) {
            ch = ' ';
            i += 2;
        }
        if (ch == '\r') {
            continue;
        }
        if (ch == '\n') {
            out.push_back('\n');
            continue;
        }
        out.push_back((char)ch);
    }
    return out;
}

static BpePretokenizerMode detect_bpe_pretokenizer_mode() {
    std::string pre = g_tokenizer_pre;
    for (char& c : pre) {
        c = (char)std::tolower((unsigned char)c);
    }

    if (pre == "qwen35" || pre == "qwen3") {
        return BpePretokenizerMode::Qwen35;
    }
    if (pre == "qwen2" || pre == "deepseek-r1-qwen" || pre == "kormo" || pre == "f2llmv2" || pre == "megrez") {
        return BpePretokenizerMode::Qwen2;
    }
    return BpePretokenizerMode::Generic;
}

bool tokenizer_load_from_list(const std::vector<std::string>& vocab) {
    // Reset all derived tokenizer state when swapping vocabularies.
    g_vocab = vocab;
    g_vid.clear();
    g_max_token_len = 0;
    for (size_t i=0;i<g_vocab.size();++i) {
        g_vid[g_vocab[i]] = (int)i;
        if (g_vocab[i].size() > g_max_token_len) g_max_token_len = g_vocab[i].size();
    }
    g_tokenizer_model.clear();
    g_tokenizer_pre.clear();
    g_tokenizer_merges.clear();
    g_bpe_ranks.clear();
    g_use_bpe = false;
    g_bpe_pretokenizer = BpePretokenizerMode::Generic;
    detect_tokenizer_markers_from_vocab();
    trie_build_from_vocab();
    return true;
}

bool tokenizer_load_from_gguf(const GGUF_File& gf) {
    if (gf.vocab_tokens.empty()) return false;
    g_vocab = gf.vocab_tokens;
    g_vid.clear();
    g_max_token_len = 0;
    for (size_t i = 0; i < g_vocab.size(); ++i) {
        g_vid[g_vocab[i]] = (int)i;
        if (g_vocab[i].size() > g_max_token_len) g_max_token_len = g_vocab[i].size();
    }
    g_tokenizer_model = gf.tokenizer_model;
    g_tokenizer_pre = gf.tokenizer_pre;
    g_tokenizer_merges = gf.tokenizer_merges;
    g_bpe_pretokenizer = detect_bpe_pretokenizer_mode();

    g_bpe_ranks.clear();
    for (size_t i = 0; i < g_tokenizer_merges.size(); ++i) {
        const std::string& m = g_tokenizer_merges[i];
        size_t sp = m.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= m.size()) continue;
        std::string a = m.substr(0, sp);
        std::string b = m.substr(sp + 1);
        if (a.empty() || b.empty()) continue;
        g_bpe_ranks[bpe_pair_key(a, b)] = (int)i;
    }
    std::string model_lc = g_tokenizer_model;
    for (char& c : model_lc) c = (char)std::tolower((unsigned char)c);
    g_use_bpe = (!g_bpe_ranks.empty()) && (model_lc.find("bpe") != std::string::npos || model_lc.find("gpt2") != std::string::npos);

    if (!g_tokenizer_model.empty()) {
        std::fprintf(stderr, "[tokenizer] model=%s pre=%s merges=%zu mode=%s\n",
            g_tokenizer_model.c_str(),
            g_tokenizer_pre.empty() ? "" : g_tokenizer_pre.c_str(),
            g_tokenizer_merges.size(),
            g_use_bpe ? "bpe" : "greedy");
    }
    detect_tokenizer_markers_from_vocab();
    trie_build_from_vocab();

    // load optional scores
    g_vocab_scores.clear();
    if (!gf.vocab_scores.empty()) {
        g_vocab_scores = gf.vocab_scores;
        if (g_vocab_scores.size() < g_vocab.size()) g_vocab_scores.resize(g_vocab.size(), 0.0f);
        else if (g_vocab_scores.size() > g_vocab.size()) g_vocab_scores.resize(g_vocab.size());
    } else {
        g_vocab_scores.assign(g_vocab.size(), 0.0f);
    }

    // load optional token types
    g_vocab_types.clear();
    if (!gf.vocab_types.empty()) {
        g_vocab_types = gf.vocab_types;
        if (g_vocab_types.size() < g_vocab.size()) g_vocab_types.resize(g_vocab.size(), 0);
        else if (g_vocab_types.size() > g_vocab.size()) g_vocab_types.resize(g_vocab.size());
    } else {
        g_vocab_types.assign(g_vocab.size(), 0);
    }

    return true;
}

void tokenizer_add_special_tokens(const std::vector<std::string>& toks) {
    if (toks.empty()) return;
    // Append new tokens at the end to avoid shifting existing IDs.
    for (const std::string& t : toks) {
        if (g_vid.find(t) != g_vid.end()) continue;
        g_vocab.push_back(t);
        g_vid[t] = (int)g_vocab.size() - 1;
        if (t.size() > g_max_token_len) g_max_token_len = t.size();
        trie_insert(t, g_vid[t]);
    }
}

int tokenizer_token_to_id(const std::string& token) {
    auto it = g_vid.find(token);
    if (it==g_vid.end()) return -1;
    return it->second;
}

std::string tokenizer_id_to_token(int id) {
    if (id < 0 || (size_t)id >= g_vocab.size()) return "<unk>";
    return g_vocab[id];
}

std::vector<int> tokenizer_encode(const std::string& text) {
    std::vector<int> out;
    if (text.empty()) return out;
    const std::string norm_src = normalize_text_for_tokenizer(text);

    if (g_use_bpe) {
        auto push_unknown_or_byte = [&](const std::string& sym) {
            auto it_dec = g_byte_decoder.find(sym);
            if (it_dec != g_byte_decoder.end()) {
                char byte_tok[7];
                std::snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned int)it_dec->second);
                int id = tokenizer_token_to_id(byte_tok);
                if (id >= 0) {
                    out.push_back(id);
                    return;
                }
            }
            int unk = -1;
            auto it_unk = g_vid.find("<unk>");
            if (it_unk != g_vid.end()) unk = it_unk->second;
            if (unk < 0 && !g_vocab.empty()) unk = 0;
            out.push_back(unk);
        };

        size_t pos = 0;
        while (pos < norm_src.size()) {
            size_t best_len = 0;
            int best_id = -1;
            if (!g_trie.empty()) {
                int node = 0;
                size_t i = pos;
                while (i < norm_src.size()) {
                    unsigned char uc = (unsigned char)norm_src[i];
                    auto it = g_trie[node].children.find(uc);
                    if (it == g_trie[node].children.end()) break;
                    node = it->second;
                    ++i;
                    if (g_trie[node].token_id >= 0) {
                        int candidate = g_trie[node].token_id;
                        if ((size_t)candidate < g_vocab.size() && is_special_like_token_text(g_vocab[(size_t)candidate])) {
                            best_id = candidate;
                            best_len = i - pos;
                        }
                    }
                }
            }

            if (best_id >= 0 && best_len > 0) {
                out.push_back(best_id);
                pos += best_len;
                continue;
            }

            size_t next_special = pos + 1;
            while (next_special < norm_src.size()) {
                size_t probe_len = 0;
                int probe_id = -1;
                if (!g_trie.empty()) {
                    int node = 0;
                    size_t i = next_special;
                    while (i < norm_src.size()) {
                        unsigned char uc = (unsigned char)norm_src[i];
                        auto it = g_trie[node].children.find(uc);
                        if (it == g_trie[node].children.end()) break;
                        node = it->second;
                        ++i;
                        if (g_trie[node].token_id >= 0) {
                            int candidate = g_trie[node].token_id;
                            if ((size_t)candidate < g_vocab.size() && is_special_like_token_text(g_vocab[(size_t)candidate])) {
                                probe_id = candidate;
                                probe_len = i - next_special;
                            }
                        }
                    }
                }
                if (probe_id >= 0 && probe_len > 0) break;
                ++next_special;
            }

            std::string raw_chunk = norm_src.substr(pos, next_special - pos);
            std::vector<std::string> parts = pretokenize_bpe_text(raw_chunk, g_bpe_pretokenizer);
            for (const std::string& part_raw : parts) {
                const std::string part = byte_level_encode(part_raw);
                std::vector<std::string> syms = split_byte_level_symbols(part);
                syms = apply_bpe_merges(std::move(syms));
                for (const std::string& tok : syms) {
                    int id = tokenizer_token_to_id(tok);
                    if (id >= 0) {
                        out.push_back(id);
                    } else {
                        push_unknown_or_byte(tok);
                    }
                }
            }
            pos = next_special;
        }
        return out;
    }

    const std::string norm_text = byte_level_encode(norm_src);

    auto match_exact_prefix = [&](const std::string& src, size_t pos, size_t& matched_len) -> int {
        matched_len = 0;
        if (g_trie.empty() || pos >= src.size()) return -1;
        int node = 0;
        int best_id = -1;
        size_t best_len = 0;
        size_t i = pos;
        while (i < src.size()) {
            unsigned char uc = (unsigned char)src[i];
            auto it = g_trie[node].children.find(uc);
            if (it == g_trie[node].children.end()) break;
            node = it->second;
            ++i;
            if (g_trie[node].token_id >= 0) {
                best_id = g_trie[node].token_id;
                best_len = i - pos;
                if (best_len == g_max_token_len) break;
            }
        }
        matched_len = best_len;
        return best_id;
    };

    auto push_unknown_or_byte = [&](unsigned char b) {
        char byte_tok[7];
        std::snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned int)b);
        int id = tokenizer_token_to_id(byte_tok);
        if (id >= 0) {
            out.push_back(id);
            return;
        }

        if (b == '\n') {
            int nl_id = tokenizer_token_to_id(g_newline_token);
            if (nl_id < 0) nl_id = tokenizer_token_to_id("\\n");
            if (nl_id >= 0) {
                out.push_back(nl_id);
                return;
            }
        }

        int unk = -1;
        auto it = g_vid.find("<unk>");
        if (it != g_vid.end()) unk = it->second;
        if (unk < 0 && !g_vocab.empty()) unk = 0;
        out.push_back(unk);
    };

    auto decode_one_byte_level_symbol = [&](const std::string& src, size_t pos, uint8_t& out_byte, size_t& consumed_len) -> bool {
        consumed_len = 0;
        for (int len = 3; len >= 1; --len) {
            if (pos + (size_t)len > src.size()) continue;
            std::string sub = src.substr(pos, (size_t)len);
            auto it = g_byte_decoder.find(sub);
            if (it == g_byte_decoder.end()) continue;
            out_byte = it->second;
            consumed_len = (size_t)len;
            return true;
        }
        return false;
    };

    size_t pos = 0;
    while (pos < norm_text.size()) {
        size_t exact_len = 0;
        int exact_id = match_exact_prefix(norm_text, pos, exact_len);
        if (exact_id >= 0) {
            out.push_back(exact_id);
            pos += exact_len;
            continue;
        }

        uint8_t raw_byte = 0;
        size_t consumed_len = 0;
        if (decode_one_byte_level_symbol(norm_text, pos, raw_byte, consumed_len)) {
            push_unknown_or_byte(raw_byte);
            pos += consumed_len;
        } else {
            push_unknown_or_byte((unsigned char)norm_text[pos]);
            ++pos;
        }
    }
    return out;
}

std::string tokenizer_render_piece(const std::string& piece) {
    if (piece.empty()) return piece;

    std::string normalized;
    normalized.reserve(piece.size() + 1);

    const std::string spm = "\xE2\x96\x81";
    const std::string spm_raw = "\xC4\xA0";
    const std::string nl_raw = "\xC4\x8A";
    const std::string spm2 = "\xC3\x84\xC2\xA0";
    const std::string spm3 = "\xC3\x84\xC2\x8A";

    for (size_t i = 0; i < piece.size();) {
        if (i + spm.size() <= piece.size() && piece.compare(i, spm.size(), spm) == 0) {
            normalized.push_back(' ');
            i += spm.size();
            continue;
        }
        if (i + spm_raw.size() <= piece.size() && piece.compare(i, spm_raw.size(), spm_raw) == 0) {
            normalized.push_back(' ');
            i += spm_raw.size();
            continue;
        }
        if (i + spm2.size() <= piece.size() && piece.compare(i, spm2.size(), spm2) == 0) {
            normalized.push_back(' ');
            i += spm2.size();
            continue;
        }
        if (i + nl_raw.size() <= piece.size() && piece.compare(i, nl_raw.size(), nl_raw) == 0) {
            normalized.push_back('\n');
            i += nl_raw.size();
            continue;
        }
        if (i + spm3.size() <= piece.size() && piece.compare(i, spm3.size(), spm3) == 0) {
            normalized.push_back('\n');
            i += spm3.size();
            continue;
        }
        normalized.push_back(piece[i++]);
    }

    return byte_level_decode(normalized);
}

std::string tokenizer_decode(const std::vector<int>& ids) {
    std::string s;
    for (size_t i = 0; i < ids.size(); ++i) {
        std::string tok = tokenizer_id_to_token(ids[i]);

        // Decode byte-fallback token like <0xE3> back to raw byte.
        if (tok.size() == 6 && tok.rfind("<0x", 0) == 0 && tok[5] == '>') {
            auto hex_to_nibble = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                return -1;
            };
            int hi = hex_to_nibble(tok[3]);
            int lo = hex_to_nibble(tok[4]);
            if (hi >= 0 && lo >= 0) {
                unsigned char b = (unsigned char)((hi << 4) | lo);
                s.push_back((char)b);
                continue;
            }
        }

        s += tok;
    }
    return byte_level_decode(s);
}

size_t tokenizer_vocab_size() {
    return g_vocab.size();
}
