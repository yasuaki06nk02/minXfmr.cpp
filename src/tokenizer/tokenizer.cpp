#include "tokenizer/tokenizer.h"
#include "../../third_party/gguf/gguf_reader.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdio>
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
                // replace common non-breaking/fullwidth spaces left as bytes
                size_t pos = 0;
                while ((pos = out.find("\xC2\xA0", pos)) != std::string::npos) out.replace(pos, 2, " ");
                pos = 0;
                while ((pos = out.find("\xE3\x80\x80", pos)) != std::string::npos) out.replace(pos, 3, " ");
                // compress whitespace, but preserve newlines because chat templates rely on them.
                std::string comp;
                comp.reserve(out.size());
                bool last_space = false;
                for (unsigned char c : out) {
                    if (c == '\r') {
                        continue;
                    }
                    if (c == '\n') {
                        comp.push_back('\n');
                        last_space = false;
                        continue;
                    }
                    if (c == ' ' || c == '\t' || c == '\v' || c == '\f') {
                        if (!last_space) comp.push_back(' ');
                        last_space = true;
                    } else {
                        comp.push_back((char)c);
                        last_space = false;
                    }
                }
                while (!comp.empty() && comp.front() == ' ') comp.erase(comp.begin());
                while (!comp.empty() && comp.back() == ' ') comp.pop_back();
                return comp;
            }
        }
    }
#endif
    // Cross-platform fallback: keep behavior conservative and predictable.
    std::string s = s_in;
    std::string out;
    out.reserve(s.size());
    bool last_was_space = false;
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
            last_was_space = false;
            continue;
        }
        if (ch == '\t' || ch == '\v' || ch == '\f') {
            ch = ' ';
        }
        if (ch == ' ') {
            if (last_was_space) continue;
            last_was_space = true;
            out.push_back(' ');
            continue;
        }
        last_was_space = false;
        out.push_back((char)ch);
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
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
    const std::string norm_text = byte_level_encode(normalize_text_for_tokenizer(text));

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
