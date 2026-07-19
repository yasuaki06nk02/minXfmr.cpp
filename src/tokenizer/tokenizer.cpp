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
static const std::string kSpmMarker("\xE2\x96\x81");

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
                // compress whitespace
                std::string comp;
                comp.reserve(out.size());
                bool last_space = false;
                for (unsigned char c : out) {
                    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f') {
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
    // Fallback conservative normalization
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
        if (ch == '\r' || ch == '\n' || ch == '\t' || ch == '\v' || ch == '\f') {
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
    g_vocab = vocab;
    g_vid.clear();
    for (size_t i=0;i<g_vocab.size();++i) g_vid[g_vocab[i]] = (int)i;
    return true;
}

bool tokenizer_load_from_gguf(const GGUF_File& gf) {
    if (gf.vocab_tokens.empty()) return false;
    g_vocab = gf.vocab_tokens;
    g_vid.clear();
    for (size_t i = 0; i < g_vocab.size(); ++i) g_vid[g_vocab[i]] = (int)i;

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
    const std::string norm_text = normalize_text_for_tokenizer(text);

    auto match_exact_prefix = [&](const std::string& src, size_t pos, size_t& matched_len) -> int {
        int best_id = -1;
        matched_len = 0;
        for (size_t i = 0; i < g_vocab.size(); ++i) {
            const std::string& tok = g_vocab[i];
            if (tok.empty()) continue;
            if (pos + tok.size() <= src.size() && src.compare(pos, tok.size(), tok) == 0) {
                if (tok.size() > matched_len) {
                    best_id = (int)i;
                    matched_len = tok.size();
                }
            }
        }
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

        int unk = -1;
        auto it = g_vid.find("<unk>");
        if (it != g_vid.end()) unk = it->second;
        if (unk < 0 && !g_vocab.empty()) unk = 0;
        out.push_back(unk);
    };

    size_t pos = 0;
    bool at_word_start = true;
    while (pos < norm_text.size()) {
        size_t exact_len = 0;
        int exact_id = match_exact_prefix(norm_text, pos, exact_len);
        if (exact_id >= 0) {
            out.push_back(exact_id);
            for (size_t i = 0; i < exact_len; ++i) {
                if (std::isspace((unsigned char)norm_text[pos + i])) at_word_start = true;
            }
            pos += exact_len;
            continue;
        }

        if (std::isspace((unsigned char)norm_text[pos])) {
            at_word_start = true;
            ++pos;
            continue;
        }

        size_t word_end = pos;
        while (word_end < norm_text.size() && !std::isspace((unsigned char)norm_text[word_end])) ++word_end;

        std::string piece;
        if (at_word_start) piece = kSpmMarker;
        piece += norm_text.substr(pos, word_end - pos);

        size_t piece_pos = 0;
        while (piece_pos < piece.size()) {
            size_t best_len = 0;
            int best_id = match_exact_prefix(piece, piece_pos, best_len);
            if (best_id >= 0) {
                out.push_back(best_id);
                piece_pos += best_len;
            } else {
                push_unknown_or_byte((unsigned char)piece[piece_pos]);
                ++piece_pos;
            }
        }

        pos = word_end;
        at_word_start = false;
    }
    return out;
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

        size_t p = 0;
        while (p < tok.size()) {
            if (p + kSpmMarker.size() <= tok.size() && tok.compare(p, kSpmMarker.size(), kSpmMarker) == 0) {
                s.push_back(' ');
                p += kSpmMarker.size();
            } else {
                s.push_back(tok[p]);
                p += 1;
            }
        }
    }
    // trim at most one leading space to avoid over-trimming multi-token outputs
    if (!s.empty() && s[0] == ' ') s.erase(0, 1);
    return s;
}

size_t tokenizer_vocab_size() {
    return g_vocab.size();
}
