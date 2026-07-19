#pragma once
#include <string>
#include <vector>

struct GGUF_File;

// Very small tokenizer: whitespace-split vocabulary with map
bool tokenizer_load_from_list(const std::vector<std::string>& vocab);

// Load tokenizer vocabulary and optional metadata from a GGUF_File
bool tokenizer_load_from_gguf(const GGUF_File& gf);

int tokenizer_token_to_id(const std::string& token);
std::string tokenizer_id_to_token(int id);

std::vector<int> tokenizer_encode(const std::string& text);
std::string tokenizer_decode(const std::vector<int>& ids);

size_t tokenizer_vocab_size();

// Ensure special tokens exist in the vocab (adds missing tokens at front)
void tokenizer_add_special_tokens(const std::vector<std::string>& toks);
