#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct minxfmr_context minxfmr_context;

minxfmr_context* minxfmr_open(const char* model_path);
minxfmr_context* minxfmr_open_with_layer(const char* model_path, int projection_layer);

int minxfmr_generate(minxfmr_context* ctx, const char* prompt, void (*callback)(const char* token), double temperature, int top_k);

void minxfmr_reset(minxfmr_context* ctx);

void minxfmr_close(minxfmr_context* ctx);

// Print summary of loaded projection weights (if any)
void minxfmr_print_weights(minxfmr_context* ctx);

// Accessors for optional chat metadata extracted from GGUF
const char* minxfmr_get_chat_template(minxfmr_context* ctx);
size_t minxfmr_get_special_tokens_count(minxfmr_context* ctx);
const char* minxfmr_get_special_token(minxfmr_context* ctx, size_t idx);

#ifdef __cplusplus
}
#endif
