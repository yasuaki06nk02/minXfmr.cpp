#pragma once
#include "../tensor/tensor.h"
#include <vector>

struct KVCache {
    size_t layers;
    size_t seq_max;
    size_t dim;
    std::vector<Tensor*> keys; // per-layer [seq_max x dim]
    std::vector<Tensor*> vals; // per-layer [seq_max x dim]
    std::vector<size_t> lengths; // current length per layer
};

KVCache* kvcache_create(size_t layers, size_t seq_max, size_t dim);
void kvcache_free(KVCache* c);
bool kvcache_append(KVCache* c, size_t layer, const float* key_row, const float* val_row);
void kvcache_reset(KVCache* c);
