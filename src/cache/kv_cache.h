#pragma once
#include "../tensor/tensor.h"
#include <vector>

struct KVCache {
    size_t layers;
    size_t seq_max;
    size_t dim;
    std::vector<Tensor*> keys; // per-layer [seq_max x dim] (views into keys_buf)
    std::vector<Tensor*> vals; // per-layer [seq_max x dim] (views into vals_buf)
    // contiguous backing storage to avoid reallocations at append time
    std::vector<float> keys_buf; // size = layers * seq_max * dim
    std::vector<float> vals_buf; // size = layers * seq_max * dim
    std::vector<size_t> lengths; // current length per layer
};

KVCache* kvcache_create(size_t layers, size_t seq_max, size_t dim);
void kvcache_free(KVCache* c);
bool kvcache_append(KVCache* c, size_t layer, const float* key_row, const float* val_row);
void kvcache_reset(KVCache* c);
