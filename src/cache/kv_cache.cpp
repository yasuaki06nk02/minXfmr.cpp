#include "kv_cache.h"
#include <cstdlib>
#include <cstring>

KVCache* kvcache_create(size_t layers, size_t seq_max, size_t dim) {
    KVCache* c = new (std::nothrow) KVCache();
    if (!c) return nullptr;
    c->layers = layers; c->seq_max = seq_max; c->dim = dim;
    c->keys.assign(layers, nullptr);
    c->vals.assign(layers, nullptr);
    c->lengths.assign(layers, 0);
    for (size_t i=0;i<layers;++i) {
        c->keys[i] = tensor_create_f32(seq_max, dim);
        c->vals[i] = tensor_create_f32(seq_max, dim);
        if (!c->keys[i] || !c->vals[i]) {
            kvcache_free(c); return nullptr;
        }
    }
    fprintf(stderr, "[kvcache] create layers=%zu seq_max=%zu dim=%zu\n", layers, seq_max, dim);
    return c;
}

void kvcache_free(KVCache* c) {
    if (!c) return;
    for (size_t i=0;i<c->layers;++i) {
        if (c->keys[i]) tensor_free(c->keys[i]);
        if (c->vals[i]) tensor_free(c->vals[i]);
    }
    delete c;
}

bool kvcache_append(KVCache* c, size_t layer, const float* key_row, const float* val_row) {
    if (!c) return false;
    if (layer >= c->layers) return false;
    size_t pos = c->lengths[layer];
    if (pos >= c->seq_max) {
        // cache full: evict oldest row by shifting everything left by one
        float* kd = (float*)c->keys[layer]->data;
        float* vd = (float*)c->vals[layer]->data;
        size_t d = c->dim;
        // shift rows 1..seq_max-1 -> 0..seq_max-2
        memmove(kd, kd + d, sizeof(float)*(c->seq_max-1)*d);
        memmove(vd, vd + d, sizeof(float)*(c->seq_max-1)*d);
        pos = c->seq_max - 1;
        c->lengths[layer] = c->seq_max; // stays full
        fprintf(stderr, "[kvcache] evict oldest for layer=%zu to make room\n", layer);
    }
    // copy key_row and val_row into tensors at row=pos
    float* kd = (float*)c->keys[layer]->data;
    float* vd = (float*)c->vals[layer]->data;
    for (size_t j=0;j<c->dim;++j) {
        kd[pos * c->dim + j] = key_row[j];
        vd[pos * c->dim + j] = val_row[j];
    }
    if (c->lengths[layer] < c->seq_max) c->lengths[layer]++;
    fprintf(stderr, "[kvcache] append layer=%zu pos=%zu newlen=%zu\n", layer, pos, c->lengths[layer]);
    return true;
}

void kvcache_reset(KVCache* c) {
    if (!c) return;
    for (size_t i=0;i<c->layers;++i) c->lengths[i] = 0;
}
