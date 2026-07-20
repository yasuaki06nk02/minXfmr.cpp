#include "kv_cache.h"
#include <cstdlib>
#include <cstring>
#include <vector>

KVCache* kvcache_create(size_t layers, size_t seq_max, size_t dim) {
    KVCache* c = new (std::nothrow) KVCache();
    if (!c) return nullptr;
    c->layers = layers; c->seq_max = seq_max; c->dim = dim;
    c->keys.assign(layers, nullptr);
    c->vals.assign(layers, nullptr);
    c->lengths.assign(layers, 0);
    c->heads.assign(layers, 0);

    // allocate contiguous backing buffers to avoid repeated reallocations during append
    size_t per_layer = seq_max * dim;
    c->keys_buf.clear(); c->vals_buf.clear();
    try {
        c->keys_buf.resize(layers * per_layer);
        c->vals_buf.resize(layers * per_layer);
    } catch (...) {
        kvcache_free(c); return nullptr;
    }

    // create tensor views pointing into the contiguous buffers
    for (size_t i = 0; i < layers; ++i) {
        float* kb = c->keys_buf.data() + i * per_layer;
        float* vb = c->vals_buf.data() + i * per_layer;
        c->keys[i] = tensor_create_f32_view(seq_max, dim, kb);
        c->vals[i] = tensor_create_f32_view(seq_max, dim, vb);
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
    c->keys.clear(); c->vals.clear();
    c->keys_buf.clear(); c->vals_buf.clear();
    delete c;
}

bool kvcache_append(KVCache* c, size_t layer, const float* key_row, const float* val_row) {
    if (!c || layer >= c->layers) return false;
    size_t d = c->dim;
    size_t seq_max = c->seq_max;
    size_t head = c->heads[layer];
    size_t len = c->lengths[layer];

    size_t write_pos;
    if (len < seq_max) {
        write_pos = len;
        c->lengths[layer]++;
    } else {
        // buffer is full, overwrite oldest (at head) and move head forward
        write_pos = head;
        c->heads[layer] = (head + 1) % seq_max;
    }

    // copy key_row and val_row into tensors at row=write_pos
    float* kd = (float*)c->keys[layer]->data;
    float* vd = (float*)c->vals[layer]->data;
    memcpy(kd + write_pos * d, key_row, sizeof(float) * d);
    memcpy(vd + write_pos * d, val_row, sizeof(float) * d);

    // fprintf(stderr, "[kvcache] append layer=%zu write_pos=%zu newlen=%zu head=%zu\n", layer, write_pos, c->lengths[layer], c->heads[layer]);
    return true;
}

void kvcache_reset(KVCache* c) {
    if (!c) return;
    for (size_t i=0;i<c->layers;++i) {
        c->lengths[i] = 0;
        c->heads[i] = 0;
    }
}
