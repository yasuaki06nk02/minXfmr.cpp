#include "tensor/tensor.h"
#include <cstdlib>
#include <new>
#include <cstring>
#include <vector>
#include <unordered_map>
#ifdef _WIN32
#include <malloc.h>
#endif

struct TensorImpl {
    Tensor t;
    void* storage;
};

namespace {
struct F32StorageCache {
    std::unordered_map<size_t, std::vector<void*>> buckets;
    size_t bytes_cached = 0;
};

static thread_local F32StorageCache g_f32_cache;
static constexpr size_t kMaxCachedBytes = 64u * 1024u * 1024u;
static constexpr size_t kMaxBucketEntries = 16;

static void* alloc_aligned_64(size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, 64);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
    return p;
#endif
}

static void free_aligned_64(void* p) {
    if (!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

static void* acquire_f32_storage(size_t bytes) {
    auto it = g_f32_cache.buckets.find(bytes);
    if (it != g_f32_cache.buckets.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        g_f32_cache.bytes_cached -= bytes;
        return p;
    }
    return alloc_aligned_64(bytes);
}

static void release_f32_storage(void* p, size_t bytes) {
    if (!p || bytes == 0) return;
    std::vector<void*>& bucket = g_f32_cache.buckets[bytes];
    if (bucket.size() >= kMaxBucketEntries || g_f32_cache.bytes_cached + bytes > kMaxCachedBytes) {
        free_aligned_64(p);
        return;
    }
    bucket.push_back(p);
    g_f32_cache.bytes_cached += bytes;
}
}

size_t tensor_q4_k_row_bytes(size_t cols) {
    if (cols == 0 || (cols % TENSOR_Q4_K_QK_K) != 0) return 0;
    return (cols / TENSOR_Q4_K_QK_K) * TENSOR_Q4_K_BLOCK_SIZE;
}

Tensor* tensor_create_f32(size_t rows, size_t cols) {
    Tensor* t = tensor_create_f32_noinit(rows, cols);
    if (!t) return nullptr;
    std::memset(t->data, 0, t->bytes);
    return t;
}

Tensor* tensor_create_f32_noinit(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return nullptr;
    TensorImpl* impl = new (std::nothrow) TensorImpl();
    if (!impl) return nullptr;
    size_t bytes = rows * cols * sizeof(float);
    impl->storage = acquire_f32_storage(bytes);
    if (!impl->storage) { delete impl; return nullptr; }
    impl->t.data = impl->storage;
    impl->t.type = DataType::F32;
    impl->t.rows = rows;
    impl->t.cols = cols;
    impl->t.bytes = bytes;
    return &impl->t;
}

Tensor* tensor_create_f32_view(size_t rows, size_t cols, float* buffer) {
    if (rows == 0 || cols == 0 || buffer == nullptr) return nullptr;
    TensorImpl* impl = new (std::nothrow) TensorImpl();
    if (!impl) return nullptr;
    impl->storage = nullptr; // we don't own the external buffer
    impl->t.data = buffer;
    impl->t.type = DataType::F32;
    impl->t.rows = rows;
    impl->t.cols = cols;
    impl->t.bytes = rows * cols * sizeof(float);
    return &impl->t;
}

Tensor* tensor_create_q4_k_from_bytes(size_t rows, size_t cols, const uint8_t* packed, size_t packed_bytes) {
    if (rows == 0 || cols == 0 || packed == nullptr) return nullptr;

    const size_t row_bytes = tensor_q4_k_row_bytes(cols);
    if (row_bytes == 0) return nullptr;

    const size_t need = rows * row_bytes;
    if (packed_bytes < need) return nullptr;

    TensorImpl* impl = new (std::nothrow) TensorImpl();
    if (!impl) return nullptr;

#ifdef _WIN32
    impl->storage = _aligned_malloc(need, 64);
    if (!impl->storage) { delete impl; return nullptr; }
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, need) != 0) { delete impl; return nullptr; }
    impl->storage = p;
#endif

    std::memcpy(impl->storage, packed, need);

    impl->t.data = impl->storage;
    impl->t.type = DataType::Q4_K;
    impl->t.rows = rows;
    impl->t.cols = cols;
    impl->t.bytes = need;
    return &impl->t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    TensorImpl* impl = (TensorImpl*)((char*)t - offsetof(TensorImpl, t));
    if (impl->storage) {
        if (impl->t.type == DataType::F32 && impl->t.bytes > 0) {
            release_f32_storage(impl->storage, impl->t.bytes);
        } else {
            free_aligned_64(impl->storage);
        }
    }
    delete impl;
}

float tensor_get_f32(const Tensor* t, size_t r, size_t c) {
    if (!t || t->type != DataType::F32) return 0.0f;
    if (r >= t->rows || c >= t->cols) return 0.0f;
    const float* storage = (const float*)t->data;
    return storage[r * t->cols + c];
}

void tensor_set_f32(Tensor* t, size_t r, size_t c, float v) {
    if (!t || t->type != DataType::F32) return;
    if (r >= t->rows || c >= t->cols) return;
    float* storage = (float*)t->data;
    storage[r * t->cols + c] = v;
}

Tensor* tensor_transpose_f32(const Tensor* in) {
    if (!in || in->type != DataType::F32) return nullptr;
    Tensor* out = tensor_create_f32(in->cols, in->rows);
    if (!out) return nullptr;
    const float* src = (const float*)in->data;
    float* dst = (float*)out->data;
    for (size_t r=0;r<in->rows;++r) {
        for (size_t c=0;c<in->cols;++c) dst[c*in->rows + r] = src[r*in->cols + c];
    }
    return out;
}
