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
    // Public handle returned to callers.
    Tensor t;
    // Non-null only when this object owns allocated storage.
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
    // Reuse same-size buffers to reduce malloc/free churn in repeated inference calls.
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
    // Keep cache bounded to avoid unbounded memory growth.
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

size_t tensor_q5_0_row_bytes(size_t cols) {
    if (cols == 0 || (cols % TENSOR_Q5_0_QK) != 0) return 0;
    return (cols / TENSOR_Q5_0_QK) * TENSOR_Q5_0_BLOCK_SIZE;
}

size_t tensor_q8_0_row_bytes(size_t cols) {
    if (cols == 0 || (cols % TENSOR_Q8_0_QK) != 0) return 0;
    return (cols / TENSOR_Q8_0_QK) * TENSOR_Q8_0_BLOCK_SIZE;
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

Tensor* tensor_create_q5_0_from_bytes(size_t rows, size_t cols, const uint8_t* packed, size_t packed_bytes) {
    if (rows == 0 || cols == 0 || packed == nullptr) return nullptr;

    const size_t row_bytes = tensor_q5_0_row_bytes(cols);
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
    impl->t.type = DataType::Q5_0;
    impl->t.rows = rows;
    impl->t.cols = cols;
    impl->t.bytes = need;
    return &impl->t;
}

Tensor* tensor_create_q8_0_from_bytes(size_t rows, size_t cols, const uint8_t* packed, size_t packed_bytes) {
    if (rows == 0 || cols == 0 || packed == nullptr) return nullptr;

    const size_t row_bytes = tensor_q8_0_row_bytes(cols);
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
    impl->t.type = DataType::Q8_0;
    impl->t.rows = rows;
    impl->t.cols = cols;
    impl->t.bytes = need;
    return &impl->t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    // `t` points to TensorImpl::t. Recover enclosing allocation.
    TensorImpl* impl = (TensorImpl*)((char*)t - offsetof(TensorImpl, t));
    if (impl->storage) {
        if (impl->t.type == DataType::F32 && impl->t.bytes > 0) {
            // F32 buffers are pooled per-thread for reuse.
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

float tensor_fp16_to_fp32(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t f = h & 0x3ff;
    uint32_t out;
    if (e == 0) {
        if (f == 0) {
            out = s << 31;
        } else {
            e = 1;
            while ((f & 0x400) == 0) { f <<= 1; --e; }
            f &= 0x3ff;
            out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7f800000u | (f << 13);
    } else {
        out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
    }
    float v;
    std::memcpy(&v, &out, sizeof(v));
    return v;
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 4] >> 6) << 4);
    }
}

void tensor_dequant_q4_k_block(const uint8_t* blk, float* dst256) {
    // If NEON is available use the optimized implementation in cpu_backend_neon.cpp
#ifdef __ARM_NEON
    extern void tensor_dequant_q4_k_block_neon(const uint8_t* blk, float* dst256);
    tensor_dequant_q4_k_block_neon(blk, dst256);
    return;
#else
    uint16_t hd = 0, hm = 0;
    std::memcpy(&hd, blk + 0, sizeof(hd));
    std::memcpy(&hm, blk + 2, sizeof(hm));
    const float d = tensor_fp16_to_fp32(hd);
    const float dmin = tensor_fp16_to_fp32(hm);
    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;
    int is = 0;
    for (int j = 0; j < (int)TENSOR_Q4_K_QK_K; j += 64) {
        uint8_t sc = 0, m = 0;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) dst256[j + l] = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) dst256[j + 32 + l] = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
#endif
}

void tensor_dequant_q5_0_block(const uint8_t* blk, float* dst32) {
    uint16_t hd = 0;
    std::memcpy(&hd, blk, sizeof(hd));
    const float d = tensor_fp16_to_fp32(hd);
    const uint8_t* qh = blk + 2;
    const uint8_t* qs = blk + 6;
    uint32_t hmask = (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) |
                     ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
    for (int i = 0; i < 16; ++i) {
        const uint8_t ql = qs[i];
        const int q0 = (((int)((hmask >> i) & 1u)) << 4) | (ql & 0x0F);
        const int q1 = (((int)((hmask >> (i + 16)) & 1u)) << 4) | (ql >> 4);
        dst32[i]      = d * (float)(q0 - 16);
        dst32[i + 16] = d * (float)(q1 - 16);
    }
}

void tensor_dequant_q8_0_block(const uint8_t* blk, float* dst32) {
    uint16_t hd = 0;
    std::memcpy(&hd, blk, sizeof(hd));
    const float d = tensor_fp16_to_fp32(hd);
    const int8_t* qs = (const int8_t*)(blk + 2);
    for (int i = 0; i < 32; ++i) dst32[i] = d * (float)qs[i];
}

bool tensor_dequant_row(const Tensor* t, size_t row, float* dst) {
    if (!t || !dst || row >= t->rows) return false;
    if (t->type == DataType::F32) {
        std::memcpy(dst, (const float*)t->data + row * t->cols, sizeof(float) * t->cols);
        return true;
    }
    if (t->type == DataType::Q4_K) {
        const size_t row_bytes = tensor_q4_k_row_bytes(t->cols);
        if (row_bytes == 0) return false;
        const uint8_t* src = (const uint8_t*)t->data + row * row_bytes;
        const size_t blocks = t->cols / TENSOR_Q4_K_QK_K;
        for (size_t b = 0; b < blocks; ++b)
            tensor_dequant_q4_k_block(src + b * TENSOR_Q4_K_BLOCK_SIZE, dst + b * TENSOR_Q4_K_QK_K);
        return true;
    }
    if (t->type == DataType::Q5_0) {
        const size_t row_bytes = tensor_q5_0_row_bytes(t->cols);
        if (row_bytes == 0) return false;
        const uint8_t* src = (const uint8_t*)t->data + row * row_bytes;
        const size_t blocks = t->cols / TENSOR_Q5_0_QK;
        for (size_t b = 0; b < blocks; ++b)
            tensor_dequant_q5_0_block(src + b * TENSOR_Q5_0_BLOCK_SIZE, dst + b * TENSOR_Q5_0_QK);
        return true;
    }
    if (t->type == DataType::Q8_0) {
        const size_t row_bytes = tensor_q8_0_row_bytes(t->cols);
        if (row_bytes == 0) return false;
        const uint8_t* src = (const uint8_t*)t->data + row * row_bytes;
        const size_t blocks = t->cols / TENSOR_Q8_0_QK;
        for (size_t b = 0; b < blocks; ++b)
            tensor_dequant_q8_0_block(src + b * TENSOR_Q8_0_BLOCK_SIZE, dst + b * TENSOR_Q8_0_QK);
        return true;
    }
    return false;
}
