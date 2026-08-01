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
        d = (q[j + 4] & 0xF) | ((q[j] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
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

// Helper: convert f32 -> IEEE-754 binary16 (round-to-nearest even)
static uint16_t f32_to_fp16_bits(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000u;
    uint32_t mant = f & 0x007FFFFFu;
    int32_t exp = (int32_t)((f >> 23) & 0xFFu) - 127;

    if (exp == 128) {
        // Inf or NaN
        if (mant == 0) return (uint16_t)(sign | 0x7C00u);
        // NaN -> canonical qNaN
        return (uint16_t)(sign | 0x7E00u);
    }

    int32_t exp16 = exp + 15;
    if (exp16 >= 31) {
        // overflow -> Inf
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp16 <= 0) {
        // subnormal or zero
        if (exp16 < -10) {
            return (uint16_t)sign; // underflow to zero
        }
        // convert to subnormal half
        mant |= 0x00800000u; // add implicit leading 1
        int32_t shift = 14 - exp16;
        uint32_t m = mant >> shift;
        // rounding: check bit (shift-1)
        uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        m += round_bit; // round to nearest (ties to even not strictly implemented)
        return (uint16_t)(sign | (m & 0x03FFu));
    }

    // Normalized
    uint32_t m16 = mant >> 13;
    // round to nearest
    uint32_t round_bits = (mant >> 12) & 1u;
    m16 += round_bits;
    if (m16 & 0x0400u) {
        // mantissa overflow
        m16 = 0;
        exp16 += 1;
        if (exp16 >= 31) return (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | ((exp16 & 0x1Fu) << 10) | (m16 & 0x03FFu));
}

// Quantize one Q8_0 block: write fp16 scale then 32 int8 values.
static void tensor_quant_q8_0_block(const float* src32, uint8_t* out_blk) {
    constexpr size_t K = TENSOR_Q8_0_QK;
    float max_abs = 0.0f;
    for (size_t i = 0; i < K; ++i) {
        float a = src32[i];
        float aa = std::fabs(a);
        if (aa > max_abs) max_abs = aa;
    }
    float d = 0.0f;
    if (max_abs > 1e-12f) d = max_abs / 127.0f;
    uint16_t hd = f32_to_fp16_bits(d);
    std::memcpy(out_blk + 0, &hd, sizeof(hd));
    int8_t* qout = (int8_t*)(out_blk + 2);
    if (d == 0.0f) {
        for (size_t i = 0; i < K; ++i) qout[i] = 0;
        return;
    }
    for (size_t i = 0; i < K; ++i) {
        int32_t qi = (int32_t)std::lround(src32[i] / d);
        if (qi < -128) qi = -128;
        if (qi > 127) qi = 127;
        qout[i] = (int8_t)qi;
    }
}

// Quantize one Q5_0 block: write fp16 scale, 4-byte hmask, and 16 bytes of packed low-nibbles.
static void tensor_quant_q5_0_block(const float* src32, uint8_t* out_blk) {
    constexpr size_t K = TENSOR_Q5_0_QK;
    float max_abs = 0.0f;
    for (size_t i = 0; i < K; ++i) {
        float a = src32[i];
        float aa = std::fabs(a);
        if (aa > max_abs) max_abs = aa;
    }
    float d = 0.0f;
    if (max_abs > 1e-12f) d = max_abs / 15.0f; // q range [-16..15]
    uint16_t hd = f32_to_fp16_bits(d);
    std::memcpy(out_blk + 0, &hd, sizeof(hd));
    uint32_t hmask = 0;
    uint8_t qs[16];
    for (size_t i = 0; i < 16; ++i) qs[i] = 0;

    if (d == 0.0f) {
        // all zeros -> q0 = 16
        for (size_t i = 0; i < K; ++i) {
            uint8_t q0 = 16u;
            if (i < 16) qs[i] = (qs[i] & 0xF0) | (q0 & 0x0F);
            else qs[i - 16] = (qs[i - 16] & 0x0F) | ((q0 & 0x0F) << 4);
            if ((q0 >> 4) & 1u) hmask |= (1u << i);
        }
    } else {
        for (size_t i = 0; i < K; ++i) {
            int32_t qi = (int32_t)std::lround(src32[i] / d);
            if (qi < -16) qi = -16;
            if (qi > 15) qi = 15;
            uint8_t q0 = (uint8_t)(qi + 16); // 0..31
            if (i < 16) {
                qs[i] = (qs[i] & 0xF0) | (q0 & 0x0F);
            } else {
                qs[i - 16] = (qs[i - 16] & 0x0F) | ((q0 & 0x0F) << 4);
            }
            if ((q0 >> 4) & 1u) hmask |= (1u << i);
        }
    }
    // write hmask little-endian
    out_blk[2] = (uint8_t)(hmask & 0xFFu);
    out_blk[3] = (uint8_t)((hmask >> 8) & 0xFFu);
    out_blk[4] = (uint8_t)((hmask >> 16) & 0xFFu);
    out_blk[5] = (uint8_t)((hmask >> 24) & 0xFFu);
    // write low-nibble packed bytes
    std::memcpy(out_blk + 6, qs, 16);
}

// Quantize one Q4_K block (256 values) into GGUF Q4_K block layout.
// This is an approximate re-quantizer matching the dequant expectations.
static void tensor_quant_q4_k_block(const float* src256, uint8_t* out_blk) {
    constexpr size_t K = TENSOR_Q4_K_QK_K; // 256
    // Compute per-32-group min/max
    const size_t groups = 8; // 8 groups of 32
    float gmin[groups];
    float gmax[groups];
    for (size_t g = 0; g < groups; ++g) {
        gmin[g] = 1e30f; gmax[g] = -1e30f;
        for (size_t i = 0; i < 32; ++i) {
            float v = src256[g * 32 + i];
            if (v < gmin[g]) gmin[g] = v;
            if (v > gmax[g]) gmax[g] = v;
        }
    }

    // compute independent base scales for sc (delta) and mm (min) to fit into 6 bits
    float d_scale = 0.0f;
    float dmin_scale = 0.0f;
    for (size_t g = 0; g < groups; ++g) {
        float minv = gmin[g];
        float maxv = gmax[g];
        float delta = maxv - minv;
        if (delta > 0.0f) {
            float need = delta / (15.0f * 63.0f);
            if (need > d_scale) d_scale = need;
        }
        if (minv < 0.0f) {
            float need = (-minv) / 63.0f;
            if (need > dmin_scale) dmin_scale = need;
        }
    }
    if (d_scale <= 0.0f) d_scale = 1e-8f;
    if (dmin_scale <= 0.0f) dmin_scale = d_scale;

    // compute sc (0..63) and mm (0..63) per group using independent scales
    uint8_t sc[groups];
    uint8_t mm[groups];
    for (size_t g = 0; g < groups; ++g) {
        float minv = gmin[g];
        float maxv = gmax[g];
        float delta = maxv - minv;
        int isc = (int)std::lround((double)(delta / (15.0f * d_scale)));
        if (isc < 1) isc = 1;
        if (isc > 63) isc = 63;
        sc[g] = (uint8_t)isc;
        int imm = (int)std::lround((double)((minv < 0.0f) ? (-minv / dmin_scale) : 0.0f));
        if (imm < 0) imm = 0;
        if (imm > 63) imm = 63;
        mm[g] = (uint8_t)imm;
    }

    // pack header d and dmin
    uint16_t hd = f32_to_fp16_bits(d_scale);
    uint16_t hm = f32_to_fp16_bits(dmin_scale);
    std::memcpy(out_blk + 0, &hd, sizeof(hd));
    std::memcpy(out_blk + 2, &hm, sizeof(hm));

    // pack 12-byte scales area exactly as dequant expects
    uint8_t S[12];
    std::memset(S, 0, sizeof(S));
    // groups 0..3: store sc low6 in S[0..3], mm low6 in S[4..7]
    for (size_t i = 0; i < 4; ++i) {
        S[i] = sc[i] & 0x3Fu;
        S[i + 4] = mm[i] & 0x3Fu;
    }
    // groups 4..7: low 4 bits go into S[8..11] nibbles; top 2 bits stored in S[4..7] bits 6..7
    for (size_t j = 4; j < 8; ++j) {
        uint8_t low4_d = sc[j] & 0xFu;
        uint8_t low4_m = mm[j] & 0xFu;
        // put low nibbles into S[8..11]
        S[8 + (j - 4)] = (uint8_t)((low4_d & 0xFu) | ((low4_m & 0xFu) << 4));
        // top bits derived from sc (use sc top bits)
        uint8_t hi_d = (sc[j] >> 4) & 0x3u;
        S[4 + (j - 4)] |= (uint8_t)((hi_d & 0x3u) << 6);
    }
    std::memcpy(out_blk + 4, S, sizeof(S));

    // now pack 128 bytes of nibbles
    // layout: 4 segments of 32 bytes. For each seg, for l in 0..31: low nibble = q_low, high nibble = q_high
    // q for element index idx
    for (size_t seg = 0; seg < 4; ++seg) {
        size_t base64 = seg * 64;
        for (size_t l = 0; l < 32; ++l) {
            size_t idx_low = base64 + l;         // first 32
            size_t idx_high = base64 + 32 + l;   // next 32
            // determine group numbers (0..7)
            size_t g_low = idx_low / 32;
            size_t g_high = idx_high / 32;
            // compute q values within 0..15
            int ql = 0;
            int qh = 0;
            // avoid division by zero
            float dsc_low = (float)sc[g_low] * d_scale;
            float dsc_high = (float)sc[g_high] * d_scale;
            if (dsc_low == 0.0f) ql = 0; else ql = (int)std::lround((double)((src256[idx_low] + dmin_scale * mm[g_low]) / dsc_low));
            if (dsc_high == 0.0f) qh = 0; else qh = (int)std::lround((double)((src256[idx_high] + dmin_scale * mm[g_high]) / dsc_high));
            if (ql < 0) ql = 0; if (ql > 15) ql = 15;
            if (qh < 0) qh = 0; if (qh > 15) qh = 15;
            out_blk[16 + seg * 32 + l] = (uint8_t)((qh << 4) | (ql & 0xF));
        }
    }
}

bool tensor_transpose_packed_inplace(Tensor*& t) {
    if (!t) return false;
    if (t->type == DataType::Q4_K) {
        const size_t R = t->rows;
        const size_t C = t->cols;
        const size_t K = TENSOR_Q4_K_QK_K;
        if ((C % K) != 0) return false;
        if ((R % K) != 0) return false;
        const size_t blocks_per_row_src = C / K;
        const size_t blocks_per_row_dst = R / K;
        const size_t src_row_bytes = tensor_q4_k_row_bytes(C);
        const size_t dst_row_bytes = tensor_q4_k_row_bytes(R);
        const size_t new_rows = C;
        const size_t new_cols = R;
        const size_t total_bytes = new_rows * dst_row_bytes;
        std::vector<uint8_t> out(total_bytes);
        const uint8_t* srcp = (const uint8_t*)t->data;
        uint8_t* outp = out.data();

        // Temporary tile of K rows x K cols (may be large: 256*256 floats)
        std::vector<float> tile;
        try { tile.resize(K * K); } catch(...) { return false; }
        std::vector<float> vals;
        try { vals.resize(K); } catch(...) { return false; }

        for (size_t bcol = 0; bcol < blocks_per_row_src; ++bcol) {
            for (size_t bdrow = 0; bdrow < blocks_per_row_dst; ++bdrow) {
                // dequantize K rows for this tile
                for (size_t rrel = 0; rrel < K; ++rrel) {
                    size_t row = bdrow * K + rrel;
                    const uint8_t* src_blk = srcp + row * src_row_bytes + bcol * TENSOR_Q4_K_BLOCK_SIZE;
                    float* dst = tile.data() + rrel * K;
                    tensor_dequant_q4_k_block(src_blk, dst);
                }
                // For each position in block column, build output block
                for (size_t pos = 0; pos < K; ++pos) {
                    size_t new_row = bcol * K + pos;
                    uint8_t* dst_blk = outp + new_row * dst_row_bytes + bdrow * TENSOR_Q4_K_BLOCK_SIZE;
                    // gather K values (column from tile)
                    for (size_t k = 0; k < K; ++k) vals[k] = tile[k * K + pos];
                    tensor_quant_q4_k_block(vals.data(), dst_blk);
                }
            }
        }

        Tensor* newt = tensor_create_q4_k_from_bytes(new_rows, new_cols, out.data(), out.size());
        if (!newt) return false;
        tensor_free(t);
        t = newt;
        return true;
    }
    if (t->type == DataType::Q8_0) {
        const size_t R = t->rows;
        const size_t C = t->cols;
        const size_t K = TENSOR_Q8_0_QK;
        if ((C % K) != 0) return false;
        if ((R % K) != 0) return false;
        const size_t blocks_per_row_src = C / K;
        const size_t blocks_per_row_dst = R / K;
        const size_t src_row_bytes = tensor_q8_0_row_bytes(C);
        const size_t dst_row_bytes = tensor_q8_0_row_bytes(R);
        const size_t new_rows = C;
        const size_t new_cols = R;
        const size_t total_bytes = new_rows * dst_row_bytes;
        std::vector<uint8_t> out(total_bytes);
        const uint8_t* srcp = (const uint8_t*)t->data;
        uint8_t* outp = out.data();

        // Temporary tile of K rows x K cols
        std::vector<float> tile(K * K);

        for (size_t bcol = 0; bcol < blocks_per_row_src; ++bcol) {
            for (size_t bdrow = 0; bdrow < blocks_per_row_dst; ++bdrow) {
                // dequantize K rows for this tile
                for (size_t rrel = 0; rrel < K; ++rrel) {
                    size_t row = bdrow * K + rrel;
                    const uint8_t* src_blk = srcp + row * src_row_bytes + bcol * TENSOR_Q8_0_BLOCK_SIZE;
                    float* dst = tile.data() + rrel * K;
                    tensor_dequant_q8_0_block(src_blk, dst);
                }
                // For each position in block column, build output block
                for (size_t pos = 0; pos < K; ++pos) {
                    size_t new_row = bcol * K + pos;
                    uint8_t* dst_blk = outp + new_row * dst_row_bytes + bdrow * TENSOR_Q8_0_BLOCK_SIZE;
                    // gather K values
                    // src value at (row = bdrow*K + rrel, col = bcol*K + pos) is tile[rrel][pos]
                    std::vector<float> vals(K);
                    for (size_t k = 0; k < K; ++k) vals[k] = tile[k * K + pos];
                    tensor_quant_q8_0_block(vals.data(), dst_blk);
                }
            }
        }

        Tensor* newt = tensor_create_q8_0_from_bytes(new_rows, new_cols, out.data(), out.size());
        if (!newt) return false;
        tensor_free(t);
        t = newt;
        return true;
    }

    if (t->type == DataType::Q5_0) {
        const size_t R = t->rows;
        const size_t C = t->cols;
        const size_t K = TENSOR_Q5_0_QK;
        if ((C % K) != 0) return false;
        if ((R % K) != 0) return false;
        const size_t blocks_per_row_src = C / K;
        const size_t blocks_per_row_dst = R / K;
        const size_t src_row_bytes = tensor_q5_0_row_bytes(C);
        const size_t dst_row_bytes = tensor_q5_0_row_bytes(R);
        const size_t new_rows = C;
        const size_t new_cols = R;
        const size_t total_bytes = new_rows * dst_row_bytes;
        std::vector<uint8_t> out(total_bytes);
        const uint8_t* srcp = (const uint8_t*)t->data;
        uint8_t* outp = out.data();

        // Temporary tile of K rows x K cols
        std::vector<float> tile(K * K);

        for (size_t bcol = 0; bcol < blocks_per_row_src; ++bcol) {
            for (size_t bdrow = 0; bdrow < blocks_per_row_dst; ++bdrow) {
                // dequantize K rows for this tile
                for (size_t rrel = 0; rrel < K; ++rrel) {
                    size_t row = bdrow * K + rrel;
                    const uint8_t* src_blk = srcp + row * src_row_bytes + bcol * TENSOR_Q5_0_BLOCK_SIZE;
                    float* dst = tile.data() + rrel * K;
                    tensor_dequant_q5_0_block(src_blk, dst);
                }
                // For each position in block column, build output block
                for (size_t pos = 0; pos < K; ++pos) {
                    size_t new_row = bcol * K + pos;
                    uint8_t* dst_blk = outp + new_row * dst_row_bytes + bdrow * TENSOR_Q5_0_BLOCK_SIZE;
                    // gather K values
                    std::vector<float> vals(K);
                    for (size_t k = 0; k < K; ++k) vals[k] = tile[k * K + pos];
                    tensor_quant_q5_0_block(vals.data(), dst_blk);
                }
            }
        }

        Tensor* newt = tensor_create_q5_0_from_bytes(new_rows, new_cols, out.data(), out.size());
        if (!newt) return false;
        tensor_free(t);
        t = newt;
        return true;
    }

    // Q4_K not currently supported for packed transpose; caller should
    // fall back to dequant->F32 path in that case.
    return false;
}
