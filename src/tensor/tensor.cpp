#include "tensor/tensor.h"
#include <cstdlib>
#include <new>
#include <cstring>
#ifdef _WIN32
#include <malloc.h>
#endif

struct TensorImpl {
    Tensor t;
    float* storage;
};

Tensor* tensor_create_f32(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return nullptr;
    TensorImpl* impl = new (std::nothrow) TensorImpl();
    if (!impl) return nullptr;
    size_t bytes = rows * cols * sizeof(float);
#ifdef _WIN32
    impl->storage = (float*)_aligned_malloc(bytes, 64);
    if (!impl->storage) { delete impl; return nullptr; }
    std::memset(impl->storage, 0, bytes);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) { delete impl; return nullptr; }
    impl->storage = (float*)p;
    std::memset(impl->storage, 0, bytes);
#endif
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

void tensor_free(Tensor* t) {
    if (!t) return;
    TensorImpl* impl = (TensorImpl*)((char*)t - offsetof(TensorImpl, t));
    if (impl->storage) {
#ifdef _WIN32
        _aligned_free(impl->storage);
#else
        std::free(impl->storage);
#endif
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
