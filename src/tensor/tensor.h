#pragma once
#include <cstddef>

enum class DataType {
    F32,
};

struct Tensor {
    void* data = nullptr;
    DataType type = DataType::F32;
    size_t rows = 0;
    size_t cols = 0;
    size_t bytes = 0;

    Tensor() noexcept = default;
    ~Tensor() noexcept = default;

    // Make Tensor move-only to prevent accidental expensive copies.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    size_t elem_count() const { return rows * cols; }
};

// allocate a float matrix (rows x cols)
Tensor* tensor_create_f32(size_t rows, size_t cols);
void tensor_free(Tensor* t);
float tensor_get_f32(const Tensor* t, size_t r, size_t c);
void tensor_set_f32(Tensor* t, size_t r, size_t c, float v);

// return a new tensor which is the transpose of `in`. Caller owns the result.
Tensor* tensor_transpose_f32(const Tensor* in);
