#pragma once
#include <cstddef>
#include <cstdint>

enum class DataType {
    // Standard 32-bit float tensor.
    F32,
    // Packed GGUF Q4_K blocks.
    Q4_K,
};

struct Tensor {
    // Points to contiguous tensor payload. Actual element format depends on `type`.
    void* data = nullptr;
    // Declares how bytes in `data` should be interpreted.
    DataType type = DataType::F32;
    // Matrix shape is always row-major [rows x cols].
    size_t rows = 0;
    size_t cols = 0;
    // Total bytes available at `data`.
    size_t bytes = 0;

    Tensor() noexcept = default;
    ~Tensor() noexcept = default;

    // Make Tensor move-only to prevent accidental expensive copies.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Number of logical elements for dense matrix operations.
    size_t elem_count() const { return rows * cols; }
};

// Allocate and zero-initialize an F32 matrix (rows x cols).
// Returns nullptr on invalid shape or allocation failure.
Tensor* tensor_create_f32(size_t rows, size_t cols);
// Allocate a float matrix without zero-initializing the buffer.
// Use only when the caller fully overwrites all elements before reading.
Tensor* tensor_create_f32_noinit(size_t rows, size_t cols);
// Create a tensor view into externally managed float buffer (does not take ownership of buffer).
Tensor* tensor_create_f32_view(size_t rows, size_t cols, float* buffer);
// Create a packed GGML Q4_K tensor and copy raw block data into owned storage.
Tensor* tensor_create_q4_k_from_bytes(size_t rows, size_t cols, const uint8_t* packed, size_t packed_bytes);

// Q4_K constants used by GGUF and CPU backend paths.
constexpr size_t TENSOR_Q4_K_QK_K = 256;
constexpr size_t TENSOR_Q4_K_BLOCK_SIZE = 2 + 2 + 12 + (TENSOR_Q4_K_QK_K / 2);

// Returns 0 when cols is not representable by Q4_K blocks.
size_t tensor_q4_k_row_bytes(size_t cols);

void tensor_free(Tensor* t);

// Safe F32 getter. Returns 0.0f on type mismatch or out-of-range access.
float tensor_get_f32(const Tensor* t, size_t r, size_t c);
// Safe F32 setter. No-op on type mismatch or out-of-range access.
void tensor_set_f32(Tensor* t, size_t r, size_t c, float v);

// return a new tensor which is the transpose of `in`. Caller owns the result.
Tensor* tensor_transpose_f32(const Tensor* in);
