# 13_TENSOR_SPEC.md

# minXfmr.cpp Tensor Specification (Beginner Friendly)

> Detailed specification for `src/tensor` used by model loading and compute paths.

---

# 1. Purpose

The Tensor module provides a small and safe matrix container.

It is responsible for:

* holding tensor bytes
* describing tensor type and shape
* allocating and releasing tensor memory
* basic F32 read/write helpers
* creating transpose copies for F32 matrices

It is **not** responsible for:

* running attention or feed-forward logic
* GGUF file parsing
* backend-specific math kernels

---

# 2. Data Model

## 2.1 Tensor fields

```cpp
struct Tensor {
    void* data;
    DataType type;
    size_t rows;
    size_t cols;
    size_t bytes;
};
```

Meaning of each field:

* `data`: start address of contiguous tensor payload
* `type`: payload format (`F32` or `Q4_K`)
* `rows`, `cols`: logical matrix shape
* `bytes`: total bytes at `data`

## 2.2 DataType

```cpp
enum class DataType {
    F32,
    Q4_K,
};
```

* `F32`: regular 32-bit float matrix
* `Q4_K`: packed GGUF quantized blocks

---

# 3. Memory Ownership

Two ownership modes exist.

## 3.1 Owned tensor

Created by:

* `tensor_create_f32`
* `tensor_create_f32_noinit`
* `tensor_create_q4_k_from_bytes`

Behavior:

* tensor allocates aligned storage
* `tensor_free` eventually releases or caches storage

## 3.2 View tensor

Created by:

* `tensor_create_f32_view`

Behavior:

* tensor wraps external `float*` buffer
* tensor does not own wrapped data memory
* `tensor_free` destroys wrapper only

Caller rule:

* external buffer must remain valid while tensor is used

---

# 4. Layout Rules

## 4.1 F32 layout

F32 is row-major:

```text
index = row * cols + col
```

Address formula:

```text
((float*)data)[row * cols + col]
```

## 4.2 Q4_K layout

Q4_K is packed by GGUF block layout.

Constants:

```cpp
TENSOR_Q4_K_QK_K = 256
TENSOR_Q4_K_BLOCK_SIZE = 2 + 2 + 12 + (256 / 2)
```

Row validity:

```text
cols % 256 == 0
```

Row byte size:

```text
row_bytes = (cols / 256) * TENSOR_Q4_K_BLOCK_SIZE
```

If row format is invalid, `tensor_q4_k_row_bytes` returns `0`.

---

# 5. API Contract

## 5.1 `tensor_create_f32(rows, cols)`

Purpose:

* create owned F32 matrix and zero-initialize bytes

Preconditions:

* `rows > 0`
* `cols > 0`

Returns:

* `Tensor*` on success
* `nullptr` on invalid shape or allocation failure

## 5.2 `tensor_create_f32_noinit(rows, cols)`

Purpose:

* create owned F32 matrix without memset

Use when:

* caller will fully overwrite all elements

Risk:

* reading before write is undefined behavior

## 5.3 `tensor_create_f32_view(rows, cols, buffer)`

Purpose:

* wrap external float memory as tensor

Preconditions:

* `rows > 0`
* `cols > 0`
* `buffer != nullptr`

Ownership:

* no ownership of `buffer`

## 5.4 `tensor_create_q4_k_from_bytes(rows, cols, packed, packed_bytes)`

Purpose:

* create owned Q4_K tensor by copying packed bytes

Preconditions:

* valid shape
* `packed != nullptr`
* `packed_bytes >= rows * tensor_q4_k_row_bytes(cols)`

Returns `nullptr` if any check fails.

## 5.5 `tensor_free(t)`

Purpose:

* release tensor wrapper and owned storage (if any)

Safe behavior:

* `tensor_free(nullptr)` is allowed

## 5.6 `tensor_get_f32(t, r, c)`

Purpose:

* read one F32 element safely

Fallback behavior:

* returns `0.0f` on type mismatch or out-of-range index

## 5.7 `tensor_set_f32(t, r, c, v)`

Purpose:

* write one F32 element safely

Fallback behavior:

* no-op on type mismatch or out-of-range index

## 5.8 `tensor_transpose_f32(in)`

Purpose:

* create owned transposed copy of an F32 tensor

Input shape:

* `[rows x cols]`

Output shape:

* `[cols x rows]`

---

# 6. Alignment and Reuse

Owned storage is aligned to 64 bytes to help SIMD-friendly access.

For F32 owned tensors, implementation uses a thread-local cache keyed by byte size.

Cache policy:

* max total cached bytes: `64 MiB` per thread
* max entries per size bucket: `16`

Why:

* reduces allocation churn in repetitive inference loops

---

# 7. Error Handling Philosophy

This module prefers defensive behavior over throwing exceptions.

Common failure signals:

* creation APIs: `nullptr`
* q4 row byte helper: `0`
* safe F32 read: `0.0f`
* safe F32 write: no-op

This supports simple MVP call sites and prevents crash cascades.

---

# 8. Typical Usage

```cpp
Tensor* a = tensor_create_f32(2, 3);
if (!a) {
    // handle allocation failure
}

for (size_t r = 0; r < a->rows; ++r) {
    for (size_t c = 0; c < a->cols; ++c) {
        tensor_set_f32(a, r, c, static_cast<float>(r * 10 + c));
    }
}

Tensor* at = tensor_transpose_f32(a);
if (!at) {
    tensor_free(a);
    // handle transpose failure
}

tensor_free(at);
tensor_free(a);
```

---

# 9. Non-goals and Future Work

Current non-goals:

* ND tensor rank beyond 2D matrix abstraction
* shape broadcasting rules
* arithmetic operations inside tensor module

Possible future additions:

* explicit `owner` flag in public metadata
* debug assertions for integer overflow checks
* optional strict APIs that report detailed error codes
