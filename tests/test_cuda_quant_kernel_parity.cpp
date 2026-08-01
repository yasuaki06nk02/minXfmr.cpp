#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "../src/tensor/tensor.h"
#include "../src/backend/cuda/cuda_backend.h"

int main() {
    std::cout << "[TEST] CUDA quant kernel parity (host dequant vs direct)" << std::endl;
    if (!cuda_backend_is_available()) {
        std::cout << "CUDA not available; skipping test" << std::endl;
        return 0;
    }

    const size_t K = TENSOR_Q4_K_QK_K; // 256
    const size_t N = TENSOR_Q4_K_QK_K; // 256

    const size_t row_bytes = tensor_q4_k_row_bytes(N);
    if (row_bytes == 0) {
        std::cerr << "Invalid quant layout" << std::endl;
        return 2;
    }

    const size_t rows = K;
    const size_t cols = N;
    const size_t packed_bytes = rows * row_bytes;
    std::vector<uint8_t> packed(packed_bytes, 0);

    // Build one block that exercises j >= 4 branch similarly to regression test.
    std::vector<uint8_t> blk(TENSOR_Q4_K_BLOCK_SIZE, 0);
    uint16_t hd = 0x3c00; // 1.0f
    uint16_t hm = 0x3c00; // 1.0f
    std::memcpy(blk.data() + 0, &hd, sizeof(hd));
    std::memcpy(blk.data() + 2, &hm, sizeof(hm));
    // prepare scales so that one element becomes -1.0f after dequant
    blk[4 + 4] = 0x40;
    blk[4 + 8] = 0x10;
    blk[4 + 0] = 0x00;
    // set low nibble in the third chunk's first q entry
    blk[16 + 64] = 0x01;

    // Fill every row with the same block
    for (size_t r = 0; r < rows; ++r) {
        std::memcpy(packed.data() + r * row_bytes, blk.data(), row_bytes);
    }

    Tensor* A = tensor_create_f32(1, K);
    Tensor* Bq = tensor_create_q4_k_from_bytes(rows, cols, packed.data(), packed.size());
    Tensor* Cout = tensor_create_f32(1, cols);
    if (!A || !Bq || !Cout) {
        std::cerr << "Allocation failed" << std::endl;
        return 3;
    }

    // Fill A with ones
    float* ad = (float*)A->data;
    for (size_t i = 0; i < K; ++i) ad[i] = 1.0f;

    // Run parity (host dequant via device F32 cache)
    cuda_backend_set_quant_parity_mode(1);
    bool ok1 = cuda_backend_matmul(A, Bq, Cout);
    if (!ok1) {
        std::cerr << "cuda_backend_matmul (parity) failed: " << cuda_backend_last_error_msg() << std::endl;
        return 4;
    }
    std::vector<float> out_parity(cols);
    std::memcpy(out_parity.data(), Cout->data, cols * sizeof(float));

    // Run direct quant kernels
    cuda_backend_set_quant_parity_mode(0);
    bool ok2 = cuda_backend_matmul(A, Bq, Cout);
    if (!ok2) {
        std::cerr << "cuda_backend_matmul (direct) failed: " << cuda_backend_last_error_msg() << std::endl;
        return 5;
    }
    std::vector<float> out_direct(cols);
    std::memcpy(out_direct.data(), Cout->data, cols * sizeof(float));

    // Compare
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < cols; ++i) {
        double d = std::fabs((double)out_parity[i] - (double)out_direct[i]);
        sum_abs += d;
        if (d > max_abs) max_abs = d;
    }
    std::cout << "Max abs diff: " << max_abs << ", mean abs diff: " << (sum_abs / cols) << std::endl;

    tensor_free(A); tensor_free(Bq); tensor_free(Cout);
    return (max_abs < 1e-3) ? 0 : 6;
}
