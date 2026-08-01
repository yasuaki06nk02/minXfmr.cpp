#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include "../src/tensor/tensor.h"
#include "../src/backend/cpu/cpu_backend.h"
#include "../src/backend/cuda/cuda_backend.h"

bool allclose(const float* a, const float* b, size_t n, float rtol = 1e-3f, float atol = 1e-4f) {
    for (size_t i = 0; i < n; ++i) {
        float abs_diff = std::fabs(a[i] - b[i]);
        float rel_err = abs_diff / (std::fabs(b[i]) + 1e-8f);
        if (rel_err > rtol && abs_diff > atol) return false;
    }
    return true;
}

bool test_q4k_dequant_regression() {
    std::vector<uint8_t> block(TENSOR_Q4_K_BLOCK_SIZE, 0);
    uint16_t hd = 0x3c00; // 1.0f
    uint16_t hm = 0x3c00; // 1.0f
    std::memcpy(block.data() + 0, &hd, sizeof(hd));
    std::memcpy(block.data() + 2, &hm, sizeof(hm));

    // Set scale/min values that exercise the j >= 4 path.
    // For the 3rd chunk, the correct path should produce value -1.0f.
    block[4 + 4] = 0x40; // contributes high bits for scale/min
    block[4 + 8] = 0x10; // contributes low bits for scale/min
    block[4 + 0] = 0x00;

    // Put a low nibble in the third chunk's first q entry.
    block[16 + 64] = 0x01;

    std::vector<float> out(TENSOR_Q4_K_QK_K, 0.0f);
    tensor_dequant_q4_k_block(block.data(), out.data());

    const float expected = -1.0f;
    const float actual = out[128];
    const bool ok = std::fabs(actual - expected) < 1e-5f;
    std::cout << (ok ? "  ✓ Q4_K dequant regression passed" : "  ✗ Q4_K dequant regression failed") << std::endl;
    return ok;
}

int main() {
    std::cout << "[TEST] Simple matmul parity test" << std::endl;
    if (!test_q4k_dequant_regression()) return 1;

    Tensor* A = tensor_create_f32(2, 3);
    Tensor* B = tensor_create_f32(3, 2);
    Tensor* C_cpu = tensor_create_f32(2, 2);
    Tensor* C_cuda = tensor_create_f32(2, 2);
    if (!A || !B || !C_cpu || !C_cuda) {
        std::cerr << "Allocation failed" << std::endl;
        return 2;
    }

    float valsA[6] = {1,2,3, 4,5,6};
    float valsB[6] = {7,8, 9,10, 11,12};
    std::memcpy(A->data, valsA, sizeof(valsA));
    std::memcpy(B->data, valsB, sizeof(valsB));

    bool ok_cpu = cpu_matmul(A, B, C_cpu);
    if (!ok_cpu) {
        std::cerr << "cpu_matmul failed" << std::endl;
        return 3;
    }

    bool cuda_ok = false;
    if (cuda_backend_is_available()) {
        cuda_ok = cuda_backend_matmul(A, B, C_cuda);
        if (!cuda_ok) {
            std::cerr << "cuda_matmul failed or unavailable" << std::endl;
        }
    } else {
        std::cout << "CUDA not available; skipping GPU parity check" << std::endl;
    }

    if (cuda_ok) {
        if (allclose((float*)C_cpu->data, (float*)C_cuda->data, 4, 1e-3f, 1e-4f)) {
            std::cout << "  ✓ CPU and CUDA outputs match" << std::endl;
        } else {
            std::cout << "  ✗ CPU and CUDA outputs differ" << std::endl;
            return 1;
        }
    }

    tensor_free(A); tensor_free(B); tensor_free(C_cpu); tensor_free(C_cuda);
    std::cout << "Test finished" << std::endl;
    return 0;
}
