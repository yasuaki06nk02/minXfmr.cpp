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
    // Test multiple quant formats: Q4_K, Q5_0, Q8_0
    bool all_ok = true;
    struct TestFmt { int id; const char* name; } fmts[] = {
        {0, "Q4_K"},
        {1, "Q5_0"},
        {2, "Q8_0"}
    };

    for (auto f : fmts) {
        size_t K = 0, N = 0;
        size_t row_bytes = 0;
        std::vector<uint8_t> packed;
        if (f.id == 0) {
            K = TENSOR_Q4_K_QK_K; N = TENSOR_Q4_K_QK_K;
            row_bytes = tensor_q4_k_row_bytes(N);
            if (row_bytes == 0) { std::cerr << "Invalid Q4 layout" << std::endl; return 2; }
            const size_t rows = K, cols = N;
            packed.assign(rows * row_bytes, 0);
            std::vector<uint8_t> blk(TENSOR_Q4_K_BLOCK_SIZE, 0);
            uint16_t hd = 0x3c00; uint16_t hm = 0x3c00; // 1.0f
            std::memcpy(blk.data() + 0, &hd, sizeof(hd));
            std::memcpy(blk.data() + 2, &hm, sizeof(hm));
            // scales set to produce one -1.0f value in the block
            blk[4 + 4] = 0x40;
            blk[4 + 8] = 0x10;
            blk[4 + 0] = 0x00;
            blk[16 + 64] = 0x01; // set low nibble in the third chunk's first q entry
            for (size_t r = 0; r < rows; ++r) std::memcpy(packed.data() + r * row_bytes, blk.data(), row_bytes);

        } else if (f.id == 1) {
            K = TENSOR_Q5_0_QK; N = TENSOR_Q5_0_QK;
            row_bytes = tensor_q5_0_row_bytes(N);
            if (row_bytes == 0) { std::cerr << "Invalid Q5 layout" << std::endl; return 2; }
            const size_t rows = K, cols = N;
            packed.assign(rows * row_bytes, 0);
            std::vector<uint8_t> blk(TENSOR_Q5_0_BLOCK_SIZE, 0);
            uint16_t hd = 0x3c00; // 1.0f
            std::memcpy(blk.data(), &hd, sizeof(hd));
            // set qs[0] low nibble = 15 to make one element = -1 (d*(15-16) = -1)
            blk[2 + 4 + 0] = 0x0F; // qs[0]
            for (size_t r = 0; r < rows; ++r) std::memcpy(packed.data() + r * row_bytes, blk.data(), row_bytes);

        } else {
            K = TENSOR_Q8_0_QK; N = TENSOR_Q8_0_QK;
            row_bytes = tensor_q8_0_row_bytes(N);
            if (row_bytes == 0) { std::cerr << "Invalid Q8 layout" << std::endl; return 2; }
            const size_t rows = K, cols = N;
            packed.assign(rows * row_bytes, 0);
            std::vector<uint8_t> blk(TENSOR_Q8_0_BLOCK_SIZE, 0);
            uint16_t hd = 0x3c00; // 1.0f
            std::memcpy(blk.data(), &hd, sizeof(hd));
            // set first int8 to -1
            blk[2] = 0xFF; // -1
            for (size_t r = 0; r < rows; ++r) std::memcpy(packed.data() + r * row_bytes, blk.data(), row_bytes);
        }

        std::cout << "\n[TEST] Format: " << (f.id == 0 ? "Q4_K" : (f.id == 1 ? "Q5_0" : "Q8_0")) << std::endl;

        Tensor* A = tensor_create_f32(1, K);
        Tensor* Bq = nullptr;
        Tensor* Cout = tensor_create_f32(1, N);
        if (f.id == 0) Bq = tensor_create_q4_k_from_bytes(K, N, packed.data(), packed.size());
        else if (f.id == 1) Bq = tensor_create_q5_0_from_bytes(K, N, packed.data(), packed.size());
        else Bq = tensor_create_q8_0_from_bytes(K, N, packed.data(), packed.size());

        if (!A || !Bq || !Cout) {
            std::cerr << "Allocation failed for " << (f.id == 0 ? "Q4" : (f.id == 1 ? "Q5" : "Q8")) << std::endl;
            return 3;
        }

        // Device vs host dequant compare for debugging
        cuda_backend_compare_dequant(Bq, 16);

        // Fill A with ones
        float* ad = (float*)A->data;
        for (size_t i = 0; i < K; ++i) ad[i] = 1.0f;

        // Run parity (host dequant via device F32 cache)
        cuda_backend_set_quant_parity_mode(1);
        bool ok1 = cuda_backend_matmul(A, Bq, Cout);
        if (!ok1) {
            std::cerr << "cuda_backend_matmul (parity) failed: " << cuda_backend_last_error_msg() << std::endl;
            tensor_free(A); tensor_free(Bq); tensor_free(Cout);
            return 4;
        }
        std::vector<float> out_parity(N);
        std::memcpy(out_parity.data(), Cout->data, N * sizeof(float));

        // Run direct quant kernels
        cuda_backend_set_quant_parity_mode(0);
        bool ok2 = cuda_backend_matmul(A, Bq, Cout);
        if (!ok2) {
            std::cerr << "cuda_backend_matmul (direct) failed: " << cuda_backend_last_error_msg() << std::endl;
            tensor_free(A); tensor_free(Bq); tensor_free(Cout);
            return 5;
        }
        std::vector<float> out_direct(N);
        std::memcpy(out_direct.data(), Cout->data, N * sizeof(float));

        // Compare
        double max_abs = 0.0;
        double sum_abs = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double d = std::fabs((double)out_parity[i] - (double)out_direct[i]);
            sum_abs += d;
            if (d > max_abs) max_abs = d;
            if (i < 8) {
                std::cout << "col[" << i << "] parity=" << out_parity[i] << " direct=" << out_direct[i] << " diff=" << d << std::endl;
            }
            if (d > 1e-6) {
                std::cerr << "mismatch @ col=" << i << " parity=" << out_parity[i] << " direct=" << out_direct[i] << " diff=" << d << std::endl;
            }
        }
        std::cout << "Max abs diff: " << max_abs << ", mean abs diff: " << (sum_abs / N) << std::endl;

        if (max_abs > 1e-3) all_ok = false;

        tensor_free(A); tensor_free(Bq); tensor_free(Cout);
    }

    return all_ok ? 0 : 6;
}
