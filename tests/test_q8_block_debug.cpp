#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include "../src/tensor/tensor.h"
#include "../src/backend/cuda/cuda_backend.h"

int main() {
    std::cout << "[DBG] Q8_0 single-block full dump" << std::endl;
    if (!cuda_backend_is_available()) {
        std::cout << "CUDA not available; skipping" << std::endl;
        return 0;
    }

    const size_t K = TENSOR_Q8_0_QK;
    const size_t N = TENSOR_Q8_0_QK;
    const size_t row_bytes = tensor_q8_0_row_bytes(N);
    if (row_bytes == 0) {
        std::cerr << "invalid row bytes" << std::endl;
        return 2;
    }

    // Create a single block patterned to trigger signedness checks
    std::vector<uint8_t> blk(row_bytes, 0);
    uint16_t hd = 0x3c00; // 1.0f
    std::memcpy(blk.data() + 0, &hd, sizeof(hd));
    // set first int8 to -1
    blk[2] = 0xFF; // -1

    // replicate the block across rows
    std::vector<uint8_t> packed(K * row_bytes);
    for (size_t r = 0; r < K; ++r) std::memcpy(packed.data() + r * row_bytes, blk.data(), row_bytes);

    Tensor* Bq = tensor_create_q8_0_from_bytes(K, N, packed.data(), packed.size());
    if (!Bq) { std::cerr << "tensor_create_q8_0_from_bytes failed" << std::endl; return 3; }

    std::vector<float> host(K), device(K);
    tensor_dequant_q8_0_block(blk.data(), host.data());

    bool ok = cuda_backend_dequant_block_device(Bq, 0, 0, device.data());
    if (!ok) { std::cerr << "cuda backend dequant device failed" << std::endl; tensor_free(Bq); return 4; }

    // Print header (first 4 bytes -> hd)
    uint16_t rh = 0;
    std::memcpy(&rh, blk.data() + 0, sizeof(rh));
    std::cout << "header hd=0x" << std::hex << std::setw(4) << std::setfill('0') << rh << std::dec << std::setfill(' ') << std::endl;

    // Dump the raw bytes
    std::cout << "raw blk bytes (0.." << (row_bytes-1) << "):\n";
    for (size_t i = 0; i < row_bytes; ++i) {
        printf("%02x", blk[i]);
        if ((i + 1) % 32 == 0) printf("\n"); else printf(" ");
    }

    std::cout << "\nIndex  Host(dequant)        Device(dequant)      Diff" << std::endl;
    for (size_t i = 0; i < K; ++i) {
        double h = host[i];
        double d = device[i];
        printf("%3zu  %16.6f  %16.6f  %12.6f\n", i, h, d, h - d);
    }

    std::cout << "\nMismatches (host != device):" << std::endl;
    size_t mismatches = 0;
    for (size_t i = 0; i < K; ++i) {
        if (std::fabs(host[i] - device[i]) > 1e-6) {
            printf("idx=%3zu host=%12.6f device=%12.6f diff=%12.6f\n", i, host[i], device[i], host[i] - device[i]);
            ++mismatches;
        }
    }
    if (mismatches == 0) std::cout << "(none)" << std::endl;

    tensor_free(Bq);
    return 0;
}
