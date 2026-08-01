#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include "../src/tensor/tensor.h"
#include "../src/backend/cuda/cuda_backend.h"

int main() {
    std::cout << "[DBG] Q4_K single-block full dump" << std::endl;
    if (!cuda_backend_is_available()) {
        std::cout << "CUDA not available; skipping" << std::endl;
        return 0;
    }

    const size_t K = TENSOR_Q4_K_QK_K;
    const size_t N = TENSOR_Q4_K_QK_K;
    const size_t row_bytes = tensor_q4_k_row_bytes(N);
    if (row_bytes == 0) {
        std::cerr << "invalid row bytes" << std::endl;
        return 2;
    }

    // Create a single block patterned like the parity test
    std::vector<uint8_t> blk(row_bytes, 0);
    uint16_t hd = 0x3c00; uint16_t hm = 0x3c00; // 1.0f
    std::memcpy(blk.data() + 0, &hd, sizeof(hd));
    std::memcpy(blk.data() + 2, &hm, sizeof(hm));
    // scales area and nibble patterns as in the parity harness
    blk[4 + 4] = 0x40;
    blk[4 + 8] = 0x10;
    blk[4 + 0] = 0x00;
    blk[16 + 64] = 0x01;

    // replicate the block across all rows to form a full tensor
    std::vector<uint8_t> packed(K * row_bytes);
    for (size_t r = 0; r < K; ++r) std::memcpy(packed.data() + r * row_bytes, blk.data(), row_bytes);

    Tensor* Bq = tensor_create_q4_k_from_bytes(K, N, packed.data(), packed.size());
    if (!Bq) { std::cerr << "tensor_create_q4_k_from_bytes failed" << std::endl; return 3; }

    std::vector<float> host(K), device(K);
    tensor_dequant_q4_k_block(blk.data(), host.data());

    bool ok = cuda_backend_dequant_block_device(Bq, 0, 0, device.data());
    if (!ok) { std::cerr << "cuda backend dequant device failed" << std::endl; tensor_free(Bq); return 4; }

    // Print header (first 4 bytes -> hd, hm)
    uint16_t rh = 0, rm = 0;
    std::memcpy(&rh, blk.data() + 0, sizeof(rh));
    std::memcpy(&rm, blk.data() + 2, sizeof(rm));
    std::cout << "header hd=0x" << std::hex << std::setw(4) << std::setfill('0') << rh
              << " hm=0x" << std::setw(4) << rm << std::dec << std::setfill(' ') << std::endl;

    // Dump the full 144 raw bytes
    std::cout << "raw blk bytes (0..143):" << std::endl;
    for (size_t i = 0; i < row_bytes; ++i) {
        printf("%02x", blk[i]);
        if ((i + 1) % 32 == 0) printf("\n"); else printf(" ");
    }

    // Also print scales area (4..15) and nibble area (16..143)
    std::cout << "scales S[0..11] (4..15):" << std::endl;
    for (int i = 4; i < 16; ++i) printf("%02x ", blk[i]);
    printf("\n");
    std::cout << "nibble bytes (16..143):" << std::endl;
    for (int i = 16; i < (int)row_bytes; ++i) {
        printf("%02x", blk[i]);
        if (((i - 16 + 1) % 32) == 0) printf("\n"); else printf(" ");
    }

    // Print full host vs device arrays side-by-side and differences
    std::cout << "\nIndex  Host(dequant)        Device(dequant)      Diff" << std::endl;
    for (size_t i = 0; i < K; ++i) {
        double h = host[i];
        double d = device[i];
        printf("%3zu  %16.6f  %16.6f  %12.6f\n", i, h, d, h - d);
    }

    // Print only mismatches
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
