#!/usr/bin/env bash
set -e

echo "=== minXfmr.cpp Optimization Tests ==="

mkdir -p build
cmake -B build -DMINXFMR_ENABLE_CUDA=ON -DMINXFMR_ENABLE_TILED_MATMUL=ON ..
cmake --build build --config Release

if [ -f build/test_cuda_quantized_matmul ]; then
    build/test_cuda_quantized_matmul
fi

echo "✓ Optimization tests completed"
