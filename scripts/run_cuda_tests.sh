#!/usr/bin/env bash
set -euo pipefail

echo "Configuring build with CUDA enabled (parity mode)..."
mkdir -p build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMINXFMR_ENABLE_CUDA=ON -DMINXFMR_CUDA_AVAILABLE=ON

echo "Building test target..."
cmake --build build --config Release --target test_cuda_quantized_matmul

echo "Running CUDA parity test (MINXFMR_CUDA_QUANT_PARITY=1)..."
export MINXFMR_CUDA_QUANT_PARITY=1
./build/test_cuda_quantized_matmul

echo "Done."
