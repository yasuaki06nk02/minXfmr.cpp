param()

Write-Host "Configuring build with CUDA enabled (parity mode)..."
if (!(Test-Path -Path build)) { New-Item -ItemType Directory -Path build | Out-Null }
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DMINXFMR_ENABLE_CUDA=ON -DMINXFMR_CUDA_AVAILABLE=ON

Write-Host "Building test target..."
cmake --build build --config Release --target test_cuda_quantized_matmul

Write-Host "Running CUDA parity test (MINXFMR_CUDA_QUANT_PARITY=1)..."
$env:MINXFMR_CUDA_QUANT_PARITY = "1"
.\build\test_cuda_quantized_matmul.exe

Write-Host "Done."
