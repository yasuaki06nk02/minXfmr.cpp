add_library(minxfmr STATIC
    api/minxfmr.cpp
    api/runtime_config.cpp
    tensor/tensor.cpp
    backend/backend_runtime.cpp
    backend/backend_context.cpp
    backend/cpu/cpu_backend.cpp
    transformer/rmsnorm.cpp
    transformer/rope.cpp
    transformer/attention.cpp
    transformer/softmax.cpp
    transformer/feed_forward.cpp
    transformer/transformer.cpp
    tokenizer/tokenizer.cpp
    cache/kv_cache.cpp
    io/gguf_loader.cpp
    ../third_party/gguf/gguf_stub.cpp
    ../third_party/gguf/gguf_reader.cpp
)

# Optional tiled matmul optimization
option(MINXFMR_ENABLE_TILED_MATMUL "Enable tiled matmul optimization" ON)
if(MINXFMR_ENABLE_TILED_MATMUL)
    target_sources(minxfmr PRIVATE backend/cpu/cpu_matmul_tiled.cpp)
    target_compile_definitions(minxfmr PRIVATE MINXFMR_TILED_MATMUL)
endif()

# Optional ARM NEON optimized dequant/matmul (only add on aarch64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    if(EXISTS ${CMAKE_SOURCE_DIR}/src/backend/cpu/cpu_backend_neon.cpp)
        target_sources(minxfmr PRIVATE backend/cpu/cpu_backend_neon.cpp)
        target_compile_options(minxfmr PRIVATE -march=armv8-a+simd)
    endif()
endif()

if(MINXFMR_CUDA_AVAILABLE)
    target_sources(minxfmr PRIVATE
        backend/cuda/cuda_backend.cu
    )
    target_compile_definitions(minxfmr PRIVATE MINXFMR_ENABLE_CUDA=1)
    set_target_properties(minxfmr PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
endif()

find_package(OpenMP)

target_include_directories(minxfmr PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(minxfmr PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/gguf
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(minxfmr PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MINXFMR_CUDA_AVAILABLE)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(minxfmr PRIVATE CUDA::cudart CUDA::cublas)
endif()

if(MINXFMR_CUDA_AVAILABLE)
    enable_testing()
    add_executable(test_cuda_quantized_matmul
        ${CMAKE_SOURCE_DIR}/tests/test_cuda_quantized_matmul.cpp
    )
    target_link_libraries(test_cuda_quantized_matmul PRIVATE minxfmr)
    add_test(NAME CudaQuantizedMatmul COMMAND test_cuda_quantized_matmul)

    add_executable(test_cuda_quant_kernel_parity
        ${CMAKE_SOURCE_DIR}/tests/test_cuda_quant_kernel_parity.cpp
    )
    target_link_libraries(test_cuda_quant_kernel_parity PRIVATE minxfmr)
    add_test(NAME CudaQuantKernelParity COMMAND test_cuda_quant_kernel_parity)

    add_executable(test_q4k_block_debug
        ${CMAKE_SOURCE_DIR}/tests/test_q4k_block_debug.cpp
    )
    target_link_libraries(test_q4k_block_debug PRIVATE minxfmr)
    add_test(NAME Q4KBlockDebug COMMAND test_q4k_block_debug)

    add_executable(test_q8_block_debug
        ${CMAKE_SOURCE_DIR}/tests/test_q8_block_debug.cpp
    )
    target_link_libraries(test_q8_block_debug PRIVATE minxfmr)
    add_test(NAME Q8BlockDebug COMMAND test_q8_block_debug)
endif()

if(MSVC)
    target_compile_options(minxfmr PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:/arch:AVX2>
    )
else()
    target_compile_options(minxfmr PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-O3>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-march=native>
    )
endif()

add_executable(minxfmr_cli
    main.cpp
)

target_link_libraries(minxfmr_cli PRIVATE minxfmr)
