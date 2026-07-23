add_library(minxfmr STATIC
    api/minxfmr.cpp
    tensor/tensor.cpp
    backend/backend_runtime.cpp
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

if(MINXFMR_CUDA_AVAILABLE)
    target_sources(minxfmr PRIVATE
        backend/cuda/cuda_backend.cu
    )
    target_compile_definitions(minxfmr PRIVATE MINXFMR_ENABLE_CUDA=1)
    set_target_properties(minxfmr PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
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

if(MSVC)
    target_compile_options(minxfmr PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:/O2>
        $<$<NOT:$<CONFIG:Debug>>:/arch:AVX2>
    )
else()
    target_compile_options(minxfmr PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:-O3>
        $<$<NOT:$<CONFIG:Debug>>:-march=native>
    )
endif()

add_executable(minxfmr_cli
    main.cpp
)

target_link_libraries(minxfmr_cli PRIVATE minxfmr)
