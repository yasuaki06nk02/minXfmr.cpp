add_library(minxfmr STATIC
    api/minxfmr.cpp
    tensor/tensor.cpp
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
