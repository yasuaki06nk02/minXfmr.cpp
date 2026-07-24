cmake_minimum_required(VERSION 3.20)

project(minxfmr LANGUAGES CXX)

# =========================================================
# Options
# =========================================================
option(MINXFMR_ENABLE_CUDA "Enable CUDA backend when toolkit/compiler are available" ON)

# =========================================================
# C++ settings
# =========================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Ninja/Makefiles) default to Release.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Multi-config generator detected. Use --config Release for optimized builds.")
endif()

# =========================================================
# Compiler warnings / optimization
# =========================================================
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif()

# =========================================================
# CUDA detection
# =========================================================
set(MINXFMR_CUDA_AVAILABLE OFF)

if(MINXFMR_ENABLE_CUDA)
    # On Windows, check_language(CUDA) can miss nvcc even when installed.
    # Pre-resolve the compiler from common env vars / PATH so detection is stable.
    if(NOT DEFINED CMAKE_CUDA_COMPILER OR NOT CMAKE_CUDA_COMPILER)
        if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
            set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler" FORCE)
        else()
            find_program(MINXFMR_NVCC_EXECUTABLE
                NAMES nvcc
                HINTS
                    "$ENV{CUDA_PATH}/bin"
                    "$ENV{CUDA_BIN_PATH}"
                PATH_SUFFIXES bin
            )
            if(MINXFMR_NVCC_EXECUTABLE)
                set(CMAKE_CUDA_COMPILER "${MINXFMR_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
            endif()
        endif()
    endif()

    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # RTX 3070 = Compute Capability 8.6
        set(CMAKE_CUDA_ARCHITECTURES 86)

        find_package(CUDAToolkit REQUIRED)

        set(MINXFMR_CUDA_AVAILABLE ON)

        message(STATUS "CUDA backend enabled")
        message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "CUDA compiler not found: building CPU-only backend")
    endif()
endif()

# Export to subdirectories
set(MINXFMR_CUDA_AVAILABLE ${MINXFMR_CUDA_AVAILABLE} CACHE BOOL "CUDA backend availability")

# =========================================================
# Source tree
# =========================================================
add_subdirectory(src)
