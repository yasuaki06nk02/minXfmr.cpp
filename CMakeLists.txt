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
    add_compile_options(/W4 /permissive-)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =========================================================
# CUDA detection
# =========================================================
set(MINXFMR_CUDA_AVAILABLE OFF)

if(MINXFMR_ENABLE_CUDA)
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
