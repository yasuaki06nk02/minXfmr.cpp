cmake_minimum_required(VERSION 3.15)
project(minxfmr CXX)

option(MINXFMR_ENABLE_CUDA "Enable CUDA backend when toolkit/compiler are available" ON)

set(MINXFMR_CUDA_AVAILABLE OFF)
if(MINXFMR_ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		set(MINXFMR_CUDA_AVAILABLE ON)
		message(STATUS "CUDA backend enabled")
	else()
		message(STATUS "CUDA compiler not found: building CPU-only backend")
	endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Single-config generators (Ninja/Makefiles) default to Release for performance.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Multi-config generator detected. Use --config Release for optimized builds.")
endif()

add_subdirectory(src)
