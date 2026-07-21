cmake_minimum_required(VERSION 3.15)
project(minxfmr CXX)

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
