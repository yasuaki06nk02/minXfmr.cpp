# 11_BUILD_SYSTEM.md

# minXfmr.cpp Build System Design

> **Minimal Cross Platform C++ Build Architecture**

---

# Overview

minXfmr.cpp uses CMake as the single build system.

Supported environments:

```text id="u2f3k8"
Linux

↓

Jetson Nano

↓

Android NDK

↓

ARM64 Device
```

---

# Design Philosophy

The build system follows:

* Simple configuration
* Minimal dependencies
* Reproducible builds
* Platform separation

---

# Build Architecture

```text id="j3k8p1"
                 CMake

                   |

        +----------+----------+

        |                     |

        v                     v

   Linux Build          Android Build

        |                     |

        v                     v

 libminxfmr.a       libminxfmr.so

        |

        v

 Transformer Core
```

---

# Directory Structure

```text id="m9q2aa"
minxfmr.cpp/

├── CMakeLists.txt

├── include/

│   └── minxfmr/

│       └── minxfmr.h

│

├── src/

│   ├── api/

│   ├── runtime/

│   ├── model/

│   ├── gguf/

│   ├── transformer/

│   ├── backend/

│   └── tokenizer/

│

├── tests/

│

├── examples/

│

└── android/

    ├── CMakeLists.txt

    └── app/
```

---

# Build Targets

The project creates several targets.

---

# Core Library

Target:

```text id="f7z0i8"
minxfmr
```

Type:

```text id="ytx2v9"
STATIC library
```

---

Purpose:

Main inference engine.

Contains:

* GGUF reader
* Transformer
* CPU backend
* Tokenizer

---

# Shared Library

Target:

```text id="4q3k5r"
minxfmr_shared
```

Type:

```text id="x3i4fz"
SHARED library
```

---

Used by:

Android JNI.

Output:

```text id="4h9x8n"
libminxfmr.so
```

---

# Example Application

Target:

```text id="7t8r0k"
minxfmr_cli
```

---

Purpose:

Linux testing.

Example:

```bash
./minxfmr_cli model.gguf "Hello"
```

---

# Test Target

Target:

```text id="h8c7f4"
minxfmr_test
```

---

Purpose:

Unit testing.

---

# Compiler Requirements

Minimum:

```text id="5m8k3b"
C++17
```

---

Reason:

* Modern syntax
* Good compiler support
* Android NDK support

---

# Compiler Options

Recommended:

```text id="p4x8km"
-O2

-Wall

-Wextra
```

---

Debug:

```text id="k3r9v5"
-g

-DDEBUG
```

---

Release:

```text id="8n0h2a"
-O3

-DNDEBUG
```

---

# Dependency Policy

Allowed:

```text id="j8k2l0"
C++ Standard Library

CMake

Android NDK
```

---

Avoid:

```text id="z7s1pm"
Large ML Framework

BLAS Dependency

Python Runtime

GPU Framework
```

---

# Root CMakeLists.txt

Concept:

```cmake
cmake_minimum_required(
    VERSION 3.22
)

project(
    minxfmr
    LANGUAGES CXX
)


add_subdirectory(src)

add_subdirectory(tests)

add_subdirectory(examples)
```

---

# Source CMake

Example:

```cmake
add_library(
    minxfmr

    runtime/context.cpp

    model/model.cpp

    gguf/reader.cpp

    transformer/transformer.cpp

    backend/cpu/cpu_backend.cpp
)
```

---

# Include Design

Public headers:

```text id="t1q7v5"
include/

└── minxfmr.h
```

---

Internal headers:

```text id="p8m2d1"
src/

└── internal headers
```

---

# Public API Boundary

Only expose:

```c
minxfmr_open()

minxfmr_generate()

minxfmr_reset()

minxfmr_close()
```

---

Internal classes remain hidden.

---

# Linux Build

Example:

```bash
mkdir build

cd build

cmake ..

make
```

---

Output:

```text id="m4s7p2"
libminxfmr.a

minxfmr_cli
```

---

# Jetson Nano Build

Environment:

```text id="n7d2s9"
Ubuntu

ARM64

GCC

CMake
```

---

Build:

```bash
cmake ..

make
```

---

No CUDA required.

---

# Android Build

Android uses:

```text id="z2q9lm"
Gradle

+

CMake

+

NDK
```

---

Flow:

```text id="e4t8r3"
Gradle

↓

CMake

↓

NDK Compiler

↓

libminxfmr.so
```

---

# Android ABI

MVP:

```text id="c1m8yx"
arm64-v8a
```

---

# Cross Compilation

Concept:

```text id="r8v5n0"
Host PC

↓

Android Toolchain

↓

ARM64 Binary
```

---

# Build Options

Use CMake options.

Example:

```cmake
option(
 ENABLE_TESTS
 ON
)


option(
 ENABLE_ANDROID
 OFF
)


option(
 ENABLE_NEON
 OFF
)
```

---

# Feature Flags

Possible:

```text id="8p2w4k"
MINXFMR_ENABLE_Q4

MINXFMR_ENABLE_NEON

MINXFMR_DEBUG
```

---

# NEON Build

Future:

```text id="h7x2q4"
CPUBackend

↓

NEON Kernel
```

enabled by:

```text id="j5k8m1"
ENABLE_NEON=ON
```

---

# Symbol Visibility

Release builds should hide internal symbols.

Expose only:

```c
minxfmr.h
```

---

# Binary Size Goal

Target:

Core runtime:

```text id="v2q6m9"
< few MB
```

excluding:

* Model
* Android UI

---

# Build Verification

Every build should verify:

```text id="p0m5r7"
Compile

↓

Link

↓

Run Test

↓

Load Model
```

---

# Continuous Integration Future

Possible:

```text id="d4n8s0"
GitHub Actions

↓

Linux Build

↓

Android Build

↓

Tests
```

---

# Forbidden Build Design

Avoid:

## Multiple Build Systems

Bad:

```text id="x8r4m2"
Makefile

+

CMake

+

Custom Script
```

---

## Platform Forks

Bad:

```text id="f3k9w1"
android.cpp

linux.cpp

jetson.cpp
```

---

Prefer:

```text id="n6p2s4"
Common C++

+

Small platform layer
```

---

# Final Build Architecture

```text id="w3k8m0"
                minXfmr.cpp

                     |

                 CMake

                     |

        +------------+------------+

        |                         |

      Linux                  Android

        |                         |

   CLI/Test                JNI App

        |

    CPUBackend

        |

       ARM64
```

---

# Final Principle

The build system should be as small as the runtime.

A lightweight AI runtime requires a lightweight build pipeline.
