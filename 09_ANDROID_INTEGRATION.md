# 09_ANDROID_INTEGRATION.md

# minXfmr.cpp Android Integration Design

> **Android NDK Integration for Minimal Transformer Runtime**

---

# Overview

minXfmr.cpp is designed as a native C++ inference engine.

Android integration uses:

```text
Android App

(Kotlin)

↓

JNI Layer

↓

C API

↓

minXfmr.cpp Core

↓

CPU Backend

↓

ARM64 CPU
```

---

# Design Philosophy

Android is the application layer.

minXfmr.cpp is the intelligence layer.

Responsibilities are separated.

---

# Android Responsibilities

Android handles:

* User interface
* User input
* Permissions
* App lifecycle
* Thread management
* Display output

---

# C++ Runtime Responsibilities

minXfmr.cpp handles:

* Model loading
* Tokenization
* Transformer execution
* KV Cache
* Token generation
* Memory management

---

# Directory Structure

Android integration is isolated.

```text
minxfmr-cpp/

├── src/

│   ├── runtime/

│   ├── transformer/

│   └── backend/


└── android/

    ├── app/

    │   └── src/main/

    │       ├── java/

    │       └── cpp/

    │
    └── CMakeLists.txt
```

---

# Android Build System

Use:

```text
Gradle

+

Android NDK

+

CMake
```

---

# Minimum Requirements

Target:

```text
Android API 26+
```

Architecture:

```text
arm64-v8a
```

---

# JNI Design

JNI should only translate calls.

Example:

```text
Kotlin

↓

JNI

↓

minxfmr.h

↓

C++
```

---

# JNI Wrapper

Example:

```cpp
class NativeEngine
{

public:

    static jlong open(
        String path);


    static void generate(
        String prompt);


    static void reset();


    static void close();

};
```

---

# JNI Does Not Contain

Forbidden:

```cpp
JNI

{

Transformer code

Attention code

Tensor code

}
```

---

# Native Handle Design

Java/Kotlin should not access C++ objects.

Use:

```text
long handle
```

---

Example:

Kotlin:

```kotlin
var handle: Long
```

---

C++:

```cpp
minxfmr_context*
```

---

Mapping:

```text
Kotlin Long

↓

C++ Pointer
```

---

# Model Loading Flow

```text
User selects GGUF file

↓

Kotlin path

↓

JNI

↓

minxfmr_open()

↓

GGUF Reader

↓

Model Loaded

```

---

# Generation Flow

```text
User Input

↓

Kotlin String

↓

JNI

↓

minxfmr_generate()

↓

Token Callback

↓

Kotlin UI Update
```

---

# Streaming Callback Design

The core API:

```c
void callback(
    const char* token
);
```

maps to Android:

```text
C++

↓

JNI callback

↓

Kotlin Flow

↓

TextView
```

---

# Recommended Kotlin Architecture

Use:

```text
ViewModel

↓

Repository

↓

Native Engine

↓

JNI
```

---

# Example Flow

```text
UI

↓

ViewModel

↓

LLMRepository

↓

MinXfmrNative

↓

JNI

↓

C++
```

---

# Threading

Inference must not run on UI thread.

Bad:

```kotlin
button.setOnClickListener {

    generate()

}
```

---

Good:

```text
Main Thread

↓

Worker Thread

↓

JNI

↓

C++
```

---

# Native Thread Model

MVP:

One inference thread.

```text
Thread

↓

Context

↓

Generate()
```

---

# Cancellation

Future:

Android:

```text
Stop Button

↓

JNI

↓

minxfmr_stop()

↓

Generation Loop Exit
```

---

# Memory Management

Android owns:

* UI memory
* Kotlin objects

C++ owns:

* Model memory
* Tensor memory
* KV Cache

---

# Lifecycle

Recommended:

```text
Activity Start

↓

open()

↓

generate()

↓

reset()

↓

close()

↓

Activity Destroy
```

---

# Background Execution

Large models may require:

* Foreground Service
* Wake Lock
* Memory monitoring

Future consideration.

---

# Model Storage

Recommended locations:

Option 1:

App internal storage

```text
/data/data/package/models/
```

---

Option 2:

User-selected storage

```text
Document Provider
```

---

# Model Download

Not part of runtime.

Application responsibility.

---

# Native Library

Output:

```text
libminxfmr.so
```

---

Architecture:

```text
app/

└── jniLibs/

    └── arm64-v8a/

        └── libminxfmr.so
```

---

# CMake Example

Concept:

```cmake
add_library(
    minxfmr
    SHARED

    native.cpp
)
```

---

# ABI Support

MVP:

```text
arm64-v8a
```

Future:

```text
armeabi-v7a

x86_64
```

---

# Android Performance Considerations

Important:

## Avoid

* JNI calls per token
* String conversion per token
* UI update per token

---

Recommended:

```text
Many tokens

↓

Batch callback

↓

UI update
```

---

# Example Optimization

Instead of:

```text
token
token
token
```

send:

```text
"Hello world"
```

periodically.

---

# Security

Model files should be treated as untrusted input.

Validate:

* GGUF header
* File size
* Tensor offsets

---

# Debug Build

Enable:

* Logging
* Memory tracking
* Tensor verification

---

# Release Build

Enable:

* Optimization
* Strip symbols
* Smaller binary

---

# Testing Strategy

Test on:

## Development

```text
Linux ARM64
```

---

## Validation

```text
Jetson Nano CPU only
```

---

## Target

```text
Android ARM64
```

---

# Android MVP Completion Criteria

Complete when:

* [ ] NDK build succeeds
* [ ] libminxfmr.so generated
* [ ] GGUF model loads
* [ ] Kotlin can call open()
* [ ] Streaming tokens reach UI
* [ ] reset works
* [ ] close releases memory

---

# Final Architecture

```text
+----------------------+
| Android Kotlin       |
| UI / App Logic       |
+----------+-----------+

           |

           v

+----------------------+
| JNI Wrapper          |
+----------+-----------+

           |

           v

+----------------------+
| minXfmr C API        |
+----------+-----------+

           |

           v

+----------------------+
| C++ Transformer Core |
+----------+-----------+

           |

           v

+----------------------+
| CPU Backend          |
+----------+-----------+

           |

           v

+----------------------+
| ARM64 CPU            |
+----------------------+
```

---

# Final Principle

Android should only know:

```c
open()

generate()

reset()

close()
```

Everything else belongs to minXfmr.cpp.
