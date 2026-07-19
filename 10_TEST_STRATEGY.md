# 10_TEST_STRATEGY.md

# minXfmr.cpp Test Strategy

> **Verification Plan for Minimal Transformer Runtime**

---

# Overview

The goal of testing is not only to confirm that the runtime runs.

The goal is to prove:

* Mathematical correctness
* Model compatibility
* Memory safety
* Runtime stability
* Performance improvement

---

# Testing Philosophy

minXfmr.cpp follows:

```text
Correctness

↓

Compatibility

↓

Performance

↓

Optimization
```

Optimization must never hide correctness problems.

---

# Test Levels

Testing is divided into:

```text
Test

├── Unit Test

├── Component Test

├── Integration Test

├── Model Test

└── Device Test
```

---

# 1. Unit Test

## Purpose

Verify individual calculations.

---

# Tensor Test

Test:

* Shape handling
* Memory allocation
* Data access

Example:

```text
Create Tensor

↓

Write values

↓

Read values

↓

Compare
```

---

# Matrix Multiplication Test

Most important numerical test.

Example:

Input:

```text
A

[1 2]

[3 4]
```

```text
B

[5 6]

[7 8]
```

Expected:

```text
C

[19 22]

[43 50]
```

---

# Activation Test

Verify:

* SiLU
* GELU
* Softmax

---

# RMSNorm Test

Compare:

```text
Reference calculation

vs

CPUBackend
```

---

# 2. GGUF Reader Test

## Purpose

Verify model loading.

---

# Header Test

Check:

* Magic number
* Version
* Metadata count
* Tensor count

---

# Metadata Test

Verify:

```text
context_length

embedding_size

layer_count

vocabulary_size
```

---

# Tensor Loading Test

Verify:

```text
Tensor name

Shape

Data type

Offset
```

---

# 3. Tokenizer Test

## Purpose

Ensure text conversion is correct.

---

Test:

```text
Text

↓

Tokens

↓

Text
```

---

Example:

```text
Hello

↓

[Token IDs]

↓

Hello
```

---

# Special Token Test

Verify:

* BOS
* EOS
* Unknown token

---

# 4. Transformer Layer Test

## Purpose

Verify model execution.

---

# Single Layer Test

Flow:

```text
Input

↓

Attention

↓

FFN

↓

Output
```

---

Compare against:

* Reference implementation
* llama.cpp output

---

# 5. End-to-End Inference Test

## Purpose

Verify complete generation.

---

Flow:

```text
GGUF

↓

Load

↓

Prompt

↓

Generate

↓

Output
```

---

# Test Model

Use small models first.

Example:

```text
Tiny model

↓

Small GGUF

↓

Fast verification
```

---

# 6. Golden Output Test

## Purpose

Detect unexpected changes.

---

Store:

```text
Prompt

+

Expected Tokens
```

Example:

```text
Prompt:

"Hello"


Expected:

" world"
```

---

# Comparison

Allow:

```text
Floating point tolerance
```

Do not require:

```text
Exact byte match
```

for quantized models.

---

# 7. Memory Test

## Purpose

Prevent leaks.

---

Check:

```text
open()

↓

memory usage

↓

generate()

↓

reset()

↓

close()

↓

memory released
```

---

# Required Measurements

Record:

```text
Initial RAM

Model RAM

Peak RAM

After close RAM
```

---

# 8. KV Cache Test

## Purpose

Verify session behavior.

---

Test:

Without reset:

```text
Prompt A

↓

Prompt B

↓

Continue
```

---

With reset:

```text
Prompt A

↓

reset()

↓

Prompt B
```

Expected:

Different contexts.

---

# 9. Quantization Test

## Purpose

Verify accuracy and performance.

---

Compare:

```text
FP16

vs

Q4
```

---

Measure:

* Memory
* Speed
* Output quality

---

# 10. Backend Test

## Purpose

Ensure hardware independence.

---

Same test:

```text
CPUBackend

↓

Reference output
```

---

Future:

```text
NEON

↓

Same output
```

---

# 11. Android Test

## Purpose

Verify mobile integration.

---

Test:

## Library Loading

```text
libminxfmr.so

↓

Loaded
```

---

## JNI Test

```text
Kotlin

↓

JNI

↓

C++
```

---

## Streaming Test

Verify:

```text
Token

↓

Callback

↓

UI
```

---

# 12. Jetson Nano Test

## Purpose

Development validation.

Environment:

```text
Jetson Nano

Ubuntu

CPU only
```

---

Measure:

* Load time
* Memory
* Token/sec

---

# Performance Benchmark

Record:

```text
Model

Quantization

RAM

Tokens/sec

Device
```

---

# Example Benchmark Table

| Device      | Model | Type | RAM | Speed |
| ----------- | ----- | ---- | --- | ----- |
| Jetson Nano | Small | Q4   | TBD | TBD   |
| Android     | Small | Q4   | TBD | TBD   |

---

# Debug Tools

Recommended:

## Logging

```text
Model loading

Tensor info

Memory usage
```

---

## Tensor Dump

Future:

```text
Save intermediate tensor
```

for comparison.

---

# Build Types

## Debug

Enable:

* Assertions
* Logs
* Checks

---

## Release

Enable:

* Optimization
* Strip symbols

---

# Continuous Testing

Every change should run:

```text
Unit Test

↓

GGUF Test

↓

Inference Test
```

---

# Development Milestones

## Phase 0

Tensor operations pass.

---

## Phase 1

GGUF loads.

---

## Phase 2

Single Transformer layer works.

---

## Phase 3

Full generation works.

---

## Phase 4

Android runs.

---

## Phase 5

Optimization begins.

---

# Failure Investigation Order

When output is incorrect:

Check:

```text
1. Tokenizer

↓

2. Tensor loading

↓

3. Tensor shapes

↓

4. MatMul

↓

5. Attention

↓

6. KV Cache

↓

7. Sampling
```

---

# Performance Investigation Order

When slow:

Check:

```text
1. Memory access

↓

2. MatMul

↓

3. Quantization

↓

4. SIMD
```

---

# Final Test Principle

A small runtime requires stronger verification.

minXfmr.cpp should always prioritize:

```text
Correct

↓

Portable

↓

Fast
```

A fast incorrect runtime has no value.
