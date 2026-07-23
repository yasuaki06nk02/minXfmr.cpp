# 06_GGUF_FORMAT.md

# minXfmr.cpp GGUF Format Design

> **Minimal GGUF Loader Specification**

---

# Overview

minXfmr.cpp uses GGUF as the model storage format.

The runtime does not define a new model format.

The goal is:

```text
GGUF File

↓

Minimal Reader

↓

Model

↓

Transformer Runtime
```

---

# Design Philosophy

GGUF is treated as a storage format only.

The runtime responsibilities are:

* Read metadata
* Locate tensors
* Load vocabulary
* Access weights

The runtime does not depend on llama.cpp.

---

# Why GGUF

Advantages:

## 1. Existing Ecosystem

Many models are already distributed as:

```text
*.gguf
```

---

## 2. Quantization Support

GGUF supports:

* FP32
* FP16
* Q8
* Q6
* Q5
* Q4
* Q3
* Q2

---

## 3. Simple Structure

GGUF consists of:

```text
Header

↓

Metadata

↓

Tensor Information

↓

Tensor Data
```

---

# File Layout

```text
GGUF File

+----------------+
| Header         |
+----------------+

+----------------+
| Metadata       |
+----------------+

+----------------+
| Tensor Info    |
+----------------+

+----------------+
| Tensor Data    |
+----------------+
```

---

# GGUF Header

## Responsibility

Identify file format.

---

Example structure:

```cpp
struct GGUFHeader
{
    uint32_t magic;

    uint32_t version;

    uint64_t tensor_count;

    uint64_t metadata_count;
};
```

---

# Header Validation

Reader checks:

```text
Magic

↓

Version

↓

Supported Format
```

---

# Metadata

Metadata describes the model.

Examples:

```text
Architecture

Model Name

Context Length

Embedding Size

Layer Count

Vocabulary Size
```

---

# Metadata Example

```text
general.architecture

↓

llama


llama.context_length

↓

4096


llama.embedding_length

↓

4096
```

---

# Model Configuration

GGUF metadata is converted into:

```cpp
struct ModelConfig
{
    int context_length;

    int embedding_size;

    int layer_count;

    int vocabulary_size;
};
```

---

# Tensor Information

Each tensor has:

```cpp
struct TensorInfo
{
    string name;

    DataType type;

    vector<int> shape;

    uint64_t offset;
};
```

---

# Tensor Loading Flow

```text
Tensor Name

↓

TensorInfo lookup

↓

Offset calculation

↓

Load data

↓

Tensor object
```

---

# Tensor Naming

Example:

```text
token_embd.weight

blk.0.attn_q.weight

blk.0.attn_k.weight

blk.0.ffn_down.weight

output.weight
```

Current loader behavior:

* prepare candidate names per tensor role (Q/K/V, norm, FFN, lm_head)
* search candidates in order
* load first match

This allows one loader path to support naming differences across architectures/exporters.

---

# Model Tensor Mapping

The runtime maps:

```text
GGUF Tensor

↓

Internal Tensor
```

Example:

```text
blk.0.attn_q.weight

↓

Layer[0].Attention.Query
```

---

# Vocabulary

GGUF stores tokenizer data.

Contains:

* Token strings
* Token IDs
* Special tokens

---

# Vocabulary Loading

Flow:

```text
GGUF

↓

Vocabulary Table

↓

Tokenizer
```

---

# Quantization

GGUF stores quantized tensors.

Example:

```text
FP16

↓

16 bits/value


Q4

↓

4 bits/value
```

---

# Tensor Type

Internal representation:

```cpp
enum class DataType
{
    F32,

    Q4_K
};
```

---

# MVP Supported Types

Initial implementation:

```text
F32

Q4_K
```

---

# Quantized Tensor Design

A quantized tensor contains:

```text
Quantized Data

+

Scale

+

Zero Point
```

---

Example:

```text
Original:

1.23456


↓

Q4


Stored:

0101

+

scale
```

---

# GGUF Loader Interface (Concept)

```cpp
class GGUFReader
{
public:

    bool open(
        const char* path);


    bool read(
        Model& model);


private:

    FILE* file;

};
```

---

# Reader Responsibilities

Does:

* Open file
* Parse header
* Read metadata
* Read tensor table
* Read vocabulary

---

Does not:

* Execute inference
* Perform matrix multiplication

---

# Memory Loading Strategy

## MVP

Load tensors into RAM.

```text
GGUF

↓

malloc

↓

Tensor Memory
```

---

## Future

Memory mapping:

```text
GGUF

↓

mmap

↓

Direct Access
```

---

# Model Loading Flow

Complete process:

```text
minxfmr_open()

↓

GGUF Loader

↓

ModelConfig

↓

Tensor Allocation

↓

Weight Loading

↓

Weight Orientation Normalization

↓

Tokenizer Loading

↓

Ready
```

---

# Error Handling

Reader errors:

```text
Invalid Magic

Unsupported Version

Missing Tensor

Invalid Offset

Memory Error
```

Return:

```cpp
false
```

---

# Compatibility

The loader should support:

* LLaMA architecture
* Mistral architecture
* Qwen architecture

where Transformer structure is compatible.

---

# Unsupported Features

MVP does not support:

* Multiple architecture conversion
* Training formats
* Safetensors
* Original PyTorch checkpoints

---

# Future Format Support

Possible:

```text
GGUF

+

Other Readers
```

Architecture:

```text
ModelLoader

├── GGUFReader

├── SafeTensorReader

└── CustomReader
```

---

# Minimal Implementation Order

## Step 1

Read header.

---

## Step 2

Read metadata.

---

## Step 3

List tensors.

---

## Step 4

Load one tensor.

---

## Step 5

Load complete model.

---

# Design Rule

GGUF is not the runtime.

GGUF is only the container.

The runtime should remain independent from the file format.

---

# Final Goal

The GGUF loader should be:

* Small
* Understandable
* Compatible
* Replaceable

The model format may change.

The Transformer engine should not.
