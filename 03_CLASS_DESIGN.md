# 03_CLASS_DESIGN.md

# minXfmr.cpp Class Design

> **Minimal Object Design for Transformer Runtime**

---

# Design Philosophy

minXfmr.cpp avoids complex object-oriented architecture.

The project follows these rules:

* Prefer simple classes
* Avoid inheritance
* Avoid virtual functions
* Prefer composition
* Keep ownership explicit

The goal is not abstraction.

The goal is understanding.

---

# Class Overview

MVP core classes:

```text
Context

Model

GGUFReader

Tokenizer

Tensor

Transformer

Attention

FeedForward

RMSNorm

RoPE

KVCache

Sampler

CPUBackend
```

---

# Class Relationship

```text
                  Context
                     |
        +------------+------------+
        |                         |
        v                         v
      Model                   KVCache
        |
        |
        v
   Transformer
        |
 +------+------+------+
 |      |      |      |
 v      v      v      v
Attention FFN RMSNorm RoPE

        |
        v

      Tensor

        |
        v

   CPUBackend
```

---

# Context

## Responsibility

Runtime session manager.

---

## Header

```cpp
class Context
{
public:

    Context();

    ~Context();


private:

    Model* model;

    KVCache* cache;

    Tokenizer* tokenizer;

};
```

---

## Owns

* Model
* Tokenizer
* KVCache

---

## Does not

* Calculate tensors
* Parse GGUF directly

---

# Model

## Responsibility

Represents loaded Transformer model.

---

## Header

```cpp
class Model
{
public:

    bool load(
        const char* path);


    Tensor* get_tensor(
        const char* name);


private:

    std::vector<Tensor> tensors;


    int hidden_size;

    int layer_count;

};
```

---

## Owns

* Tensor metadata
* Model configuration

---

## Does not

* Execute inference

---

# GGUFReader

## Responsibility

Read GGUF model files.

---

## Header

```cpp
class GGUFReader
{
public:

    bool open(
        const char* path);


    bool read_metadata(
        Model& model);


private:

    FILE* file;

};
```

---

## Does not

* Store runtime state
* Perform inference

---

# Tokenizer

## Responsibility

Convert between text and tokens.

---

## Header

```cpp
class Tokenizer
{
public:

    std::vector<int> encode(
        const char* text);


    std::string decode(
        int token);


private:

    std::vector<std::string> vocabulary;

};
```

---

## Does not

* Know Transformer structure

---

# Tensor

## Responsibility

Basic numerical data container.

---

## Header

```cpp
struct Tensor
{

    void* data;


    int type;


    std::vector<int> shape;


    size_t size;


};
```

---

## Design Rule

Tensor contains data only.

No inference logic.

---

# Transformer

## Responsibility

Execute decoder-only Transformer.

---

## Header

```cpp
class Transformer
{
public:

    Tensor forward(
        Tensor& input);


private:

    Model* model;


    Attention attention;


    FeedForward feedforward;


    RMSNorm norm;


};
```

---

## Flow

```text
Token

↓

Embedding

↓

Attention

↓

FeedForward

↓

Logits
```

---

# Attention

## Responsibility

Self-attention calculation.

---

## Header

```cpp
class Attention
{
public:

    Tensor forward(
        Tensor& input,
        KVCache& cache);


private:

    Tensor query;

    Tensor key;

    Tensor value;

};
```

---

## Does not

* Manage cache lifetime

---

# FeedForward

## Responsibility

Transformer MLP layer.

---

## Header

```cpp
class FeedForward
{
public:

    Tensor forward(
        Tensor& input);


private:

    Tensor gate;

    Tensor up;

    Tensor down;

};
```

---

# RMSNorm

## Responsibility

Normalization.

---

## Header

```cpp
class RMSNorm
{
public:

    Tensor forward(
        Tensor& input);


private:

    float epsilon;

};
```

---

# RoPE

## Responsibility

Rotary positional embedding.

---

## Header

```cpp
class RoPE
{
public:

    void apply(
        Tensor& query,
        Tensor& key);

};
```

---

# KVCache

## Responsibility

Store previous attention states.

---

## Header

```cpp
class KVCache
{
public:

    void reset();


    Tensor& get_key(
        int layer);


    Tensor& get_value(
        int layer);


private:

    std::vector<Tensor> keys;


    std::vector<Tensor> values;

};
```

---

# Sampler

## Responsibility

Select next token.

---

## Header

```cpp
class Sampler
{
public:

    int sample(
        Tensor& logits);


};
```

---

## MVP

Only:

```text
Greedy Sampling
```

---

# CPUBackend

## Responsibility

Low-level numerical operations.

---

## Header

```cpp
class CPUBackend
{
public:

    void matmul(
        Tensor& a,
        Tensor& b,
        Tensor& out);


    void add(
        Tensor& a,
        Tensor& b,
        Tensor& out);

};
```

---

## Does not

* Know tokens
* Know models
* Know prompts

---

# Ownership Model

The ownership tree:

```text
Context

 ├── Model

 ├── Tokenizer

 └── KVCache


Transformer

 └── uses Model


Attention

 └── uses KVCache


Tensor

 └── owned by Model
```

---

# Constructor Rule

Constructors should not perform heavy work.

Bad:

```cpp
Model()
{
    load("model.gguf");
}
```

---

Good:

```cpp
Model model;

model.load(path);
```

---

# Destructor Rule

All resources must have clear ownership.

No global cleanup.

No singleton.

---

# Error Handling

Classes return:

```cpp
bool
```

or

```cpp
nullptr
```

No exceptions in core runtime.

---

# Class Size Guideline

Target:

```text
Class:
< 200 lines

Function:
< 50 lines
```

---

# Future Extension

Possible additions:

* VulkanBackend
* NeonBackend
* MetalBackend

They should extend implementation, not redesign the core.

---

# Final Rule

Before adding a class:

Ask:

> "Can this responsibility belong to an existing class?"

If yes,

do not create a new class.
