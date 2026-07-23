# DESIGN_PHILOSOPHY.md

# minXfmr.cpp Design Philosophy

> **Minimal Transformer Runtime**

---

# Mission

minXfmr.cpp aims to build the smallest, most understandable, and most maintainable Transformer runtime for Android.

The project is inspired by the design philosophy of SQLite:

* Small API
* Small codebase
* High readability
* Long-term maintainability

The goal is **not** to implement every feature.

The goal is to implement only what is necessary.

---

# Core Philosophy

## Minimalism

Every feature has a maintenance cost.

Before adding any feature, ask:

> **"Can this project remain simpler without it?"**

If the answer is yes,

**do not implement it.**

---

## Readability First

Code is read far more often than it is written.

A future developer should understand any source file within minutes.

Readable code is more valuable than clever code.

---

## Long-term Maintainability

This project should remain maintainable for many years.

Avoid unnecessary abstraction.

Avoid over-engineering.

Prefer simple solutions.

---

## Android First

The runtime is designed primarily for Android devices.

Linux is used as the development and testing platform.

Desktop platforms are not the primary target.

---

# Public API Philosophy

The public API must remain extremely small.

MVP exposes only four functions.

```c
minxfmr_context* minxfmr_open(const char* model_path);

int minxfmr_generate(
    minxfmr_context* ctx,
    const char* prompt,
    void (*callback)(const char* token),
    double temperature,
    int top_k);

void minxfmr_reset(
    minxfmr_context* ctx);

void minxfmr_close(
    minxfmr_context* ctx);
```

Everything else belongs to the internal implementation.

The public API should rarely change.

---

# Internal Architecture Philosophy

Complexity belongs inside.

Simplicity belongs outside.

Internal modules may evolve.

Public interfaces should remain stable.

---

# Single Responsibility

Each module should have exactly one responsibility.

Examples:

* GGUF Loader
* Tokenizer
* Tensor
* Attention
* Sampler
* KV Cache

Avoid "God Classes."

---

# Small Files

Large source files are difficult to understand.

Recommended limits:

* Source file: approximately 500 lines
* Function: approximately 50 lines

These are guidelines, not strict rules.

---

# Struct First

Prefer simple data structures.

Use:

```cpp
struct Tensor;
struct Context;
struct KVCache;
```

Avoid unnecessary classes.

Behavior should be implemented as free functions whenever practical.

---

# No Unnecessary Abstraction

Avoid abstraction unless it solves a real problem.

Do not introduce layers merely because they are common in software architecture.

---

# Design Rules

Avoid:

* Large classes
* Deep inheritance
* Complex template metaprogramming
* Overuse of design patterns
* Hidden control flow

Prefer explicit code.

---

# Dependency Policy

Dependencies increase maintenance costs.

Every dependency must justify its existence.

Prefer implementing small utilities internally rather than introducing large external libraries.

---

# Error Handling

Errors should be explicit.

Return clear status information.

Avoid exceptions in the runtime core.

---

# Performance Philosophy

Performance is important.

Readability is equally important.

Do not sacrifice maintainability for insignificant performance gains.

Optimize only after measuring.

---

# Development Philosophy

Develop incrementally.

Every commit should leave the project in a working state.

Prefer many small improvements over large rewrites.

---

# Future Growth

New features must not compromise the project's philosophy.

The project should remain:

* Small
* Predictable
* Readable
* Easy to debug

Feature growth must never become architecture growth.

---

# Guiding Principle

Every pull request should answer one question:

> **Does this make minXfmr.cpp simpler?**

If not,

reconsider the design.
