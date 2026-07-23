# 05_INFERENCE_FLOW.md

# minXfmr.cpp Inference Flow

> **Minimal Transformer Token Generation Pipeline**

---

# Overview

minXfmr.cpp performs inference using a decoder-only Transformer.

The runtime does not generate text directly.

It performs:

```text
Text

↓

Tokens

↓

Transformer

↓

Next Token

↓

Callback

↓

Text
```

---

# Core Principle

A language model generates one token at a time.

The fundamental loop is:

```text
Input Tokens

↓

Predict Next Token

↓

Append Token

↓

Predict Next Token

↓

Repeat
```

---

# Complete Generation Flow

```text
minxfmr_generate()

        |

        v

+----------------+
| Tokenizer      |
+----------------+

        |

        v

+----------------+
| Token IDs      |
+----------------+

        |

        v

+----------------+
| Transformer    |
+----------------+

        |

        v

+----------------+
| Logits         |
+----------------+

        |

        v

+----------------+
| Sampler        |
+----------------+

        |

        v

+----------------+
| Next Token     |
+----------------+

        |

        v

+----------------+
| Callback       |
+----------------+

        |

        v

Repeat
```

---

# Flow-to-Code Mapping

This section maps each inference step to the current implementation files.

| Flow Step | Main responsibility | Code location |
| --- | --- | --- |
| Entry point | Start one generation request | [src/api/minxfmr.cpp](src/api/minxfmr.cpp) (`minxfmr_generate`) |
| Prompt encoding | Convert prompt text to token IDs | [src/tokenizer/tokenizer.cpp](src/tokenizer/tokenizer.cpp) (`tokenizer_encode`) |
| Prompt prefill + token loop | Run prompt tokens, then autoregressive token steps | [src/api/minxfmr.cpp](src/api/minxfmr.cpp) (`minxfmr_generate`) |
| Per-layer forward | Execute RMSNorm/Attention/FFN path | [src/transformer/transformer.cpp](src/transformer/transformer.cpp) (`transformer_forward_single_layer`) |
| Stack execution | Run all layers and manage reusable buffers | [src/api/minxfmr.cpp](src/api/minxfmr.cpp) (`run_stack_forward`) |
| KV cache append/reset | Store past K/V and reset per-call state | [src/cache/kv_cache.cpp](src/cache/kv_cache.cpp), [src/api/minxfmr.cpp](src/api/minxfmr.cpp) |
| Logits projection | Multiply hidden state by output head | [src/api/minxfmr.cpp](src/api/minxfmr.cpp), [src/backend/backend_runtime.cpp](src/backend/backend_runtime.cpp) |
| Backend dispatch | Route math to CPU/CUDA and fallback | [src/backend/backend_runtime.cpp](src/backend/backend_runtime.cpp) |
| GGUF model load (open time) | Load metadata, tensors, tokenizer assets | [src/io/gguf_loader.cpp](src/io/gguf_loader.cpp), [src/api/minxfmr.cpp](src/api/minxfmr.cpp) (`minxfmr_open_with_layer`) |
| Token callback output | Stream token text back to application | [src/api/minxfmr.cpp](src/api/minxfmr.cpp) (`minxfmr_generate` callback path) |

Quick reading path for beginners:

1. Read `minxfmr_generate` in [src/api/minxfmr.cpp](src/api/minxfmr.cpp)
2. Follow calls to `tokenizer_encode` and `run_stack_forward`
3. Open `transformer_forward_single_layer` in [src/transformer/transformer.cpp](src/transformer/transformer.cpp)
4. Check KV behavior in [src/cache/kv_cache.cpp](src/cache/kv_cache.cpp)
5. Check math dispatch in [src/backend/backend_runtime.cpp](src/backend/backend_runtime.cpp)

---

# Step 1: Prompt Encoding

## Input

Example:

```text
Hello
```

---

## Tokenizer

Converts text:

```text
Hello

↓

[15496]
```

---

## Output

Token IDs:

```cpp
std::vector<int> tokens;
```

---

# Step 2: Token Processing

Prompt tokens are first processed sequentially (prefill stage).
Then generation runs one-token-at-a-time (autoregressive stage).

Example:

Input:

```text
[10, 25, 300]
```

Prefill processing:

```text
10

↓

Transformer

↓

25

↓

Transformer

↓

300

↓

Transformer


Autoregressive processing:

```text
last_token

↓

Transformer

↓

next_token

↓

repeat
```
```

---

# Step 3: Embedding

Token IDs are converted into vectors.

Example:

```text
Token ID

↓

Embedding Table

↓

Hidden Vector
```

---

Concept:

```text
token_id

↓

embedding[token_id]
```

---

# Step 4: Decoder Block Processing

Each Transformer layer executes:

```text
Input

↓

RMSNorm

↓

Attention

↓

Residual Add

↓

RMSNorm

↓

Feed Forward

↓

Residual Add

↓

Output
```

---

# Decoder Block Structure

```text
        Input

          |

          v

      RMSNorm

          |

          v

    Self Attention

          |

          v

      Residual

          |

          v

      RMSNorm

          |

          v

     Feed Forward

          |

          v

      Residual

          |

          v

        Output
```

---

# Step 5: RMSNorm

Purpose:

Normalize hidden state.

Formula:

```text
x / sqrt(mean(x²)+epsilon)
```

---

Input:

```text
Hidden Vector
```

Output:

```text
Normalized Vector
```

---

# Step 6: Self Attention

Attention calculates relationships between tokens.

---

Input:

```text
Query

Key

Value
```

---

Calculation:

```text
Attention

=

softmax(
Q × K / sqrt(d)
)

× V
```

---

# Step 7: KV Cache

During generation, previous keys and values are stored.

Without cache:

```text
Token 100

↓

Recalculate token 1-99
```

---

With cache:

```text
Token 100

↓

Reuse previous K/V
```

---

# KV Cache Flow

```text
Layer 0

K,V

↓

Cache


Layer 1

K,V

↓

Cache


...


Layer N

K,V

↓

Cache
```

---

# Step 8: Feed Forward Network

The MLP section.

Typical structure:

```text
Input

↓

Gate Projection

↓

Activation

↓

Up Projection

↓

Down Projection

↓

Output
```

---

# Step 9: Final Normalization

After all layers:

```text
Hidden State

↓

Final RMSNorm

↓

Logits
```

---

# Step 10: Logits

The Transformer outputs scores.

In current implementation, logits are computed by multiplying hidden state
with output head (`Wout`) when available.

Example:

```text
Vocabulary size: 50000
```

Output:

```text
[
 token0 : 0.1
 token1 : 3.5
 token2 : -1.2
 ...
]
```

---

# Step 11: Sampling

Sampler selects next token.

MVP:

```text
Greedy or Top-K + Temperature
```

Algorithm:

```text
temperature <= 0  -> greedy

temperature > 0   -> sample from top-k candidates
```

Example:

```text
token1 = 3.5

↓

selected
```

---

# Step 12: Token Callback

Selected token is sent immediately.

```c
callback(token);
```

Example:

```text
Token:

"Hello"
```

Application receives:

```text
Hello
```

---

# Step 13: Generation Loop

The full loop:

```text
while not finished:

    token = transformer.forward()

    next = sampler.sample()

    callback(next)

    append(next)

```

---

# Stop Conditions

Generation stops when:

## EOS Token

Model outputs:

```text
<eos>
```

Role/template marker tokens are also suppressed from user callback output.

---

## Maximum Token Count

Example:

```text
max_tokens = 512
```

---

## Future User Cancellation

Possible:

```text
stop flag
```

---

# Session State

The Context contains:

```text
Context

├── Model

├── Tokenizer

├── KV Cache

└── Current Sequence
```

---

# Multiple Generate Calls

Example:

```c
generate(
"Hello"
);

generate(
"My name is"
);
```

Each `generate()` call resets KV Cache at start.

So the second call does not continue prior call context unless the caller
includes previous conversation text in the new prompt.

---

# Reset Flow

reset():

```text
KV Cache

↓

Clear


Sequence

↓

Clear
```

Model remains loaded.

Note:

`generate()` already performs KV reset for per-call isolation.

---

# Memory Flow

During generation:

```text
Model Weights

        +

KV Cache

        +

Temporary Buffers

        |

        v

Transformer

        |

        v

Token
```

---

# Performance Critical Points

The most expensive operations:

1. Matrix Multiplication

2. Attention

3. Feed Forward

---

# Optimization Priority

Do not optimize:

* API
* Tokenizer
* Sampler

Optimize:

1. Matmul
2. Memory access
3. Cache reuse

---

# Debug Mode

Future:

Each stage can expose:

```text
Input Tensor

Output Tensor

Execution Time
```

for verification.

---

# Implementation Order

Recommended:

1. Single token forward
2. Single Transformer layer
3. Multiple layers
4. KV Cache
5. Token generation
6. Streaming callback

---

# Final Principle

The runtime is not a chatbot.

It is a minimal machine that repeatedly performs:

```text
Predict

↓

Select

↓

Emit

↓

Repeat
```

This simplicity is the foundation of minXfmr.cpp.
