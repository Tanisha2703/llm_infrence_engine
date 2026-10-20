# Lesson 1: The model (`model.py`, `loader.py`, `test_model_parity.py`)

## Why we wrote the model ourselves
Every optimization in this project (KV cache, batching, paging, speculation) changes
what happens *inside* the forward pass. If the model is a black box from a library, you
can only wrap it. Because we own `model.py`, we can change it. It is also the thing an
interviewer can probe: "walk me through a decoder layer" has a precise answer in this file.

## The big picture: what an LLM does
Text -> tokens (integers) -> **embedding** (each token becomes a vector of 896 numbers)
-> 24 identical **decoder layers** -> final norm -> **lm_head** (vector -> a score for
each of 151,936 possible next tokens) -> pick one -> append -> repeat.

That repeat loop is *autoregressive decoding*. Everything in inference engineering is
about making that loop fast.

## `model.py` walked through

### `ModelConfig`
Plain numbers read from `config.json`: 24 layers, hidden size 896, 14 query heads,
2 key/value heads, etc. `head_dim = 896 / 14 = 64`.

### `RMSNorm`
Rescales a vector so its root-mean-square is 1, then multiplies by a learned weight.
LayerNorm also subtracts the mean and adds a bias; RMSNorm skips both and works just as
well, so it is cheaper. We upcast to float32 for the reduction because bf16 has few
mantissa bits and the sum of squares loses precision.

### RoPE (`rope_cos_sin`, `apply_rope`)
Attention needs to know *where* a token is. RoPE encodes position by **rotating** the
query and key vectors by an angle proportional to the position. Dot products of rotated
vectors depend only on the *relative* distance between tokens. Each pair of features
rotates at its own frequency: `inv_freq = 1 / theta^(2i/d)`, theta = 1,000,000 for Qwen.
Key detail for later: the rotation uses the token's **absolute position**, so cached keys
stay valid as long as their positions do not change. That is why a KV cache works.

### `Attention` (the heart)
1. Project the input into queries (14 heads), keys (2 heads), values (2 heads).
2. Rotate q and k with RoPE.
3. **GQA (grouped-query attention):** 14 query heads but only 2 KV heads, so each KV head
   is shared by 7 query heads (`repeat_interleave`). Why? The KV cache stores K and V for
   every past token, so fewer KV heads means a **7x smaller cache**, which is the main
   memory cost when serving. Quality loss is small.
4. `scaled_dot_product_attention(..., is_causal=True)`: softmax(q·kᵀ / sqrt(d)) · v, with a
   mask so token i only sees tokens <= i.
5. Merge heads, `o_proj`.

### `MLP` (SwiGLU)
`down(silu(gate(x)) * up(x))`. Two parallel projections up to 4864 dims, one gates the
other, then project back to 896. Most of the model's parameters live here.

### `DecoderLayer`
`x = x + attention(norm(x))` then `x = x + mlp(norm(x))`. Pre-norm with residual
connections. The `x + ...` residual path is what lets 24 layers train stably.

### `Qwen2`
Embedding -> layers -> norm -> `lm_head`. **Tied embeddings:** `lm_head.weight` *is*
`embed_tokens.weight` (one 151936x896 matrix, ~136M of the 494M parameters, used twice).

## `loader.py`
- `torch.device("meta")` builds the model with shapes but no memory, then
  `load_state_dict(assign=True)` plugs the loaded tensors straight in. This avoids
  allocating the model twice.
- Our parameter names equal the checkpoint's keys, so there is no remapping.
- We reuse Hugging Face's *tokenizer*; reimplementing BPE teaches little about inference.

## `test_model_parity.py`
Runs the same prompt through Hugging Face and through ours; logits must agree within
1e-3 and argmax must be identical. **Every later optimization is tested against this
baseline.** "I verify correctness against a reference implementation" is a sentence
interviewers like.

## Measured so far
Naive generation (no cache): 40 tokens in 19.1 s on CPU. Each step recomputes the whole
sequence. Lesson 2 removes that waste.

## Interview questions this lesson prepares you for
1. *Walk through one forward pass of a decoder-only transformer.*
2. *Why GQA? What does it save?* (KV cache memory and bandwidth, not parameters.)
3. *What does RoPE do and why is it relative?*
4. *RMSNorm vs LayerNorm?*
5. *Why do you upcast to fp32 in the norm?*
6. *Why does the causal mask exist?* (Training parallelism and consistency: token i must not see the future.)
7. *What are tied embeddings and why use them?*

## Watch (small, in this order)
Search these titles on YouTube:
1. 3Blue1Brown, **"Attention in transformers, step-by-step"** (~26 min). The intuition.
2. Andrej Karpathy, **"Let's build GPT: from scratch, in code, spelled out"** (the
   attention section; skim the rest). It maps almost line for line to `Attention`.
3. Search **"Rotary Positional Embeddings explained"** (any ~10-15 min video) for RoPE.
You do not need to master the math. You need to explain each box in your own words.
