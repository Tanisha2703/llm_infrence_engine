# Lesson 1: How an LLM works inside (`model.py`, `loader.py`)

**How to use this lesson:** read it once top to bottom without stopping. Then run
`python examples/walkthrough.py` and read the output next to Part 2. Then read Part 3
(the code). Do the self-test at the end. If a part is still fuzzy, tell me which part
and I will explain it differently.

You do **not** need math beyond multiplication and addition.

---

## Part 0: The one-sentence version

> An LLM is a machine that looks at some text and answers one question:
> **"what word is most likely to come next?"**
> To write a paragraph it asks that question again and again, adding each answer to the text.

That is the whole thing. Everything below is *how it answers that one question*, and
later lessons are *how to do it faster*.

---

## Part 1: The pipeline in plain words

Our prompt is `"The capital of France is"`. Six stages:

```
text  ->  tokens  ->  vectors  ->  24 layers of "thinking"  ->  scores  ->  pick one word
"The capital..."  [785, 6722, ...]  [[-0.03, ...], ...]       (repeat 24x)   [Paris: 30%...]   " Paris"
```

### Stage 1: Tokenization (text -> integers)
Computers cannot read letters, only numbers. A **tokenizer** chops text into pieces
called **tokens** and gives each piece a number from a fixed dictionary of 151,936 entries.

Real output from your machine:
```
token ids   : [785, 6722, 315, 9625, 374]
token pieces: ['The', ' capital', ' of', ' France', ' is']
```
A token is usually a word or part of a word (`"unbelievable"` might become `un` + `believ` + `able`).
Roughly, 1 token = 3/4 of an English word. Note the space belongs to the token (`' capital'`).

### Stage 2: Embedding (integer -> list of numbers)
The number 6722 means nothing by itself (6723 is not "similar" to 6722). So the model has a big
table: row 6722 is a list of **896 decimal numbers** that represent the *meaning* of
`' capital'`. This list is called a **vector** or **embedding**.

```
shape: (1, 5, 896)  = 1 prompt, 5 tokens, 896 numbers each
first 6 numbers of token 0: [-0.032, -0.002, 0.012, -0.016, 0.016, -0.03]
```

Picture each token as a point in a 896-dimensional space. Words with similar meaning end up
near each other ("Paris" near "London"; "king" near "queen"). The numbers were *learned* during
training. We never set them by hand.

From now on, the model never sees words again, only these lists of 896 numbers.

### Stage 3: The 24 layers
The 5 vectors pass through **24 identical blocks** (the "decoder layers"). Each block
has the same structure but its **own learned numbers** (weights). Each block does two things:

1. **Attention**: let each word *look at the other words* and pull in useful information.
2. **MLP**: let each word *think on its own* about what it just collected.

After 24 rounds, the vector for the last word `' is'` has absorbed "we are talking about the
capital of France, and a city name is expected next."

### Stage 4: Scores for every possible next word
The final vector is multiplied against the dictionary (151,936 words) to get **one score
per word**. High score = likely next word. These raw scores are called **logits**.
A function called **softmax** turns them into probabilities that sum to 100%.

Real output:
```
' Paris'     probability 30.2%
' ______'    probability 12.3%
':\n'        probability  6.6%
```
The model is not "sure", it is just most confident in `Paris`.

### Stage 5: Pick one word and repeat
**Greedy decoding** = always pick the highest. (Other strategies add randomness. That is
what "temperature" controls.) Append the winner to the text and run everything again:

```
The capital of France is Paris
The capital of France is Paris.
The capital of France is Paris. It
The capital of France is Paris. It is
```
This loop is called **autoregressive decoding**. Hold on to it. **Every optimization in this
project attacks the cost of this loop**, because each pass re-runs all 24 layers.

---

## Part 2: The two ideas that need slow explanation

### Idea A: Attention (how words look at each other)
Consider: *"The animal didn't cross the street because **it** was too tired."*
What does "it" refer to? To answer, the word "it" must look back at other words and decide
that "animal" matters. Attention is that mechanism.

**The library analogy.** Every token plays three roles at once, each described by a list of numbers:

| Role | Question it answers | Library analogy |
|---|---|---|
| **Query (Q)** | "What am I looking for?" | the question you bring to the library |
| **Key (K)** | "What do I contain / what am I about?" | the label on each book's spine |
| **Value (V)** | "What information do I hand over if you pick me?" | the content inside the book |

For each token:
1. Compare its **query** with every earlier token's **key** (a dot product gives a similarity
   score: bigger = better match).
2. Run the scores through **softmax** -> weights that sum to 1 (e.g. 79% book A, 21% book B).
3. Take the **weighted average of the values**. That blend is the token's new vector.

Here are the real attention weights from layer 0, head 0 on your model:

```
               The  capital       of   France       is
      The     1.00     0.00     0.00     0.00     0.00
  capital     0.79     0.21     0.00     0.00     0.00
       of     0.82     0.13     0.05     0.00     0.00
   France     0.45     0.17     0.16     0.22     0.00
       is     0.57     0.07     0.20     0.09     0.08
```
Read it row by row: row `France` says "when updating the vector for France, take 45% from
'The', 17% from 'capital', 16% from 'of', 22% from itself."

**Notice the triangle of zeros** in the top right. That is the **causal mask**: a word may
only look at words *before* it (and itself), never at the future. Why? Because when the model
generates text, the future does not exist yet. Training uses the same rule so it behaves the
same way at inference time.

**Multi-head.** One attention pattern is not enough. The model runs 14 of them in parallel
(**14 heads**), each free to learn different things (one tracks grammar, another tracks "what
noun does this refer to", and so on). Each head works on a slice of 64 numbers (14 x 64 = 896).
Results are glued together and mixed by one more matrix (`o_proj`).

**Grouped-query attention (GQA), the cost-saver.** Normally each of the 14 heads has its
own Q, K and V. Qwen has 14 query heads but only **2** key/value heads, so 7 query heads
share one K/V set. From the real output:
```
queries: (1, 5, 896)   -> 14 heads x 64
keys   : (1, 5, 128)   ->  2 heads x 64   <- 7x smaller!
values : (1, 5, 128)   ->  2 heads x 64
```
Why do we care? **Later we will store the K and V of every past token in memory (the "KV cache").**
With GQA that is 24 layers x (K+V) x 128 numbers x 2 bytes = **12 KB per token** instead of
about 86 KB. A 1,000-token conversation costs 12 MB instead of 86 MB. When you serve hundreds
of users at once, this decides how many users fit in memory.

### Idea B: Position (RoPE)
Attention as described has a flaw: it treats the text as a *bag of words*. "Dog bites man"
and "Man bites dog" would look identical. The model needs to know **where** each word is.

**Rotary position embedding (RoPE)** solves it with a clock analogy. Take each query/key
vector and treat its numbers in pairs; each pair is a point on a 2D plane, like the tip of a
clock hand. **Rotate each hand by an angle proportional to the token's position**: position 0
rotates by 0 degrees, position 1 a bit, position 2 a bit more, and so on.

Why that works: when you compare a query and a key (dot product), what survives is the
*difference* between their angles, which equals the *distance between the two words*.
So attention automatically knows "that word is 3 positions back." Different pairs rotate at
different speeds (fast hands see small distances, slow hands see big distances), like seconds,
minutes and hours hands.

**Remember this for Lesson 2:** a token's rotation depends only on its own position. So once
we compute a token's K, it never changes. This is the exact fact that lets us cache it.

### The MLP, norms and residuals in one paragraph each
- **MLP (feed-forward):** after attention gathers information, each token's vector is
  expanded from 896 to 4,864 numbers, passed through a non-linear filter, and squeezed back to
  896. This is where a lot of the model's "knowledge" (facts) lives. It uses a design
  called SwiGLU: two expansions, one of which acts as a *gate* that decides how much of the
  other passes through.
- **Normalization (RMSNorm):** after many multiplications numbers can drift large or tiny.
  Before each sub-block we rescale the vector to a steady size. Like turning the volume
  knob back to a standard level.
- **Residual connection (`x = x + ...`):** each block *adds* its result to the input rather
  than replacing it. Information has a "highway" straight through all 24 layers and each block
  only contributes a correction. Without this, very deep networks are close to impossible to train.

---

## Part 3: Reading the code

Open [src/miniserve/model.py](../src/miniserve/model.py) next to this. Top to bottom:

| Code | Plain-English meaning |
|---|---|
| `ModelConfig` | The model's size settings (24 layers, 896 wide, 14 heads...) read from `config.json`. |
| `RMSNorm.forward` | Divide the vector by its typical size, times a learned scale. Done in 32-bit precision because 16-bit loses accuracy when summing squares. |
| `rope_cos_sin` | Pre-computes the rotation angles (as cos and sin) for each position. |
| `apply_rope` | Does the actual rotation: `x*cos + rotate_half(x)*sin` is the formula for rotating a 2D point. |
| `Attention.__init__` | Four learned matrices: `q_proj`, `k_proj`, `v_proj` produce Q, K, V from the input; `o_proj` mixes the heads back together. |
| `Attention.forward` | (1) make Q, K, V (2) rotate Q and K (3) `repeat_interleave` copies each KV head 7x so shapes line up for GQA (4) `scaled_dot_product_attention` does softmax(Q.K / 8) . V with the causal mask (`8` = sqrt(64), it keeps scores from getting huge) (5) merge heads, `o_proj`. |
| `MLP.forward` | `down( silu(gate(x)) * up(x) )`: the SwiGLU described above. |
| `DecoderLayer.forward` | Two lines: `x = x + attention(norm(x))` then `x = x + mlp(norm(x))`. This is the whole layer. |
| `Qwen2.forward` | Embed the token ids -> run 24 `DecoderLayer`s -> final norm -> `lm_head` produces scores for all 151,936 words. |
| tied weights | `lm_head.weight = embed_tokens.weight`. The "dictionary" used to turn words into vectors is reused to turn vectors back into words. It saves 136M parameters. |

[src/miniserve/loader.py](../src/miniserve/loader.py): downloads the released weights (a
`.safetensors` file; the model's ~494 million learned numbers) and pours them into our
classes. It works with no renaming because we named our variables exactly like the file's keys.

[tests/test_model_parity.py](../tests/test_model_parity.py): feeds the same prompt to Hugging
Face's official implementation and to ours, and checks the output scores agree to 3 decimals.
This proves our hand-written version is right. We rerun this style of check after every later
optimization ("did the speedup change the answers?").

---

## Part 4: Why this matters for the rest of the project

Look again at stage 5 (the loop). To produce `"... Paris. It is"` we re-ran **the entire
prompt through all 24 layers each time**, even though the first words never changed. That is
why our naive version took **19 seconds for 40 tokens**.

Lesson 2 fixes that: remember each token's K and V so each new word costs one token of work,
not the whole sentence. That memory is the **KV cache**, and the rest of the project
(batching, paging, speculative decoding) builds on it.

---

## Glossary
| Term | Meaning |
|---|---|
| Token | A piece of text (word or part of a word) that has a number |
| Embedding / vector | A list of numbers representing a token |
| Weights / parameters | The learned numbers in the model (494 million here) |
| Layer | One repeated block of attention + MLP |
| Attention | Each token gathers information from earlier tokens |
| Q, K, V | Query, Key, Value: the three roles a token plays in attention |
| Head | One independent attention pattern |
| GQA | Several query heads share one K/V head to save memory |
| Causal mask | Rule that a token cannot see later tokens |
| RoPE | Position encoding by rotating Q and K |
| Logits | Raw score per possible next word |
| Softmax | Converts scores into probabilities that sum to 1 |
| Greedy decoding | Always take the highest-probability word |
| Autoregressive | Generating one token at a time, feeding output back as input |

---

## Self-test (answer out loud; answers below)
1. What single question does an LLM answer?
2. Why does the model need embeddings instead of using the token number directly?
3. In attention, what do query, key and value each do?
4. Why is there a triangle of zeros in the attention table?
5. What does GQA save, and why does that matter for serving?
6. Why does a token's position not break caching later?
7. Why does generating 40 tokens without a cache take so long?

<details><summary>Answers</summary>

1. What is the most likely next token, given everything so far.
2. Token numbers are arbitrary ids with no notion of similarity. Embeddings are learned vectors where similar meanings are close together, which the network can compute with.
3. Query = what this token is looking for. Key = what each earlier token offers to be matched against. Value = the information actually passed on, blended by the match weights.
4. The causal mask: a token may not see later tokens, because they do not exist yet at generation time.
5. It cuts the number of key/value heads (14 -> 2), making the KV cache 7x smaller, so far more concurrent requests fit in memory.
6. RoPE rotates a token by its own absolute position, so once computed its K never changes as new tokens arrive.
7. Each step re-processes the whole sequence through all 24 layers, repeating work already done for the earlier tokens.
</details>

---

## Interview Q&A (with model answers)

**Q: Walk me through a forward pass of a decoder-only transformer.**
"Tokenize, look up embeddings, then for each of N layers: RMSNorm, attention with a causal mask
(Q, K, V projections, rotary position applied to Q and K, softmax(QKᵀ/√d)V, output projection)
added back via a residual connection; then RMSNorm and a SwiGLU MLP, also with a residual.
Finally a norm and a linear head projects to vocabulary logits; softmax gives next-token
probabilities. I implemented this from scratch and verified logits against Hugging Face."

**Q: Why grouped-query attention?**
"It shares each key/value head across several query heads. It barely hurts quality but shrinks
the KV cache by the sharing factor, 7x in Qwen2.5-0.5B. Decoding is memory-bandwidth-bound, so
a smaller cache means less data read per token and more concurrent sequences fit in memory."

**Q: What does RoPE do?**
"It encodes position by rotating query and key vectors by position-dependent angles. Their dot
product then depends only on relative distance. It needs no learned position table and keeps
cached keys valid because each token's rotation depends only on its own position."

**Q: Why divide by √d in attention?**
"Dot products of d-dimensional vectors grow with d, which pushes softmax into saturation
(near one-hot) and gives tiny gradients. Dividing by √d keeps the variance near 1."

**Q: Why is the loss of precision a concern in RMSNorm?**
"bf16 has only 8 bits of mantissa. Summing squares across 896 values in bf16 loses accuracy,
so we upcast to fp32 for the reduction and cast back."

---

## Watch (short, in this order; search the titles)
1. **3Blue1Brown: "But what is a GPT? Visual intro to transformers"** (~27 min). Best
   overall picture. Start here; it covers Part 1.
2. **3Blue1Brown: "Attention in transformers, step-by-step"** (~26 min). Covers Idea A
   with animations.
3. **Andrej Karpathy: "Let's build GPT: from scratch, in code, spelled out"**. Watch only the
   self-attention section (the "mathematical trick" and "self-attention" parts). Matches our
   `Attention` class closely.
4. Optional: search **"Rotary Positional Embeddings"** for a 10-15 minute RoPE explainer.

Do not try to master the math. The goal is to explain each box in your own words.
