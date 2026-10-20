# Lesson 1: How an LLM works inside

Nine short segments. **One segment = one sitting (10–20 minutes).** Do not binge them.
Depth comes from re-explaining each one out loud, not from reading more.

## The rule for every segment
1. **Read** the segment.
2. **Run** its demo: `python examples/lesson1_demos.py 1.3` (use `.venv\Scripts\python.exe`). Compare
   the output with the segment. Change a number and rerun. Breaking things teaches more than reading.
3. **Close the file** and explain the idea out loud as if to an interviewer, in under 90 seconds.
   Use a phone voice recorder if it helps. If you get stuck, reopen and fix that gap only.
4. **Climb the interview ladder** at the bottom of the segment. Each segment has three levels:
   - **Level 1:** the opening question every interviewer asks.
   - **Level 2:** the follow-up that separates "read about it" from "understands it".
   - **Level 3:** the deep or trap question. Reaching here is what a top candidate does.
5. Only then move on.

## Segments

| # | Topic | Core question you will be able to answer | Demo |
|---|---|---|---|
| [1.1](1.1-tokens.md) | Tokens | What is a token, and why does it decide cost and latency? | `1.1` |
| [1.2](1.2-embeddings.md) | Embeddings | How does a number become meaning? | `1.2` |
| [1.3](1.3-attention.md) | Attention (Q, K, V) | How do tokens exchange information? | `1.3` |
| [1.4](1.4-causal-mask.md) | Causal mask, prefill vs decode | Why can't a token see the future, and what does that mean for serving? | `1.4` |
| [1.5](1.5-heads-and-gqa.md) | Multi-head and GQA | Why 14 query heads but 2 key/value heads? | `1.5` |
| [1.6](1.6-rope.md) | RoPE | How does attention know word order? | `1.6` |
| [1.7](1.7-mlp-norm-residual.md) | MLP, norm, residual | Where do the parameters live, and why 24 layers train at all? | `1.7` |
| [1.8](1.8-output-and-sampling.md) | Logits and sampling | How do scores become a word? What is temperature? | `1.8` |
| [1.9](1.9-whole-picture.md) | The whole picture | Explain the full forward pass in two minutes | none |

**Pace suggestion:** one or two segments a day. Lesson 2 (the KV cache) builds directly on 1.3, 1.4
and 1.5, so those three matter most. Do not skip them.

## If you get stuck
Tell me the segment number and the sentence where it stopped making sense. I will explain that
piece a different way and add it to the segment.

## Files this lesson covers
[model.py](../../src/miniserve/model.py), [loader.py](../../src/miniserve/loader.py),
[test_model_parity.py](../../tests/test_model_parity.py).
