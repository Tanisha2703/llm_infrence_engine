"""Turn logits into a chosen token id: greedy, temperature, top-k, top-p.

``logits_to_probs`` is the single place where the sampling distribution is defined. Speculative decoding (Stage 5) relies on
applying EXACTLY the same transformation to the draft and the target model, because its correctness guarantee is
"the output is distributed as the target's *processed* distribution".
"""
from __future__ import annotations

import torch


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """logits [..., vocab] -> the probability distribution tokens are sampled from, same shape.

    temperature <= 0 gives a one-hot distribution on the argmax (greedy decoding as a degenerate distribution).
    Otherwise: scale by 1/temperature, keep the top_k tokens and/or the smallest set whose mass reaches top_p, renormalize.
    """
    if temperature <= 0.0:
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)

    logits = logits.float() / temperature

    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        # Probability mass *before* each token; drop a token if the mass before it already reaches top_p.
        # This always keeps the first (most likely) token.
        before = probs.cumsum(dim=-1) - probs
        sorted_logits = sorted_logits.masked_fill(before >= top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return logits.softmax(dim=-1)


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: [B, vocab] -> token ids [B]. temperature == 0 means greedy (argmax); otherwise draw from ``logits_to_probs``."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = logits_to_probs(logits, temperature, top_k, top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
