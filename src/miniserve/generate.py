"""Text generation: prefill the prompt, then decode one token at a time.

``stream_tokens`` is the core: a generator that yields each new token id as soon as it exists
(this is what a streaming API needs). ``generate`` wraps it and records TTFT / per-token latency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import torch

from miniserve.cache import KVCache
from miniserve.model import Qwen2
from miniserve.sampler import sample


@dataclass
class GenerationResult:
    token_ids: list[int]  # generated tokens only (stop token excluded)
    prompt_len: int
    ttft: float  # seconds from call start to first token (= prefill + first sample)
    total_time: float
    step_times: list[float] = field(default_factory=list)  # seconds per decode step (after the first token)
    finish_reason: str = "length"  # "length" or "stop"
    cache_bytes: int = 0

    @property
    def decode_tps(self) -> float:
        """Steady-state decode speed: tokens per second after the first token."""
        t = sum(self.step_times)
        return len(self.step_times) / t if t > 0 else 0.0

    @property
    def ms_per_token(self) -> float:
        return 1000 * sum(self.step_times) / len(self.step_times) if self.step_times else 0.0


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def stream_tokens(
    model: Qwen2,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    eos_ids: Sequence[int] = (),
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    cache_cls=KVCache,
) -> Iterator[int]:
    """Yield generated token ids one by one. Stops at an end token or after ``max_new_tokens``."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    eos = set(eos_ids)

    cache = cache_cls(model.cfg, 1, len(prompt_ids) + max_new_tokens, dtype, device) if use_cache else None

    # Prefill: the whole prompt in one pass (compute-bound). Only the last position's logits matter.
    logits = model(ids, cache=cache, last_only=True)[:, -1]

    for step in range(max_new_tokens):
        nxt = sample(logits, temperature, top_k, top_p, generator)
        tok = int(nxt.item())
        if tok in eos:
            return
        yield tok
        if step == max_new_tokens - 1:
            return  # budget reached: don't pay for a forward pass whose result we'd discard
        # Decode: feed back only the new token (cached) or the whole sequence again (uncached baseline).
        if cache is not None:
            logits = model(nxt[:, None], cache=cache, last_only=True)[:, -1]
        else:
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            logits = model(ids, last_only=True)[:, -1]


def generate(
    model: Qwen2,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 64,
    eos_ids: Sequence[int] = (),
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    use_cache: bool = True,
    seed: int | None = None,
    cache_cls=KVCache,
) -> GenerationResult:
    """Generate and measure. ``use_cache=False`` is the recompute-everything baseline."""
    device = next(model.parameters()).device
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    _sync(device)
    t0 = time.perf_counter()
    out: list[int] = []
    step_times: list[float] = []
    last = t0
    ttft = 0.0
    for tok in stream_tokens(
        model, prompt_ids, max_new_tokens, eos_ids, temperature, top_k, top_p, use_cache, gen, cache_cls
    ):
        _sync(device)
        now = time.perf_counter()
        if not out:
            ttft = now - t0
        else:
            step_times.append(now - last)
        last = now
        out.append(tok)
    _sync(device)
    total = time.perf_counter() - t0

    return GenerationResult(
        token_ids=out,
        prompt_len=len(prompt_ids),
        ttft=ttft if out else total,
        total_time=total,
        step_times=step_times,
        finish_reason="length" if len(out) >= max_new_tokens else "stop",
        cache_bytes=(
            KVCache.bytes_per_token(model.cfg, next(model.parameters()).dtype) * (len(prompt_ids) + len(out))
            if use_cache
            else 0
        ),
    )
