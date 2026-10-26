"""Speculative decoding (Leviathan et al., ICML 2023; Chen et al., 2023).

Decode is memory-bound: a step costs about the same whether it processes 1 token or several. Speculative
decoding exploits that. A cheap *drafter* proposes ``k`` tokens; the expensive *target* model scores all of them in ONE forward
pass; a rejection-sampling rule then accepts a prefix of the proposals and always adds one more token from the target. Each
round emits between 1 and k+1 tokens for one target forward pass, and the emitted tokens are distributed EXACTLY as if the
target had sampled them one by one (greedy decoding: identical tokens).

One round, with ``ctx`` = prompt + tokens so far and ``last = ctx[-1]`` (not yet in the target cache):

  1. drafter proposes x_1..x_k with distributions p_1..p_k (p_i is over the token after ctx + x_1..x_{i-1})
  2. target runs on [last, x_1, ..., x_k] in one pass -> q_0..q_k  (q_i is the target's distribution after ctx + x_1..x_i)
  3. for i = 1..k: accept x_i with probability min(1, q_{i-1}(x_i) / p_i(x_i)); on the first rejection sample a replacement
     from normalize(max(0, q_{i-1} - p_i)) and stop; if all k are accepted, sample a bonus token from q_k
  4. roll both caches back to the committed prefix (rejected tokens' K/V are discarded)

Drafters: ``ModelDrafter`` (a small model with its own KV cache) and ``NGramDrafter`` (prompt lookup: copy the continuation of an
earlier occurrence of the current suffix; no model, deterministic proposals).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import torch

from miniserve.cache import KVCache
from miniserve.model import Qwen2
from miniserve.sampler import logits_to_probs, sample


# --------------------------------------------------------------------------------------------- the acceptance rule
def verify_and_sample(
    draft_tokens: Sequence[int],
    draft_probs: Optional[torch.Tensor],
    target_probs: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> tuple[list[int], int]:
    """The heart of speculative sampling. Pure function, tested statistically (tests/test_speculative.py).

    draft_tokens : x_1..x_k proposed by the drafter
    draft_probs  : [k, V] the drafter's distributions p_i (processed with the same temperature/top-k/top-p as the target),
                   or None for a deterministic drafter (each p_i is a one-hot on x_i)
    target_probs : [k+1, V] the target's processed distributions q_0..q_k
    Returns (emitted tokens, number of draft tokens accepted); len(emitted) == accepted + 1 always.
    """
    k = len(draft_tokens)
    emitted: list[int] = []
    for i, x in enumerate(draft_tokens):
        q = target_probs[i]
        qx = float(q[x])
        px = 1.0 if draft_probs is None else float(draft_probs[i][x])
        ratio = qx / px if px > 0 else (1.0 if qx > 0 else 0.0)
        u = float(torch.rand((), generator=generator))
        if u < min(1.0, ratio):  # accept with probability min(1, q/p)
            emitted.append(x)
            continue
        # rejected: sample from the residual distribution norm(max(0, q - p)); this is what makes the output exact
        if draft_probs is None:
            residual = q.clone()
            residual[x] = 0.0
        else:
            residual = (q - draft_probs[i]).clamp_(min=0.0)
        total = float(residual.sum())
        if total <= 0.0:  # only possible through rounding when q == p; fall back to the target distribution
            residual, total = q, float(q.sum())
        y = int(torch.multinomial(residual / total, 1, generator=generator))
        return emitted + [y], len(emitted)
    # every proposal accepted: one bonus token from the target's distribution after the last proposal
    y = int(torch.multinomial(target_probs[k], 1, generator=generator))
    return emitted + [y], k


def expected_tokens_per_round(alpha: float, k: int) -> float:
    """E[tokens emitted per round] = (1 - alpha^(k+1)) / (1 - alpha), for per-token acceptance rate alpha (Leviathan et al.)."""
    return float(k + 1) if alpha >= 1.0 else (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def predicted_speedup(alpha: float, k: int, draft_cost: float, verify_cost: float = 1.0) -> float:
    """Speedup over plain decoding, in units of one target decode step.
    draft_cost = cost of one draft step / cost of one target step (c); verify_cost = cost of the (k+1)-token target pass / one step."""
    return expected_tokens_per_round(alpha, k) / (k * draft_cost + verify_cost)


# --------------------------------------------------------------------------------------------- drafters
class NGramDrafter:
    """Prompt-lookup drafting: find the most recent earlier occurrence of the context's last n tokens and propose what followed it.
    Costs almost nothing and works when the output copies from the context (summaries, quotes, code edits, RAG answers)."""

    def __init__(self, max_ngram: int = 3, min_ngram: int = 1):
        self.max_ngram, self.min_ngram = max_ngram, min_ngram

    def reset(self) -> None:
        pass

    def propose(self, ctx: list[int], k: int, *_unused) -> tuple[list[int], Optional[torch.Tensor]]:
        n_ctx = len(ctx)
        for n in range(self.max_ngram, self.min_ngram - 1, -1):
            if n_ctx <= n:
                continue
            suffix = ctx[-n:]
            for i in range(n_ctx - n - 1, -1, -1):  # most recent earlier occurrence first
                if ctx[i : i + n] == suffix:
                    cont = ctx[i + n : i + n + k]
                    if cont:
                        return cont, None  # deterministic proposals: p_i is one-hot
        return [], None

    def rollback(self, committed_len: int) -> None:
        pass


class ModelDrafter:
    """A small model with its own KV cache. ``propose`` first feeds every committed token the drafter has not seen yet
    (after a full-accept round that is 2 tokens, otherwise 1), then drafts ``k`` tokens autoregressively."""

    def __init__(self, model: Qwen2, max_len: int):
        p = next(model.parameters())
        self.model = model
        self.max_len = max_len
        self.cache = KVCache(model.cfg, 1, max_len, p.dtype, p.device)
        self.device = p.device

    def reset(self) -> None:
        self.cache.length = 0

    @torch.inference_mode()
    def propose(self, ctx, k, temperature, top_k, top_p, generator) -> tuple[list[int], Optional[torch.Tensor]]:
        pending = ctx[self.cache.length :]  # tokens committed but not yet in the draft cache (at least `last`)
        ids = torch.tensor([pending], dtype=torch.long, device=self.device)
        logits = self.model(ids, cache=self.cache, last_only=True)[:, -1]
        tokens, probs = [], []
        for i in range(k):
            p = logits_to_probs(logits[0], temperature, top_k, top_p)
            x = int(p.argmax()) if temperature <= 0 else int(torch.multinomial(p, 1, generator=generator))
            tokens.append(x)
            probs.append(p)
            if i < k - 1:
                logits = self.model(torch.tensor([[x]], device=self.device), cache=self.cache, last_only=True)[:, -1]
        return tokens, torch.stack(probs)

    def rollback(self, committed_len: int) -> None:
        """Discard cache entries beyond the prefix that is still valid (stale K/V past ``length`` are masked and overwritten)."""
        self.cache.length = min(self.cache.length, committed_len)


# --------------------------------------------------------------------------------------------- the decoding loop
@dataclass
class RoundStat:
    k: int  # proposals made
    accepted: int  # proposals accepted
    emitted: int  # tokens produced this round (accepted + 1)
    draft_s: float
    verify_s: float


@dataclass
class SpecResult:
    token_ids: list[int]
    prompt_len: int
    ttft: float
    total_time: float
    rounds: list[RoundStat] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)  # seconds since the call started, one per emitted token
    finish_reason: str = "length"

    @property
    def proposed(self) -> int:
        return sum(r.k for r in self.rounds)

    @property
    def accepted(self) -> int:
        return sum(r.accepted for r in self.rounds)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tested(self) -> int:
        """Proposals the target actually judged: the accepted ones plus the first rejected one (later proposals are never examined)."""
        return sum(r.accepted + (1 if r.accepted < r.k else 0) for r in self.rounds)

    @property
    def alpha_hat(self) -> float:
        """Per-token acceptance rate for the (1-a^(k+1))/(1-a) formula: accepted / tested. Unlike ``acceptance_rate`` it does not count
        proposals that were discarded after an earlier rejection as failures."""
        return self.accepted / self.tested if self.tested else 0.0

    @property
    def mean_emitted_per_round(self) -> float:
        return sum(r.emitted for r in self.rounds) / len(self.rounds) if self.rounds else 0.0

    @property
    def decode_tps(self) -> float:
        """Tokens per second after the first token (excludes prefill)."""
        n = len(self.token_ids) - 1
        t = self.total_time - self.ttft
        return n / t if n > 0 and t > 0 else 0.0


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def speculative_stream(
    target: Qwen2,
    drafter,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    k: int = 4,
    eos_ids: Sequence[int] = (),
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
    inspect: Optional[dict] = None,
) -> Iterator[tuple[list[int], Optional[RoundStat]]]:
    """Yield (new tokens, round stats) after every round; the first yield is the prefill's token (stats None).
    ``inspect`` (tests only): if given, receives the live target cache and context under the keys 'cache' and 'ctx'."""
    device = next(target.parameters()).device
    dtype = next(target.parameters()).dtype
    eos = set(eos_ids)
    P = len(prompt_ids)
    tc = KVCache(target.cfg, 1, P + max_new_tokens + k + 2, dtype, device)
    drafter.reset()

    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    logits = target(ids, cache=tc, last_only=True)[:, -1]
    first = int(sample(logits, temperature, top_k, top_p, generator).item())
    if first in eos:
        return
    ctx = list(prompt_ids) + [first]  # invariant: the target cache holds ctx[:-1]
    if inspect is not None:
        inspect["cache"], inspect["ctx"] = tc, ctx
    yield [first], None
    produced = 1

    while produced < max_new_tokens:
        k_eff = min(k, max_new_tokens - produced - 1)  # a round emits at most k_eff + 1 <= tokens still wanted
        t0 = time.perf_counter()
        draft_tokens, draft_probs = drafter.propose(ctx, k_eff, temperature, top_k, top_p, generator) if k_eff > 0 else ([], None)
        _sync(device)
        t1 = time.perf_counter()

        # one target pass over [last, x_1..x_k]: logits row i is the distribution after ctx + x_1..x_i
        inp = torch.tensor([[ctx[-1]] + list(draft_tokens)], dtype=torch.long, device=device)
        t_logits = target(inp, cache=tc)[0]
        target_probs = logits_to_probs(t_logits, temperature, top_k, top_p)
        _sync(device)
        t2 = time.perf_counter()

        emitted, n_acc = verify_and_sample(draft_tokens, draft_probs, target_probs, generator)

        old_len = len(ctx)
        out: list[int] = []
        stop = False
        for tok in emitted:
            if tok in eos:
                stop = True
                break
            out.append(tok)
        ctx.extend(out)
        produced += len(out)

        # roll back: the target cache keeps ctx_new[:-1]; the drafter keeps only the still-valid prefix
        tc.length = old_len + n_acc
        drafter.rollback(old_len + min(n_acc, max(len(draft_tokens) - 1, 0)))

        stat = RoundStat(len(draft_tokens), n_acc, len(emitted), t1 - t0, t2 - t1)
        if out:
            yield out, stat
        if stop:
            return


def generate_speculative(
    target: Qwen2,
    drafter,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 64,
    k: int = 4,
    eos_ids: Sequence[int] = (),
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> SpecResult:
    device = next(target.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    _sync(device)
    t0 = time.perf_counter()
    out: list[int] = []
    times: list[float] = []
    rounds: list[RoundStat] = []
    ttft = 0.0
    for toks, stat in speculative_stream(target, drafter, prompt_ids, max_new_tokens, k, eos_ids, temperature, top_k, top_p, gen):
        now = time.perf_counter() - t0
        if not out:
            ttft = now
        out.extend(toks)
        times.extend([now] * len(toks))
        if stat is not None:
            rounds.append(stat)
    total = time.perf_counter() - t0
    return SpecResult(out, len(prompt_ids), ttft if out else total, total, rounds, times,
                      "length" if len(out) >= max_new_tokens else "stop")
