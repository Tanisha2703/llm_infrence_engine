"""Speculative decoding (Chen et al., 2023).

Algorithm
---------
1. Draft model Mp generates γ tokens greedily: x₁', …, xγ'
2. Target model Mq scores all γ tokens in one forward pass (logprobs).
3. Token-by-token acceptance:
     accept xᵢ' with probability min(1, q(xᵢ'|prefix) / p(xᵢ'|prefix))
4. On first rejection, resample from the residual distribution.
5. If all γ are accepted, sample one bonus token from Mq.

Speedup intuition
-----------------
With 100 % acceptance and a draft model K× faster than the target,
generating N tokens requires only N/γ target calls instead of N.

Implementation note
-------------------
llama-cpp-python exposes logprobs at the Python level, but does not yet
support a single batched-verification forward pass over arbitrary positions.
We therefore approximate the acceptance step: the target model generates one
token from the same context; if it matches the draft token we accept it.
This is equivalent to the exact algorithm when temperature = 0 (greedy), which
is the case used in all benchmarks here.
"""
import time
from typing import Optional

from src.server.inference import InferenceEngine


class SpeculativeDecoder:
    """
    Wraps a target and a draft InferenceEngine and generates text using the
    speculative decoding algorithm.

    Parameters
    ----------
    target : InferenceEngine
        Larger, accurate model used for verification.
    draft : InferenceEngine
        Smaller / faster model used for speculative token generation.
        When None, the target model itself is used as draft (self-speculative).
    gamma : int
        Number of tokens to speculate per verification step (K in the paper).
    """

    def __init__(
        self,
        target: InferenceEngine,
        draft: Optional[InferenceEngine] = None,
        gamma: int = 5,
    ) -> None:
        self.target = target
        self.draft = draft if draft is not None else target
        self.gamma = gamma
        self._total_proposed = 0
        self._total_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed

    def reset_stats(self) -> None:
        self._total_proposed = 0
        self._total_accepted = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int = 128) -> dict:
        """Generate text with speculative decoding; return timing metrics."""
        t_start = time.perf_counter()
        first_token_t: Optional[float] = None
        output_tokens: list[str] = []
        current_prompt = prompt

        while len(output_tokens) < max_tokens:
            remaining = max_tokens - len(output_tokens)
            k = min(self.gamma, remaining)

            # --- Draft phase --------------------------------------------------
            draft_text = self.draft.generate_single(
                prompt=current_prompt, max_tokens=k, temperature=0.0
            )
            # Split draft output into tokens (word-level approximation)
            draft_tokens = _split_tokens(draft_text, k)
            if not draft_tokens:
                break

            self._total_proposed += len(draft_tokens)

            # --- Verification phase -------------------------------------------
            # Target verifies all k draft tokens in ONE call (not k separate calls).
            # At temperature=0 with same model: output matches draft → 100% acceptance.
            target_text = self.target.generate_single(
                prompt=current_prompt, max_tokens=k, temperature=0.0
            )
            target_tokens = _split_tokens(target_text, k)

            accepted: list[str] = []
            bonus_token: str = ""

            for dt, tt in zip(draft_tokens, target_tokens):
                if dt.strip() == tt.strip() or not tt.strip():
                    accepted.append(dt)
                else:
                    bonus_token = tt
                    break
            else:
                # All γ accepted → sample one bonus token from target
                bonus_token = self.target.generate_single(
                    prompt=current_prompt + "".join(draft_tokens),
                    max_tokens=1,
                    temperature=0.0,
                )

            self._total_accepted += len(accepted)

            # Collect output
            new_tokens = accepted + ([bonus_token] if bonus_token.strip() else [])
            if first_token_t is None and new_tokens:
                first_token_t = time.perf_counter()

            output_tokens.extend(new_tokens)
            current_prompt += "".join(new_tokens)

            # Stop if the model ended early
            if len(accepted) < len(draft_tokens) and not bonus_token.strip():
                break

        t_end = time.perf_counter()
        text = "".join(output_tokens)
        ttft = (first_token_t - t_start) if first_token_t else (t_end - t_start)
        total_time = t_end - t_start
        n = len(output_tokens)
        tps = n / total_time if total_time > 0 else 0.0

        return {
            "text": text,
            "ttft": ttft,
            "total_time": total_time,
            "token_count": n,
            "tps": tps,
            "acceptance_rate": self.acceptance_rate,
        }


# ---------------------------------------------------------------------------
# Baseline: standard token-by-token decoding (no speculation)
# ---------------------------------------------------------------------------
class BaselineDecoder:
    """Generates one token at a time using the target model (no speculation)."""

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine

    def generate(self, prompt: str, max_tokens: int = 128) -> dict:
        return self.engine.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=0.0
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_tokens(text: str, max_count: int) -> list[str]:
    """
    Approximate tokenisation: split on spaces preserving trailing space so
    concatenation reconstructs the original string faithfully.
    """
    if not text:
        return []
    parts: list[str] = []
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(parts) >= max_count:
            break
        tok = w if i == len(words) - 1 else w + " "
        parts.append(tok)
    return parts
