"""Evaluation helpers: WikiText-2 text, calibration windows, and perplexity.

Perplexity = exp(mean next-token cross-entropy) on held-out text: "how surprised is the model by real text". It is the standard
quick measure of how much quantization (or any approximation) damages a language model. Lower is better; compare numbers only within the same
protocol (same text, window length and number of windows).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def load_wikitext_text(split: str = "test") -> str:
    """WikiText-2 (raw) from the public Hugging Face dataset ``Salesforce/wikitext`` (no login needed; requires pyarrow)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    return "\n\n".join(pq.read_table(path).column("text").to_pylist())


def wikitext_ids(tok, split: str = "test", max_tokens: Optional[int] = None) -> torch.Tensor:
    text = load_wikitext_text(split)
    if max_tokens is not None:
        text = text[: max_tokens * 6]  # about 4-5 characters per token; plenty of margin
    ids = tok(text).input_ids
    if max_tokens is not None:
        ids = ids[:max_tokens]
    return torch.tensor(ids, dtype=torch.long)


def calibration_windows(ids: torch.Tensor, n_seq: int, length: int, seed: int = 0) -> torch.Tensor:
    """n_seq random windows of ``length`` tokens (for GPTQ Hessians and SmoothQuant statistics). Use TRAIN text, never the test text."""
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - length - 1, (n_seq,), generator=g)
    return torch.stack([ids[s : s + length] for s in starts.tolist()])


@torch.inference_mode()
def perplexity(model, ids: torch.Tensor, window: int = 512, n_windows: int = 40, cache_cls=None) -> dict:
    """Mean next-token loss over ``n_windows`` consecutive, non-overlapping windows of ``window`` tokens.
    ``cache_cls``: evaluate with attention reading K/V through a (quantized) cache, e.g. Int8KVCache."""
    total, count = 0.0, 0
    n_windows = min(n_windows, len(ids) // window)
    for w in range(n_windows):
        x = ids[w * window : (w + 1) * window][None]
        cache = None if cache_cls is None else cache_cls(model.cfg, 1, window, torch.float32)
        logits = model(x, cache=cache)
        total += F.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="sum").item()
        count += window - 1
    nll = total / count
    return {"ppl": math.exp(nll), "nll": nll, "tokens": count, "windows": n_windows}
