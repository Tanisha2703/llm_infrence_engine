"""The KV cache is an optimization: it must not change what the model says.

Every test compares against the no-cache path, which the parity test already proved equals Hugging Face.
"""
import pytest
import torch

from miniserve.cache import KVCache
from miniserve.generate import generate
from miniserve.loader import DEFAULT_MODEL, load_model


@pytest.fixture(scope="module")
def ours():
    return load_model(DEFAULT_MODEL, dtype=torch.float32)


def _prompt(tok, text="The three primary colors are"):
    return tok(text).input_ids


def test_cached_generation_matches_uncached_exactly(ours):
    model, tok = ours
    p = _prompt(tok)
    fast = generate(model, p, max_new_tokens=24, use_cache=True)
    slow = generate(model, p, max_new_tokens=24, use_cache=False)
    assert fast.token_ids == slow.token_ids


def test_incremental_logits_match_full_forward(ours):
    """Prefill n-1 tokens, feed the last one with the cache: same logits as one full pass."""
    model, tok = ours
    ids = torch.tensor([_prompt(tok, "Machine learning models are trained on")])
    with torch.inference_mode():
        full = model(ids)[:, -1]
        cache = KVCache(model.cfg, 1, ids.shape[1], torch.float32)
        model(ids[:, :-1], cache=cache)
        step = model(ids[:, -1:], cache=cache)[:, -1]
    assert cache.length == ids.shape[1]
    assert (full - step).abs().max().item() < 1e-3
    assert torch.equal(full.argmax(-1), step.argmax(-1))


def test_chunked_prefill_matches_single_prefill(ours):
    """Several new tokens arriving after cached ones exercises the explicit-mask path."""
    model, tok = ours
    ids = torch.tensor([_prompt(tok, "Once upon a time in a small village by the sea there lived")])
    n = ids.shape[1]
    with torch.inference_mode():
        full = model(ids)
        cache = KVCache(model.cfg, 1, n, torch.float32)
        a = model(ids[:, : n // 2], cache=cache)
        b = model(ids[:, n // 2 :], cache=cache)
    assert (full[:, : n // 2] - a).abs().max().item() < 1e-3
    assert (full[:, n // 2 :] - b).abs().max().item() < 1e-3


def test_last_only_returns_last_position(ours):
    model, tok = ours
    ids = torch.tensor([_prompt(tok)])
    with torch.inference_mode():
        full = model(ids)
        last = model(ids, last_only=True)
    assert last.shape == (1, 1, model.cfg.vocab_size)
    # Not bit-for-bit: a [1,T,d] and a [1,1,d] matmul may use different BLAS kernels, and which ones depends on the CPU
    # (this test failed on a cloud CPU while passing on a laptop with identical printed values). Same function, same token.
    assert torch.allclose(full[:, -1:], last, atol=1e-4, rtol=1e-4)
    assert torch.equal(full[:, -1:].argmax(-1), last.argmax(-1))


def test_stops_at_end_token(ours):
    model, tok = ours
    p = _prompt(tok)
    r = generate(model, p, max_new_tokens=8)
    stop_id = r.token_ids[3]
    r2 = generate(model, p, max_new_tokens=8, eos_ids=[stop_id])
    assert r2.finish_reason == "stop"
    # generation halts at the FIRST occurrence of the stop id, before emitting it
    assert r2.token_ids == r.token_ids[: r.token_ids.index(stop_id)]


def test_cache_overflow_raises(ours):
    model, tok = ours
    ids = torch.tensor([_prompt(tok)])
    cache = KVCache(model.cfg, 1, ids.shape[1] - 1, torch.float32)
    with pytest.raises(ValueError):
        with torch.inference_mode():
            model(ids, cache=cache)


def test_cache_size_matches_hand_calculation(ours):
    model, _ = ours
    assert KVCache.bytes_per_token(model.cfg, torch.bfloat16) == 12288  # 24 x 2 x 2 x 64 x 2 bytes
