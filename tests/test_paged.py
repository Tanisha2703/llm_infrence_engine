"""Paged KV cache, prefix caching and preemption must never change what a request says.
Reference: the single-sequence path from Stage 2."""
import random

import pytest
import torch

from miniserve.cache import KVCache
from miniserve.engine import Engine, Request
from miniserve.generate import generate
from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.paged import NULL_BLOCK, BlockAllocator, OutOfBlocks, PagedCache, block_hashes
from miniserve.paged_engine import PagedEngine

# ------------------------------------------------------------------------------ allocator and hashing (no model)


def test_allocator_basic_and_null_block_reserved():
    a = BlockAllocator(num_blocks=5, block_size=4)
    got = [a.allocate() for _ in range(4)]
    assert sorted(got) == [1, 2, 3, 4] and NULL_BLOCK not in got
    with pytest.raises(OutOfBlocks):
        a.allocate()
    for b in got:
        a.decref(b)
    assert a.num_available() == 4 and a.num_in_use() == 0
    a.check_invariants()


def test_double_free_is_detected():
    a = BlockAllocator(4, 4)
    b = a.allocate()
    a.decref(b)
    with pytest.raises(ValueError):
        a.decref(b)


def test_shared_block_survives_until_last_reference_released():
    a = BlockAllocator(4, 4)
    b = a.allocate()
    a.register(b, b"h")
    a.incref(b)  # a second request shares it
    a.decref(b)
    assert a.ref[b] == 1 and a.lookup(b"h") == b  # still owned by the first request
    a.decref(b)
    assert a.ref[b] == 0 and a.lookup(b"h") == b  # idle but still cached (evictable)
    a.check_invariants()


def test_evicts_least_recently_released_cached_block_first():
    a = BlockAllocator(4, 4)  # 3 usable blocks
    b1, b2, b3 = a.allocate(), a.allocate(), a.allocate()
    for b, h in ((b1, b"h1"), (b2, b"h2"), (b3, b"h3")):
        a.register(b, h)
    a.decref(b1), a.decref(b2), a.decref(b3)  # released in this order => b1 is the oldest
    assert a.num_available() == 3
    got = a.allocate()  # no free block, so the oldest cached one is evicted
    assert got == b1 and a.lookup(b"h1") is None and a.lookup(b"h2") == b2 and a.evictions == 1
    a.check_invariants()


def test_incref_rescues_cached_block_from_eviction():
    a = BlockAllocator(3, 4)
    b1, b2 = a.allocate(), a.allocate()
    a.register(b1, b"h1"), a.register(b2, b"h2")
    a.decref(b1), a.decref(b2)
    a.incref(b1)  # a new request reuses the prefix block
    assert a.allocate() == b2  # b1 is pinned, so b2 gets evicted instead
    assert a.lookup(b"h1") == b1
    a.check_invariants()


def test_prefix_cache_can_be_disabled():
    a = BlockAllocator(4, 4, enable_prefix_cache=False)
    b = a.allocate()
    a.register(b, b"h")
    assert a.lookup(b"h") is None
    a.decref(b)
    assert a.num_available() == 3
    a.check_invariants()


def test_chain_hash_depends_on_the_whole_prefix():
    bs = 4
    same_tail = [9, 9, 9, 9]
    h1 = block_hashes([1, 2, 3, 4] + same_tail, bs)
    h2 = block_hashes([5, 6, 7, 8] + same_tail, bs)
    assert h1[0] != h2[0]
    assert h1[1] != h2[1]  # identical block content, different prefix => different identity
    assert block_hashes([1, 2, 3, 4, 7, 7, 7, 7], bs)[0] == h1[0]  # same prefix => same hash
    assert len(block_hashes(list(range(10)), bs)) == 2  # only FULL blocks get a hash


# ------------------------------------------------------------------------------ with the model
PROMPTS = [
    "The three primary colors are",
    "Machine learning models are trained on large amounts of data, and then",
    "Once upon a time in a small village by the sea there lived an old fisherman who",
    "def fibonacci(n):",
]


@pytest.fixture(scope="module")
def ours():
    return load_model(DEFAULT_MODEL, dtype=torch.float32)


@pytest.fixture(scope="module")
def prompts(ours):
    _, tok = ours
    return [tok(p).input_ids for p in PROMPTS]


@pytest.fixture(scope="module")
def ref(ours):
    model, _ = ours
    cache = {}

    def get(prompt, n):
        key = (tuple(prompt), n)
        if key not in cache:
            cache[key] = generate(model, prompt, n).token_ids
        return cache[key]

    return get


def lcp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def run(model, reqs, **kw):
    eng = PagedEngine(model, **kw)
    for r in reqs:
        eng.add_request(r)
    eng.run_until_idle()
    return eng


def test_paged_logits_match_single_sequence_with_shuffled_blocks(ours, prompts):
    """Different lengths (including block-boundary lengths), non-contiguous randomly ordered physical blocks."""
    model, _ = ours
    bs = 4
    pc = PagedCache(model.cfg, 40, bs)
    ids = list(range(1, 40))
    random.Random(0).shuffle(ids)
    seqs = [prompts[0][:3], prompts[1][:4], prompts[2][:5], prompts[2][:8]]  # lengths 3, 4, 5, 8
    tables, lens = [], []
    with torch.inference_mode():
        for p in seqs:
            nblocks = -(-(len(p) + 1) // bs)  # room for the prompt plus one decoded token
            tab = [ids.pop() for _ in range(nblocks)]
            model(torch.tensor([p]), cache=pc.view([tab], [0]), last_only=True)
            tables.append(tab)
            lens.append(len(p))
        nxt = torch.tensor([[1000], [1500], [2000], [2500]])
        batched = model(nxt, cache=pc.view(tables, lens), last_only=True)[:, -1]
        for row, p in enumerate(seqs):
            kc = KVCache(model.cfg, 1, 64)
            model(torch.tensor([p]), cache=kc)
            single = model(nxt[row : row + 1], cache=kc)[:, -1]
            assert (batched[row] - single[0]).abs().max().item() < 1e-3
            assert batched[row].argmax() == single[0].argmax()


def test_paged_engine_matches_individual_outputs(ours, prompts, ref):
    model, _ = ours
    reqs = [Request(p, max_new_tokens=12) for p in prompts]
    eng = run(model, reqs, num_blocks=60, block_size=4, max_len=128)
    for r, p in zip(reqs, prompts):
        assert r.output_ids == ref(p, 12)
    assert eng.allocator.num_in_use() == 0
    eng.allocator.check_invariants()


def test_prefix_cache_hit_skips_prefill_and_keeps_output(ours, prompts, ref):
    model, tok = ours
    bs = 4
    prefix = tok("You are a careful assistant. Always answer briefly and accurately. Question:").input_ids
    a = prefix + tok(" What is the capital of France?").input_ids
    b = prefix + tok(" Name a primary color.").input_ids
    ra, rb = Request(a, max_new_tokens=8), Request(b, max_new_tokens=8)
    eng = PagedEngine(model, num_blocks=80, block_size=bs, max_len=128)
    eng.add_request(ra)
    eng.run_until_idle()  # a's prompt blocks are now cached
    eng.add_request(rb)
    eng.run_until_idle()
    assert ra.output_ids == ref(a, 8) and rb.output_ids == ref(b, 8)
    expected = min(lcp(a, b) // bs, (len(b) - 1) // bs) * bs
    assert expected >= 8 and rb.prefix_hit_tokens == expected
    s = eng.stats()
    assert s["prefill_tokens_from_cache"] == expected and s["prefix_hit_rate"] > 0.2

    off = PagedEngine(model, num_blocks=80, block_size=bs, max_len=128, enable_prefix_cache=False)
    for r in (Request(a, max_new_tokens=8), Request(b, max_new_tokens=8)):
        off.add_request(r)
        off.run_until_idle()
    assert off.stats()["prefill_tokens_from_cache"] == 0
    assert off.stats()["prefill_tokens_computed"] == eng.stats()["prefill_tokens_computed"] + expected


def test_fully_cached_prompt_still_computes_last_block(ours, ref):
    """If the whole prompt is cached we must still compute logits for its last token: the last block is recomputed."""
    model, tok = ours
    bs = 4
    p = tok("The quick brown fox jumps over the lazy dog and keeps on running far away").input_ids
    p = p[: (len(p) // bs) * bs]  # exact multiple of the block size
    assert len(p) % bs == 0 and len(p) >= 12
    r1, r2 = Request(p, max_new_tokens=6), Request(p, max_new_tokens=6)
    eng = PagedEngine(model, num_blocks=60, block_size=bs, max_len=128)
    eng.add_request(r1)
    eng.run_until_idle()
    eng.add_request(r2)
    eng.run_until_idle()
    assert r2.prefix_hit_tokens == len(p) - bs  # everything but the last block
    assert r1.output_ids == r2.output_ids == ref(p, 6)


def test_same_block_content_after_different_prefix_is_not_shared(ours, ref):
    """Why hashes are chained. Block X computed after prefix Q holds different K/V than block X after prefix P.

    Matching walks block by block and stops at the first miss, so false sharing needs TWO earlier requests:
      r1 = [P, Y] caches P (at position 0) and r2 = [Q, X] caches X (computed after Q). Then r3 = [P, X] hits P (from r1)
    and, with a content-only hash, would also hit X (from r2), reusing K/V that were computed after the wrong prefix.
    """
    model, tok = ours
    bs = 4
    P_ = tok("Alpha beta gamma delta epsilon zeta").input_ids[:bs]
    Q_ = tok("One two three four five six seven").input_ids[:bs]
    X_ = tok(" the cat sat on the mat today").input_ids[:bs]
    Y_ = tok(" a dog ran across the wide field").input_ids[:bs]
    tail = tok(" and then").input_ids[:2]
    r1, r2, r3 = (Request(P_ + Y_ + tail, max_new_tokens=6), Request(Q_ + X_ + tail, max_new_tokens=6),
                  Request(P_ + X_ + tail, max_new_tokens=6))
    eng = PagedEngine(model, num_blocks=60, block_size=bs, max_len=128)
    for r in (r1, r2, r3):
        eng.add_request(r)
        eng.run_until_idle()
    assert r3.prefix_hit_tokens == bs  # only P is shared; X after a different prefix is a different block
    for r in (r1, r2, r3):
        assert r.output_ids == ref(r.prompt_ids, 6)


def test_preemption_under_memory_pressure_preserves_outputs(ours, prompts, ref):
    model, _ = ours
    n = 14
    reqs = [Request(p, max_new_tokens=n) for p in prompts]
    # each request needs ~6-8 blocks of 4 tokens; 14 usable blocks cannot hold all four at once
    eng = run(model, reqs, num_blocks=15, block_size=4, max_len=128)
    assert eng.preemptions > 0
    assert any(r.preemptions > 0 for r in reqs)
    for r, p in zip(reqs, prompts):
        assert r.output_ids == ref(p, n), "a preempted-and-resumed request must produce the same text"
        assert r.finish_reason == "length"
    assert eng.allocator.num_in_use() == 0
    eng.allocator.check_invariants()


def test_paged_memory_holds_more_concurrent_requests_than_slots(ours, prompts):
    """Same number of bytes: 2 slots of 64 positions (= 128 positions) vs a pool of 32 blocks of 4 tokens."""
    model, _ = ours
    short = [prompts[3]] * 6  # 6 short requests, ~6 tokens + 12 new each
    slot = Engine(model, max_slots=2, max_len=64)
    for p in short:
        slot.add_request(Request(p, max_new_tokens=12))
    peak_slot = 0
    while slot.has_work():
        slot.step()
        peak_slot = max(peak_slot, len(slot.running))
    paged = PagedEngine(model, num_blocks=33, block_size=4, max_len=64)
    assert paged.cache.total_bytes() - slot.cache.total_bytes() == paged.cache.total_bytes() // 33  # + the null block
    for p in short:
        paged.add_request(Request(p, max_new_tokens=12))
    paged.run_until_idle()
    assert peak_slot == 2
    assert paged.stats()["peak_running"] >= 5


def test_block_invariants_hold_after_every_step(ours, prompts):
    model, _ = ours
    eng = PagedEngine(model, num_blocks=15, block_size=4, max_len=128)
    for p in prompts:
        eng.add_request(Request(p, max_new_tokens=10))
    while eng.has_work():
        eng.step()
        eng.allocator.check_invariants()
        owned = [b for r in eng.running for b in r.blocks]
        for b in set(owned):
            assert eng.allocator.ref[b] == owned.count(b)  # refcount equals the number of owners
    assert eng.allocator.num_in_use() == 0


def test_request_that_can_never_fit_is_rejected(ours):
    model, _ = ours
    eng = PagedEngine(model, num_blocks=5, block_size=4, max_len=512)  # 4 usable blocks = 16 tokens
    with pytest.raises(ValueError):
        eng.add_request(Request(list(range(10)), max_new_tokens=10))
