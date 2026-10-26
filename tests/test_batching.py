"""Batching must not change what any request says. Reference = the single-sequence path from Stage 2
(itself proven equal to the uncached model and to Hugging Face)."""
import pytest
import torch

from miniserve.cache import KVCache, SlotCache, bytes_per_token
from miniserve.engine import Engine, Request
from miniserve.generate import generate
from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.metrics import summarize

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
def reference(ours, prompts):
    """Single-sequence greedy outputs for 12 and 5 new tokens, computed once."""
    model, _ = ours
    return {n: [generate(model, p, n).token_ids for p in prompts] for n in (1, 5, 12)}


def test_batched_decode_logits_match_single_sequence(ours, prompts):
    """Two sequences of different lengths in non-adjacent slots, decoded in ONE batched forward pass."""
    model, _ = ours
    a, b = prompts[0], prompts[2]
    assert len(a) != len(b)
    sc = SlotCache(model.cfg, 3, 64)
    with torch.inference_mode():
        for slot, p in ((0, a), (2, b)):
            model(torch.tensor([p]), cache=sc.view([slot]), last_only=True)
        nxt = torch.tensor([[1000], [2000]])
        batched = model(nxt, cache=sc.view([0, 2]), last_only=True)[:, -1]

        for row, (p, t) in enumerate(((a, 1000), (b, 2000))):
            kc = KVCache(model.cfg, 1, 64)
            model(torch.tensor([p]), cache=kc)
            single = model(torch.tensor([[t]]), cache=kc)[:, -1]
            assert (batched[row] - single[0]).abs().max().item() < 1e-3
            assert batched[row].argmax() == single[0].argmax()
    assert sc.lengths.tolist() == [len(a) + 1, 0, len(b) + 1]


def test_continuous_batching_matches_individual(ours, prompts, reference):
    model, _ = ours
    eng = Engine(model, max_slots=4, max_len=128, policy="continuous")
    reqs = [eng.add_request(Request(p, max_new_tokens=12)) for p in prompts]
    eng.run_until_idle()
    for r, ref in zip(reqs, reference[12]):
        assert r.output_ids == ref
        assert r.finish_reason == "length"


def test_request_joining_mid_flight_matches_individual(ours, prompts, reference):
    model, _ = ours
    eng = Engine(model, max_slots=3, max_len=128, policy="continuous")
    first = eng.add_request(Request(prompts[2], max_new_tokens=12))
    for _ in range(4):  # first request is already decoding when the others arrive
        eng.step()
    assert len(first.output_ids) > 1
    late = [eng.add_request(Request(prompts[i], max_new_tokens=12)) for i in (0, 1)]
    eng.run_until_idle()
    assert first.output_ids == reference[12][2]
    assert late[0].output_ids == reference[12][0]
    assert late[1].output_ids == reference[12][1]


def test_slot_reuse_never_leaks_previous_occupant(ours, prompts, reference):
    """More requests than slots: later requests reuse slots full of another request's stale K/V."""
    model, _ = ours
    eng = Engine(model, max_slots=2, max_len=128, policy="continuous")
    order = [2, 0, 3, 1, 0]  # long prompt first, so the short prompts that follow reuse its slot
    reqs = [eng.add_request(Request(prompts[i], max_new_tokens=5)) for i in order]
    eng.run_until_idle()
    for r, i in zip(reqs, order):
        assert r.output_ids == reference[5][i]
    assert sorted(eng.free_slots) == [0, 1]


def test_static_batching_with_zombie_rows_matches_individual(ours, prompts, reference):
    model, _ = ours
    eng = Engine(model, max_slots=4, max_len=128, policy="static")
    lens = [3, 12, 5, 1]  # rows finish at very different times; finished rows keep being computed
    reqs = [eng.add_request(Request(p, max_new_tokens=n)) for p, n in zip(prompts, lens)]
    eng.run_until_idle()
    for r, i, n in zip(reqs, range(4), lens):
        assert r.output_ids == (reference[12][i][:n])
    s = summarize(reqs, eng.steps, 1.0)
    assert s["wasted_row_fraction"] > 0.3  # static batching burns compute on finished rows
    assert sorted(eng.free_slots) == [0, 1, 2, 3]


def test_continuous_wastes_no_decode_rows(ours, prompts):
    model, _ = ours
    eng = Engine(model, max_slots=4, max_len=128, policy="continuous")
    reqs = [eng.add_request(Request(p, max_new_tokens=n)) for p, n in zip(prompts, [3, 12, 5, 8])]
    eng.run_until_idle()
    assert summarize(reqs, eng.steps, 1.0)["wasted_row_fraction"] == 0.0


def test_continuous_needs_fewer_decode_steps_than_static(ours, prompts):
    """Scheduling logic only (no timing): short requests free slots that waiting requests can take at once."""
    model, _ = ours
    lens = [2, 14, 2, 2, 14, 2]

    def decode_steps(policy):
        eng = Engine(model, max_slots=2, max_len=128, policy=policy)
        for i, n in enumerate(lens):
            eng.add_request(Request(prompts[i % 4], max_new_tokens=n))
        eng.run_until_idle()
        return sum(1 for s in eng.steps if s.decode_batch > 0)

    assert decode_steps("continuous") < decode_steps("static")


def test_stop_token_and_single_token_requests(ours, prompts, reference):
    model, _ = ours
    eng = Engine(model, max_slots=2, max_len=128)
    stop_id = reference[12][0][4]
    stopper = eng.add_request(Request(prompts[0], max_new_tokens=12, eos_ids=(stop_id,)))
    one = eng.add_request(Request(prompts[1], max_new_tokens=1))
    eng.run_until_idle()
    assert stopper.finish_reason == "stop"
    assert stopper.output_ids == reference[12][0][: reference[12][0].index(stop_id)]
    assert one.output_ids == reference[1][1]
    assert sorted(eng.free_slots) == [0, 1]


def test_timestamps_are_consistent(ours, prompts):
    model, _ = ours
    eng = Engine(model, max_slots=2, max_len=128)
    reqs = [eng.add_request(Request(p, max_new_tokens=4)) for p in prompts[:3]]
    eng.run_until_idle()
    for r in reqs:
        assert r.arrival_time <= r.admitted_time <= r.first_token_time <= r.finish_time
        assert len(r.token_times) == 4 and r.token_times == sorted(r.token_times)
        assert r.ttft >= r.queue_wait >= 0
    assert reqs[2].queue_wait > reqs[0].queue_wait  # the third request had to wait for a free slot


def test_add_request_validation(ours):
    model, _ = ours
    eng = Engine(model, max_slots=1, max_len=16)
    with pytest.raises(ValueError):
        eng.add_request(Request(list(range(10)), max_new_tokens=10))  # needs 20 > 16
    with pytest.raises(ValueError):
        eng.add_request(Request([], max_new_tokens=2))


def test_slot_cache_accounting(ours):
    model, _ = ours
    sc = SlotCache(model.cfg, 4, 100)
    assert sc.total_bytes() == bytes_per_token(model.cfg, torch.float32) * 4 * 100
    assert sc.bytes_used() == 0


def test_batched_sampling_decides_greedy_rows_together_and_leaves_sampled_rows_unchanged(ours):
    """_sample_rows does one argmax for all greedy rows; greedy rows must equal per-row argmax, sampled rows must equal the
    old per-row draw from a generator with the same seed, and finished rows get no token."""
    from miniserve.sampler import sample

    model, _ = ours
    eng = Engine(model, max_slots=4, max_len=32)
    torch.manual_seed(0)
    logits = torch.randn(5, 500)
    reqs = [Request([1], max_new_tokens=4, temperature=t, seed=s) for t, s in
            ((0.0, None), (0.8, 5), (0.0, None), (0.8, 6), (0.0, None))]
    for r in reqs:
        eng.add_request(r)
    reqs[4].state = "finished"  # a zombie row in static mode
    toks = eng._sample_rows(reqs, logits)
    for i in (0, 2):
        assert toks[i] == int(logits[i].argmax())
    for i in (1, 3):
        want = int(sample(logits[i : i + 1], 0.8, 0, 1.0, torch.Generator().manual_seed(reqs[i].seed)).item())
        assert toks[i] == want
    assert toks[4] is None
