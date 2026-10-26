"""CUDA-graph decoding must change how fast a step runs, never what it computes.

Two layers of tests:
* CPU (always run): the *static-shape* cache view that makes capture possible gives the same answers as the ordinary view, and padding
  rows can never touch a real request's cache.
* GPU (skipped without CUDA; run on Modal, see benchmarks/modal_run.py --suite gputests): graph replay equals eager decoding, for every
  batch size including padded ones, and the engine with graphs equals the engine without.
"""
import pytest
import torch

from miniserve.cache import SlotCache, StaticSlotBatch
from miniserve.engine import Engine, Request
from miniserve.generate import generate
from miniserve.graph_decode import GraphDecoder, default_batch_sizes
from miniserve.loader import DEFAULT_MODEL, load_model

PROMPTS = [
    "The three primary colors are",
    "Machine learning models are trained on large amounts of data, and then",
    "Once upon a time in a small village by the sea there lived an old fisherman who",
    "def fibonacci(n):",
    "The capital of France is",
]
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="module")
def ours():
    return load_model(DEFAULT_MODEL, dtype=torch.float32)


@pytest.fixture(scope="module")
def prompts(ours):
    _, tok = ours
    return [tok(p).input_ids for p in PROMPTS]


def _prefill(model, cache, slot, ids):
    with torch.inference_mode():
        model(torch.tensor([ids], device=cache.device), cache=cache.view([slot]), last_only=True)


# ---------------------------------------------------------------------------------------------- CPU
def test_default_batch_sizes_are_powers_of_two_ending_at_the_maximum():
    assert default_batch_sizes(8) == [1, 2, 4, 8]
    assert default_batch_sizes(6) == [1, 2, 4, 6]
    assert default_batch_sizes(1) == [1]


def test_static_view_decodes_like_the_dynamic_view(ours, prompts):
    """Two sequences of different lengths, decoded for several steps with both cache views: same logits, same lengths."""
    model, _ = ours
    cfg = model.cfg
    dyn, sta = SlotCache(cfg, 4, 64), SlotCache(cfg, 4, 64)
    for cache in (dyn, sta):
        _prefill(model, cache, 0, prompts[0])
        _prefill(model, cache, 2, prompts[2])
    assert len(prompts[0]) != len(prompts[2])
    ids = torch.tensor([[1000], [2000]])
    slots = torch.tensor([0, 2])
    with torch.inference_mode():
        for _ in range(5):
            a = model(ids, cache=dyn.view([0, 2]), last_only=True)[:, -1]
            b = model(ids, cache=StaticSlotBatch(sta, slots), last_only=True)[:, -1]
            assert (a - b).abs().max().item() < 1e-3
            assert torch.equal(a.argmax(-1), b.argmax(-1))
            ids = a.argmax(-1, keepdim=True)
    assert dyn.lengths.tolist() == sta.lengths.tolist()


def test_padding_rows_in_the_scratch_slot_never_touch_real_slots(ours, prompts):
    model, _ = ours
    cache = SlotCache(model.cfg, 4, 64)  # slots 0-2 are real, slot 3 is the scratch slot
    _prefill(model, cache, 0, prompts[0])
    before = [t.clone() for t in cache.k]
    ref = SlotCache(model.cfg, 4, 64)
    _prefill(model, ref, 0, prompts[0])
    with torch.inference_mode():
        # one real row plus two padding rows, exactly as the graph decoder would run it
        out = model(torch.tensor([[1000], [0], [0]]), cache=StaticSlotBatch(cache, torch.tensor([0, 3, 3])), last_only=True)[:, -1]
        want = model(torch.tensor([[1000]]), cache=ref.view([0]), last_only=True)[:, -1]
    assert (out[0] - want[0]).abs().max().item() < 1e-3
    for layer in range(len(cache.k)):
        assert torch.equal(cache.k[layer][1], before[layer][1]) and torch.equal(cache.k[layer][2], before[layer][2])
    assert cache.lengths[1].item() == 0 and cache.lengths[2].item() == 0
    assert cache.lengths[0].item() == len(prompts[0]) + 1


def test_cuda_graphs_are_rejected_without_a_gpu_or_the_continuous_policy(ours):
    model, _ = ours
    if not next(model.parameters()).is_cuda:
        with pytest.raises(ValueError, match="CUDA"):
            Engine(model, max_slots=2, max_len=64, cuda_graphs=True)
    with pytest.raises(ValueError, match="continuous"):
        Engine(model, max_slots=2, max_len=64, policy="static", cuda_graphs=True)


# ---------------------------------------------------------------------------------------------- GPU
@pytest.fixture(scope="module")
def gpu(ours):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    model, tok = load_model(DEFAULT_MODEL, dtype=torch.float32, device="cuda")
    return model, tok


def _scripted_run(engine, ids_list, n_new):
    """A fixed schedule with mid-flight joins, so graph and eager engines see exactly the same sequence of events."""
    reqs = [engine.add_request(Request(ids_list[i], max_new_tokens=n_new)) for i in (0, 1)]
    for _ in range(3):
        engine.step()
    reqs += [engine.add_request(Request(ids_list[i], max_new_tokens=n_new)) for i in (2, 3, 4)]
    engine.run_until_idle()
    return [r.output_ids for r in reqs]


@needs_cuda
def test_engine_with_cuda_graphs_equals_engine_without(gpu, prompts):
    model, _ = gpu
    eager = _scripted_run(Engine(model, max_slots=4, max_len=128), prompts, 12)
    graph_engine = Engine(model, max_slots=4, max_len=128, cuda_graphs=True)
    graphed = _scripted_run(graph_engine, prompts, 12)
    assert graphed == eager
    assert sorted(graph_engine.free_slots) == [0, 1, 2, 3]  # every real slot came back; the scratch slot (4) was never handed out


@needs_cuda
def test_engine_with_cuda_graphs_equals_single_sequence_generation(gpu, prompts):
    model, _ = gpu
    engine = Engine(model, max_slots=4, max_len=128, cuda_graphs=True)
    reqs = [engine.add_request(Request(p, max_new_tokens=10)) for p in prompts[:4]]
    engine.run_until_idle()
    for r, p in zip(reqs, prompts):
        assert r.output_ids == generate(model, p, 10).token_ids


@needs_cuda
def test_every_captured_batch_size_including_padded_ones_matches_eager_decoding(gpu, prompts):
    model, _ = gpu
    cache = SlotCache(model.cfg, 8, 128, torch.float32, "cuda")  # 7 real slots + scratch
    decoder = GraphDecoder(model, cache, default_batch_sizes(7))
    for n in range(1, 8):  # n = 3, 5, 6, 7 need padding up to the next captured size
        ref = SlotCache(model.cfg, 8, 128, torch.float32, "cuda")
        for s in range(n):
            ids = prompts[s % len(prompts)]
            _prefill(model, cache, s, ids)
            _prefill(model, ref, s, ids)
        last = [1000 + s for s in range(n)]
        got = decoder.run(last, list(range(n))).clone()[:, -1]
        with torch.inference_mode():
            want = model(torch.tensor([[t] for t in last], device="cuda"), cache=ref.view(range(n)), last_only=True)[:, -1]
        assert (got - want).abs().max().item() < 1e-3, f"batch of {n}"
        assert torch.equal(got.argmax(-1), want.argmax(-1))
        for s in range(n):
            cache.reset_slot(s)


@needs_cuda
def test_graph_decoder_can_be_used_for_many_steps_without_drift(gpu, prompts):
    """Replaying the same graph 30 times must keep tracking the eager path (lengths advance on the device each time)."""
    model, _ = gpu
    cache = SlotCache(model.cfg, 3, 128, torch.float32, "cuda")
    decoder = GraphDecoder(model, cache, [1, 2])
    ref = SlotCache(model.cfg, 3, 128, torch.float32, "cuda")
    for c in (cache, ref):
        _prefill(model, c, 0, prompts[0])
        _prefill(model, c, 1, prompts[1])
    tok = [500, 600]
    for _ in range(30):
        got = decoder.run(tok, [0, 1]).clone()[:, -1]
        with torch.inference_mode():
            want = model(torch.tensor([[t] for t in tok], device="cuda"), cache=ref.view([0, 1]), last_only=True)[:, -1]
        assert torch.equal(got.argmax(-1), want.argmax(-1))
        tok = got.argmax(-1).tolist()
    assert cache.lengths[:2].tolist() == ref.lengths[:2].tolist()
