"""Quantization: the arithmetic must be exact where it can be, and the approximations must behave the way the theory says."""
import copy

import pytest
import torch

from miniserve.cache import KVCache
from miniserve.evalutil import perplexity
from miniserve.generate import generate
from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.quant import (
    Int8KVCache,
    QuantizedWeight,
    QuantLinear,
    QuantSpec,
    apply_activation_quant,
    collect_act_absmax,
    dequantize_weight,
    fake_quant_act,
    gptq_quantize,
    model_size_bytes,
    pack4,
    quantize_model_gptq,
    quantize_model_rtn,
    quantize_weight,
    smooth_model,
    unpack4,
)

# ------------------------------------------------------------------------------------------ the format (no model)

SPECS = [QuantSpec(8, 0), QuantSpec(8, -1), QuantSpec(4, 0), QuantSpec(4, 32), QuantSpec(4, 128), QuantSpec(4, 64, False), QuantSpec(3, 32)]


def _w(seed=0, shape=(64, 256), scale=0.05):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * scale


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_error_is_at_most_half_a_quantization_step(spec):
    w = _w()
    qw = quantize_weight(w, spec)
    deq = dequantize_weight(qw)
    step = qw.scale
    if spec.group_size < 0:
        step_full = step.expand(1, 1).reshape(1, 1).expand_as(w)
    elif spec.group_size == 0:
        step_full = step.expand(w.shape[0], w.shape[1])
    else:
        step_full = step.repeat_interleave(spec.group_size, dim=1)
    assert ((w - deq).abs() <= 0.5 * step_full + 1e-6).all()


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_codes_stay_in_range(spec):
    qw = quantize_weight(_w(), spec)
    lo, hi = (-spec.qmax, spec.qmax) if spec.symmetric else (0, spec.qmax)
    assert qw.codes.min().item() >= lo and qw.codes.max().item() <= hi


@pytest.mark.parametrize("symmetric", [True, False])
def test_int4_packing_roundtrips_exactly(symmetric):
    qw = quantize_weight(_w(1), QuantSpec(4, 32, symmetric))
    packed = pack4(qw.codes, symmetric)
    assert packed.dtype == torch.uint8 and packed.shape == (qw.codes.shape[0], qw.codes.shape[1] // 2)
    assert torch.equal(unpack4(packed, symmetric), qw.codes)


def test_packing_layout_is_even_column_low_nibble():
    codes = torch.tensor([[1, 2, 3, 4]], dtype=torch.int8)  # symmetric: stored as value + 8
    packed = pack4(codes, True)
    assert packed.tolist() == [[(1 + 8) | ((2 + 8) << 4), (3 + 8) | ((4 + 8) << 4)]]


def _rel_mse(w, spec, rows):
    deq = dequantize_weight(quantize_weight(w, spec))
    return (((w - deq)[rows]) ** 2).sum() / (w[rows] ** 2).sum()


def test_per_tensor_scale_is_ruined_by_one_outlier_row_but_per_channel_is_not():
    w = _w(2)
    w[0] *= 100  # one row ~100x larger than the rest
    others = slice(1, None)
    tensor_err = _rel_mse(w, QuantSpec(8, -1), others)
    channel_err = _rel_mse(w, QuantSpec(8, 0), others)
    assert tensor_err > 50 * channel_err


def test_smaller_groups_help_when_outliers_are_sparse_along_the_input_dimension():
    w = _w(3, (64, 512))
    w[:, ::97] *= 30  # a few outlier input columns, like LLM activation/weight outliers
    errs = {g: _rel_mse(w, QuantSpec(4, g), slice(None)) for g in (0, 128, 32)}
    assert errs[32] < errs[128] < errs[0]


def test_asymmetric_quantization_helps_skewed_weights():
    w = torch.rand(64, 256, generator=torch.Generator().manual_seed(4)) * 0.1 + 0.05  # all positive
    assert _rel_mse(w, QuantSpec(4, 64, False), slice(None)) < 0.7 * _rel_mse(w, QuantSpec(4, 64, True), slice(None))


def test_scales_are_exactly_representable_in_fp16():
    qw = quantize_weight(_w(5), QuantSpec(4, 32))
    assert torch.equal(qw.scale, qw.scale.half().float())  # what we quantize with is what we store


# ------------------------------------------------------------------------------------------ QuantLinear


def test_quantlinear_equals_matmul_with_the_dequantized_weight_and_counts_bytes():
    lin = torch.nn.Linear(256, 64, bias=True)
    for spec, expected_weight_bytes, expected_scale_bytes in ((QuantSpec(4, 128), 64 * 256 // 2, 64 * 2 * 2), (QuantSpec(8, 0), 64 * 256, 64 * 2)):
        ql = QuantLinear.from_linear(lin, spec)
        x = torch.randn(3, 256)
        ref = torch.nn.functional.linear(x, ql.dequantize(), lin.bias)
        assert torch.allclose(ql(x), ref, atol=1e-5)
        assert ql.qweight.numel() * ql.qweight.element_size() == expected_weight_bytes
        assert ql.scale.numel() * ql.scale.element_size() == expected_scale_bytes
        assert ql.stored_bytes() == expected_weight_bytes + expected_scale_bytes + 64 * 4  # + fp32 bias


def test_cached_dequantization_gives_identical_outputs():
    lin = torch.nn.Linear(128, 32)
    a, b = QuantLinear.from_linear(lin, QuantSpec(4, 32)), QuantLinear.from_linear(lin, QuantSpec(4, 32), cache_dequant=True)
    x = torch.randn(2, 128)
    assert torch.equal(a(x), b(x)) and torch.equal(b(x), b(x))


# ------------------------------------------------------------------------------------------ GPTQ (no model)


def _layer_problem(seed=0, n=2048, in_f=128, out_f=64):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(in_f, in_f, generator=g) / in_f**0.5 + torch.eye(in_f)  # correlated input features, like real activations
    x = torch.randn(n, in_f, generator=g) @ mix
    w = torch.randn(out_f, in_f, generator=g) * 0.1
    return x, w, 2.0 * (x.T @ x) / n


def _output_err(x, w, qw):
    return ((x @ w.T - x @ dequantize_weight(qw).T) ** 2).sum().item()


@pytest.mark.parametrize("spec", [QuantSpec(4, 0), QuantSpec(4, 32), QuantSpec(3, 32)], ids=lambda s: s.name)
def test_gptq_beats_round_to_nearest_on_layer_output_error(spec):
    x, w, h = _layer_problem()
    rtn = _output_err(x, w, quantize_weight(w, spec))
    gptq = _output_err(x, w, gptq_quantize(w, h, spec))
    assert gptq < 0.85 * rtn


def test_gptq_blocked_updates_are_equivalent_to_column_by_column_updates():
    """Lazy block updates are an optimization: with 256 columns, blocksize 64 (4 blocks) must give the same result as one block of 256."""
    x, w, h = _layer_problem(seed=3, in_f=256)
    spec = QuantSpec(4, 32)
    one = gptq_quantize(w, h, spec, blocksize=256)
    many = gptq_quantize(w, h, spec, blocksize=64)
    assert (one.codes == many.codes).float().mean().item() > 0.99  # identical up to floating-point ties
    assert abs(_output_err(x, w, one) - _output_err(x, w, many)) < 0.02 * _output_err(x, w, one)


def test_gptq_with_an_identity_hessian_has_nothing_to_compensate_and_equals_rtn():
    _, w, _ = _layer_problem()
    for spec in (QuantSpec(4, 0), QuantSpec(4, 32)):
        g = gptq_quantize(w, torch.eye(w.shape[1]), spec)
        r = quantize_weight(w, spec)
        assert torch.equal(g.codes, r.codes) and torch.allclose(g.scale, r.scale)


def test_gptq_output_is_a_valid_quantized_weight():
    x, w, h = _layer_problem()
    for spec in (QuantSpec(4, 32), QuantSpec(4, 32, False)):
        qw = gptq_quantize(w, h, spec)
        lo, hi = (-spec.qmax, spec.qmax) if spec.symmetric else (0, spec.qmax)
        assert qw.codes.min() >= lo and qw.codes.max() <= hi
        assert qw.scale.shape == (64, 4)
        QuantLinear(qw)  # can be stored and packed
        assert _output_err(x, w, qw) < _output_err(x, w, QuantizedWeight(torch.zeros_like(qw.codes), qw.scale, qw.zero, spec))


# ------------------------------------------------------------------------------------------ activations


def test_per_token_activation_scales_beat_per_tensor_when_one_token_has_outliers():
    x = torch.randn(16, 64)
    x[3] *= 80
    keep = [i for i in range(16) if i != 3]
    err_tensor = ((fake_quant_act(x, "tensor") - x)[keep] ** 2).mean()
    err_token = ((fake_quant_act(x, "token") - x)[keep] ** 2).mean()
    assert err_tensor > 100 * err_token


# ------------------------------------------------------------------------------------------ with the real model


@pytest.fixture(scope="module")
def ours():
    return load_model(DEFAULT_MODEL, dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_ids(ours):
    _, tok = ours
    text = ("The history of computing stretches from mechanical calculators to modern data centers. Engineers learned that making "
            "programs fast depends less on raw speed and more on avoiding repeated work and moving less data. ") * 3
    return torch.tensor([tok(text).input_ids[:96]])


def _agreement(a, b):
    return (a.argmax(-1) == b.argmax(-1)).float().mean().item()


def test_int8_per_channel_model_is_nearly_lossless(ours, sample_ids):
    model, tok = ours
    q = quantize_model_rtn(copy.deepcopy(model), QuantSpec(8, 0))
    with torch.inference_mode():
        a, b = model(sample_ids), q(sample_ids)
    assert _agreement(a, b) >= 0.95
    assert (a - b).abs().mean().item() < 0.25  # measured 0.107 on logits whose mean magnitude is ~2.5 (4%); int4 g64 gives ~0.87
    ids = tok("The three primary colors are").input_ids
    assert generate(q, ids, 16).token_ids == generate(model, ids, 16).token_ids


def test_quantized_models_are_smaller_by_the_expected_factor(ours):
    model, _ = ours
    fp = model_size_bytes(model)
    q8 = model_size_bytes(quantize_model_rtn(copy.deepcopy(model), QuantSpec(8, 0)))
    q4 = model_size_bytes(quantize_model_rtn(copy.deepcopy(model), QuantSpec(4, 128)))
    assert q8["other"] == fp["other"] == q4["other"]  # embeddings and norms untouched
    assert q8["layer_linears"] < 0.27 * fp["layer_linears"]  # about 1/4 plus scales and biases
    assert q4["layer_linears"] < 0.15 * fp["layer_linears"]  # about 1/8 plus scales and biases


def test_int4_degrades_gracefully_and_int8_beats_int4(ours, sample_ids):
    model, _ = ours
    with torch.inference_mode():
        ref = model(sample_ids)
        err = {}
        for spec in (QuantSpec(8, 0), QuantSpec(4, 64)):
            q = quantize_model_rtn(copy.deepcopy(model), spec)
            err[spec.name] = ((q(sample_ids) - ref) ** 2).mean().item()
    assert err["int8-channel-sym"] < err["int4-g64-sym"]
    assert _agreement(ref, quantize_model_rtn(copy.deepcopy(model), QuantSpec(4, 64))(sample_ids)) > 0.6


def test_smoothquant_leaves_the_floating_point_model_unchanged(ours, sample_ids):
    model, _ = ours
    m = copy.deepcopy(model)
    stats = collect_act_absmax(m, sample_ids.repeat(2, 1))
    smooth_model(m, stats, alpha=0.5)
    with torch.inference_mode():
        a, b = model(sample_ids), m(sample_ids)
    assert (a - b).abs().max().item() < 5e-3  # same function, different internal scaling


def test_smoothquant_shrinks_the_activation_outliers_it_targets(ours, sample_ids):
    model, _ = ours
    m = copy.deepcopy(model)
    ids = sample_ids.repeat(2, 1)
    before = collect_act_absmax(m, ids)
    smooth_model(m, before, alpha=0.5)
    after = collect_act_absmax(m, ids)
    ratio = lambda s: float(sum((v.max() / v.median()) for v in s.values()) / len(s))
    assert ratio(after) < 0.6 * ratio(before)  # outlier channels relative to the typical channel


def test_naive_w8a8_hurts_more_than_weight_only_int8(ours, sample_ids):
    model, _ = ours
    with torch.inference_mode():
        ref = model(sample_ids)
        w8 = quantize_model_rtn(copy.deepcopy(model), QuantSpec(8, 0))
        w8a8 = apply_activation_quant(quantize_model_rtn(copy.deepcopy(model), QuantSpec(8, 0)), per="tensor")
        e_w8 = ((w8(sample_ids) - ref) ** 2).mean().item()
        e_w8a8 = ((w8a8(sample_ids) - ref) ** 2).mean().item()
    assert e_w8a8 > 3 * e_w8  # activation outliers make per-tensor int8 activations lossy


def test_gptq_on_a_two_layer_model_beats_round_to_nearest(ours):
    """End-to-end sequential GPTQ (Hessian collection, quantization, propagation) on the first 2 decoder layers."""
    model, tok = ours
    small = copy.deepcopy(model)
    small.model.layers = small.model.layers[:2]
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 20000, (24, 96), generator=g)  # random tokens suffice to excite the layers
    calib, test = ids[:16], ids[16:]
    spec = QuantSpec(4, 64)
    with torch.inference_mode():
        ref = small(test)
        rtn = quantize_model_rtn(copy.deepcopy(small), spec)
        gptq = quantize_model_gptq(copy.deepcopy(small), calib, spec, batch=8)
        e_rtn = ((rtn(test) - ref) ** 2).mean().item()
        e_gptq = ((gptq(test) - ref) ** 2).mean().item()
    assert e_gptq < e_rtn


# ------------------------------------------------------------------------------------------ KV cache and perplexity


def test_int8_kv_cache_roundtrip_error_is_half_a_step():
    x = torch.randn(1, 2, 5, 64)
    q, s = Int8KVCache._quant(x)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    assert ((q.float() * s.float() - x).abs() <= 0.5 * s.float() + 1e-6).all()


def test_int8_kv_cache_is_close_and_about_half_the_size_of_bf16(ours, sample_ids):
    model, tok = ours
    assert Int8KVCache.bytes_per_token_int8(model.cfg) == 24 * 2 * 2 * (64 + 2)  # 6,336 bytes vs 12,288 for a bf16 cache
    assert Int8KVCache.bytes_per_token_int8(model.cfg) < 0.52 * KVCache.bytes_per_token(model.cfg, torch.bfloat16)
    with torch.inference_mode():
        full = model(sample_ids, cache=KVCache(model.cfg, 1, 96))
        q8 = model(sample_ids, cache=Int8KVCache(model.cfg, 1, 96))
    assert _agreement(full, q8) >= 0.95 and (full - q8).abs().mean().item() < 0.3  # measured 0.154 (6%)
    ids = tok("The three primary colors are").input_ids
    assert generate(model, ids, 12, cache_cls=Int8KVCache).token_ids == generate(model, ids, 12).token_ids


def test_perplexity_is_lower_for_real_text_than_for_shuffled_tokens(ours, sample_ids):
    model, _ = ours
    ids = sample_ids[0]
    shuffled = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))]
    real = perplexity(model, ids, window=48, n_windows=2)
    junk = perplexity(model, shuffled, window=48, n_windows=2)
    assert 1.0 < real["ppl"] < junk["ppl"] and real["tokens"] == 2 * 47


def test_static_per_channel_key_scales_are_calibrated_per_layer_head_and_channel(ours):
    from miniserve.quant import calibrate_key_scales

    model, tok = ours
    ids = torch.tensor([tok("Once upon a time in a small village by the sea there lived an old fisherman who loved the waves. " * 4).input_ids[:64]])
    sc = calibrate_key_scales(model, ids)
    assert sc.shape == (24, 2, 64) and (sc > 0).all()
    assert sc.std(dim=-1).mean() > 0  # channels genuinely differ: a single scale per head would not do


def test_int8_kv_with_static_key_scales_is_more_accurate_than_per_token_key_scales_and_smaller(ours, sample_ids):
    from miniserve.quant import calibrate_key_scales

    model, tok = ours
    paras = ["Programs run faster when engineers avoid repeated work and move less data between memory and processor. ",
             "Once upon a time in a small village by the sea there lived an old fisherman who loved the waves and the gulls. ",
             "The French Revolution began in 1789 and transformed the political landscape of Europe over the following decades. ",
             "To make the sauce, melt the butter in a pan, stir in the flour, and slowly add the warm milk while whisking constantly. ",
             "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen inside the chloroplasts of plant cells. ",
             "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n# the function returns the sum of two numbers\n"]
    large = torch.tensor([tok(p * 6).input_ids[:96] for p in paras])  # 576 tokens from 6 different texts
    small = torch.tensor([tok(paras[0] * 6).input_ids[:64]])  # 64 tokens from one text
    sc, sc_small = calibrate_key_scales(model, large), calibrate_key_scales(model, small)
    with torch.inference_mode():
        full = model(sample_ids, cache=KVCache(model.cfg, 1, 96))
        dyn = model(sample_ids, cache=Int8KVCache(model.cfg, 1, 96))
        stat = model(sample_ids, cache=Int8KVCache(model.cfg, 1, 96, key_scales=sc))
        stat_small = model(sample_ids, cache=Int8KVCache(model.cfg, 1, 96, key_scales=sc_small))
    e_dyn, e_stat, e_small = [(full - o).abs().mean().item() for o in (dyn, stat, stat_small)]
    assert e_stat < 0.5 * e_dyn  # measured 0.046 vs 0.154 (0.30x)
    assert e_small > 2 * e_stat  # static scales need representative calibration: 64 tokens of one text gave 0.212 (worse than per-token)
    assert _agreement(full, stat) >= 0.97
    assert Int8KVCache.bytes_per_token_int8(model.cfg, static_keys=True) == 24 * 2 * (2 * 64 + 2) < Int8KVCache.bytes_per_token_int8(model.cfg)
    cache = Int8KVCache(model.cfg, 1, 96, key_scales=sc)
    assert cache.bytes_used() == 0
    ids = tok("The three primary colors are").input_ids
    from functools import partial

    assert generate(model, ids, 12, cache_cls=partial(Int8KVCache, key_scales=sc)).token_ids == generate(model, ids, 12).token_ids
