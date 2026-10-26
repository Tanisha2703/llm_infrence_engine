"""Benchmark: quantization (weights, activations, KV cache): quality, size and speed.

  python benchmarks/bench_quant.py --only stats      # outlier statistics of real weights and activations
  python benchmarks/bench_quant.py --only ppl        # perplexity + size for every scheme (the long one: ~25 minutes on CPU)
  python benchmarks/bench_quant.py --only speed      # decode / prefill speed, incl. PyTorch's real int8 kernel
  python benchmarks/bench_quant.py --only gen        # greedy generation agreement with the fp32 model

Protocol: WikiText-2 (raw) TEST text, non-overlapping windows of 512 tokens, 40 windows (40 x 511 = 20,440 predicted tokens). GPTQ and SmoothQuant are
calibrated on 32 random 256-token windows of the TRAIN split (never the test text). Model: Qwen2.5-0.5B-Instruct computed in fp32 (the checkpoint
is bf16, so 'fp32' here is lossless relative to the released weights).
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import time
from functools import partial
from pathlib import Path

import torch

from miniserve.cache import KVCache
from miniserve.evalutil import calibration_windows, perplexity, wikitext_ids
from miniserve.generate import generate
from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.quant import (
    LINEAR_SLOTS,
    Int8KVCache,
    QuantLinear,
    QuantSpec,
    apply_activation_quant,
    calibrate_key_scales,
    collect_act_absmax,
    model_size_bytes,
    quantize_model_gptq,
    quantize_model_rtn,
    smooth_model,
)

RESULTS = Path("results/raw")


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def mib(n):
    return n / 2**20


# ----------------------------------------------------------------------------------------------- outlier statistics
def stats(model, tok, calib):
    print("\nWEIGHT OUTLIERS (why per-tensor scales fail)")
    print(f"{'layer.matrix':<18} {'max|w|':>8} {'median|w|':>10} {'max/median':>11} {'row-absmax: max/median':>23}")
    out = {"weights": {}, "activations": {}}
    for li in (0, 12, 23):
        layer = model.model.layers[li]
        for parent_name, name in LINEAR_SLOTS:
            if name not in ("q_proj", "down_proj"):
                continue
            w = getattr(getattr(layer, parent_name), name).weight.data.abs()
            rows = w.amax(1)
            row = f"{li}.{name}"
            out["weights"][row] = {"max": w.max().item(), "median": w.median().item(), "row_spread": (rows.max() / rows.median()).item()}
            print(f"{row:<18} {w.max():>8.3f} {w.median():>10.4f} {w.max() / w.median():>11.0f} {rows.max() / rows.median():>23.1f}")
    print("\nACTIVATION OUTLIERS at the inputs of the linear layers (per-channel max over calibration tokens)")
    st = collect_act_absmax(model, calib)
    down_in = {}
    handles = []
    for li, layer in enumerate(model.model.layers):
        def hook(_m, inp, li=li):
            cur = inp[0].abs().reshape(-1, inp[0].shape[-1]).amax(0)
            down_in[li] = cur if li not in down_in else torch.maximum(down_in[li], cur)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    with torch.inference_mode():
        for b in range(0, calib.shape[0], 8):
            model(calib[b : b + 8], last_only=True)
    for h in handles:
        h.remove()
    print(f"{'input of':<22} {'layer':>5} {'largest channel':>16} {'median channel':>15} {'ratio':>8}")
    for key, label in ((("attn", None), "q/k/v (after norm)"), (("mlp", None), "gate/up (after norm)"), (("down", None), "down_proj")):
        for li in (0, 12, 23):
            v = down_in[li] if key[0] == "down" else st[(key[0], li)]
            out["activations"][f"{label}@{li}"] = {"max": v.max().item(), "median": v.median().item(), "ratio": (v.max() / v.median()).item()}
            print(f"{label:<22} {li:>5} {v.max():>16.2f} {v.median():>15.3f} {v.max() / v.median():>8.0f}")
    return out


# ----------------------------------------------------------------------------------------------- perplexity + size
def build_variants(model, calib, only=None):
    """name -> (builder returning a (model, cache_cls) pair, description). Builders construct lazily so only one extra model lives in memory."""
    spec = QuantSpec

    def rtn(sp):
        return lambda: (quantize_model_rtn(copy.deepcopy(model), sp), None)

    def gptq(sp):
        return lambda: (quantize_model_gptq(copy.deepcopy(model), calib, sp, progress=lambda i, n: print(f"    GPTQ layer {i}/{n}", end="\r", flush=True)), None)

    def w8a8(per, smooth):
        def build():
            m = copy.deepcopy(model)
            if smooth:
                smooth_model(m, collect_act_absmax(m, calib), alpha=0.5)
            return apply_activation_quant(quantize_model_rtn(m, spec(8, 0)), per=per), None
        return build

    v = {
        "fp32 (baseline)": lambda: (model, None),
        "RTN int8 per-tensor": rtn(spec(8, -1)),
        "RTN int8 per-channel": rtn(spec(8, 0)),
        "RTN int4 per-channel": rtn(spec(4, 0)),
        "RTN int4 g128": rtn(spec(4, 128)),
        "RTN int4 g64": rtn(spec(4, 64)),
        "RTN int4 g32": rtn(spec(4, 32)),
        "RTN int4 g128 asymmetric": rtn(spec(4, 128, False)),
        "RTN int3 g32": rtn(spec(3, 32)),
        "GPTQ int4 per-channel": gptq(spec(4, 0)),
        "GPTQ int4 g128": gptq(spec(4, 128)),
        "GPTQ int4 g32": gptq(spec(4, 32)),
        "GPTQ int3 g32": gptq(spec(3, 32)),
        "W8A8 per-tensor act": w8a8("tensor", False),
        "W8A8 per-token act": w8a8("token", False),
        "W8A8 per-tensor act + SmoothQuant": w8a8("tensor", True),
        "fp32 + int8 KV cache": lambda: (model, Int8KVCache),
        "fp32 + int8 KV cache (static per-channel keys)": lambda: (model, partial(Int8KVCache, key_scales=calibrate_key_scales(model, calib))),
    }
    return {k: f for k, f in v.items() if only is None or k in only}


def quantized_bytes(m):
    sz = model_size_bytes(m)
    ideal = 0.0
    for mod in m.modules():
        if isinstance(mod, QuantLinear):
            ideal += mod.ideal_bytes() + (0 if mod.bias is None else mod.bias.numel() * 4)
    return sz, ideal


def ppl_section(model, tok, test_ids, calib, windows, only, out_path, bf16_total):
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    print(f"\nPERPLEXITY on WikiText-2 test ({windows} windows x 512 tokens)\n")
    print(f"{'scheme':<36} {'ppl':>8} {'vs fp32':>8} {'layer weights MiB':>18} {'model MiB':>10} {'x smaller (fp32)':>17} {'quantize s':>11}")
    variants = build_variants(model, calib, only)
    for name, build in variants.items():
        if name in results and not only:
            continue
        t0 = time.perf_counter()
        m, cache_cls = build()
        t_q = time.perf_counter() - t0
        sz, ideal = quantized_bytes(m)
        p = perplexity(m, test_ids, 512, windows, cache_cls)
        results[name] = {"ppl": p["ppl"], "nll": p["nll"], "tokens": p["tokens"], "layer_linear_bytes": sz["layer_linears"], "model_bytes": sz["total"],
                         "ideal_layer_linear_bytes": ideal, "quantize_seconds": t_q}
        base = results.get("fp32 (baseline)", {}).get("ppl", p["ppl"])
        fp_total = results.get("fp32 (baseline)", {}).get("model_bytes", sz["total"])
        print(f"{name:<36} {p['ppl']:>8.3f} {p['ppl'] - base:>+8.3f} {mib(sz['layer_linears']):>18.1f} {mib(sz['total']):>10.1f} "
              f"{fp_total / sz['total']:>16.2f}x {t_q:>11.1f}")
        save(out_path, results)
        if m is not model:
            del m
    return results


# ----------------------------------------------------------------------------------------------- speed
def decode_speed(model, ids, cache_cls=KVCache, new_tokens=24, repeats=2):
    best = None
    for _ in range(repeats + 1):  # first is warm-up
        r = generate(model, ids, new_tokens, cache_cls=cache_cls)
        if _ > 0 and (best is None or r.ms_per_token < best):
            best = r.ms_per_token
    return best


def prefill_speed(model, n=256, repeats=2):
    ids = torch.randint(1000, 5000, (1, n))
    best = float("inf")
    with torch.inference_mode():
        for r in range(repeats + 1):
            s = time.perf_counter()
            model(ids, last_only=True)
            if r > 0:
                best = min(best, time.perf_counter() - s)
    return best


def dyn_quant(model):
    from torch.ao.quantization import per_channel_dynamic_qconfig, quantize_dynamic

    return quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear: per_channel_dynamic_qconfig}, dtype=torch.qint8)


def serialized_mib(m):
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return mib(buf.tell())


def speed_section(model, tok, test_ids, windows):
    ids = tok("The history of computing stretches from mechanical calculators to modern data centers. Engineers learned that").input_ids
    print("\nSPEED (CPU, fp32 activations; decode = ms per generated token after the first, prefill = 256 tokens)\n")
    print(f"{'variant':<44} {'decode ms/tok':>14} {'tok/s':>7} {'vs fp32':>8} {'prefill ms':>11} {'weights MiB':>12}")
    rows = {}

    def run(name, m, size_mib, cache_cls=KVCache):
        d = decode_speed(m, ids, cache_cls)
        pf = prefill_speed(m) * 1000
        rows[name] = {"decode_ms": d, "decode_tok_s": 1000 / d, "prefill_ms": pf, "weights_mib": size_mib}
        base = rows.get("fp32 baseline", {}).get("decode_ms", d)
        print(f"{name:<44} {d:>14.1f} {1000 / d:>7.2f} {base / d:>7.2f}x {pf:>11.0f} {size_mib:>12.0f}")

    run("fp32 baseline", model, mib(model_size_bytes(model)["total"]))
    for label, sp in (("int8 per-channel", QuantSpec(8, 0)), ("int4 g128", QuantSpec(4, 128))):
        m = quantize_model_rtn(copy.deepcopy(model), sp)
        run(f"{label}: dequantize on every call (ours)", m, mib(model_size_bytes(m)["total"]))
        del m
        m = quantize_model_rtn(copy.deepcopy(model), sp, cache_dequant=True)
        run(f"{label}: dequantize once, cache fp32 (ours)", m, mib(model_size_bytes(m)["total"]))
        del m
    try:
        dq = dyn_quant(model)
        run("int8 W+A dynamic, PyTorch oneDNN kernel", dq, serialized_mib(dq))
        p = perplexity(dq, test_ids, 512, windows)
        rows["int8 W+A dynamic, PyTorch oneDNN kernel"]["ppl"] = p["ppl"]
        print(f"    (perplexity of the PyTorch int8 kernel model: {p['ppl']:.3f})")
        del dq
    except Exception as e:  # keep the benchmark running if this build lacks the kernel
        print("    PyTorch dynamic int8 unavailable:", repr(e)[:150])
    ctx_ids = tok("The history of computing " * 40).input_ids[:256]
    d_fp = decode_speed(model, ctx_ids, KVCache)
    d_q8 = decode_speed(model, ctx_ids, Int8KVCache)
    rows["kv"] = {"fp32_cache_ms": d_fp, "int8_cache_ms": d_q8}
    print(f"\nKV cache at 256 tokens of context: fp32 cache {d_fp:.1f} ms/token, int8 cache {d_q8:.1f} ms/token "
          f"(cache bytes/token: {KVCache.bytes_per_token(model.cfg, torch.float32)} fp32, {KVCache.bytes_per_token(model.cfg, torch.bfloat16)} bf16, "
          f"{Int8KVCache.bytes_per_token_int8(model.cfg)} int8)")
    return rows


# ----------------------------------------------------------------------------------------------- greedy agreement
def gen_section(model, tok, calib):
    prompts = ["The three primary colors are", "Explain what a cache is in one sentence.", "def fibonacci(n):",
               "The capital of France is", "Once upon a time in a small village by the sea"]
    ids = [tok(p).input_ids for p in prompts]
    n = 32
    ref = [generate(model, i, n).token_ids for i in ids]
    print("\nGREEDY GENERATION vs the fp32 model (5 prompts x 32 tokens)\n")
    print(f"{'scheme':<30} {'tokens identical':>17} {'first divergence (mean token index)':>38}")
    out = {}
    for name, build in {"RTN int8 per-channel": lambda: quantize_model_rtn(copy.deepcopy(model), QuantSpec(8, 0)),
                        "RTN int4 g128": lambda: quantize_model_rtn(copy.deepcopy(model), QuantSpec(4, 128)),
                        "RTN int4 g32": lambda: quantize_model_rtn(copy.deepcopy(model), QuantSpec(4, 32)),
                        "GPTQ int4 g128": lambda: quantize_model_gptq(copy.deepcopy(model), calib, QuantSpec(4, 128))}.items():
        m = build()
        same = div = 0
        for i, r in zip(ids, ref):
            g = generate(m, i, n).token_ids
            same += sum(a == b for a, b in zip(g, r))
            k = next((j for j, (a, b) in enumerate(zip(g, r)) if a != b), n)
            div += k
        out[name] = {"identical_fraction": same / (n * len(ids)), "mean_first_divergence": div / len(ids)}
        print(f"{name:<30} {100 * same / (n * len(ids)):>16.0f}% {div / len(ids):>38.1f}")
        del m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["stats", "ppl", "speed", "gen"], default=None)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--schemes", nargs="*", default=None, help="restrict --only ppl to these scheme names")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok = load_model(DEFAULT_MODEL, dtype=torch.float32)
    bf16_total = sum(p.numel() for p in model.parameters()) * 2
    need_data = args.only in (None, "stats", "ppl", "speed", "gen")
    test_ids = wikitext_ids(tok, "test", max_tokens=args.windows * 512 + 64) if need_data else None
    train_ids = wikitext_ids(tok, "train", max_tokens=60000)
    calib = calibration_windows(train_ids, 32, 256, seed=0)
    out = {}
    if args.only in (None, "stats"):
        out["stats"] = stats(model, tok, calib)
        save(RESULTS / "quant_stats.json", out["stats"])
    if args.only in (None, "ppl"):
        out["ppl"] = ppl_section(model, tok, test_ids, calib, args.windows, args.schemes, Path(args.out or RESULTS / "quant_ppl.json"), bf16_total)
    if args.only in (None, "speed"):
        out["speed"] = speed_section(model, tok, test_ids, min(args.windows, 10))
        save(RESULTS / "quant_speed.json", out["speed"])
    if args.only in (None, "gen"):
        out["gen"] = gen_section(model, tok, calib)
        save(RESULTS / "quant_gen.json", out["gen"])


if __name__ == "__main__":
    main()
