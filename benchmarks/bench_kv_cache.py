"""Benchmark: generation with and without the KV cache.

Run:  python benchmarks/bench_kv_cache.py [--prompt-lens 16 128 256] [--new-tokens 32]
Writes results/raw/kv_cache_<device>_<dtype>.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch

from miniserve.cache import KVCache
from miniserve.generate import generate
from miniserve.loader import DEFAULT_MODEL, load_model

PASSAGE = (
    "The history of computing stretches from mechanical calculators to modern data centers. "
    "Early machines filled entire rooms and performed a few operations per second, while today's "
    "processors execute billions of operations in the time it takes to blink. Engineers learned that "
    "making programs fast depends less on raw speed and more on avoiding repeated work, moving less "
    "data, and keeping every part of the machine busy. "
)


def make_prompt(tok, n_tokens: int) -> list[int]:
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids += tok(PASSAGE).input_ids
    return ids[:n_tokens]


def run(model, prompt, new_tokens, use_cache, repeats):
    runs = [generate(model, prompt, new_tokens, use_cache=use_cache) for _ in range(repeats)]
    best = min(runs, key=lambda r: r.total_time)  # least-noisy run
    return best, [r.total_time for r in runs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 128, 256])
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, tok = load_model(DEFAULT_MODEL, dtype=dtype, device=args.device)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    env = {
        "device": args.device,
        "dtype": args.dtype,
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "weight_bytes": weight_bytes,
    }
    print(env)

    # Warm-up (first call pays one-time allocation / kernel-selection costs)
    generate(model, make_prompt(tok, 8), 4)

    rows = []
    for L in args.prompt_lens:
        prompt = make_prompt(tok, L)
        fast, fast_all = run(model, prompt, args.new_tokens, True, args.repeats)
        slow, slow_all = run(model, prompt, args.new_tokens, False, max(1, args.repeats - 1))
        identical = fast.token_ids == slow.token_ids
        if args.dtype == "float32":
            assert identical, "cache changed the output!"  # in fp32 cached and uncached must agree exactly
        elif not identical:  # bf16/fp16: different kernels round differently, a near-tied argmax may flip; report it
            print(f"  note: prompt {L}: cached and uncached outputs differ in {args.dtype} (reduced-precision rounding)")
        rows.append(
            {
                "prompt_len": L,
                "outputs_identical": identical,
                "new_tokens": len(fast.token_ids),
                "uncached": {
                    "ttft_s": slow.ttft,
                    "ms_per_token": slow.ms_per_token,
                    "decode_tps": slow.decode_tps,
                    "total_s": slow.total_time,
                },
                "cached": {
                    "ttft_s": fast.ttft,
                    "ms_per_token": fast.ms_per_token,
                    "decode_tps": fast.decode_tps,
                    "total_s": fast.total_time,
                    "cache_mb": fast.cache_bytes / 2**20,
                    "effective_weight_bandwidth_gbs": weight_bytes / (fast.ms_per_token / 1000) / 1e9,
                },
                "speedup_total": slow.total_time / fast.total_time,
                "speedup_decode": slow.ms_per_token / fast.ms_per_token,
                "total_s_spread_cached": [min(fast_all), statistics.mean(fast_all), max(fast_all)],
            }
        )
        r = rows[-1]
        print(
            f"prompt={L:4d} | uncached {r['uncached']['ms_per_token']:8.1f} ms/tok "
            f"| cached {r['cached']['ms_per_token']:7.1f} ms/tok | decode speedup {r['speedup_decode']:5.1f}x "
            f"| end-to-end {r['speedup_total']:5.1f}x"
        )

    out = Path(args.out or f"results/raw/kv_cache_{args.device.split(':')[0]}_{args.dtype}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env, "rows": rows}, indent=2))
    print(f"\nwrote {out}")

    print("\n| prompt | mode | TTFT | ms/token | tokens/s | total |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        for mode in ("uncached", "cached"):
            m = r[mode]
            print(
                f"| {r['prompt_len']} | {mode} | {m['ttft_s']:.2f}s | {m['ms_per_token']:.1f} "
                f"| {m['decode_tps']:.2f} | {m['total_s']:.2f}s |"
            )
    c = rows[-1]["cached"]
    print(
        f"\ncached decode streams {weight_bytes/1e9:.2f} GB of weights per token => "
        f"~{c['effective_weight_bandwidth_gbs']:.1f} GB/s effective bandwidth (compare with 1.13's ceiling)"
    )
    _ = KVCache  # imported so benchmark fails fast if the cache module breaks


if __name__ == "__main__":
    main()
