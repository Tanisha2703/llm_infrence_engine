"""Benchmark: batching policies.

  python benchmarks/bench_batching.py                 # scaling + both scenarios
  python benchmarks/bench_batching.py --only scaling  # just decode-step cost vs batch size
  python benchmarks/bench_batching.py --requests 12 --slots 8

1. scaling : cost of ONE decode step as the batch grows.
2. burst   : all requests arrive at t=0.
3. staggered: Poisson arrivals, so requests join while others are mid-generation.
Policies: sequential (static, 1 slot)  |  static batching (batch = slots)  |  continuous batching (slots).
Every request is checked against... nothing here: correctness is tested in tests/test_batching.py. This file only measures.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from miniserve.cache import SlotCache
from miniserve.engine import Engine, Request
from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.metrics import summarize

PASSAGE = (
    "The history of computing stretches from mechanical calculators to modern data centers. Early machines filled "
    "entire rooms and performed a few operations per second, while today's processors execute billions of operations in "
    "the time it takes to blink. Engineers learned that making programs fast depends less on raw speed and more on "
    "avoiding repeated work, moving less data, and keeping every part of the machine busy. Meanwhile libraries, "
    "compilers and operating systems hid the complexity so that most programmers never see it. "
)
OUTPUT_LENGTHS = [8, 12, 16, 32, 40, 48]  # mixed short and long answers: what makes continuous batching shine


def make_workload(tok, n, seed, mean_gap):
    rng = random.Random(seed)
    base = tok(PASSAGE * 3).input_ids
    specs, t = [], 0.0
    for _ in range(n):
        plen = rng.randint(16, 96)
        start = rng.randint(0, len(base) - plen)
        specs.append({"prompt": base[start : start + plen], "max_new": rng.choice(OUTPUT_LENGTHS), "arrival": t})
        if mean_gap > 0:
            t += rng.expovariate(1 / mean_gap)
    return specs


def run_policy(model, specs, policy, slots, max_len, cuda_graphs=False):
    eng = Engine(model, max_slots=slots, max_len=max_len, policy=policy, cuda_graphs=cuda_graphs)
    pending = sorted(specs, key=lambda s: s["arrival"])
    reqs, i = [], 0
    t0 = time.perf_counter()
    while i < len(pending) or eng.has_work():
        now = time.perf_counter() - t0
        while i < len(pending) and pending[i]["arrival"] <= now:
            s = pending[i]
            r = eng.add_request(Request(s["prompt"], max_new_tokens=s["max_new"]))
            r.arrival_time = t0 + s["arrival"]  # the scheduled time, not "whenever the loop got to it"
            reqs.append(r)
            i += 1
        if eng.has_work():
            eng.step()
        else:
            time.sleep(max(0.0, pending[i]["arrival"] - now))
    wall = time.perf_counter() - t0
    out = summarize(reqs, eng.steps, wall)
    out["cache_reserved_mb"] = eng.cache.total_bytes() / 2**20
    return out


def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def scaling(model, ctx=128, batches=(1, 2, 4, 8, 16), repeats=3):
    rows = []
    wbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    for B in batches:
        sc = SlotCache(model.cfg, B, ctx + 8, next(model.parameters()).dtype, next(model.parameters()).device)
        ids = torch.randint(1000, 5000, (B, 1), device=next(model.parameters()).device)
        best = float("inf")
        with torch.inference_mode():
            for r in range(repeats + 1):  # first iteration is warm-up
                sc.lengths[:] = ctx  # pretend every row already holds ctx tokens (K/V zeros are fine for timing)
                sync(ids.device)  # CUDA launches are asynchronous: without these, only the enqueue time is measured
                s = time.perf_counter()
                model(ids, cache=sc.view(range(B)), last_only=True)
                sync(ids.device)
                dt = time.perf_counter() - s
                if r > 0:
                    best = min(best, dt)
        rows.append({"batch": B, "step_ms": best * 1000, "tokens_per_s": B / best})
    for r in rows:
        r["speedup_vs_b1"] = r["tokens_per_s"] / rows[0]["tokens_per_s"]
        r["ms_per_token_per_row"] = r["step_ms"] / r["batch"]
        r["weight_gbs"] = wbytes / (r["step_ms"] / 1000) / 1e9
    return rows


def fmt(d):
    return (
        f"throughput {d['throughput_tok_s']:6.2f} tok/s | wall {d['wall_time_s']:6.1f}s | TTFT mean {d['ttft_mean_s']:5.2f}s "
        f"p95 {d['ttft_p95_s']:5.2f}s | E2E mean {d['e2e_mean_s']:5.1f}s p95 {d['e2e_p95_s']:5.1f}s | "
        f"batch {d['mean_decode_batch']:4.1f} | wasted rows {100 * d['wasted_row_fraction']:3.0f}%"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["scaling", "burst", "staggered"], default=None)
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--mean-gap", type=float, default=1.5, help="mean seconds between arrivals in the staggered scenario")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cuda-graphs", action="store_true", help="also run continuous batching with CUDA-graph decoding (GPU only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok = load_model(DEFAULT_MODEL, dtype=getattr(torch, args.dtype), device=args.device)
    max_len = 96 + max(OUTPUT_LENGTHS) + 8
    results = {"env": {"device": args.device, "dtype": args.dtype, "threads": torch.get_num_threads(), "torch": torch.__version__,
                       "slots": args.slots, "requests": args.requests, "seed": args.seed}}

    # warm-up so the first measured scenario does not pay one-time costs
    run_policy(model, make_workload(tok, 2, 99, 0.0), "continuous", 2, max_len)
    if args.cuda_graphs:
        run_policy(model, make_workload(tok, 2, 99, 0.0), "continuous", 2, max_len, cuda_graphs=True)

    if args.only in (None, "scaling"):
        rows = scaling(model)
        results["scaling"] = rows
        print("\nDECODE STEP COST vs BATCH SIZE (every row has 128 cached tokens)")
        print(f"{'batch':>5} {'step ms':>9} {'tokens/s':>9} {'speedup':>8} {'ms/token/row':>13} {'weights GB/s':>13}")
        for r in rows:
            print(f"{r['batch']:>5} {r['step_ms']:>9.1f} {r['tokens_per_s']:>9.2f} {r['speedup_vs_b1']:>7.2f}x {r['ms_per_token_per_row']:>13.1f} {r['weight_gbs']:>13.1f}")

    for scenario, gap in (("burst", 0.0), ("staggered", args.mean_gap)):
        if args.only not in (None, scenario):
            continue
        specs = make_workload(tok, args.requests, args.seed, gap)
        total_tokens = sum(s["max_new"] for s in specs)
        print(f"\n{scenario.upper()}: {args.requests} requests, {total_tokens} output tokens, "
              f"arrivals span {specs[-1]['arrival']:.1f}s, slots {args.slots}")
        results[scenario] = {}
        for name, policy, slots in (("sequential", "static", 1), (f"static batch={args.slots}", "static", args.slots),
                                    (f"continuous slots={args.slots}", "continuous", args.slots),
                                    *(((f"continuous slots={args.slots} + CUDA graphs", "continuous", args.slots),) if args.cuda_graphs else ())):
            d = run_policy(model, specs, policy, slots, max_len, cuda_graphs=name.endswith("CUDA graphs"))
            results[scenario][name] = d
            print(f"  {name:22} {fmt(d)}")

    out = Path(args.out or f"results/raw/batching_{args.device.split(':')[0]}_{args.dtype}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
