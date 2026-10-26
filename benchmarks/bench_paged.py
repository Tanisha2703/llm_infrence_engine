"""Benchmark: paged KV cache and prefix caching.

  python benchmarks/bench_paged.py                       # everything
  python benchmarks/bench_paged.py --only memory         # equal-memory comparison: slots vs paged blocks
  python benchmarks/bench_paged.py --only prefix         # shared system prompt, prefix cache on vs off
  python benchmarks/bench_paged.py --only pressure       # tiny pool: cost of preemption
  python benchmarks/bench_paged.py --only overhead       # decode-step cost of the block-table gather

Correctness is tested in tests/test_paged.py; this file only measures.
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
from miniserve.paged import PagedCache
from miniserve.paged_engine import PagedEngine

PASSAGE = (
    "The history of computing stretches from mechanical calculators to modern data centers. Early machines filled "
    "entire rooms and performed a few operations per second, while today's processors execute billions of operations in "
    "the time it takes to blink. Engineers learned that making programs fast depends less on raw speed and more on "
    "avoiding repeated work, moving less data, and keeping every part of the machine busy. Meanwhile libraries, "
    "compilers and operating systems hid the complexity so that most programmers never see it. "
)
OUTPUT_LENGTHS = [8, 12, 16, 32, 40, 48]
BLOCK = 16


def workload_mixed(tok, n, seed, mean_gap=0.0):
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


def workload_shared_prefix(tok, n, seed, mean_gap, prefix_len=192, new_tokens=16):
    """n requests that all start with the same long 'system prompt' followed by a short unique question."""
    rng = random.Random(seed)
    base = tok(PASSAGE * 6).input_ids
    system = base[:prefix_len]
    specs, t = [], 0.0
    for _ in range(n):
        start = rng.randint(prefix_len, len(base) - 40)
        suffix = base[start : start + rng.randint(16, 32)]
        specs.append({"prompt": system + suffix, "max_new": new_tokens, "arrival": t})
        t += rng.expovariate(1 / mean_gap)
    return specs


def drive(eng, specs, util_fn=None):
    """Feed requests at their scheduled arrival times and step the engine until everything finishes."""
    pending = sorted(specs, key=lambda s: s["arrival"])
    reqs, i, utils, peak = [], 0, [], 0
    t0 = time.perf_counter()
    while i < len(pending) or eng.has_work():
        now = time.perf_counter() - t0
        while i < len(pending) and pending[i]["arrival"] <= now:
            s = pending[i]
            r = eng.add_request(Request(s["prompt"], max_new_tokens=s["max_new"]))
            r.arrival_time = t0 + s["arrival"]
            reqs.append(r)
            i += 1
        if eng.has_work():
            eng.step()
            peak = max(peak, len(eng.running))
            if util_fn:
                u = util_fn(eng)
                if u is not None:
                    utils.append(u)
        else:
            time.sleep(max(0.0, pending[i]["arrival"] - now))
    wall = time.perf_counter() - t0
    out = summarize(reqs, eng.steps, wall)
    out["peak_running"] = peak
    out["mean_committed_memory_useful_fraction"] = sum(utils) / len(utils) if utils else float("nan")
    return out, reqs


def slot_util(eng):
    """Of the memory COMMITTED to running requests (whole slots), what fraction holds real tokens?"""
    if not eng.running:
        return None
    return int(eng.cache.lengths.sum()) / (len(eng.running) * eng.max_len)


def paged_util(eng):
    """Of the blocks held by running requests, what fraction of their positions holds real tokens?"""
    in_use = eng.allocator.num_in_use()
    if not eng.running or in_use == 0:
        return None
    return min(1.0, sum(r.cached_len for r in eng.running) / (in_use * eng.block_size))


def fmt(d, extra=""):
    return (f"throughput {d['throughput_tok_s']:6.2f} tok/s | wall {d['wall_time_s']:5.1f}s | TTFT mean {d['ttft_mean_s']:5.2f}s "
            f"p95 {d['ttft_p95_s']:5.2f}s | E2E mean {d['e2e_mean_s']:5.1f}s | batch {d['mean_decode_batch']:4.1f}{extra}")


def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["memory", "prefix", "pressure", "overhead"], default=None)
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok = load_model(DEFAULT_MODEL, dtype=getattr(torch, args.dtype), device=args.device)
    max_len = 96 + max(OUTPUT_LENGTHS) + 8  # 152
    results = {"env": {"device": args.device, "dtype": args.dtype, "threads": torch.get_num_threads(),
                       "torch": torch.__version__, "block_size": BLOCK, "seed": args.seed}}
    bpt = SlotCache(model.cfg, 1, 1, next(model.parameters()).dtype).total_bytes()  # bytes per token

    # warm-up
    w = PagedEngine(model, num_blocks=30, block_size=BLOCK, max_len=max_len)
    drive(w, workload_mixed(tok, 2, 99))

    if args.only in (None, "memory", "pressure"):
        specs = workload_mixed(tok, args.requests, args.seed)
        slots = 8
        slot_positions = slots * max_len  # the memory budget, in token positions
        pool_blocks = slot_positions // BLOCK + 1  # same bytes (plus the reserved null block)

    if args.only in (None, "memory"):
        print(f"\nMEMORY: {args.requests} requests burst; both caches get {slot_positions} token positions "
              f"({slot_positions * bpt / 2**20:.1f} MiB)")
        results["memory"] = {}
        e1 = Engine(model, max_slots=slots, max_len=max_len, policy="continuous")
        d, _ = drive(e1, specs, slot_util)
        results["memory"]["slots (8 x 152)"] = d
        print(f"  slots (8 x {max_len}){'':8} {fmt(d)} | peak running {d['peak_running']} | useful fraction of committed memory {d['mean_committed_memory_useful_fraction']:.0%}")
        e2 = PagedEngine(model, num_blocks=pool_blocks, block_size=BLOCK, max_seqs=32, max_len=max_len)
        d, _ = drive(e2, specs, paged_util)
        d.update({k: v for k, v in e2.stats().items() if k != "peak_running"})
        results["memory"][f"paged ({pool_blocks - 1} blocks x {BLOCK})"] = d
        print(f"  paged ({pool_blocks - 1} blocks x {BLOCK}) {fmt(d)} | useful fraction of committed memory {d['mean_committed_memory_useful_fraction']:.0%}"
              f" | peak running {d['peak_running']} | preemptions {d['preemptions']}")

    if args.only in (None, "pressure"):
        results["pressure"] = {}
        for small in (39, 20):
            print(f"\nPRESSURE: same {args.requests} requests, pool shrunk to {small - 1} blocks ({(small - 1) * BLOCK} positions)")
            e3 = PagedEngine(model, num_blocks=small, block_size=BLOCK, max_seqs=32, max_len=max_len)
            d, _ = drive(e3, specs, paged_util)
            d.update({k: v for k, v in e3.stats().items() if k != "peak_running"})
            results["pressure"][f"paged ({small - 1} blocks)"] = d
            print(f"  paged ({small - 1} blocks)   {fmt(d)} | peak running {d['peak_running']} | preemptions {d['preemptions']} "
                  f"| prefill tokens computed {d['prefill_tokens_computed']}, from cache {d['prefill_tokens_from_cache']}")

    if args.only in (None, "prefix"):
        specs_p = workload_shared_prefix(tok, 12, args.seed, mean_gap=1.2)
        total_prompt = sum(len(s["prompt"]) for s in specs_p)
        print(f"\nPREFIX CACHING: 12 requests sharing a 192-token system prompt (+16-32 unique tokens, 16 new tokens), "
              f"Poisson arrivals over {specs_p[-1]['arrival']:.1f}s; {total_prompt} prompt tokens in total")
        results["prefix"] = {}
        for label, on in (("prefix cache OFF", False), ("prefix cache ON", True)):
            e = PagedEngine(model, num_blocks=200, block_size=BLOCK, max_seqs=16, max_len=320, enable_prefix_cache=on)
            d, _ = drive(e, specs_p, paged_util)
            d.update({k: v for k, v in e.stats().items() if k != "peak_running"})
            results["prefix"][label] = d
            print(f"  {label:17} {fmt(d)} | prefill tokens computed {d['prefill_tokens_computed']:5d} "
                  f"| from cache {d['prefill_tokens_from_cache']:5d} ({100 * d['prefix_hit_rate']:.0f}%)")

    if args.only in (None, "overhead"):
        print("\nOVERHEAD: one batched decode step, every row holding 128 cached tokens (best of 3)")
        results["overhead"] = []
        dt = next(model.parameters()).dtype
        print(f"{'batch':>5} {'slot cache ms':>14} {'paged cache ms':>15} {'paged / slot':>13}")
        for B in (1, 4, 8):
            ids = torch.randint(1000, 5000, (B, 1), device=next(model.parameters()).device)
            sc = SlotCache(model.cfg, B, 136, dt, next(model.parameters()).device)
            nb = 136 // BLOCK + 1
            pc = PagedCache(model.cfg, B * nb + 1, BLOCK, dt, next(model.parameters()).device)
            tables = [list(range(1 + b * nb, 1 + (b + 1) * nb)) for b in range(B)]
            t_slot = t_paged = float("inf")
            with torch.inference_mode():
                for r in range(4):  # first iteration is warm-up
                    sc.lengths[:] = 128
                    sync(ids.device)  # CUDA launches are asynchronous: time the work, not the enqueue
                    s = time.perf_counter()
                    model(ids, cache=sc.view(range(B)), last_only=True)
                    sync(ids.device)
                    a = time.perf_counter() - s
                    s = time.perf_counter()
                    model(ids, cache=pc.view(tables, [128] * B), last_only=True)
                    sync(ids.device)
                    b_ = time.perf_counter() - s
                    if r > 0:
                        t_slot, t_paged = min(t_slot, a), min(t_paged, b_)
            results["overhead"].append({"batch": B, "slot_ms": t_slot * 1000, "paged_ms": t_paged * 1000})
            print(f"{B:>5} {t_slot * 1000:>14.1f} {t_paged * 1000:>15.1f} {t_paged / t_slot:>12.2f}x")

    out = Path(args.out or f"results/raw/paged_{args.device.split(':')[0]}_{args.dtype}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
