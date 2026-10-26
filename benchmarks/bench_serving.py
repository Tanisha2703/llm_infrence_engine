"""Serving benchmark: the three schedulers behind the real HTTP server, driven by the load tester.

  python benchmarks/bench_serving.py                         # policies, closed loop, 2 repeats each
  python benchmarks/bench_serving.py --only sweep            # open-loop arrival-rate sweep on the continuous engine
  python benchmarks/bench_serving.py --device cuda --dtype bfloat16 --slots 16 --concurrency 16 --requests 48

Each configuration starts a FRESH server process (so state, caches and metrics never leak between runs), warms it up,
runs the workload, scrapes the server's own counters, and stops it. Repeats are separate server starts; the table reports
the median and the min-max range, because a laptop CPU's speed drifts between runs (Stage 2-5 saw 110-150 ms/token).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import loadtest  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "raw"
KEYS = ["throughput_tok_s", "ttft_p50_s", "ttft_p95_s", "tpot_p50_ms", "e2e_p50_s", "e2e_p95_s"]


def one_run(server_args: str, workload: list[dict], concurrency: int, rate: float | None, warmup: int, seed: int) -> dict:
    proc, url = loadtest.spawn_server(server_args)
    try:
        if warmup:
            asyncio.run(loadtest.run_load(url, workload[:warmup], 1, None, seed))
        before = loadtest.scrape(url)
        results, wall = asyncio.run(loadtest.run_load(url, workload[warmup:], concurrency, rate, seed))
        after = loadtest.scrape(url)
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    s = loadtest.summarize(results, wall)
    d = {k: after[k] - before[k] for k in after}
    steps = d["miniserve_decode_batch_size_count"]
    s["mean_decode_batch"] = d["miniserve_decode_batch_size_sum"] / steps if steps else 0.0
    s["prefix_hit_rate"] = (d["miniserve_prefix_cache_hit_tokens_total"] / d["miniserve_prompt_tokens_total"]
                            if d["miniserve_prompt_tokens_total"] else 0.0)
    s["preemptions"] = d["miniserve_preemptions_total"]
    return s


def span(runs: list[dict], key: str) -> str:
    v = [r[key] for r in runs]
    return f"{statistics.median(v):.3g} [{min(v):.3g}-{max(v):.3g}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["policies", "sweep"], default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.4, 0.8, 1.6], help="open-loop arrivals/s for the sweep")
    ap.add_argument("--sweep-requests", type=int, default=24)
    ap.add_argument("--out-tokens", type=int, nargs="+", default=[16, 32, 48])
    ap.add_argument("--shared-prefix-words", type=int, default=40)
    ap.add_argument("--max-queue", type=int, default=64)
    ap.add_argument("--tag", default="cpu_float32")
    a = ap.parse_args()

    base = f"--device {a.device} --dtype {a.dtype} --max-len 512 --max-queue {a.max_queue}"
    wl = loadtest.make_workload(a.requests + 2, 0, (30, 90), a.out_tokens, a.shared_prefix_words)
    out: dict = {"args": vars(a), "policies": {}, "sweep": {}}

    if a.only in (None, "policies"):
        configs = {
            "static batching": f"{base} --engine static --max-slots {a.slots}",
            "continuous batching": f"{base} --engine continuous --max-slots {a.slots}",
            "paged + prefix cache": f"{base} --engine paged --max-seqs {a.slots}",
        }
        print(f"\nPOLICIES: closed loop, {a.concurrency} clients, {a.requests} requests, outputs {a.out_tokens}, "
              f"shared prefix {a.shared_prefix_words} words, {a.repeats} repeats (median [min-max])\n")
        for name, args in configs.items():
            runs = []
            for i in range(a.repeats):
                t0 = time.time()
                runs.append(one_run(args, wl, a.concurrency, None, 2, i))
                print(f"  {name} repeat {i + 1}: {runs[-1]['throughput_tok_s']:.1f} tok/s ({time.time() - t0:.0f}s)", flush=True)
            out["policies"][name] = runs
        print(f"\n{'scheme':<24}" + "".join(f"{k:>24}" for k in KEYS + ["mean_decode_batch", "prefix_hit_rate"]))
        for name, runs in out["policies"].items():
            print(f"{name:<24}" + "".join(f"{span(runs, k):>24}" for k in KEYS + ["mean_decode_batch", "prefix_hit_rate"]))

    if a.only in (None, "sweep"):
        args = f"{base} --engine continuous --max-slots {a.slots}"
        wl2 = loadtest.make_workload(a.sweep_requests + 2, 1, (30, 90), a.out_tokens, 0)
        print(f"\nSWEEP: open loop (Poisson arrivals), continuous batching, queue limit {a.max_queue}\n")
        print(f"{'arrivals/s':>10} {'ok':>5} {'429':>5} {'tok/s':>8} {'ttft p50':>9} {'ttft p95':>9} {'e2e p95':>8}")
        for rate in a.rates:
            s = one_run(args, wl2, 1, rate, 2, 0)
            out["sweep"][str(rate)] = s
            print(f"{rate:>10.2f} {s['requests_ok']:>5} {s['failures'].get('429', 0):>5} {s['throughput_tok_s']:>8.1f} "
                  f"{s['ttft_p50_s']:>8.2f}s {s['ttft_p95_s']:>8.2f}s {s['e2e_p95_s']:>7.1f}s", flush=True)

    path = RESULTS / f"serving_{a.tag}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
