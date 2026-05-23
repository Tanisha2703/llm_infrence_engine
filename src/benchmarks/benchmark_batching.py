"""Benchmark: request scheduling strategies.

Simulates a burst of concurrent requests and measures:
  - Throughput (requests / second)
  - Mean wait time (time from arrival to inference start)
  - End-to-end latency per request

Strategies compared:
  1. No batching  – sequential, one request at a time
  2. Static batching – collect up to `batch_size` before processing
  3. Continuous batching – fill slots as they free up

Usage
-----
    python -m src.benchmarks.benchmark_batching
"""
import sys
import time
import threading
from pathlib import Path
from statistics import mean, median

from src.server.inference import InferenceEngine
from src.optimizations.batching import (
    Request,
    NoBatchScheduler,
    StaticBatchScheduler,
    ContinuousBatchScheduler,
)
from src.benchmarks.metrics import print_table

MODEL_PATH = "models/tinyllama-1.1b-Q4_K_M.gguf"
N_THREADS = 4
MAX_TOKENS = 128
N_CTX = 2048
N_REQUESTS = 8  # total requests in the burst
BATCH_SIZE = 4
BATCH_TIMEOUT = 0.5  # seconds

PROMPTS = [
    "What is the speed of light?",
    "Who wrote Hamlet?",
    "What is DNA?",
    "Define entropy.",
    "What is the Pythagorean theorem?",
    "Who invented the telephone?",
    "What is a black hole?",
    "How does the heart pump blood?",
]


def _run_scheduler(scheduler, prompts: list[str]) -> dict:
    """Send all prompts concurrently and collect timing statistics."""
    requests = [Request(prompt=p, max_tokens=MAX_TOKENS) for p in prompts]

    # Record global start (all requests sent at t=0)
    t_global_start = time.perf_counter()
    # Reset arrival times relative to global start
    for r in requests:
        r.arrival_time = t_global_start

    threads = []
    results_store: list[dict] = [{}] * len(requests)

    def _submit(idx: int, req: Request) -> None:
        result = scheduler.submit(req)
        results_store[idx] = result

    for i, req in enumerate(requests):
        t = threading.Thread(target=_submit, args=(i, req), daemon=True)
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)

    t_global_end = time.perf_counter()
    elapsed = t_global_end - t_global_start

    wait_times = [r.get("wait_time", 0.0) for r in results_store if r]
    ttfts = [r.get("ttft", 0.0) for r in results_store if r]

    return {
        "elapsed_s": elapsed,
        "throughput": len(prompts) / elapsed,
        "mean_wait": mean(wait_times) if wait_times else 0.0,
        "median_wait": median(wait_times) if wait_times else 0.0,
        "mean_ttft": mean(ttfts) if ttfts else 0.0,
        "n_completed": len([r for r in results_store if r]),
    }


def main() -> None:
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}\nSee SETUP.md.")
        sys.exit(1)

    print("=" * 60)
    print("Batching Strategy Benchmark")
    print("=" * 60)
    print(f"Model      : {MODEL_PATH}")
    print(f"Requests   : {N_REQUESTS}  (sent concurrently)")
    print(f"Max tokens : {MAX_TOKENS}")
    print()

    engine = InferenceEngine(MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS)
    infer_fn = engine.generate

    schedulers = [
        ("No batching",        NoBatchScheduler(infer_fn)),
        ("Static batching",    StaticBatchScheduler(infer_fn, batch_size=BATCH_SIZE, timeout=BATCH_TIMEOUT)),
        ("Continuous batching", ContinuousBatchScheduler(infer_fn, max_concurrent=2)),
    ]

    rows = []
    baseline_wait: float | None = None

    for name, sched in schedulers:
        print(f"Running: {name} …", flush=True)
        r = _run_scheduler(sched, PROMPTS[:N_REQUESTS])
        if baseline_wait is None:
            baseline_wait = r["mean_wait"]
        rows.append((
            name,
            f"{r['throughput']:.3f} req/s",
            f"{r['mean_wait']:.1f}s",
            f"{r['median_wait']:.1f}s",
            f"{r['mean_ttft']:.2f}s",
        ))
        print(f"  → {r['throughput']:.3f} req/s  wait={r['mean_wait']:.1f}s  TTFT={r['mean_ttft']:.2f}s")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print_table(rows, ["Scheduler", "Throughput", "Mean wait", "Median wait", "Mean TTFT"])


if __name__ == "__main__":
    main()
