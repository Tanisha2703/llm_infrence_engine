"""Benchmark: KV cache eviction policies.

Compares three cache policies (no cache, FIFO, LRU, sliding window) on a
workload where a subset of prompts repeats.  A cache hit returns the stored
response immediately; a miss goes to the inference engine.

Usage
-----
    python -m src.benchmarks.benchmark_kv_cache
"""
import sys
import time
from pathlib import Path

from src.server.inference import InferenceEngine
from src.optimizations.kv_cache import LRUCache, FIFOCache, SlidingWindowCache
from src.benchmarks.metrics import print_table

MODEL_PATH = "models/tinyllama-1.1b-Q4_K_M.gguf"
N_THREADS = 4
MAX_TOKENS = 256
N_CTX = 2048
CACHE_CAPACITY = 32

# Workload: 6 unique prompts, repeated 3 times each (18 total requests)
UNIQUE_PROMPTS = [
    "Explain the water cycle.",
    "What is Newton's first law?",
    "How does the internet work?",
    "What is photosynthesis?",
    "Explain machine learning in one paragraph.",
    "What causes earthquakes?",
]
REPEAT_FACTOR = 3
WORKLOAD = UNIQUE_PROMPTS * REPEAT_FACTOR  # 18 requests


def _run_with_cache(engine: InferenceEngine, cache, prompts: list[str]) -> dict:
    """Run all prompts through the cache → engine pipeline and collect metrics."""
    cache.reset_stats()
    ttfts: list[float] = []

    for prompt in prompts:
        cached = cache.get(prompt)
        if cached is not None:
            ttfts.append(0.0)  # instant cache hit
            continue

        result = engine.generate(prompt, max_tokens=MAX_TOKENS, temperature=0.0)
        cache.put(prompt, result["text"])
        ttfts.append(result["ttft"])

    non_zero = [t for t in ttfts if t > 0]
    avg_ttft_miss = sum(non_zero) / len(non_zero) if non_zero else 0.0
    avg_ttft_all = sum(ttfts) / len(ttfts) if ttfts else 0.0
    return {
        "avg_ttft_all": avg_ttft_all,
        "avg_ttft_miss": avg_ttft_miss,
        "hit_rate": cache.hit_rate,
        "hits": cache.hits,
        "misses": cache.misses,
    }


def _run_no_cache(engine: InferenceEngine, prompts: list[str]) -> dict:
    ttfts = []
    for prompt in prompts:
        result = engine.generate(prompt, max_tokens=MAX_TOKENS, temperature=0.0)
        ttfts.append(result["ttft"])
    avg = sum(ttfts) / len(ttfts) if ttfts else 0.0
    return {"avg_ttft_all": avg, "avg_ttft_miss": avg, "hit_rate": 0.0,
            "hits": 0, "misses": len(prompts)}


def main() -> None:
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}\nSee SETUP.md.")
        sys.exit(1)

    print("=" * 60)
    print("KV Cache Benchmark")
    print("=" * 60)
    print(f"Model       : {MODEL_PATH}")
    print(f"Workload    : {len(WORKLOAD)} requests ({len(UNIQUE_PROMPTS)} unique × {REPEAT_FACTOR})")
    print(f"Capacity    : {CACHE_CAPACITY} entries")
    print()

    engine = InferenceEngine(MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS)

    configs = [
        ("No cache",       None),
        ("FIFO",           FIFOCache(capacity=CACHE_CAPACITY)),
        ("LRU",            LRUCache(capacity=CACHE_CAPACITY)),
        ("Sliding window", SlidingWindowCache(capacity=CACHE_CAPACITY, window_seconds=600)),
    ]

    rows = []
    baseline_ttft = None

    for name, cache in configs:
        print(f"Running: {name} …", flush=True)
        if cache is None:
            r = _run_no_cache(engine, WORKLOAD)
        else:
            r = _run_with_cache(engine, cache, WORKLOAD)

        if baseline_ttft is None:
            baseline_ttft = r["avg_ttft_all"]

        reduction_pct = (
            (1 - r["avg_ttft_all"] / baseline_ttft) * 100
            if baseline_ttft and baseline_ttft > 0 else 0.0
        )
        rows.append((
            name,
            f"{r['avg_ttft_all']:.2f}s",
            f"{r['hit_rate']*100:.1f}%",
            f"{r['hits']}/{r['hits']+r['misses']}",
            f"{reduction_pct:.1f}%",
        ))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print_table(
        rows,
        ["Policy", "Avg TTFT", "Hit rate", "Hits/Total", "TTFT Reduction"],
    )


if __name__ == "__main__":
    main()
