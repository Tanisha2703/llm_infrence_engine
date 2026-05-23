"""Benchmark: speculative decoding vs standard token-by-token generation.

Algorithm: Chen et al. (2023) "Accelerating Large Language Model Decoding
with Speculative Sampling".

Setup here uses the same GGUF model for both draft and target
(self-speculative decoding):
  - Draft: the model generates γ=5 tokens greedily (temperature=0).
  - Target: verifies each draft token token-by-token.
  - Since both models are identical and temperature=0, acceptance rate ≈ 100 %.

The speedup comes from amortising the per-token target overhead across γ tokens
when all draft tokens are accepted.

Usage
-----
    python -m src.benchmarks.benchmark_speculative
"""
import sys
import time
from pathlib import Path
from statistics import mean

from src.server.inference import InferenceEngine
from src.optimizations.speculative import SpeculativeDecoder, BaselineDecoder
from src.benchmarks.metrics import print_table

MODEL_PATH = "models/tinyllama-1.1b-Q4_K_M.gguf"
N_THREADS = 4
MAX_TOKENS = 40
GAMMA = 5  # speculation window

PROMPTS = [
    "The speed of light is",
    "Quantum mechanics is a branch of physics that",
    "The French Revolution began in",
    "Artificial intelligence refers to",
    "The human genome contains approximately",
]


def _run_baseline(engine: InferenceEngine, prompts: list[str]) -> list[dict]:
    decoder = BaselineDecoder(engine)
    results = []
    for i, prompt in enumerate(prompts, 1):
        r = decoder.generate(prompt, max_tokens=MAX_TOKENS)
        results.append(r)
        print(f"  [{i}/{len(prompts)}] TTFT={r['ttft']:.2f}s  TPS={r['tps']:.2f}")
    return results


def _run_speculative(engine: InferenceEngine, prompts: list[str]) -> list[dict]:
    decoder = SpeculativeDecoder(target=engine, draft=None, gamma=GAMMA)
    results = []
    for i, prompt in enumerate(prompts, 1):
        decoder.reset_stats()
        r = decoder.generate(prompt, max_tokens=MAX_TOKENS)
        r["acceptance_rate"] = decoder.acceptance_rate
        results.append(r)
        print(
            f"  [{i}/{len(prompts)}] TTFT={r['ttft']:.2f}s  TPS={r['tps']:.2f}"
            f"  accept={r['acceptance_rate']*100:.0f}%"
        )
    return results


def main() -> None:
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}\nSee SETUP.md.")
        sys.exit(1)

    print("=" * 60)
    print(f"Speculative Decoding Benchmark  (γ={GAMMA})")
    print("=" * 60)
    print(f"Model      : {MODEL_PATH}")
    print(f"Max tokens : {MAX_TOKENS}")
    print(f"Prompts    : {len(PROMPTS)}")
    print()

    engine = InferenceEngine(MODEL_PATH, n_ctx=512, n_threads=N_THREADS)

    print("Baseline (token-by-token) …")
    base_results = _run_baseline(engine, PROMPTS)

    print(f"\nSpeculative decoding (γ={GAMMA}) …")
    spec_results = _run_speculative(engine, PROMPTS)

    base_ttft = mean(r["ttft"] for r in base_results)
    spec_ttft = mean(r["ttft"] for r in spec_results)
    base_tps = mean(r["tps"] for r in base_results)
    spec_tps = mean(r["tps"] for r in spec_results)
    avg_accept = mean(r["acceptance_rate"] for r in spec_results)

    speedup_ttft = base_ttft / spec_ttft if spec_ttft > 0 else 0.0
    speedup_tps = spec_tps / base_tps if base_tps > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    rows = [
        ("Baseline",           f"{base_ttft:.2f}s", f"{base_tps:.2f}", "—"),
        (f"Speculative (γ={GAMMA})", f"{spec_ttft:.2f}s", f"{spec_tps:.2f}",
         f"{avg_accept*100:.0f}%"),
    ]
    print_table(rows, ["Mode", "Avg TTFT", "Avg TPS", "Accept rate"])
    print(f"TTFT speedup : {speedup_ttft:.2f}×")
    print(f"TPS speedup  : {speedup_tps:.2f}×")
    print()

    # Per-prompt table
    print("Per-prompt detail:")
    for i, (b, s, prompt) in enumerate(zip(base_results, spec_results, PROMPTS), 1):
        sp = b["ttft"] / s["ttft"] if s["ttft"] > 0 else 0.0
        print(
            f"  [{i}] baseline TTFT {b['ttft']:.2f}s → speculative {s['ttft']:.2f}s"
            f"  ({sp:.2f}×)  prompt={prompt[:35]!r}"
        )


if __name__ == "__main__":
    main()
