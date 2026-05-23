"""Benchmark: float16 GGUF vs Q4_K_M GGUF.

Usage
-----
    python -m src.benchmarks.benchmark_quantization

Requires two model files (see SETUP.md):
  models/tinyllama-1.1b-f16.gguf    (~2.2 GB)
  models/tinyllama-1.1b-Q4_K_M.gguf (~0.7 GB)
"""
import sys
from pathlib import Path

from src.optimizations.quantization import QuantizationBenchmark
from src.benchmarks.metrics import print_table, rouge_l

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "What are the main causes of climate change?",
    "Describe how photosynthesis works.",
    "What is the difference between machine learning and deep learning?",
    "Summarise the plot of Romeo and Juliet.",
]

F16_PATH = "models/tinyllama-1.1b-f16.gguf"
Q4_PATH = "models/tinyllama-1.1b-Q4_K_M.gguf"
MAX_TOKENS = 64
N_THREADS = 4


def main() -> None:
    missing = [p for p in (F16_PATH, Q4_PATH) if not Path(p).exists()]
    if missing:
        print("ERROR: Missing model file(s):")
        for m in missing:
            print(f"  {m}")
        print("\nSee SETUP.md for download instructions.")
        sys.exit(1)

    bench = QuantizationBenchmark(
        f16_path=F16_PATH,
        q4_path=Q4_PATH,
        n_ctx=512,
        n_threads=N_THREADS,
    )

    print("=" * 60)
    print("Quantization Benchmark: F16  vs  Q4_K_M")
    print("=" * 60)
    print(f"Prompts : {len(PROMPTS)}")
    print(f"Max tokens : {MAX_TOKENS}")
    print()

    results = bench.run(prompts=PROMPTS, max_tokens=MAX_TOKENS)
    f16 = results["f16"]
    q4 = results["q4_k_m"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    rows = [
        ("F16 (baseline)", f"{f16['model_size_mb']:.0f} MB",
         f"{f16['avg_ttft']:.2f}s", f"{f16['avg_tps']:.2f}", "—"),
        ("Q4_K_M",         f"{q4['model_size_mb']:.0f} MB",
         f"{q4['avg_ttft']:.2f}s", f"{q4['avg_tps']:.2f}",
         f"{q4['rouge_l']:.3f}"),
    ]
    print_table(rows, ["Format", "Size", "Avg TTFT", "Avg TPS", "ROUGE-L"])

    speedup = results["speedup_ttft"]
    quality_drop = results["quality_drop_pct"]
    print(f"TTFT speedup   : {speedup:.2f}×")
    print(f"Quality drop   : {quality_drop:.1f}% (ROUGE-L vs F16)")
    print()

    # Per-prompt detail
    print("Per-prompt breakdown (Q4_K_M):")
    for i, (f_r, q_r, prompt) in enumerate(
        zip(f16["results"], q4["results"], PROMPTS), 1
    ):
        rl = rouge_l(q_r["text"], f_r["text"])
        print(
            f"  [{i}] TTFT {q_r['ttft']:.2f}s  TPS {q_r['tps']:.2f}"
            f"  ROUGE-L {rl:.3f}  prompt={prompt[:40]!r}"
        )


if __name__ == "__main__":
    main()
