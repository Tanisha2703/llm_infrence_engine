"""Quantization comparison helpers.

Loads a float16 (F16) GGUF and a Q4_K_M GGUF of the same model, runs
identical prompts through both, and reports TTFT, TPS, and ROUGE-L quality.

Usage
-----
    from src.optimizations.quantization import QuantizationBenchmark
    bench = QuantizationBenchmark(f16_path="models/...-f16.gguf",
                                   q4_path="models/...-Q4_K_M.gguf")
    results = bench.run(prompts=[...], max_tokens=64)
"""
import os
import time
from pathlib import Path
from typing import Optional

from src.server.inference import InferenceEngine


class QuantizationBenchmark:
    """Compare a high-precision (F16) model against a Q4_K_M quantised model."""

    def __init__(
        self,
        f16_path: str,
        q4_path: str,
        n_ctx: int = 512,
        n_threads: int = 4,
    ) -> None:
        self.f16_path = f16_path
        self.q4_path = q4_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._f16: Optional[InferenceEngine] = None
        self._q4: Optional[InferenceEngine] = None

    def _load_f16(self) -> InferenceEngine:
        if self._f16 is None:
            print(f"Loading F16 model from {self.f16_path} …")
            self._f16 = InferenceEngine(
                model_path=self.f16_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
            )
        return self._f16

    def _load_q4(self) -> InferenceEngine:
        if self._q4 is None:
            print(f"Loading Q4_K_M model from {self.q4_path} …")
            self._q4 = InferenceEngine(
                model_path=self.q4_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
            )
        return self._q4

    def _model_size_mb(self, path: str) -> float:
        try:
            return os.path.getsize(path) / (1024 ** 2)
        except OSError:
            return 0.0

    def run(self, prompts: list[str], max_tokens: int = 64) -> dict:
        """
        Run both models on the same prompts and return aggregated metrics
        plus per-prompt ROUGE-L scores.
        """
        from src.benchmarks.metrics import rouge_l

        f16 = self._load_f16()
        q4 = self._load_q4()

        f16_results, q4_results = [], []

        print(f"\nRunning {len(prompts)} prompt(s) through F16 model …")
        for i, prompt in enumerate(prompts, 1):
            r = f16.generate(prompt, max_tokens=max_tokens, temperature=0.0)
            f16_results.append(r)
            print(f"  [{i}/{len(prompts)}] TTFT={r['ttft']:.2f}s  TPS={r['tps']:.2f}")

        print(f"\nRunning {len(prompts)} prompt(s) through Q4_K_M model …")
        for i, prompt in enumerate(prompts, 1):
            r = q4.generate(prompt, max_tokens=max_tokens, temperature=0.0)
            q4_results.append(r)
            print(f"  [{i}/{len(prompts)}] TTFT={r['ttft']:.2f}s  TPS={r['tps']:.2f}")

        # Average metrics
        def avg(lst, key):
            vals = [x[key] for x in lst if x[key] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        # ROUGE-L quality: Q4 vs F16 as reference
        rouge_scores = [
            rouge_l(q["text"], f["text"])
            for q, f in zip(q4_results, f16_results)
        ]
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        f16_ttft = avg(f16_results, "ttft")
        q4_ttft = avg(q4_results, "ttft")
        speedup = f16_ttft / q4_ttft if q4_ttft > 0 else 0.0

        return {
            "f16": {
                "avg_ttft": f16_ttft,
                "avg_tps": avg(f16_results, "tps"),
                "model_size_mb": self._model_size_mb(self.f16_path),
                "results": f16_results,
            },
            "q4_k_m": {
                "avg_ttft": q4_ttft,
                "avg_tps": avg(q4_results, "tps"),
                "model_size_mb": self._model_size_mb(self.q4_path),
                "rouge_l": avg_rouge,
                "results": q4_results,
            },
            "speedup_ttft": speedup,
            "quality_drop_pct": (1.0 - avg_rouge) * 100 if avg_rouge > 0 else 0.0,
        }
