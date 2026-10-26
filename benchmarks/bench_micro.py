"""Two small measurements quoted in docs/RESULTS.md.

1. How much prefill time `last_only` saves (skip logits for every prompt position).
2. This machine's memory read bandwidth, the ceiling for memory-bound decode.

Run: python benchmarks/bench_micro.py [--device cpu]
"""
import argparse
import time

import torch

from miniserve.loader import load_model


def best_time(fn, n=3):
    fn()  # warm-up
    best = float("inf")
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        s = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - s)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--prompt-len", type=int, default=256)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, _ = load_model(dtype=dtype, device=args.device)
    ids = torch.randint(1000, 5000, (1, args.prompt_len), device=args.device)
    with torch.inference_mode():
        full = best_time(lambda: model(ids))
        last = best_time(lambda: model(ids, last_only=True))
    logits_mb = args.prompt_len * model.cfg.vocab_size * torch.finfo(dtype).bits // 8 / 2**20
    print(
        f"prefill {args.prompt_len} tokens: full logits {full*1000:.0f} ms | last_only {last*1000:.0f} ms "
        f"| saved {100*(1-last/full):.0f}% | skipped logits tensor {logits_mb:.0f} MB"
    )

    x = torch.randn(256 * 1024 * 1024 // 4, device=args.device)  # 256 MB, larger than any cache
    t = min(best_time(lambda: x.sum(), 5) for _ in range(3))
    print(f"memory read bandwidth (sum over 256 MB): {x.numel() * 4 / t / 1e9:.1f} GB/s")


if __name__ == "__main__":
    main()
