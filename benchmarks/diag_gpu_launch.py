"""Is a decode step limited by memory bandwidth (as the roofline model predicts) or by Python/launch overhead?

A GPU runs kernels asynchronously: Python enqueues them and moves on. Two times per decode step tell the story:
  total   : enqueue + wait for the GPU to finish (what a request experiences)
  enqueue : time for the Python call to RETURN (the CPU's cost of issuing every kernel), GPU possibly still busy
If enqueue ~ total, the GPU is idle part of the time waiting for the CPU to issue work (launch-bound).
If enqueue << total, the GPU is the bottleneck. The ideal memory-bound time is weight bytes / bandwidth.

  python benchmarks/diag_gpu_launch.py --device cuda --dtype bfloat16 --out results/raw/gpu_launch_diag.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from miniserve.cache import SlotCache
from miniserve.loader import DEFAULT_MODEL, load_model


def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def read_bandwidth(device, mib=256) -> float:
    """GB/s achieved by summing a large tensor (a streaming read, like loading weights)."""
    x = torch.randn(mib * 2**20 // 4, device=device)
    x.sum()
    sync(device)
    best = float("inf")
    for _ in range(5):
        s = time.perf_counter()
        x.sum()
        sync(device)
        best = min(best, time.perf_counter() - s)
    return x.numel() * 4 / best / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model, _ = load_model(DEFAULT_MODEL, dtype=getattr(torch, a.dtype), device=a.device)
    dt = next(model.parameters()).dtype
    wbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    bw = read_bandwidth(a.device)
    ideal_ms = wbytes / (bw * 1e9) * 1000
    print(f"weights {wbytes / 1e9:.2f} GB | measured read bandwidth {bw:.0f} GB/s | ideal memory-bound step {ideal_ms:.2f} ms\n")
    print(f"{'batch':>5} {'total ms':>9} {'enqueue ms':>11} {'enqueue/total':>14} {'vs ideal':>9} {'tokens/s':>9} {'kernels/step':>13}")

    rows = []
    for B in a.batches:
        sc = SlotCache(model.cfg, B, a.ctx + 8, dt, a.device)
        ids = torch.randint(1000, 5000, (B, 1), device=a.device)
        tot = enq = float("inf")
        with torch.inference_mode():
            for r in range(6):  # first iterations warm up
                sc.lengths[:] = a.ctx
                sync(a.device)
                s = time.perf_counter()
                model(ids, cache=sc.view(range(B)), last_only=True)
                e = time.perf_counter() - s
                sync(a.device)
                t = time.perf_counter() - s
                if r > 1:
                    tot, enq = min(tot, t), min(enq, e)
            kernels = None
            if torch.device(a.device).type == "cuda":
                from torch.profiler import ProfilerActivity, profile
                sc.lengths[:] = a.ctx
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    model(ids, cache=sc.view(range(B)), last_only=True)
                    sync(a.device)
                kernels = sum(1 for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA)
        row = {"batch": B, "total_ms": tot * 1000, "enqueue_ms": enq * 1000, "enqueue_over_total": enq / tot,
               "vs_ideal": tot * 1000 / ideal_ms, "tokens_per_s": B / tot, "kernels_per_step": kernels}
        rows.append(row)
        print(f"{B:>5} {row['total_ms']:>9.1f} {row['enqueue_ms']:>11.1f} {row['enqueue_over_total']:>14.2f} "
              f"{row['vs_ideal']:>8.1f}x {row['tokens_per_s']:>9.0f} {kernels if kernels is not None else '-':>13}")

    if torch.device(a.device).type == "cuda":  # the same step, replayed from a CUDA graph (one launch instead of ~1,400)
        from miniserve.graph_decode import GraphDecoder

        cache_g = SlotCache(model.cfg, max(a.batches) + 1, a.ctx + 8, dt, a.device)  # + the scratch slot
        decoder = GraphDecoder(model, cache_g, a.batches)
        print(f"\nCUDA graph replay (same step, includes the host-to-device input copies):")
        print(f"{'batch':>5} {'graph ms':>9} {'eager ms':>9} {'speedup':>8} {'vs ideal':>9} {'tokens/s':>9}")
        for row in rows:
            B = row["batch"]
            best = float("inf")
            for r in range(8):
                cache_g.lengths[:B] = a.ctx
                sync(a.device)
                s = time.perf_counter()
                decoder.run([1000 + i for i in range(B)], list(range(B)))
                sync(a.device)
                t = time.perf_counter() - s
                if r > 1:
                    best = min(best, t)
            row.update({"graph_ms": best * 1000, "graph_tokens_per_s": B / best, "graph_speedup": row["total_ms"] / (best * 1000),
                        "graph_vs_ideal": best * 1000 / ideal_ms})
            print(f"{B:>5} {row['graph_ms']:>9.2f} {row['total_ms']:>9.1f} {row['graph_speedup']:>7.1f}x {row['graph_vs_ideal']:>8.1f}x {row['graph_tokens_per_s']:>9.0f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"device": a.device, "dtype": a.dtype, "ctx": a.ctx, "weight_bytes": wbytes,
                                           "read_bandwidth_gbs": bw, "ideal_ms": ideal_ms, "rows": rows,
                                           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None}, indent=2))


if __name__ == "__main__":
    main()
