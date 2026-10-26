"""Regenerate the README figures from the raw result files in results/raw/ (nothing is typed in by hand except labels).

  python benchmarks/make_figures.py          # writes results/figures/*.png

Colors follow a validated categorical palette (slots 1-3: blue, orange, aqua; all-pairs colorblind-distance and
normal-vision checks pass). Aqua is below 3:1 contrast on the light surface, so every mark carries a visible value label.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "raw"  # the data the charts are drawn from
OUT = ROOT / "results" / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, in order

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "savefig.facecolor": SURFACE,
})


def title(fig, text, sub):
    fig.text(0.015, 0.965, text, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.905, sub, fontsize=9.5, color=INK2, va="top")


def fig_launch_bound():
    d = json.load(open(RES / "gpu_launch_diag.json"))
    rows = d["rows"]
    xs = [r["batch"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.07, right=0.98, wspace=0.22)
    title(fig, "On a GPU, a decode step barely slows down as the batch grows from 1 to 32",
          f"NVIDIA A10, bf16, Qwen2.5-0.5B, eager PyTorch. {rows[0]['kernels_per_step']:,} kernel launches per step; "
          f"97-98% of the time is the CPU issuing them.")
    a.plot(xs, [r["total_ms"] for r in rows], color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
    a.axhline(d["ideal_ms"], color=INK2, lw=1.2, ls=(0, (4, 3)))
    a.text(xs[-1], d["ideal_ms"] + 0.8, f"memory-bandwidth ideal {d['ideal_ms']:.1f} ms", ha="right", color=INK2, fontsize=9)
    for r in rows:
        a.annotate(f"{r['total_ms']:.0f}", (r["batch"], r["total_ms"]), textcoords="offset points", xytext=(0, 9), ha="center", color=INK, fontsize=9)
    a.set_ylim(0, 30), a.set_xticks(xs), a.set_xlabel("batch size (requests decoded together)"), a.set_ylabel("time per decode step (ms)")
    a.set_title("Step time (flat)", loc="left", fontsize=10.5, color=INK, pad=8)
    b.plot(xs, [r["tokens_per_s"] for r in rows], color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
    for r in rows:
        b.annotate(f"{r['tokens_per_s']:,.0f}", (r["batch"], r["tokens_per_s"]), textcoords="offset points", xytext=(-4, 9), ha="center", color=INK, fontsize=9)
    b.set_ylim(0, 1600), b.set_xticks(xs), b.set_xlabel("batch size (requests decoded together)"), b.set_ylabel("tokens per second")
    b.set_title("Throughput (scales with batch)", loc="left", fontsize=10.5, color=INK, pad=8)
    fig.savefig(OUT / "gpu_launch_bound.png", dpi=200)
    plt.close(fig)


def fig_cuda_graphs():
    """Eager vs CUDA-graph decode, measured in the same container on the same GPU (results/raw/gpu_graphs_launch_diag.json)."""
    d = json.load(open(RES / "gpu_graphs_launch_diag.json"))
    rows = d["rows"]
    xs = [r["batch"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.07, right=0.98, wspace=0.22)
    title(fig, "CUDA graphs make the decode step 3-4x faster",
          "NVIDIA A10, bf16, Qwen2.5-0.5B. Same step, same container: eager PyTorch (1,358 launches) vs one replayed CUDA graph.")
    for ax, ek, gk, ylabel, ttl, fmt, top in (
        (a, "total_ms", "graph_ms", "time per decode step (ms)", "Step time (lower is better)", "{:.0f}", 30),
        (b, "tokens_per_s", "graph_tokens_per_s", "tokens per second", "Throughput (higher is better)", "{:,.0f}", 4600),
    ):
        ax.plot(xs, [r[ek] for r in rows], color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label="eager")
        ax.plot(xs, [r[gk] for r in rows], color=ORANGE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label="CUDA graph")
        for r in rows:
            low = False  # eager points sit on the x-axis here: label them to the right instead
            ax.annotate(fmt.format(r[ek]), (r["batch"], r[ek]), textcoords="offset points",
                        xytext=((14, -12) if low else (0, -16 if ek == "tokens_per_s" else 9)), ha="left" if low else "center", color=INK, fontsize=9)
            ax.annotate(f"{r[gk]:.1f}" if ek == "total_ms" else fmt.format(r[gk]), (r["batch"], r[gk]), textcoords="offset points",
                        xytext=(0, 9 if ek == "tokens_per_s" else -16), ha="center", color=INK, fontsize=9, fontweight="bold")
        ax.set_ylim(-420 if ek == "tokens_per_s" else 0, top), ax.set_xticks(xs), ax.set_xlabel("batch size (requests decoded together)"), ax.set_ylabel(ylabel)
        ax.set_title(ttl, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.legend(frameon=False, loc="center right" if ek == "total_ms" else "upper left", labelcolor=INK2)
    fig.savefig(OUT / "gpu_cuda_graphs.png", dpi=200)
    plt.close(fig)


def _serving_runs():
    """policy -> list of (throughput tok/s, TTFT p95 s) over the three GPU runs (JSON for runs 2-3, console transcript for run 1)."""
    out = {"static": [], "continuous": [], "paged": []}
    for p in out:
        for suffix in ("", "_rep2"):
            c = json.load(open(RES / f"gpu_serving_{p}_c16{suffix}.json"))["client"]
            out[p].append((c["throughput_tok_s"], c["ttft_p95_s"]))
    text = (RES / "gpu_serving_run1_A10G_console.txt").read_text()
    for p in out:
        block = text.split(f"serve_{p}_c16")[1]
        tput = float(re.search(r"([\d.]+) tok/s", block).group(1))
        ttft = float(re.search(r"TTFT\s+p50\s+[\d.]+s\s+p95\s+([\d.]+)s", block).group(1))
        out[p].append((tput, ttft))
    return out


def fig_serving():
    runs = _serving_runs()
    order = [("static", "static batching", BLUE), ("continuous", "continuous batching", ORANGE), ("paged", "paged + prefix cache", AQUA)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    title(fig, "Continuous batching: higher throughput, ~4x lower tail time-to-first-token",
          "HTTP server, 16 concurrent clients, 64 requests, NVIDIA A10 / A10G, bf16. Bar = median of 3 runs, whisker = min-max.")
    for ax, idx, ylabel, fmt, ttl in ((a, 0, "output tokens per second", "{:.0f}", "Throughput (higher is better)"),
                                       (b, 1, "seconds", "{:.2f}", "Time to first token, p95 (lower is better)")):
        for i, (key, label, color) in enumerate(order):
            vals = [r[idx] for r in runs[key]]
            med, lo, hi = statistics.median(vals), min(vals), max(vals)
            ax.bar(i, med, width=0.55, color=color, zorder=3)
            ax.vlines(i, lo, hi, color=INK, lw=1.4, zorder=4)
            ax.text(i, hi + (max(r[idx] for v in runs.values() for r in v) * 0.03), fmt.format(med), ha="center", color=INK, fontsize=10, fontweight="bold")
        ax.set_xticks(range(3)), ax.set_xticklabels([o[1] for o in order], fontsize=9.5)
        ax.set_ylabel(ylabel), ax.set_title(ttl, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12), ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "gpu_serving.png", dpi=200)
    plt.close(fig)


def fig_quantization():
    q = json.load(open(RES / "quant_ppl.json"))
    base = q["fp32 (baseline)"]["ppl"]
    pairs = [("int4, one scale per output channel", "RTN int4 per-channel", "GPTQ int4 per-channel"),
             ("int4, groups of 128", "RTN int4 g128", "GPTQ int4 g128"),
             ("int4, groups of 32", "RTN int4 g32", "GPTQ int4 g32")]
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.27, right=0.96)
    title(fig, "GPTQ recovers most of the accuracy lost to 4-bit rounding, at identical size",
          "Qwen2.5-0.5B, WikiText-2 perplexity (lower is better), 7.1-7.75x smaller layer weights. RTN = round-to-nearest. Dashed line = fp32 baseline.")
    ys = list(range(len(pairs)))[::-1]
    for y, (label, rtn_k, gptq_k) in zip(ys, pairs):
        r, g = q[rtn_k]["ppl"], q[gptq_k]["ppl"]
        ax.plot([g, r], [y, y], color=GRID, lw=5, solid_capstyle="round", zorder=2)
        ax.plot(r, y, "o", color=BLUE, ms=11, mec=SURFACE, mew=2, zorder=3)
        ax.plot(g, y, "o", color=ORANGE, ms=11, mec=SURFACE, mew=2, zorder=3)
        ax.text(r + 0.9, y, f"RTN {r:.1f}", va="center", color=INK, fontsize=10, bbox=dict(fc=SURFACE, ec="none", pad=1.5), zorder=5)
        ax.text(g - 0.9, y, f"GPTQ {g:.1f}", va="center", ha="right", color=INK, fontsize=10, fontweight="bold", bbox=dict(fc=SURFACE, ec="none", pad=1.5), zorder=5)
    ax.axvline(base, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.text(base + 0.3, len(pairs) - 0.45, f"fp32 {base:.1f}", color=INK2, fontsize=9, va="center")
    ax.set_yticks(ys), ax.set_yticklabels([p[0] for p in pairs], fontsize=10, color=INK)
    ax.set_xlim(9, 50), ax.set_ylim(-0.6, len(pairs) - 0.2), ax.set_xlabel("perplexity on WikiText-2 test (40 windows x 512 tokens)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "quantization_quality.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_launch_bound()
    fig_cuda_graphs()
    fig_serving()
    fig_quantization()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
