"""Do the scheduling paths still say the same thing on this device and dtype?

The unit tests prove batching, paging and speculation equal the single-sequence reference on CPU in fp32. Before quoting
GPU speed numbers we check the same property where the numbers come from. Reduced precision matters here: in bf16 a
batched matmul and a single-row matmul may round differently, so an argmax between two near-tied tokens can flip and the
text diverges, with no bug anywhere. So this script REPORTS agreement (identical sequences, first divergence) instead of
asserting equality; fp32 should be (nearly) exact, bf16 will show how much drift reduced precision causes.

  python benchmarks/gpu_parity.py --device cuda --dtype float32
  python benchmarks/gpu_parity.py --device cuda --dtype bfloat16 --out results/raw/gpu_parity_bf16.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from miniserve.engine import Engine, Request
from miniserve.generate import generate
from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.paged_engine import PagedEngine
from miniserve.speculative import NGramDrafter, generate_speculative

PROMPTS = [
    "The three primary colors are",
    "Machine learning models are trained on large amounts of data, and then",
    "Once upon a time in a small village by the sea there lived an old fisherman who",
    "def fibonacci(n):",
]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(name: str, got: list[list[int]], ref: list[list[int]]) -> dict:
    div = [first_divergence(g, r) for g, r in zip(got, ref)]
    total = sum(len(r) for r in ref)
    same = sum(sum(x == y for x, y in zip(g, r)) for g, r in zip(got, ref))
    return {"path": name, "identical_sequences": sum(d is None for d in div), "sequences": len(ref),
            "token_agreement": same / total, "first_divergence": div}


def run_engine(engine, ids_list, n) -> list[list[int]]:
    reqs = [engine.add_request(Request(ids, max_new_tokens=n)) for ids in ids_list]
    engine.run_until_idle()
    return [r.output_ids for r in reqs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model, tok = load_model(DEFAULT_MODEL, dtype=getattr(torch, a.dtype), device=a.device)
    ids = [tok(p).input_ids for p in PROMPTS]
    n = a.tokens
    ref = [generate(model, p, n).token_ids for p in ids]

    rows = []
    rows.append(compare("continuous batching (4 slots)", run_engine(Engine(model, max_slots=4, max_len=256), ids, n), ref))
    rows.append(compare("static batching (4 slots)", run_engine(Engine(model, max_slots=4, max_len=256, policy="static"), ids, n), ref))
    if a.device.startswith("cuda"):
        rows.append(compare("continuous batching + CUDA graphs", run_engine(Engine(model, max_slots=4, max_len=256, cuda_graphs=True), ids, n), ref))
    paged = PagedEngine(model, num_blocks=64, block_size=16, max_seqs=4, max_len=256)
    rows.append(compare("paged continuous batching", run_engine(paged, ids, n), ref))
    rows.append(compare("paged, second pass (prefix cache hits)", run_engine(paged, ids, n), ref))
    spec = [generate_speculative(model, NGramDrafter(), p, n, k=4).token_ids for p in ids]
    rows.append(compare("speculative decoding (n-gram drafter, k=4)", spec, ref))

    hit = paged.stats()["prefix_hit_rate"]
    print(f"device={a.device} dtype={a.dtype} tokens={n} prompts={len(ids)}  (paged prefix-cache hit rate {hit:.0%})\n")
    print(f"{'path':<46} {'identical':>10} {'token agreement':>16}   first divergence per prompt")
    for r in rows:
        print(f"{r['path']:<46} {r['identical_sequences']}/{r['sequences']:>8} {r['token_agreement']:>15.1%}   {r['first_divergence']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"device": a.device, "dtype": a.dtype, "tokens": n, "rows": rows,
                                           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None}, indent=2))


if __name__ == "__main__":
    main()
