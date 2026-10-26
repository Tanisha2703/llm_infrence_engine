"""Benchmark: speculative decoding.

  python benchmarks/bench_speculative.py                 # everything (several minutes on CPU)
  python benchmarks/bench_speculative.py --only costs    # just the cost ingredients
  python benchmarks/bench_speculative.py --target Qwen/Qwen2.5-0.5B-Instruct --draft none   # n-gram only, single model

Measures (1) the ingredients of the speedup formula (draft cost c, multi-token verification cost), (2) end-to-end decode speed
and acceptance for a small draft MODEL and for the draft-free N-GRAM drafter on open-ended and copy-heavy prompts, and (3) whether
the formula, fed with the measured ingredients, predicts the measured speedup. Correctness lives in tests/test_speculative.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from miniserve.cache import KVCache
from miniserve.generate import generate
from miniserve.loader import load_model
from miniserve.speculative import ModelDrafter, NGramDrafter, generate_speculative, predicted_speedup

PARAGRAPH = (
    "The lighthouse at Cape Alder was built in 1871 to warn ships away from the rocks. Its keeper, Maria Santos, climbed "
    "the 142 steps every evening to light the lamp. During the great storm of 1894 she kept the light burning for three "
    "nights, and twelve fishing boats found their way home because of it."
)
OPEN_PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Give me three tips for learning a new programming language.",
    "What are the main differences between a list and a tuple in Python?",
    "Describe how a refrigerator works in simple terms.",
]
COPY_PROMPTS = [
    f"Repeat the following text exactly, word for word:\n\n{PARAGRAPH}",
    f"Here is a text:\n\n{PARAGRAPH}\n\nQuote the sentence that mentions the great storm.",
    "Rewrite this function with a docstring, keeping the code the same:\n\n"
    "def add_numbers(first_number, second_number):\n    result = first_number + second_number\n    return result",
    "Copy this list in the same order:\n\napple, banana, cherry, date, elderberry, fig, grape, honeydew, kiwi, lemon",
]


def chat_ids(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt",
                                   return_dict=True)["input_ids"][0].tolist()


def best_time(fn, n=3):
    fn()
    best = float("inf")
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # CUDA launches are asynchronous: time the work, not the enqueue
        s = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - s)
    return best


def measure_costs(target, draft, ctx=64, max_t=9):
    """Time one target forward over T new tokens (cache already holding `ctx` tokens), and one draft decode step."""
    dt = next(target.parameters()).dtype
    dev = next(target.parameters()).device
    out = {"verify": {}}
    with torch.inference_mode():
        cache = KVCache(target.cfg, 1, ctx + max_t + 2, dt, dev)
        target(torch.randint(1000, 5000, (1, ctx), device=dev), cache=cache, last_only=True)
        for T in (1, 2, 3, 4, 5, 6, 8, 9):
            if T > max_t:
                continue
            ids = torch.randint(1000, 5000, (1, T), device=dev)

            def run():
                cache.length = ctx
                target(ids, cache=cache)  # all T positions' logits, as the verification needs

            out["verify"][T] = best_time(run)
        if draft is not None:
            dcache = KVCache(draft.cfg, 1, ctx + 4, dt, dev)
            draft(torch.randint(1000, 5000, (1, ctx), device=dev), cache=dcache, last_only=True)
            ids = torch.randint(1000, 5000, (1, 1), device=dev)

            def drun():
                dcache.length = ctx
                draft(ids, cache=dcache, last_only=True)

            out["draft_step"] = best_time(drun)
    return out


def run_config(target, drafter, prompts, tok, max_new, k, temperature, seeds):
    tokens = 0
    time_s = 0.0
    prop = acc = rounds = emitted = 0
    draft_s = verify_s = 0.0
    accepted_hist = []  # (k, accepted) per round, for per-position acceptance
    tested = 0
    for ids, seed in zip(prompts, seeds):
        if drafter is None:  # baseline: plain cached decoding of the target
            r = generate(target, ids, max_new, temperature=temperature, seed=seed if temperature > 0 else None)
            n = len(r.token_ids) - 1
            tokens += n
            time_s += sum(r.step_times)
        else:
            r = generate_speculative(target, drafter, ids, max_new, k=k, temperature=temperature, seed=seed if temperature > 0 else None)
            n = len(r.token_ids) - 1
            tokens += n
            time_s += r.total_time - r.ttft
            prop += r.proposed
            acc += r.accepted
            rounds += len(r.rounds)
            emitted += sum(x.emitted for x in r.rounds)
            draft_s += sum(x.draft_s for x in r.rounds)
            verify_s += sum(x.verify_s for x in r.rounds)
            accepted_hist += [(x.k, x.accepted) for x in r.rounds]
            tested += r.tested
    res = {"tokens": tokens, "decode_s": time_s, "tok_per_s": tokens / time_s}
    if drafter is not None:
        res.update({"acceptance": acc / prop if prop else 0.0, "alpha_hat": acc / tested if tested else 0.0,
                    "tokens_per_round": emitted / rounds if rounds else 0.0,
                    "rounds": rounds, "draft_ms_per_round": 1000 * draft_s / rounds, "verify_ms_per_round": 1000 * verify_s / rounds,
                    "proposed": prop})
        # per-position conditional acceptance: P(x_i accepted | x_1..x_{i-1} accepted)
        pos = []
        for i in range(1, k + 1):
            reach = sum(1 for kk, a in accepted_hist if kk >= i and a >= i - 1)
            ok = sum(1 for kk, a in accepted_hist if kk >= i and a >= i)
            pos.append(ok / reach if reach else float("nan"))
        res["position_acceptance"] = pos
        prod, e_pos = 1.0, 1.0  # E[tokens/round] = 1 + sum_i prod_{j<=i} alpha_j  (no independence assumption)
        for a_i in pos:
            prod *= 0.0 if a_i != a_i else a_i
            e_pos += prod
        res["tokens_per_round_from_positions"] = e_pos
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct", help="'none' to run only the n-gram drafter")
    ap.add_argument("--only", choices=["costs", "open", "copy"], default=None)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--sampling", action="store_true", help="also run temperature 0.7")
    ap.add_argument("--no-ngram", action="store_true", help="skip the n-gram drafter")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dt = getattr(torch, args.dtype)
    target, tok = load_model(args.target, dtype=dt, device=args.device)
    draft = None if args.draft == "none" else load_model(args.draft, dtype=dt, device=args.device)[0]
    results = {"env": {"device": args.device, "dtype": args.dtype, "threads": torch.get_num_threads(), "torch": torch.__version__,
                       "target": args.target, "draft": args.draft, "max_new": args.max_new,
                       "target_params_M": target.num_params() / 1e6, "draft_params_M": draft.num_params() / 1e6 if draft else None}}
    print(results["env"])

    # warm-up
    generate(target, chat_ids(tok, "Hi"), 4)
    costs = measure_costs(target, draft)
    results["costs"] = costs
    v1 = costs["verify"][1]
    print("\nCOST INGREDIENTS (target forward over T new tokens, cache of 64; relative to T=1)")
    print(f"{'T':>3} {'ms':>8} {'x one step':>11}")
    for T, t in costs["verify"].items():
        print(f"{T:>3} {t * 1000:>8.0f} {t / v1:>10.2f}x")
    c = None
    if draft is not None:
        c = costs["draft_step"] / v1
        print(f"one draft step {costs['draft_step'] * 1000:.0f} ms = {c:.2f} of one target step (c)")
    results["c"] = c
    if args.only == "costs":
        Path("results/raw").mkdir(parents=True, exist_ok=True)
        Path(args.out or "results/raw/speculative_costs.json").write_text(json.dumps(results, indent=2, default=float))
        return

    workloads = {"open-ended": OPEN_PROMPTS, "copy-heavy": COPY_PROMPTS}
    if args.only:
        workloads = {"open-ended": OPEN_PROMPTS} if args.only == "open" else {"copy-heavy": COPY_PROMPTS}
    temps = [0.0] + ([0.7] if args.sampling else [])
    for name, texts in workloads.items():
        prompts = [chat_ids(tok, t) for t in texts]
        seeds = list(range(len(prompts)))
        for temp in temps:
            tag = f"{name} @ T={temp}"
            results[tag] = {}
            print(f"\n{tag.upper()}: {len(prompts)} prompts, {args.max_new} new tokens each")
            base = run_config(target, None, prompts, tok, args.max_new, 0, temp, seeds)
            results[tag]["baseline"] = base
            print(f"  {'baseline (target only)':26} {base['tok_per_s']:6.2f} tok/s")
            configs = []
            if draft is not None and (temp == 0.0 or name == "open-ended"):
                configs += [(f"draft model k={k}", ModelDrafter(draft, 512), k) for k in args.ks]
            if temp == 0.0 and not args.no_ngram:
                configs += [(f"n-gram k={k}", NGramDrafter(), k) for k in (4, 8)]
            for label, drafter, k in configs:
                r = run_config(target, drafter, prompts, tok, args.max_new, k, temp, seeds)
                r["speedup"] = r["tok_per_s"] / base["tok_per_s"]
                v_k = costs["verify"].get(k + 1, costs["verify"][max(costs["verify"])]) / v1
                if label.startswith("draft"):
                    r["predicted_speedup"] = predicted_speedup(r["alpha_hat"], k, c, v_k)  # constant-alpha formula with the right alpha
                    r["predicted_speedup_positions"] = r["tokens_per_round_from_positions"] / (k * c + v_k)  # per-position alphas
                results[tag][label] = r
                pred = (f" | predicted {r['predicted_speedup']:.2f}x (per-position {r['predicted_speedup_positions']:.2f}x)"
                        if "predicted_speedup" in r else "")
                print(f"  {label:26} {r['tok_per_s']:6.2f} tok/s | speedup {r['speedup']:.2f}x{pred} | accepted/proposed {100 * r['acceptance']:3.0f}% alpha {r['alpha_hat']:.2f} "
                      f"| tokens/round {r['tokens_per_round']:.2f} | draft {r['draft_ms_per_round']:.0f} ms + verify {r['verify_ms_per_round']:.0f} ms per round")

    out = Path(args.out or f"results/raw/speculative_{args.device.split(':')[0]}_{args.dtype}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
