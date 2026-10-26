"""Load tester for the miniserve HTTP server: measures what a CLIENT sees (TTFT, per-token latency, throughput).

  python benchmarks/loadtest.py --url http://127.0.0.1:8000 --concurrency 4 --requests 24
  python benchmarks/loadtest.py --spawn "--engine paged --max-len 512" --rate 2 --requests 40 --out results/raw/serving_paged.json

Two ways to generate load (they answer different questions):
* closed loop (``--concurrency N``): N clients, each sends its next request the moment the previous one finishes.
  Measures capacity: "how much work gets done with N users hammering the server". The offered load adapts to the
  server's speed, so latencies understate what an overloaded server would do to real traffic.
* open loop (``--rate R``): requests ARRIVE at R per second (Poisson) whether or not earlier ones finished, like real
  users. Latency then grows without bound once R exceeds capacity, which is exactly what you want to see.

Every request streams, so TTFT is measured at the first received text event. Prompts vary in length; outputs have a fixed
length per request (``ignore_eos``) so runs are comparable. ``--spawn`` starts the server as a subprocess so the whole
experiment is one command; otherwise point ``--url`` at a running server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from miniserve.metrics import percentile

PASSAGE = (
    "The history of computing stretches from mechanical calculators to modern data centers. Early machines filled "
    "entire rooms and performed a few operations per second, while today's processors execute billions of operations in "
    "the time it takes to blink. Engineers learned that making programs fast depends less on raw speed and more on "
    "avoiding repeated work, moving less data, and keeping every part of the machine busy. Meanwhile libraries, "
    "compilers and operating systems hid the complexity so that most programmers never see it. "
).split()
SHARED = (
    "You are a careful assistant for a company that serves language models. Answer briefly and precisely, cite the "
    "numbers you were given, and say so when you are unsure. "
).split()


def make_workload(n: int, seed: int, prompt_words: tuple[int, int], out_tokens: list[int], shared_words: int):
    rng = random.Random(seed)
    prefix = (SHARED * 10)[:shared_words]
    reqs = []
    for i in range(n):
        k = rng.randint(*prompt_words)
        start = rng.randrange(len(PASSAGE))
        body = [PASSAGE[(start + j) % len(PASSAGE)] for j in range(k)]
        reqs.append({"prompt": " ".join(prefix + body) + f" (request {i})", "max_tokens": rng.choice(out_tokens)})
    return reqs


async def one_request(client: httpx.AsyncClient, url: str, spec: dict) -> dict:
    payload = {**spec, "stream": True, "ignore_eos": True, "temperature": 0.0}
    t0 = time.perf_counter()
    rec = {"status": None, "ttft": None, "e2e": None, "tokens": 0, "prompt_tokens": 0, "gaps": []}
    try:
        async with client.stream("POST", url + "/v1/completions", json=payload) as r:
            rec["status"] = r.status_code
            if r.status_code != 200:
                await r.aread()
                rec["e2e"] = time.perf_counter() - t0
                return rec
            last = None
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                now = time.perf_counter()
                ev = json.loads(line[6:])
                if "choices" not in ev:
                    rec["status"] = "stream_error"
                    break
                if ev["choices"][0]["text"]:
                    if rec["ttft"] is None:
                        rec["ttft"] = now - t0
                    elif last is not None:
                        rec["gaps"].append(now - last)
                    last = now
                if "usage" in ev:
                    rec["tokens"] = ev["usage"]["completion_tokens"]
                    rec["prompt_tokens"] = ev["usage"]["prompt_tokens"]
            rec["e2e"] = time.perf_counter() - t0
    except Exception as e:  # connection refused, reset, timeout: count it, never hide it
        rec["status"] = f"exception:{type(e).__name__}"
        rec["e2e"] = time.perf_counter() - t0
    return rec


async def run_load(url: str, reqs: list[dict], concurrency: int, rate: float | None, seed: int) -> tuple[list[dict], float]:
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=limits) as client:
        t0 = time.perf_counter()
        if rate:  # open loop: Poisson arrivals, independent of completions
            rng = random.Random(seed + 1)
            tasks = []
            for spec in reqs:
                tasks.append(asyncio.create_task(one_request(client, url, spec)))
                await asyncio.sleep(rng.expovariate(rate))
            results = await asyncio.gather(*tasks)
        else:  # closed loop: `concurrency` workers pull from a shared list
            it = iter(reqs)
            results: list[dict] = []

            async def worker():
                for spec in it:  # the iterator is shared; each request goes to exactly one worker
                    results.append(await one_request(client, url, spec))

            await asyncio.gather(*[worker() for _ in range(concurrency)])
        return list(results), time.perf_counter() - t0


def summarize(results: list[dict], wall: float) -> dict:
    ok = [r for r in results if r["status"] == 200 and r["ttft"] is not None]
    bad: dict[str, int] = {}
    for r in results:
        if r not in ok:
            bad[str(r["status"])] = bad.get(str(r["status"]), 0) + 1
    ttft = [r["ttft"] for r in ok]
    e2e = [r["e2e"] for r in ok]
    tpot = [(r["e2e"] - r["ttft"]) / (r["tokens"] - 1) for r in ok if r["tokens"] > 1]
    gaps = [g for r in ok for g in r["gaps"]]
    tokens = sum(r["tokens"] for r in ok)

    def pct(v, p):
        return percentile(v, p) if v else float("nan")

    return {
        "requests_sent": len(results), "requests_ok": len(ok), "failures": bad,
        "output_tokens": tokens, "wall_time_s": wall,
        "throughput_tok_s": tokens / wall if wall else 0.0, "request_rate_s": len(ok) / wall if wall else 0.0,
        "ttft_p50_s": pct(ttft, 50), "ttft_p95_s": pct(ttft, 95), "ttft_p99_s": pct(ttft, 99),
        "tpot_p50_ms": 1000 * pct(tpot, 50), "tpot_p95_ms": 1000 * pct(tpot, 95), "tpot_p99_ms": 1000 * pct(tpot, 99),
        "e2e_p50_s": pct(e2e, 50), "e2e_p95_s": pct(e2e, 95), "e2e_p99_s": pct(e2e, 99),
        "inter_chunk_p99_ms": 1000 * pct(gaps, 99),
    }


def scrape(url: str) -> dict[str, float]:
    """The server's own counters (for the cross-check between client-side and server-side numbers)."""
    text = httpx.get(url + "/metrics", timeout=10).text
    out = {}
    for name in ("miniserve_preemptions_total", "miniserve_prefix_cache_hit_tokens_total", "miniserve_prompt_tokens_total",
                 "miniserve_generation_tokens_total", "miniserve_decode_batch_size_sum", "miniserve_decode_batch_size_count"):
        m = re.search(rf"^{name} ([0-9.e+-]+)$", text, re.M)
        out[name] = float(m.group(1)) if m else 0.0
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_server(extra: str, timeout_s: float = 600.0):
    port = free_port()
    log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    proc = subprocess.Popen([sys.executable, "-m", "miniserve.server", *shlex.split(extra), "--port", str(port)],
                            stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early; log: {log.name}")
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                return proc, url
        except httpx.HTTPError:
            time.sleep(1)
    proc.kill()
    raise RuntimeError(f"server not ready after {timeout_s}s; log: {log.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="server base URL (omit with --spawn)")
    ap.add_argument("--spawn", default=None, help="start `python -m miniserve.server <these args>` and test it")
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=4, help="closed loop clients")
    ap.add_argument("--rate", type=float, default=None, help="open loop: Poisson arrivals per second (overrides --concurrency)")
    ap.add_argument("--warmup", type=int, default=2, help="requests sent first and discarded")
    ap.add_argument("--prompt-words", type=int, nargs=2, default=[30, 90], metavar=("MIN", "MAX"))
    ap.add_argument("--out-tokens", type=int, nargs="+", default=[16, 32, 48], help="output lengths to sample from")
    ap.add_argument("--shared-prefix-words", type=int, default=0, help="common leading words (exercises the prefix cache)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.url and not a.spawn:
        ap.error("give --url or --spawn")

    proc = None
    try:
        if a.spawn is not None:
            proc, url = spawn_server(a.spawn)
        else:
            url = a.url.rstrip("/")
        reqs = make_workload(a.requests + a.warmup, a.seed, tuple(a.prompt_words), a.out_tokens, a.shared_prefix_words)
        if a.warmup:
            asyncio.run(run_load(url, reqs[: a.warmup], 1, None, a.seed))
        before = scrape(url)
        results, wall = asyncio.run(run_load(url, reqs[a.warmup :], a.concurrency, a.rate, a.seed))
        after = scrape(url)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    summary = summarize(results, wall)
    server = {k: after[k] - before[k] for k in after}
    ds = server["miniserve_decode_batch_size_count"]
    server["mean_decode_batch"] = server.pop("miniserve_decode_batch_size_sum") / ds if ds else 0.0
    server.pop("miniserve_decode_batch_size_count")
    prompt_tok = server["miniserve_prompt_tokens_total"]
    server["prefix_hit_rate"] = server["miniserve_prefix_cache_hit_tokens_total"] / prompt_tok if prompt_tok else 0.0

    print(f"\n{summary['requests_ok']}/{summary['requests_sent']} ok  failures={summary['failures'] or 'none'}  "
          f"wall={summary['wall_time_s']:.1f}s  {summary['throughput_tok_s']:.1f} tok/s  {summary['request_rate_s']:.2f} req/s")
    print(f"TTFT  p50 {summary['ttft_p50_s']:.3f}s  p95 {summary['ttft_p95_s']:.3f}s  p99 {summary['ttft_p99_s']:.3f}s")
    print(f"TPOT  p50 {summary['tpot_p50_ms']:.1f}ms  p95 {summary['tpot_p95_ms']:.1f}ms  p99 {summary['tpot_p99_ms']:.1f}ms")
    print(f"E2E   p50 {summary['e2e_p50_s']:.2f}s  p95 {summary['e2e_p95_s']:.2f}s  p99 {summary['e2e_p99_s']:.2f}s")
    print(f"server-side: mean decode batch {server['mean_decode_batch']:.2f}, preemptions {server['miniserve_preemptions_total']:.0f}, "
          f"prefix hit rate {server['prefix_hit_rate']:.1%}")

    if a.out:
        meta = {"args": vars(a), "python": sys.version.split()[0], "machine": platform.processor() or platform.machine(),
                "mode": "open_loop" if a.rate else "closed_loop"}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"meta": meta, "client": summary, "server": server}, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
