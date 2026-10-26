"""Summaries of a finished run: latency percentiles, throughput, batch utilisation."""
from __future__ import annotations

import math
from typing import Sequence

from miniserve.engine import Request, StepRecord


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100); nan for an empty list."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(requests: Sequence[Request], steps: Sequence[StepRecord], wall_time: float) -> dict:
    done = [r for r in requests if r.state == "finished"]
    tokens = sum(len(r.output_ids) for r in done)
    ttft = [r.ttft for r in done if r.ttft is not None]
    e2e = [r.e2e for r in done if r.e2e is not None]
    tpot = [r.tpot for r in done if r.tpot is not None]
    wait = [r.queue_wait for r in done if r.queue_wait is not None]
    decode_steps = [s for s in steps if s.decode_batch > 0]
    rows = sum(s.decode_batch for s in decode_steps)
    useful = sum(s.useful_rows for s in decode_steps)
    return {
        "requests": len(done),
        "output_tokens": tokens,
        "wall_time_s": wall_time,
        "throughput_tok_s": tokens / wall_time if wall_time > 0 else 0.0,
        "ttft_mean_s": sum(ttft) / len(ttft) if ttft else float("nan"),
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p95_s": percentile(ttft, 95),
        "queue_wait_mean_s": sum(wait) / len(wait) if wait else float("nan"),
        "e2e_mean_s": sum(e2e) / len(e2e) if e2e else float("nan"),
        "e2e_p95_s": percentile(e2e, 95),
        "tpot_mean_ms": 1000 * sum(tpot) / len(tpot) if tpot else float("nan"),
        "steps": len(steps),
        "mean_decode_batch": rows / len(decode_steps) if decode_steps else 0.0,
        "wasted_row_fraction": 1 - useful / rows if rows else 0.0,
    }
