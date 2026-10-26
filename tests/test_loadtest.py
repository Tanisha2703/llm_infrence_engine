"""The load tester is measurement code: if it is wrong, every serving number in the README is wrong. So it is tested
against a REAL server over a real socket (not a mock), and its arithmetic is tested on hand-made records."""
import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import loadtest  # noqa: E402

from miniserve.loader import DEFAULT_MODEL, load_model  # noqa: E402
from miniserve.server import create_app  # noqa: E402


@pytest.fixture(scope="module")
def live_server():
    model, tok = load_model(DEFAULT_MODEL, dtype=torch.float32)
    app = create_app(model, tok, max_slots=4, max_len=512, max_queue=2)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if loadtest.httpx.get(url + "/health", timeout=1).status_code == 200:
                break
        except loadtest.httpx.HTTPError:
            time.sleep(0.1)
    yield url
    server.should_exit = True
    t.join(timeout=10)


def test_workload_is_deterministic_and_varied():
    a = loadtest.make_workload(10, 0, (20, 60), [8, 16], 0)
    b = loadtest.make_workload(10, 0, (20, 60), [8, 16], 0)
    assert a == b
    assert len({len(r["prompt"].split()) for r in a}) > 3  # prompt lengths differ
    assert {r["max_tokens"] for r in a} <= {8, 16}
    assert loadtest.make_workload(10, 1, (20, 60), [8, 16], 0) != a  # the seed matters


def test_shared_prefix_is_really_shared():
    reqs = loadtest.make_workload(5, 0, (20, 40), [8], 12)
    prefix = " ".join(reqs[0]["prompt"].split()[:12])
    assert all(r["prompt"].startswith(prefix) for r in reqs)


def test_summary_arithmetic_on_hand_made_records():
    ok = lambda ttft, e2e, n: {"status": 200, "ttft": ttft, "e2e": e2e, "tokens": n, "prompt_tokens": 5, "gaps": [0.1]}
    results = [ok(1.0, 3.0, 5), ok(2.0, 4.0, 3), {"status": 429, "ttft": None, "e2e": 0.01, "tokens": 0, "prompt_tokens": 0, "gaps": []}]
    s = loadtest.summarize(results, wall=4.0)
    assert s["requests_ok"] == 2 and s["failures"] == {"429": 1}
    assert s["output_tokens"] == 8 and s["throughput_tok_s"] == pytest.approx(2.0)
    assert s["ttft_p50_s"] == pytest.approx(1.5)
    # TPOT = (e2e - ttft) / (tokens - 1): (3-1)/4 = 0.5 s and (4-2)/2 = 1.0 s -> median 0.75 s
    assert s["tpot_p50_ms"] == pytest.approx(750.0)
    assert s["e2e_p50_s"] == pytest.approx(3.5)


def test_closed_loop_against_a_real_server(live_server):
    reqs = loadtest.make_workload(6, 0, (10, 30), [6, 10], 0)
    results, wall = asyncio.run(loadtest.run_load(live_server, reqs, concurrency=3, rate=None, seed=0))
    assert len(results) == 6 and all(r["status"] == 200 for r in results)
    # ignore_eos: every request must have produced exactly what it asked for, as counted by the SERVER
    want = sorted(r["max_tokens"] for r in reqs)
    assert sorted(r["tokens"] for r in results) == want
    for r in results:
        # TTFT is the FIRST text event: everything after it (the rest of the stream) is the sum of the gaps, so
        # e2e - ttft >= sum(gaps), and there is roughly one text event per token.
        assert r["e2e"] - r["ttft"] >= sum(r["gaps"]) - 1e-3
        assert len(r["gaps"]) >= r["tokens"] // 2
    s = loadtest.summarize(results, wall)
    assert s["output_tokens"] == sum(want)
    assert 0 < s["ttft_p50_s"] < s["e2e_p50_s"] <= wall
    assert s["throughput_tok_s"] == pytest.approx(sum(want) / wall)


def test_open_loop_overload_produces_429s_and_they_are_counted_not_hidden(live_server):
    """Arrivals far faster than service, with a tiny queue (max_queue=2): the server must shed load, and the load
    tester must report it as failures rather than silently dropping those requests from the statistics."""
    reqs = loadtest.make_workload(16, 0, (10, 20), [24], 0)
    results, wall = asyncio.run(loadtest.run_load(live_server, reqs, concurrency=1, rate=200.0, seed=0))
    statuses = [r["status"] for r in results]
    assert 429 in statuses and 200 in statuses
    s = loadtest.summarize(results, wall)
    assert s["requests_sent"] == 16 and s["requests_ok"] + sum(s["failures"].values()) == 16
    assert s["failures"].get("429", 0) == statuses.count(429)


def test_connection_refused_is_reported_as_a_failure():
    results, wall = asyncio.run(loadtest.run_load("http://127.0.0.1:1", loadtest.make_workload(2, 0, (5, 6), [4], 0), 1, None, 0))
    assert all(str(r["status"]).startswith("exception:") for r in results)
    assert loadtest.summarize(results, wall)["requests_ok"] == 0
