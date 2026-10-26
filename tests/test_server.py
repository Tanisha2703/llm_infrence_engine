"""The server must not change what the model says, must survive misbehaving clients, and must report honest metrics.
Reference for correctness = the single-sequence path from Stage 2 (proven equal to the uncached model and to HF)."""
import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from miniserve.engine import Engine, Request
from miniserve.generate import generate
from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.paged_engine import PagedEngine
from miniserve.server import SamplingParams, Server, StreamDecoder, create_app

PROMPTS = [
    "The three primary colors are",
    "Machine learning models are trained on large amounts of data, and then",
    "def fibonacci(n):",
]


@pytest.fixture(scope="module")
def ours():
    return load_model(DEFAULT_MODEL, dtype=torch.float32)


@pytest.fixture(scope="module")
def client(ours):
    model, tok = ours
    app = create_app(model, tok, max_slots=4, max_len=256, max_queue=16)
    with TestClient(app) as c:
        c.srv = app.state.server
        yield c


def reference_text(ours, srv, prompt_ids, n, **kw):
    model, tok = ours
    out = generate(model, prompt_ids, n, eos_ids=srv.eos_ids, **kw)
    return tok.decode(out.token_ids, skip_special_tokens=True), out


def metric(client, name, labels=""):
    """Value of one sample from /metrics (0.0 if the series does not exist yet)."""
    body = client.get("/metrics").text
    m = re.search(rf"^{re.escape(name)}{re.escape(labels)} ([0-9.e+-]+)$", body, re.M)
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------------------------------------- correctness
def test_completion_equals_single_sequence_generation(client, ours):
    _, tok = ours
    ids = tok(PROMPTS[0]).input_ids
    want, ref = reference_text(ours, client.srv, ids, 12)
    r = client.post("/v1/completions", json={"prompt": PROMPTS[0], "max_tokens": 12})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["text"] == want
    assert body["usage"] == {"prompt_tokens": len(ids), "completion_tokens": len(ref.token_ids),
                             "total_tokens": len(ids) + len(ref.token_ids)}
    assert body["choices"][0]["finish_reason"] == ref.finish_reason == "length"


def test_streaming_concatenates_to_the_non_streaming_text(client):
    payload = {"prompt": PROMPTS[1], "max_tokens": 14}
    whole = client.post("/v1/completions", json=payload).json()
    full = whole["choices"][0]["text"]
    with client.stream("POST", "/v1/completions", json={**payload, "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert "".join(c["choices"][0]["text"] for c in chunks) == full
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"] == whole["usage"]  # the final chunk reports token counts
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    assert len(chunks) > 3  # really incremental, not one big chunk


def test_chat_endpoint_applies_the_chat_template(client, ours):
    _, tok = ours
    msgs = [{"role": "user", "content": "Say hi."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    want, _ = reference_text(ours, client.srv, tok(text, add_special_tokens=False).input_ids, 10)
    r = client.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 10})
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": want}


def test_chat_streaming_has_role_then_content_deltas(client):
    msgs = [{"role": "user", "content": "Count to three."}]
    with client.stream("POST", "/v1/chat/completions", json={"messages": msgs, "max_tokens": 8, "stream": True}) as r:
        lines = [ln for ln in r.iter_lines() if ln]
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[-1]["choices"][0]["finish_reason"] in ("length", "stop")


def test_concurrent_requests_each_get_their_own_correct_answer(client, ours):
    _, tok = ours
    want = [reference_text(ours, client.srv, tok(p).input_ids, 10)[0] for p in PROMPTS]
    with ThreadPoolExecutor(len(PROMPTS)) as ex:
        got = list(ex.map(lambda p: client.post("/v1/completions", json={"prompt": p, "max_tokens": 10}).json()
                          ["choices"][0]["text"], PROMPTS))
    assert got == want


def test_seeded_sampling_is_reproducible(client):
    payload = {"prompt": PROMPTS[0], "max_tokens": 10, "temperature": 0.9, "top_p": 0.95, "seed": 7}
    a = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]
    b = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]
    assert a == b


def test_ignore_eos_forces_the_requested_length(client):
    r = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 6, "ignore_eos": True}).json()
    assert r["usage"]["completion_tokens"] == 6


def test_paged_engine_behind_the_server_gives_the_same_text(ours):
    model, tok = ours
    app = create_app(model, tok, engine_kind="paged", max_len=256, max_seqs=4, num_blocks=64)
    with TestClient(app) as c:
        srv = app.state.server
        want, _ = reference_text(ours, srv, tok(PROMPTS[1]).input_ids, 12)
        got = c.post("/v1/completions", json={"prompt": PROMPTS[1], "max_tokens": 12}).json()["choices"][0]["text"]
        assert got == want
        assert metric(c, "miniserve_kv_blocks_total") == srv.engine.allocator.capacity  # usable blocks (one is reserved)
        assert metric(c, "miniserve_kv_blocks_in_use") == 0
    assert srv.engine.allocator.num_in_use() == 0  # every block was returned


# ---------------------------------------------------------------------------------------------- validation + errors
@pytest.mark.parametrize("payload", [
    {"prompt": "", "max_tokens": 4},
    {"prompt": "hi", "max_tokens": 0},
    {"prompt": "hi", "temperature": -1},
    {"prompt": "hi", "top_p": 0},
    {"max_tokens": 4},
])
def test_invalid_parameters_are_rejected_with_422(client, payload):
    assert client.post("/v1/completions", json=payload).status_code == 422


def test_request_longer_than_the_cache_is_a_400_not_a_crash(client):
    r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 10_000})
    assert r.status_code == 400 and "max_len" in r.json()["detail"]
    assert client.get("/health").status_code == 200  # the engine is still alive


def test_full_queue_answers_429_with_retry_after(ours):
    model, tok = ours
    srv = Server(model, tok, max_slots=1, max_len=64, max_queue=2)  # engine thread NOT started: nothing drains the queue

    async def go():
        for _ in range(2):
            srv.submit([1, 2, 3], SamplingParams(max_tokens=4), "completions")
        with pytest.raises(HTTPException) as e:
            srv.submit([1, 2, 3], SamplingParams(max_tokens=4), "completions")
        return e.value

    err = asyncio.run(go())
    assert err.status_code == 429 and err.headers["Retry-After"] == "1"
    assert srv.m_requests.labels("completions", "rejected")._value.get() == 1


def test_engine_crash_fails_requests_and_flips_health(ours, monkeypatch):
    model, tok = ours
    app = create_app(model, tok, max_slots=2, max_len=128)
    with TestClient(app) as c:
        srv = app.state.server
        assert c.get("/health").status_code == 200

        def boom():
            raise RuntimeError("simulated CUDA out of memory")

        monkeypatch.setattr(srv.engine, "step", boom)
        r = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4})
        assert r.status_code == 500  # failed loudly, did not hang
        h = c.get("/health")
        assert h.status_code == 503 and h.json()["status"] == "engine_crashed"
        assert c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4}).status_code == 503


# ---------------------------------------------------------------------------------------------- metrics
def test_metrics_reflect_what_happened(client):
    before = {
        "ok": metric(client, "miniserve_requests_total", '{endpoint="completions",status="ok"}'),
        "gen": metric(client, "miniserve_generation_tokens_total"),
        "ttft": metric(client, "miniserve_time_to_first_token_seconds_count"),
        "steps": metric(client, "miniserve_engine_step_seconds_count"),
    }
    r = client.post("/v1/completions", json={"prompt": PROMPTS[2], "max_tokens": 9, "ignore_eos": True}).json()
    assert r["usage"]["completion_tokens"] == 9
    assert metric(client, "miniserve_requests_total", '{endpoint="completions",status="ok"}') == before["ok"] + 1
    assert metric(client, "miniserve_generation_tokens_total") == before["gen"] + 9
    assert metric(client, "miniserve_time_to_first_token_seconds_count") == before["ttft"] + 1
    assert metric(client, "miniserve_engine_step_seconds_count") >= before["steps"] + 9  # 1 prefill + 8 decode steps
    text = client.get("/metrics").text
    assert "# TYPE miniserve_time_to_first_token_seconds histogram" in text
    assert metric(client, "miniserve_requests_waiting") == 0 and metric(client, "miniserve_requests_running") == 0


def test_list_models_and_health(client):
    assert client.get("/v1/models").json()["data"][0]["object"] == "model"
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["engine"] == "continuous"


# ---------------------------------------------------------------------------------------------- cancellation
def test_cancelled_running_request_frees_its_slot_and_does_not_disturb_the_others(ours):
    model, tok = ours
    ids = [tok(p).input_ids for p in PROMPTS[:2]]
    finished = []
    eng = Engine(model, max_slots=2, max_len=128, on_finish=lambda r: finished.append((r.id, r.finish_reason)))
    a = eng.add_request(Request(ids[0], max_new_tokens=12))
    b = eng.add_request(Request(ids[1], max_new_tokens=12))
    for _ in range(4):
        eng.step()
    assert a.state == b.state == "running"
    eng.cancel(a)
    eng.step()
    assert a.state == "finished" and a.finish_reason == "abort" and a.slot in eng.free_slots
    assert a not in eng.finished  # a cancelled request is not a completion
    eng.run_until_idle()
    assert b.output_ids == generate(model, ids[1], 12).token_ids  # b never noticed
    assert sorted(finished) == [(a.id, "abort"), (b.id, "length")]
    assert len(eng.free_slots) == 2


def test_cancelled_waiting_request_never_runs(ours):
    model, tok = ours
    eng = Engine(model, max_slots=1, max_len=128)
    a = eng.add_request(Request(tok(PROMPTS[0]).input_ids, max_new_tokens=6))
    b = eng.add_request(Request(tok(PROMPTS[1]).input_ids, max_new_tokens=6))  # waits: only one slot
    eng.cancel(b)
    eng.run_until_idle()
    assert b.finish_reason == "abort" and b.output_ids == [] and b.admitted_time is None
    assert a.finish_reason == "length"


def test_cancel_returns_paged_blocks(ours):
    model, tok = ours
    eng = PagedEngine(model, num_blocks=32, block_size=16, max_seqs=2, max_len=128)
    a = eng.add_request(Request(tok(PROMPTS[1]).input_ids, max_new_tokens=40))
    for _ in range(5):
        eng.step()
    assert eng.allocator.num_in_use() > 0
    eng.cancel(a)
    eng.step()
    assert a.finish_reason == "abort" and eng.allocator.num_in_use() == 0 and not eng.running


def test_a_client_that_goes_away_mid_stream_gets_cancelled(ours):
    """Closing the event generator (what Starlette does on disconnect) must cancel the request inside the engine."""
    model, tok = ours
    srv = Server(model, tok, max_slots=2, max_len=256)
    srv.start()

    async def go():
        req = srv.submit(tok(PROMPTS[0]).input_ids, SamplingParams(max_tokens=200, ignore_eos=True), "completions")
        events = srv._events(req, None)
        async for kind, _ in events:
            if kind == "text":
                break  # got the first piece, then "disconnect"
        await events.aclose()
        return req

    try:
        req = asyncio.run(go())
        deadline = time.time() + 10
        while req.state != "finished" and time.time() < deadline:
            time.sleep(0.05)
        assert req.finish_reason == "abort"
        assert len(req.output_ids) < 200  # stopped long before finishing
        assert len(srv.engine.free_slots) == 2
        assert srv.m_requests.labels("completions", "aborted")._value.get() == 1
    finally:
        srv.stop()


# ---------------------------------------------------------------------------------------------- incremental detokenizer
@pytest.mark.parametrize("text", ["Hello, world!", "héllo wörld", "日本語のテキスト", "smile 😀😀 done", "melting \U0001fae0 face", "a\n\nb  c"])
def test_stream_decoder_reassembles_text_and_never_emits_broken_characters(ours, text):
    _, tok = ours
    dec = StreamDecoder(tok)
    pieces = [dec.push(i) for i in tok(text).input_ids]
    pieces.append(dec.flush())
    assert "".join(pieces) == tok.decode(tok(text).input_ids, skip_special_tokens=True) == text
    assert not any("�" in p for p in pieces)


def test_stream_decoder_withholds_half_characters(ours):
    """A rare emoji (U+1FAE0) is three byte-level tokens: the first two must produce nothing rather than U+FFFD."""
    _, tok = ours
    emoji = "\U0001fae0"
    ids = tok(emoji).input_ids
    assert len(ids) == 3, "test premise: this tokenizer splits the character into byte tokens"
    dec = StreamDecoder(tok)
    pieces = [dec.push(i) for i in ids]
    assert pieces == ["", "", emoji]


def test_an_engine_thread_that_dies_outside_step_still_fails_requests_and_flips_health(ours, monkeypatch):
    """Not only an error inside step(): ANY exception that ends the engine thread must fail requests (previously the thread died
    silently and requests hung forever)."""
    model, tok = ours
    app = create_app(model, tok, max_slots=2, max_len=128)
    srv = app.state.server

    def boom():
        raise RuntimeError("simulated failure outside step()")

    monkeypatch.setattr(srv.engine, "has_work", boom)
    with TestClient(app) as c:
        deadline = time.time() + 10
        while not srv.fatal and time.time() < deadline:
            time.sleep(0.05)
        assert srv.fatal and "simulated failure outside step()" in srv.fatal
        assert c.get("/health").status_code == 503
        assert c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4}).status_code == 503


def test_sampling_profiler_reports_where_each_thread_spends_its_time(tmp_path):
    from miniserve.profiling import SamplingProfiler
    import threading

    def busy_function_for_profiling_test():
        end = time.time() + 1.0
        while time.time() < end:
            sum(i * i for i in range(2000))

    p = SamplingProfiler(interval=0.002)
    p.start()
    t = threading.Thread(target=busy_function_for_profiling_test, name="worker")
    t.start()
    t.join()
    p.stop()
    out = tmp_path / "profile.txt"
    p.report(str(out))
    text = out.read_text()
    assert "thread worker" in text and "busy_function_for_profiling_test" in text
