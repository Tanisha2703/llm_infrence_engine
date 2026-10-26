"""HTTP server around the engine: an OpenAI-compatible subset with streaming, plus health and Prometheus metrics.

Threads
-------
* The **event loop** (uvicorn) handles HTTP: it parses requests, hands them to the engine, and streams results back.
* One **engine thread** runs ``engine.step()`` in a loop. All model computation happens there, never on the event loop,
  so a slow forward pass cannot block health checks, new connections or metric scrapes.
* They communicate through the engine's ``waiting`` queue (event loop -> engine thread) and per-request
  ``asyncio.Queue`` objects (engine thread -> event loop, via ``loop.call_soon_threadsafe``).

Endpoints
---------
``POST /v1/completions``, ``POST /v1/chat/completions`` (``stream: true`` gives server-sent events),
``GET /v1/models``, ``GET /health``, ``GET /metrics`` (Prometheus text format).

Not implemented: ``stop`` strings, ``n > 1``, logprobs, authentication, TLS.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request as HttpRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from miniserve.engine import Engine, Request
from miniserve.paged_engine import PagedEngine

log = logging.getLogger("miniserve")

# Latency buckets span CPU (seconds per token) and GPU (milliseconds per token) regimes.
TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
TPOT_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1)
E2E_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


# ---------------------------------------------------------------------------------------------- request schemas
class SamplingParams(BaseModel):
    max_tokens: int = Field(64, ge=1)
    temperature: float = Field(0.0, ge=0.0, le=5.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    seed: Optional[int] = None
    stream: bool = False
    ignore_eos: bool = False  # keep generating past the end-of-text token (fixed-length load tests)


class CompletionRequest(SamplingParams):
    prompt: str = Field(..., min_length=1)
    model: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(SamplingParams):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: Optional[str] = None


# ---------------------------------------------------------------------------------------------- incremental detokenizer
class StreamDecoder:
    """Turn a growing list of token ids into text pieces.

    A token is not a character: one UTF-8 character can span several tokens, and decoding a partial sequence gives
    the replacement character U+FFFD. So we decode everything so far, withhold the tail while it ends in U+FFFD,
    and emit only the new suffix once it is complete.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.emitted = ""

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""  # incomplete character: wait for more tokens
        piece, self.emitted = text[len(self.emitted):], text
        return piece

    def flush(self) -> str:
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        piece, self.emitted = text[len(self.emitted):], text
        return piece


class _Sink:
    """Engine thread -> event loop mailbox for one request."""

    def __init__(self, loop: asyncio.AbstractEventLoop, endpoint: str):
        self.loop = loop
        self.endpoint = endpoint
        self.queue: asyncio.Queue = asyncio.Queue()

    def put(self, item) -> None:
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:  # event loop already closed (shutdown): nobody is listening
            pass


# ---------------------------------------------------------------------------------------------- the server object
class Server:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        model_name: str = "miniserve",
        engine_kind: str = "continuous",
        max_slots: int = 8,
        max_len: int = 1024,
        max_seqs: int = 32,
        num_blocks: Optional[int] = None,
        block_size: int = 16,
        max_queue: int = 64,
        enable_prefix_cache: bool = True,
        cuda_graphs: bool = False,
    ):
        self.tok = tokenizer
        self.model_name = model_name
        self.max_queue = max_queue
        self.registry = CollectorRegistry()
        self._build_metrics()

        kw = dict(on_token=self._on_token, on_finish=self._on_finish)
        if cuda_graphs and engine_kind != "continuous":
            raise ValueError("cuda_graphs is supported with the continuous engine only")
        if engine_kind == "paged":
            blocks = num_blocks or max(1, max_slots * max_len // block_size)
            self.engine: Engine = PagedEngine(model, num_blocks=blocks, block_size=block_size, max_seqs=max_seqs,
                                              max_len=max_len, enable_prefix_cache=enable_prefix_cache, **kw)
        elif engine_kind in ("continuous", "static"):
            self.engine = Engine(model, max_slots=max_slots, max_len=max_len, policy=engine_kind, cuda_graphs=cuda_graphs, **kw)
        else:
            raise ValueError(f"unknown engine kind {engine_kind!r}")
        self.engine_kind = engine_kind
        self._register_gauges()

        names = [tokenizer.eos_token] + ["<|im_end|>", "<|endoftext|>"]
        ids = {tokenizer.convert_tokens_to_ids(n) for n in names if n}
        unk = getattr(tokenizer, "unk_token_id", None)
        self.eos_ids = tuple(sorted(i for i in ids if i is not None and i != unk))

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.fatal: Optional[str] = None  # set if the engine thread crashed; the server then reports unhealthy
        self._profiler = None  # set by start() when MINISERVE_PROFILE is in the environment

    # ------------------------------------------------------------------ metrics
    def _build_metrics(self) -> None:
        r = self.registry
        self.m_requests = Counter("miniserve_requests_total", "Requests by endpoint and outcome.",
                                  ["endpoint", "status"], registry=r)
        self.m_prompt_tokens = Counter("miniserve_prompt_tokens_total", "Prompt tokens of completed requests.", registry=r)
        self.m_gen_tokens = Counter("miniserve_generation_tokens_total", "Tokens generated for completed requests.", registry=r)
        self.m_prefix_hit = Counter("miniserve_prefix_cache_hit_tokens_total",
                                    "Prompt tokens served from the prefix cache (paged engine).", registry=r)
        self.m_preempt = Counter("miniserve_preemptions_total", "Requests evicted to free KV memory (paged engine).", registry=r)
        self.m_ttft = Histogram("miniserve_time_to_first_token_seconds", "Arrival to first token, queueing included.",
                                buckets=TTFT_BUCKETS, registry=r)
        self.m_tpot = Histogram("miniserve_time_per_output_token_seconds", "Mean seconds per token after the first.",
                                buckets=TPOT_BUCKETS, registry=r)
        self.m_e2e = Histogram("miniserve_e2e_request_seconds", "Arrival to last token.", buckets=E2E_BUCKETS, registry=r)
        self.m_queue = Histogram("miniserve_queue_wait_seconds", "Arrival to admission.", buckets=TTFT_BUCKETS, registry=r)
        self.m_step = Histogram("miniserve_engine_step_seconds", "Duration of one engine step.", buckets=TPOT_BUCKETS, registry=r)
        self.m_batch = Histogram("miniserve_decode_batch_size", "Rows per decode step.", buckets=BATCH_BUCKETS, registry=r)

    def _register_gauges(self) -> None:
        """Gauges are read at scrape time from the engine's own state, so they can never drift from it."""
        r, e = self.registry, self.engine
        Gauge("miniserve_engine_up", "1 while the engine thread is healthy, 0 after it crashed.", registry=r).set_function(
            lambda: 0 if self.fatal else 1)
        Gauge("miniserve_requests_waiting", "Requests queued, not yet admitted.", registry=r).set_function(lambda: len(e.waiting))
        Gauge("miniserve_requests_running", "Requests currently decoding.", registry=r).set_function(lambda: len(e.running))
        if isinstance(e, PagedEngine):
            Gauge("miniserve_kv_blocks_in_use", "KV-cache blocks held by running requests.", registry=r).set_function(
                lambda: e.allocator.num_in_use())
            Gauge("miniserve_kv_blocks_total", "KV-cache blocks in the pool.", registry=r).set_function(lambda: e.allocator.capacity)
        else:
            Gauge("miniserve_kv_slots_in_use", "KV-cache slots held by running requests.", registry=r).set_function(
                lambda: e.max_slots - len(e.free_slots))
            Gauge("miniserve_kv_slots_total", "KV-cache slots.", registry=r).set_function(lambda: e.max_slots)

    # ------------------------------------------------------------------ engine-thread callbacks
    def _on_token(self, req: Request, tok: int) -> None:
        req.sink.put(("token", tok))

    def _on_finish(self, req: Request) -> None:
        reason = req.finish_reason
        status = {"stop": "ok", "length": "ok", "abort": "aborted"}.get(reason, "error")
        self.m_requests.labels(req.sink.endpoint, status).inc()
        if status == "ok":
            self.m_prompt_tokens.inc(len(req.prompt_ids))
            self.m_gen_tokens.inc(len(req.output_ids))
            self.m_prefix_hit.inc(req.prefix_hit_tokens)
            if req.ttft is not None:
                self.m_ttft.observe(req.ttft)
            if req.tpot is not None:
                self.m_tpot.observe(req.tpot)
            if req.queue_wait is not None:
                self.m_queue.observe(req.queue_wait)
            if req.e2e is not None:
                self.m_e2e.observe(req.e2e)
        req.sink.put(("done", reason))

    def _run(self) -> None:
        eng = self.engine
        while not self._stop.is_set():
            if not eng.has_work():
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            try:
                rec = eng.step()
            except Exception:  # a bug or out-of-memory must fail the requests loudly, not hang them
                self.fatal = traceback.format_exc()
                log.error("engine thread crashed:\n%s", self.fatal)
                self._fail_all()
                return
            self.m_step.observe(rec.duration)
            if rec.decode_batch:
                self.m_batch.observe(rec.decode_batch)
            if rec.n_preempted:
                self.m_preempt.inc(rec.n_preempted)

    def _fail_all(self) -> None:
        eng = self.engine
        with eng._lock:
            victims = list(eng.waiting) + list(eng.running)
            eng.waiting.clear()
            eng.running = []
        for r in victims:
            r.state, r.finish_reason = "finished", "error"
            self._on_finish(r)

    def _run_guarded(self) -> None:
        """Run the engine loop; ANY exception, not only one raised inside ``step()``, must fail the requests loudly.

        Without this, an error anywhere else in the thread would end it silently while the server kept accepting requests that
        nothing would ever process (they would hang forever)."""
        try:
            self._run()
        except BaseException:
            if not self.fatal:
                self.fatal = traceback.format_exc()
                log.error("engine thread died:\n%s", self.fatal)
                self._fail_all()

    def start(self) -> None:
        if os.environ.get("MINISERVE_PROFILE"):  # opt-in: find out where the server's own time goes (see profiling.py)
            from miniserve.profiling import SamplingProfiler

            self._profiler = SamplingProfiler()
            self._profiler.start()
        self._thread = threading.Thread(target=self._run_guarded, name="engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler.report("/tmp/profile_threads.txt")

    # ------------------------------------------------------------------ submission
    def submit(self, prompt_ids: list[int], p: SamplingParams, endpoint: str) -> Request:
        if self.fatal:
            raise HTTPException(503, "engine has crashed; see server log")
        if len(self.engine.waiting) >= self.max_queue:
            self.m_requests.labels(endpoint, "rejected").inc()
            raise HTTPException(429, f"queue full ({self.max_queue} waiting); retry later", headers={"Retry-After": "1"})
        req = Request(
            prompt_ids=prompt_ids, max_new_tokens=p.max_tokens, eos_ids=() if p.ignore_eos else self.eos_ids,
            temperature=p.temperature, top_k=p.top_k, top_p=p.top_p, seed=p.seed,
            sink=_Sink(asyncio.get_running_loop(), endpoint),
        )
        try:
            self.engine.add_request(req)
        except ValueError as e:  # too long for the cache, empty prompt
            self.m_requests.labels(endpoint, "rejected").inc()
            raise HTTPException(400, str(e))
        self._wake.set()
        return req

    # ------------------------------------------------------------------ response bodies
    async def _events(self, req: Request, http: Optional[HttpRequest]):
        """Yield ('text', piece) then ('done', reason). Cancels the request if the client disappears."""
        dec = StreamDecoder(self.tok)
        try:
            while True:
                try:
                    kind, val = await asyncio.wait_for(req.sink.queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if http is not None and await http.is_disconnected():
                        return
                    continue
                if kind == "token":
                    piece = dec.push(val)
                    if piece:
                        yield "text", piece
                else:
                    tail = dec.flush()
                    if tail:
                        yield "text", tail
                    yield "done", val
                    return
        finally:
            if req.state != "finished":
                self.engine.cancel(req)  # client went away (or the generator was closed): free its KV memory

    def _sse(self, req: Request, http, obj: str, rid: str, created: int, chat: bool):
        async def gen():
            if chat:
                yield self._chunk(obj, rid, created, delta={"role": "assistant"}, finish=None, chat=True)
            async for kind, val in self._events(req, http):
                if kind == "text":
                    yield self._chunk(obj, rid, created, text=val, finish=None, chat=chat)
                elif val == "error":
                    yield "data: " + json.dumps({"error": {"message": "engine error"}}) + "\n\n"
                else:
                    n_out = len(req.output_ids)  # the final chunk carries the usage, so clients can count tokens exactly
                    usage = {"prompt_tokens": len(req.prompt_ids), "completion_tokens": n_out,
                             "total_tokens": len(req.prompt_ids) + n_out}
                    yield self._chunk(obj, rid, created, text="", finish=val, chat=chat, usage=usage)
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    def _chunk(self, obj, rid, created, text="", finish=None, chat=False, delta=None, usage=None) -> str:
        if chat:
            choice = {"index": 0, "delta": delta if delta is not None else ({"content": text} if text else {}),
                      "finish_reason": finish}
        else:
            choice = {"index": 0, "text": text, "finish_reason": finish}
        body = {"id": rid, "object": obj, "created": created, "model": self.model_name, "choices": [choice]}
        if usage is not None:
            body["usage"] = usage
        return "data: " + json.dumps(body) + "\n\n"

    async def _collect(self, req: Request, http, obj: str, rid: str, created: int, chat: bool):
        text, reason = [], None
        async for kind, val in self._events(req, http):
            if kind == "text":
                text.append(val)
            else:
                reason = val
        if reason is None:  # the client disconnected; nothing to send
            return JSONResponse({"error": "client disconnected"}, status_code=499)
        if reason == "error":
            raise HTTPException(500, "engine error")
        body = "".join(text)
        n_out = len(req.output_ids)
        choice = ({"index": 0, "message": {"role": "assistant", "content": body}, "finish_reason": reason} if chat
                  else {"index": 0, "text": body, "finish_reason": reason})
        return {"id": rid, "object": obj, "created": created, "model": self.model_name, "choices": [choice],
                "usage": {"prompt_tokens": len(req.prompt_ids), "completion_tokens": n_out,
                          "total_tokens": len(req.prompt_ids) + n_out}}

    async def handle(self, p: SamplingParams, prompt_ids: list[int], http: HttpRequest, chat: bool):
        endpoint = "chat" if chat else "completions"
        req = self.submit(prompt_ids, p, endpoint)
        obj = "chat.completion" + (".chunk" if p.stream else "") if chat else "text_completion"
        rid, created = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24], int(time.time())
        if p.stream:
            return self._sse(req, http, obj, rid, created, chat)
        return await self._collect(req, http, obj, rid, created, chat)


# ---------------------------------------------------------------------------------------------- FastAPI app
def create_app(model, tokenizer, **kw) -> FastAPI:
    srv = Server(model, tokenizer, **kw)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        srv.start()
        yield
        srv.stop()

    app = FastAPI(title="miniserve", lifespan=lifespan)
    app.state.server = srv

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: HttpRequest):
        return await srv.handle(body, tokenizer(body.prompt).input_ids, request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: HttpRequest):
        text = tokenizer.apply_chat_template([m.model_dump() for m in body.messages], tokenize=False,
                                             add_generation_prompt=True)
        return await srv.handle(body, tokenizer(text, add_special_tokens=False).input_ids, request, chat=True)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": srv.model_name, "object": "model"}]}

    @app.get("/health")
    async def health():
        body = {"status": "ok" if not srv.fatal else "engine_crashed", "engine": srv.engine_kind,
                "waiting": len(srv.engine.waiting), "running": len(srv.engine.running)}
        return JSONResponse(body, status_code=200 if not srv.fatal else 503)

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(srv.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    import torch
    import uvicorn

    from miniserve.loader import DEFAULT_MODEL, load_model

    ap = argparse.ArgumentParser(description="Serve a model over HTTP.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="continuous", choices=["continuous", "static", "paged"])
    ap.add_argument("--max-slots", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-seqs", type=int, default=32, help="paged engine: concurrent sequences")
    ap.add_argument("--num-blocks", type=int, default=None, help="paged engine: pool size (default: same bytes as slots)")
    ap.add_argument("--max-queue", type=int, default=64, help="requests allowed to wait before answering 429")
    ap.add_argument("--cuda-graphs", action="store_true", help="decode with CUDA graphs (continuous engine on a GPU)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)  # our own messages; third-party libraries (huggingface, httpx) stay quiet
    log.info("loading %s (%s on %s) ...", a.model, a.dtype, a.device)
    model, tok = load_model(a.model, dtype=getattr(torch, a.dtype), device=a.device)
    app = create_app(model, tok, model_name=a.model, engine_kind=a.engine, max_slots=a.max_slots, max_len=a.max_len,
                     max_seqs=a.max_seqs, num_blocks=a.num_blocks, max_queue=a.max_queue, cuda_graphs=a.cuda_graphs)
    log.info("ready: %s engine%s, listening on http://%s:%d (docs at /docs, metrics at /metrics)", a.engine,
             " with CUDA graphs" if a.cuda_graphs else "", a.host, a.port)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
