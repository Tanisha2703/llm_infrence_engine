"""Batching engine: serve many requests at once with a scheduler that decides, every step, who runs.

One engine, three behaviours selected by ``policy`` (so the benchmark compares only the scheduling idea):

* ``static``      (batch size 1 = plain sequential serving, Stage 2's behaviour)
                  Wait until nothing is running, take up to ``max_slots`` waiting requests, run them together
                  until the LAST one finishes. Finished rows keep occupying the batch ("zombies") and new
                  requests wait for the whole batch to end.
* ``continuous``  (iteration-level scheduling, as in Orca / vLLM)
                  Re-decide at EVERY step: finished requests leave immediately, waiting requests take any free
                  slot at once, and all running requests share one batched decode step.

A step is: (1) admit + prefill new requests, (2) one batched decode step for the requests that were already running.
Prefill is done one request at a time (batch of 1); batching prefills with padding is a straightforward extension.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from miniserve.cache import SlotCache
from miniserve.graph_decode import GraphDecoder, default_batch_sizes
from miniserve.model import Qwen2
from miniserve.sampler import sample


@dataclass
class Request:
    prompt_ids: list[int]
    max_new_tokens: int = 64
    eos_ids: tuple[int, ...] = ()
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: Optional[int] = None
    sink: Optional[object] = field(default=None, repr=False)  # opaque handle for the caller (the server keeps its stream here)

    # --- filled in by the engine ---
    id: int = -1
    state: str = "waiting"  # waiting -> running -> finished
    output_ids: list[int] = field(default_factory=list)
    finish_reason: Optional[str] = None  # "stop" | "length" | "abort" (cancelled by the caller)
    cancelled: bool = False  # set by Engine.cancel(); honoured at the start of the next step
    slot: Optional[int] = None
    arrival_time: float = 0.0
    admitted_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    token_times: list[float] = field(default_factory=list)
    _gen: Optional[torch.Generator] = field(default=None, repr=False)
    # --- used by the paged engine (Stage 4) ---
    blocks: list[int] = field(default_factory=list, repr=False)  # block table: logical block i -> physical block id
    cached_len: int = 0  # tokens whose K/V are in the cache
    prefix_hit_tokens: int = 0  # leading tokens served from the prefix cache at (the latest) admission
    preemptions: int = 0
    admit_seq: int = 0  # admission order, used to pick the youngest victim

    @property
    def ttft(self) -> Optional[float]:
        """Arrival -> first token. INCLUDES time spent waiting in the queue."""
        return None if self.first_token_time is None else self.first_token_time - self.arrival_time

    @property
    def queue_wait(self) -> Optional[float]:
        return None if self.admitted_time is None else self.admitted_time - self.arrival_time

    @property
    def e2e(self) -> Optional[float]:
        return None if self.finish_time is None else self.finish_time - self.arrival_time

    @property
    def tpot(self) -> Optional[float]:
        """Average time per output token after the first."""
        n = len(self.token_times)
        if n < 2:
            return None
        return (self.token_times[-1] - self.token_times[0]) / (n - 1)


@dataclass
class StepRecord:
    n_prefill: int
    decode_batch: int  # rows in the decode step (including wasted zombie rows in static mode)
    useful_rows: int  # rows that produced a token the user wanted
    duration: float
    n_preempted: int = 0  # requests evicted this step to free memory (paged engine)


class Engine:
    def __init__(
        self,
        model: Qwen2,
        max_slots: int = 8,
        max_len: int = 512,
        policy: str = "continuous",
        max_prefills_per_step: int = 1,
        on_token: Optional[Callable[[Request, int], None]] = None,
        on_finish: Optional[Callable[[Request], None]] = None,
        clock: Callable[[], float] = time.perf_counter,
        cuda_graphs: bool = False,
    ):
        if policy not in ("continuous", "static"):
            raise ValueError("policy must be 'continuous' or 'static'")
        if cuda_graphs and policy != "continuous":
            raise ValueError("cuda_graphs needs the continuous policy (static batching keeps finished rows in the batch)")
        if cuda_graphs and not next(model.parameters()).is_cuda:
            raise ValueError("cuda_graphs needs a model on a CUDA device")
        p = next(model.parameters())
        self.model = model
        self.policy = policy
        self.max_slots = max_slots
        self.max_len = max_len
        self.max_prefills_per_step = max_prefills_per_step
        self.on_token = on_token
        self.on_finish = on_finish  # called once per request, including cancelled ones, from the thread running step()
        self.clock = clock
        self.cuda_graphs = cuda_graphs
        self.cache = self._make_cache(p)
        self._graphs = None
        if cuda_graphs:
            if not isinstance(self.cache, SlotCache):
                raise ValueError("cuda_graphs is implemented for the slot cache only (the paged engine gathers by block table)")
            self._graphs = GraphDecoder(model, self.cache, default_batch_sizes(max_slots))
        self.preemptions = 0
        self.free_slots: list[int] = list(range(max_slots))
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.steps: list[StepRecord] = []
        self._next_id = 0
        self._lock = threading.Lock()  # add_request may be called from another thread (the server)

    def _make_cache(self, p: torch.Tensor):
        extra = 1 if self.cuda_graphs else 0  # the graph decoder parks padding rows in one extra slot that no request owns
        return SlotCache(self.model.cfg, self.max_slots + extra, self.max_len, p.dtype, p.device)

    # ------------------------------------------------------------------ public API
    def add_request(self, req: Request) -> Request:
        need = len(req.prompt_ids) + req.max_new_tokens
        if need > self.max_len:
            raise ValueError(f"request needs {need} cache positions but max_len is {self.max_len}")
        if not req.prompt_ids:
            raise ValueError("empty prompt")
        with self._lock:
            req.id = self._next_id
            self._next_id += 1
            req.arrival_time = self.clock()
            req.state = "waiting"
            if req.seed is not None:
                req._gen = torch.Generator(device=self.cache.device).manual_seed(req.seed)
            self.waiting.append(req)
        return req

    def has_work(self) -> bool:
        with self._lock:
            return bool(self.waiting) or bool(self.running)

    def cancel(self, req: Request) -> None:
        """Ask the engine to drop a request (the client went away). Safe from any thread: it only sets a flag,
        and the engine thread frees the request's memory at the start of its next step."""
        req.cancelled = True

    def _reap_cancelled(self) -> None:
        now = self.clock()
        with self._lock:
            dropped = [r for r in self.waiting if r.cancelled]
            for r in dropped:
                self.waiting.remove(r)
        for r in dropped:  # never admitted (or preempted): holds no memory
            self._finish(r, "abort", now)
        for r in [r for r in self.running if r.cancelled and r.state == "running"]:
            self._finish(r, "abort", now)

    @torch.inference_mode()
    def step(self) -> StepRecord:
        t0 = self.clock()
        preempted_before = self.preemptions
        self._reap_cancelled()

        # Who decodes this step: decided BEFORE admission, so a request prefilled now joins next step.
        decode_rows = list(self.running) if self.policy == "static" else [r for r in self.running if r.state == "running"]

        n_prefill = self._admit()

        useful = 0
        if decode_rows:
            useful = self._decode(decode_rows)

        if self.policy == "static" and self.running and all(r.state == "finished" for r in self.running):
            for r in self.running:  # the whole batch ended: now (and only now) the slots are released
                self._release(r)
            self.running = []

        rec = StepRecord(n_prefill, len(decode_rows), useful, self.clock() - t0, self.preemptions - preempted_before)
        self.steps.append(rec)
        return rec

    def run_until_idle(self) -> None:
        while self.has_work():
            self.step()

    # ------------------------------------------------------------------ internals
    def _admit(self) -> int:
        """Move waiting requests into free slots and prefill them. Returns how many were prefilled."""
        if self.policy == "static":
            if self.running:
                return 0  # a batch is in flight: nothing new joins until every row has finished
            limit = self.max_slots  # at a batch boundary, fill the whole batch
        else:
            limit = self.max_prefills_per_step  # continuous: a few per step, so decodes are not starved
        n = 0
        while n < limit:
            with self._lock:
                if not self.waiting or not self.free_slots:
                    break
                req = self.waiting.popleft()
                slot = self.free_slots.pop(0)
            self._prefill(req, slot)
            n += 1
        return n

    def _prefill(self, req: Request, slot: int) -> None:
        req.slot = slot
        req.admitted_time = self.clock()
        req.state = "running"
        self.cache.reset_slot(slot)
        self.running.append(req)
        ids = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.cache.device)
        logits = self.model(ids, cache=self.cache.view([slot]), last_only=True)[:, -1]
        self._emit(req, self._sample(req, logits))

    def _decode(self, rows: Sequence[Request]) -> int:
        """One batched decode step over ``rows``; returns how many rows produced a wanted token."""
        # A finished static-mode row keeps being computed (that is the waste being modelled), but never past
        # the end of its slot.
        rows = [r for r in rows if r.state != "finished" or int(self.cache.lengths[r.slot]) < self.max_len]
        last = [r.output_ids[-1] if r.output_ids else r.prompt_ids[-1] for r in rows]
        if self._graphs is not None and len(rows) <= self._graphs.max_batch:
            logits = self._graphs.run(last, [r.slot for r in rows])[:, -1]  # one CUDA-graph launch for the whole step
        else:
            ids = torch.tensor([[t] for t in last], dtype=torch.long, device=self.cache.device)
            view = self.cache.view([r.slot for r in rows])
            logits = self.model(ids, cache=view, last_only=True)[:, -1]  # [B, vocab]
        useful = 0
        for r, tok in zip(rows, self._sample_rows(rows, logits)):
            if tok is None:
                continue  # static-mode zombie row: computed, result thrown away
            self._emit(r, tok)
            useful += 1
        return useful

    def _sample(self, req: Request, logits: torch.Tensor) -> int:
        return int(sample(logits, req.temperature, req.top_k, req.top_p, req._gen).item())

    def _sample_rows(self, rows: Sequence[Request], logits: torch.Tensor) -> list[Optional[int]]:
        """Next token for every row of ``logits`` [len(rows), vocab] (None for rows that already finished).

        All greedy rows are decided with ONE argmax and ONE device-to-host transfer instead of a launch and a synchronization
        per request (``.item()`` waits for the GPU): with 16 requests that is 16 syncs saved per step. Sampled rows keep their own
        generator, so their draws are unchanged.
        """
        toks: list[Optional[int]] = [None] * len(rows)
        live = [i for i, r in enumerate(rows) if r.state != "finished"]
        greedy = [i for i in live if rows[i].temperature <= 0.0]
        if greedy:
            picked = (logits if len(greedy) == len(rows) else logits[greedy]).argmax(dim=-1).tolist()
            for i, t in zip(greedy, picked):
                toks[i] = t
        for i in live:
            if toks[i] is None:
                toks[i] = self._sample(rows[i], logits[i : i + 1])
        return toks

    def _emit(self, req: Request, tok: int) -> None:
        now = self.clock()
        if tok in req.eos_ids:
            self._finish(req, "stop", now)
            return
        req.output_ids.append(tok)
        req.token_times.append(now)
        if req.first_token_time is None:
            req.first_token_time = now
        if self.on_token:
            self.on_token(req, tok)
        if len(req.output_ids) >= req.max_new_tokens:
            self._finish(req, "length", now)

    def _finish(self, req: Request, reason: str, now: float) -> None:
        holds_memory = req.state == "running"  # a cancelled request that never started holds nothing
        req.state = "finished"
        req.finish_reason = reason
        req.finish_time = now
        if reason != "abort":
            self.finished.append(req)  # cancelled requests are not completions: they must not enter latency statistics
        if holds_memory and self.policy == "continuous":
            self.running.remove(req)
            self._release(req)  # slot is free for the very next step
        # static: the row stays in the batch until every row has finished
        if self.on_finish:
            self.on_finish(req)

    def _release(self, req: Request) -> None:
        with self._lock:
            self.cache.reset_slot(req.slot)
            self.free_slots.append(req.slot)
            self.free_slots.sort()
