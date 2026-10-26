"""Continuous batching on a paged KV cache with prefix caching and preemption (the vLLM-style scheduler core).

Differences from ``Engine`` (which gives every request a fixed slot reserving ``max_len`` positions):

* Memory is a pool of blocks; a request holds only the blocks its tokens need and grows by one block at a time.
* Admission is by FREE BLOCKS (plus a reserve so running requests can keep growing), not by a free slot.
* On admission the prompt is matched against the prefix cache: leading blocks already in the pool are reused and
  only the remaining tokens are prefilled.
* If a running request needs a new block and none is available, the YOUNGEST running request is preempted: its blocks are
  released (cached ones stay reusable), and it goes back to the front of the waiting queue. On resume it re-prefills
  ``prompt + tokens generated so far`` (often mostly from the prefix cache) and continues; recompute-based preemption.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from miniserve.engine import Engine, Request
from miniserve.paged import OutOfBlocks, PagedCache, block_hashes


class PagedEngine(Engine):
    def __init__(
        self,
        model,
        num_blocks: int,
        block_size: int = 16,
        max_seqs: int = 32,
        max_len: int = 512,
        enable_prefix_cache: bool = True,
        max_prefills_per_step: int = 1,
        on_token: Optional[Callable[[Request, int], None]] = None,
        **kw,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        super().__init__(model, max_slots=max_seqs, max_len=max_len, policy="continuous",
                         max_prefills_per_step=max_prefills_per_step, on_token=on_token, **kw)
        # statistics
        self.prefill_tokens_computed = 0
        self.prefill_tokens_from_cache = 0
        self.peak_running = 0
        self.peak_blocks_in_use = 0
        self._admit_counter = 0

    def _make_cache(self, p: torch.Tensor) -> PagedCache:
        return PagedCache(self.model.cfg, self.num_blocks, self.block_size, p.dtype, p.device, self.enable_prefix_cache)

    @property
    def allocator(self):
        return self.cache.allocator

    # ------------------------------------------------------------------ public API
    def add_request(self, req: Request) -> Request:
        if len(req.prompt_ids) + req.max_new_tokens > self.allocator.capacity * self.block_size:
            raise ValueError("request can never fit in the block pool")
        return super().add_request(req)

    def step(self):
        rec = super().step()
        self.peak_running = max(self.peak_running, len(self.running))
        self.peak_blocks_in_use = max(self.peak_blocks_in_use, self.allocator.num_in_use())
        return rec

    def stats(self) -> dict:
        total = self.prefill_tokens_computed + self.prefill_tokens_from_cache
        return {
            "prefill_tokens_computed": self.prefill_tokens_computed,
            "prefill_tokens_from_cache": self.prefill_tokens_from_cache,
            "prefix_hit_rate": self.prefill_tokens_from_cache / total if total else 0.0,
            "preemptions": self.preemptions,
            "evictions": self.allocator.evictions,
            "peak_running": self.peak_running,
            "peak_blocks_in_use": self.peak_blocks_in_use,
            "pool_blocks": self.allocator.capacity,
            "pool_mib": self.cache.total_bytes() / 2**20,
        }

    # ------------------------------------------------------------------ admission + prefill
    def _admit(self) -> int:
        n = 0
        while n < self.max_prefills_per_step:
            with self._lock:
                if not self.waiting or len(self.running) >= self.max_slots:
                    break
                req = self.waiting[0]
            if not self._try_prefill(req):
                break  # FCFS: if the head of the queue doesn't fit yet, wait (no skipping, so no starvation)
            with self._lock:
                self.waiting.popleft()
            n += 1
        return n

    def _try_prefill(self, req: Request) -> bool:
        """Allocate blocks for ``prompt + generated-so-far`` (reusing cached prefix blocks), prefill, emit a token."""
        bs, alloc = self.block_size, self.allocator
        tokens = req.prompt_ids + req.output_ids  # output_ids is non-empty only when resuming after preemption
        P = len(tokens)
        hashes = block_hashes(tokens, bs)

        # 1. longest cached prefix, but always leave at least one token to compute (we need logits for the last one)
        matched: list[int] = []
        for i in range((P - 1) // bs):
            b = alloc.lookup(hashes[i])
            if b is None:
                break
            matched.append(b)
        for b in matched:
            alloc.incref(b)  # pin them (and take them out of the evictable list) before counting free blocks

        # 2. do we have room for the rest, plus one spare block per running request so they can keep growing?
        need_new = math.ceil(P / bs) - len(matched)
        if alloc.num_available() < need_new + len(self.running):
            for b in matched:
                alloc.decref(b)
            return False

        req.blocks = matched + [alloc.allocate() for _ in range(need_new)]
        req.cached_len = req.prefix_hit_tokens = len(matched) * bs

        # 3. prefill only the tokens not already cached
        suffix = tokens[req.cached_len :]
        ids = torch.tensor([suffix], dtype=torch.long, device=self.cache.device)
        logits = self.model(ids, cache=self.cache.view([req.blocks], [req.cached_len]), last_only=True)[:, -1]
        req.cached_len = P
        self.prefill_tokens_computed += len(suffix)
        self.prefill_tokens_from_cache += req.cached_len - len(suffix)

        # 4. blocks that are now full and were not matched become reusable prefix blocks for later requests
        for i in range(len(matched), P // bs):
            alloc.register(req.blocks[i], hashes[i])

        if req.admitted_time is None:
            req.admitted_time = self.clock()
        req.state = "running"
        self._admit_counter += 1
        req.admit_seq = self._admit_counter
        self.running.append(req)
        self._emit(req, self._sample(req, logits))
        return True

    # ------------------------------------------------------------------ decode with on-demand growth
    def _decode(self, rows) -> int:
        alloc, bs = self.allocator, self.block_size
        ready: list[Request] = []
        for r in rows:
            if r.state != "running":
                continue  # preempted earlier in this loop
            if r.cached_len // bs >= len(r.blocks):  # the new token starts a new block
                while True:
                    try:
                        r.blocks.append(alloc.allocate())
                        break
                    except OutOfBlocks:
                        victim = max(self.running, key=lambda x: x.admit_seq)  # youngest request
                        self._preempt(victim)
                        if victim is r:
                            break
            if r.state == "running":
                ready.append(r)
        ready = [r for r in ready if r.state == "running"]  # a later row may have preempted an earlier one
        if not ready:
            return 0

        ids = torch.tensor([[r.output_ids[-1]] for r in ready], dtype=torch.long, device=self.cache.device)
        view = self.cache.view([r.blocks for r in ready], [r.cached_len for r in ready])
        logits = self.model(ids, cache=view, last_only=True)[:, -1]
        for r in ready:
            r.cached_len += 1
        useful = 0
        for r, tok in zip(ready, self._sample_rows(ready, logits)):
            self._emit(r, tok)
            useful += 1
        return useful

    def _preempt(self, victim: Request) -> None:
        self.running.remove(victim)
        self._free_blocks(victim)
        victim.state = "waiting"
        victim.cached_len = 0
        victim.preemptions += 1
        self.preemptions += 1
        with self._lock:
            self.waiting.appendleft(victim)  # resume as soon as memory allows

    # ------------------------------------------------------------------ release
    def _free_blocks(self, req: Request) -> None:
        for b in req.blocks:
            self.allocator.decref(b)
        req.blocks = []

    def _release(self, req: Request) -> None:
        self._free_blocks(req)
