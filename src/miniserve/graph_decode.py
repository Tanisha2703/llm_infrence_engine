"""CUDA-graph decoding: replace ~1,400 kernel launches per decode step with ONE graph launch.

Why: on a GPU an eager decode step of this model launches about 1,358 kernels and spends 97-98% of its time with the CPU issuing
them, while the GPU waits (benchmarks/diag_gpu_launch.py). A CUDA graph records the whole step once and replays it with a single
launch, removing that overhead.

What a graph demands: the same kernels on the same memory addresses with the same shapes, and no value read back to the CPU.
So the step is run through ``StaticSlotBatch`` (shapes depend only on batch size and ``max_len``), its inputs live in static
buffers that are overwritten before every replay, and one graph is captured per batch size. A step with ``n`` running requests
uses the smallest captured size ``B >= n``; the ``B - n`` padding rows point at a *scratch slot* that no request owns, so they
read and write only garbage that nobody ever attends to.
"""
from __future__ import annotations

from typing import Sequence

import torch

from miniserve.cache import SlotCache, StaticSlotBatch
from miniserve.model import Qwen2


def default_batch_sizes(max_batch: int) -> list[int]:
    """1, 2, 4, 8, ... up to ``max_batch`` (always including ``max_batch`` itself)."""
    sizes, b = [], 1
    while b < max_batch:
        sizes.append(b)
        b *= 2
    sizes.append(max_batch)
    return sizes


class GraphDecoder:
    def __init__(self, model: Qwen2, cache: SlotCache, batch_sizes: Sequence[int]):
        if cache.device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA device")
        self.model, self.cache = model, cache
        self.scratch = cache.num_slots - 1  # the last slot belongs to no request: padding rows park here
        self.sizes = sorted({b for b in batch_sizes if 1 <= b <= self.scratch})
        if not self.sizes:
            raise ValueError("no batch size fits: the cache needs at least one real slot plus the scratch slot")
        self.max_batch = self.sizes[-1]
        self._pool = torch.cuda.graph_pool_handle()  # all graphs share one memory pool (decode activations are small)
        self._graphs: dict[int, tuple] = {}
        for b in self.sizes:
            self._capture(b)
        self.cache.lengths[self.scratch] = 0

    def _capture(self, B: int) -> None:
        dev = self.cache.device
        ids = torch.zeros(B, 1, dtype=torch.long, device=dev)
        slots = torch.full((B,), self.scratch, dtype=torch.long, device=dev)  # warm-up and capture touch only the scratch slot

        def step():
            with torch.inference_mode():
                return self.model(ids, cache=StaticSlotBatch(self.cache, slots), last_only=True)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up (kernel selection, allocator) must happen outside the capture
            for _ in range(3):
                step()
                self.cache.lengths[self.scratch] = 0
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        self.cache.lengths[self.scratch] = 0
        with torch.cuda.graph(graph, pool=self._pool):
            logits = step()
        torch.cuda.synchronize()
        self.cache.lengths[self.scratch] = 0
        self._graphs[B] = (graph, ids, slots, logits)

    @torch.inference_mode()
    def run(self, last_tokens: Sequence[int], slot_ids: Sequence[int]) -> torch.Tensor:
        """One decode step for ``len(slot_ids)`` requests. Returns logits [n, 1, vocab] (valid until the next call)."""
        n = len(slot_ids)
        B = next(b for b in self.sizes if b >= n)
        graph, ids, slots, logits = self._graphs[B]
        dev = self.cache.device
        ids[:n, 0] = torch.tensor(list(last_tokens), dtype=torch.long, device=dev)
        slots[:n] = torch.tensor(list(slot_ids), dtype=torch.long, device=dev)
        slots[n:] = self.scratch
        self.cache.lengths[self.scratch] = 0  # padding rows advanced it last time; they must always write at position 0
        graph.replay()
        return logits[:n]
