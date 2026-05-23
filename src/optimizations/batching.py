"""Request scheduling strategies.

NoBatchScheduler       – sequential, one request at a time
StaticBatchScheduler   – collect up to `batch_size` requests then process them
ContinuousBatchScheduler – fill inference slots as soon as they free up

All schedulers expose the same `submit(request) -> dict` interface.
"""
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Request:
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    arrival_time: float = field(default_factory=time.perf_counter)
    result: Optional[dict] = field(default=None, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    def set_result(self, result: dict) -> None:
        self.result = result
        self._done.set()

    def wait(self, timeout: float = 600.0) -> dict:
        self._done.wait(timeout=timeout)
        return self.result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# No batching
# ---------------------------------------------------------------------------
class NoBatchScheduler:
    """Process every request immediately and sequentially."""

    def __init__(self, inference_fn: Callable) -> None:
        self._infer = inference_fn

    def submit(self, request: Request) -> dict:
        result = self._infer(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        result["wait_time"] = 0.0
        request.set_result(result)
        return result


# ---------------------------------------------------------------------------
# Static batching
# ---------------------------------------------------------------------------
class StaticBatchScheduler:
    """
    Collect up to `batch_size` requests (or wait `timeout` seconds), then
    process them in insertion order.  Simulates vLLM-style static batching
    where a fixed window is used before starting execution.
    """

    def __init__(
        self,
        inference_fn: Callable,
        batch_size: int = 4,
        timeout: float = 0.5,
    ) -> None:
        self._infer = inference_fn
        self.batch_size = batch_size
        self.timeout = timeout
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, request: Request) -> dict:
        self._queue.put(request)
        return request.wait()

    def _worker(self) -> None:
        while True:
            batch: list[Request] = []
            try:
                # Block for the first item
                batch.append(self._queue.get(timeout=1.0))
            except queue.Empty:
                continue

            deadline = time.perf_counter() + self.timeout
            while len(batch) < self.batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=max(remaining, 0.0)))
                except queue.Empty:
                    break

            for req in batch:
                t_start = time.perf_counter()
                result = self._infer(
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
                result["wait_time"] = t_start - req.arrival_time
                req.set_result(result)


# ---------------------------------------------------------------------------
# Continuous batching
# ---------------------------------------------------------------------------
class ContinuousBatchScheduler:
    """
    Maintain `max_concurrent` inference slots.  As soon as one slot frees up,
    the next request from the queue is picked up without waiting for the rest
    of the batch.  This minimises head-of-line blocking and improves GPU/CPU
    utilisation.
    """

    def __init__(self, inference_fn: Callable, max_concurrent: int = 2) -> None:
        self._infer = inference_fn
        self._sem = threading.Semaphore(max_concurrent)
        self._queue: queue.Queue = queue.Queue()

    def submit(self, request: Request) -> dict:
        self._queue.put(request)
        t = threading.Thread(target=self._run_one, daemon=True)
        t.start()
        return request.wait()

    def _run_one(self) -> None:
        req = self._queue.get()
        self._sem.acquire()
        try:
            t_start = time.perf_counter()
            result = self._infer(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            result["wait_time"] = t_start - req.arrival_time
            req.set_result(result)
        finally:
            self._sem.release()
