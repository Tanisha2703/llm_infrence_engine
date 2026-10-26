"""A tiny sampling profiler for a running server (enabled with the environment variable MINISERVE_PROFILE=1).

Why not cProfile: it instruments every call (large overhead that distorts a latency-sensitive server) and Python 3.12 allows only one
active profiler per process, so it cannot profile the event-loop thread and the engine thread together. This sampler instead wakes
every few milliseconds, looks at the current stack of every thread (``sys._current_frames``) and counts which functions are on it.
Overhead is small and it sees all threads at once. Two numbers per function:

* **self**: samples where the function was the innermost frame (where the thread actually was),
* **inclusive**: samples where the function was anywhere on the stack (what it and its callees cost).

A thread blocked waiting (event loop in ``select``, engine thread waiting for the GPU inside a torch call) shows up as such, which is
exactly what is needed to tell "busy in Python" from "idle/waiting".
"""
from __future__ import annotations

import os
import sys
import threading
from collections import Counter, defaultdict


class SamplingProfiler:
    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.total: Counter = Counter()  # thread name -> samples
        self.self_: dict[str, Counter] = defaultdict(Counter)
        self.incl: dict[str, Counter] = defaultdict(Counter)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sampling-profiler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        me = threading.get_ident()
        while not self._stop.wait(self.interval):
            names = {t.ident: t.name for t in threading.enumerate()}
            for tid, frame in sys._current_frames().items():
                if tid == me:
                    continue
                name = names.get(tid, str(tid))
                stack = []
                f = frame
                while f is not None:
                    stack.append(f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}")
                    f = f.f_back
                self.total[name] += 1
                self.self_[name][stack[0]] += 1
                for fn in set(stack):
                    self.incl[name][fn] += 1

    def report(self, path: str, top: int = 22) -> None:
        with open(path, "w") as out:
            out.write(f"sampling interval {self.interval * 1000:.0f} ms; percentages are of the samples taken for that thread\n")
            for name, n in self.total.most_common():
                if n < 5:
                    continue
                out.write(f"\n===== thread {name}: {n} samples =====\n--- where the thread was (self) ---\n")
                for fn, c in self.self_[name].most_common(top):
                    out.write(f"{100 * c / n:5.1f}%  {fn}\n")
                out.write("--- on the stack (inclusive) ---\n")
                for fn, c in self.incl[name].most_common(top):
                    out.write(f"{100 * c / n:5.1f}%  {fn}\n")
