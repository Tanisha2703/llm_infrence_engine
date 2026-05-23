"""Application-level KV cache with three eviction policies.

These caches operate at the prompt level: a SHA-256 hash of the prompt is the
key, and the full generated response is the value.  This simulates the latency
benefit of transformer KV-cache reuse when the same prefix is seen repeatedly.

Policies
--------
LRUCache          – evict least-recently-used entry
FIFOCache         – evict oldest-inserted entry
SlidingWindowCache – evict entries older than `window_seconds`
"""
import hashlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CacheEntry:
    key: str
    response: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_accessed = time.time()


class BaseCache(ABC):
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    @abstractmethod
    def get(self, prompt: str) -> Optional[str]: ...

    @abstractmethod
    def put(self, prompt: str, response: str) -> None: ...

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0


class LRUCache(BaseCache):
    """Least-recently-used eviction."""

    def __init__(self, capacity: int = 256) -> None:
        super().__init__(capacity)
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, prompt: str) -> Optional[str]:
        key = self._hash(prompt)
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key].touch()
            self.hits += 1
            return self._store[key].response
        self.misses += 1
        return None

    def put(self, prompt: str, response: str) -> None:
        key = self._hash(prompt)
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key].response = response
            self._store[key].touch()
        else:
            if len(self._store) >= self.capacity:
                self._store.popitem(last=False)
            self._store[key] = CacheEntry(key=key, response=response)

    def __len__(self) -> int:
        return len(self._store)


class FIFOCache(BaseCache):
    """First-in first-out eviction."""

    def __init__(self, capacity: int = 256) -> None:
        super().__init__(capacity)
        self._store: dict[str, CacheEntry] = {}
        self._order: deque[str] = deque()

    def get(self, prompt: str) -> Optional[str]:
        key = self._hash(prompt)
        if key in self._store:
            self._store[key].touch()
            self.hits += 1
            return self._store[key].response
        self.misses += 1
        return None

    def put(self, prompt: str, response: str) -> None:
        key = self._hash(prompt)
        if key in self._store:
            self._store[key].response = response
            self._store[key].touch()
        else:
            if len(self._store) >= self.capacity:
                oldest = self._order.popleft()
                self._store.pop(oldest, None)
            self._store[key] = CacheEntry(key=key, response=response)
            self._order.append(key)

    def __len__(self) -> int:
        return len(self._store)


class SlidingWindowCache(BaseCache):
    """Keep only entries created within the last `window_seconds`."""

    def __init__(self, capacity: int = 256, window_seconds: float = 300.0) -> None:
        super().__init__(capacity)
        self.window_seconds = window_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    def _evict_expired(self) -> None:
        cutoff = time.time() - self.window_seconds
        expired = [k for k, v in self._store.items() if v.created_at < cutoff]
        for k in expired:
            del self._store[k]

    def get(self, prompt: str) -> Optional[str]:
        self._evict_expired()
        key = self._hash(prompt)
        if key in self._store:
            self._store[key].touch()
            self.hits += 1
            return self._store[key].response
        self.misses += 1
        return None

    def put(self, prompt: str, response: str) -> None:
        self._evict_expired()
        key = self._hash(prompt)
        if key not in self._store and len(self._store) >= self.capacity:
            self._store.popitem(last=False)
        self._store[key] = CacheEntry(key=key, response=response)

    def __len__(self) -> int:
        return len(self._store)
