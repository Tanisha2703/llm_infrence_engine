"""Paged KV cache with prefix caching: the core ideas of PagedAttention (Kwon et al., SOSP 2023).

Instead of reserving ``max_len`` contiguous positions per request (SlotCache), the cache is a POOL of fixed-size
blocks (``block_size`` tokens each). A request owns a *block table*: a list of physical block ids, one per
``block_size`` logical positions, like a page table in an operating system. Blocks are allocated on demand,
so the only wasted memory per request is the unused tail of its last block.

Prefix caching: a full block of prompt tokens is identified by a *chain hash* (hash of the previous block's hash
plus this block's tokens). If another request's prompt starts with the same tokens, it reuses the same physical blocks
(reference-counted) and skips computing their K/V. Chaining matters: a block's K/V depend on every token before it,
so two blocks with identical tokens but different prefixes must NOT be shared.

``BlockAllocator``: free list, reference counts, hash -> block map, LRU eviction of unreferenced cached blocks.
``PagedCache``    : the K/V pool (one tensor pair per layer) plus its allocator.
``PagedBatch``    : a view for one forward pass over several requests; implements the same cache interface the model
                    already uses (prepare / positions / update / mask / advance), so the model code is unchanged.

What this is NOT: a custom attention kernel. ``PagedBatch.update`` gathers the blocks into a dense tensor before calling
PyTorch's attention, which costs a copy that real PagedAttention kernels avoid by reading blocks in place.
"""
from __future__ import annotations

import hashlib
import math
import struct
from collections import OrderedDict, deque
from typing import Optional, Sequence

import torch

from miniserve.cache import bytes_per_token
from miniserve.model import ModelConfig

NULL_BLOCK = 0  # reserved padding target for block tables; never allocated, always masked out


class OutOfBlocks(Exception):
    """No free or evictable block is available."""


def block_hashes(tokens: Sequence[int], block_size: int) -> list[bytes]:
    """Chain hashes of every FULL block of ``tokens``: h_i = H(h_{i-1} || tokens of block i)."""
    hashes: list[bytes] = []
    h = b"miniserve-root"
    for i in range(len(tokens) // block_size):
        blk = tokens[i * block_size : (i + 1) * block_size]
        h = hashlib.blake2b(h + struct.pack(f"<{block_size}i", *blk), digest_size=16).digest()
        hashes.append(h)
    return hashes


class BlockAllocator:
    """Bookkeeping for the block pool (no tensors here, only ids).

    States of a block:  free (ref 0, no hash)  ->  in use (ref >= 1)  ->  cached-evictable (ref 0, has a hash;
    stays reusable until memory is needed)  ->  evicted back to free.
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        if num_blocks < 2:
            raise ValueError("need at least 2 blocks (block 0 is reserved)")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.ref = [0] * num_blocks
        self._hash: list[Optional[bytes]] = [None] * num_blocks
        self._by_hash: dict[bytes, int] = {}
        self._free: deque[int] = deque(range(1, num_blocks))  # block 0 is the null block
        self._evictable: "OrderedDict[int, None]" = OrderedDict()  # oldest first
        self.evictions = 0

    # --- capacity ---------------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        return self.num_blocks - 1

    def num_available(self) -> int:
        """Blocks that can be handed out right now (free + evictable cached blocks)."""
        return len(self._free) + len(self._evictable)

    def num_in_use(self) -> int:
        return self.capacity - self.num_available()

    # --- allocation -------------------------------------------------------------------------
    def allocate(self) -> int:
        if self._free:
            b = self._free.popleft()
        elif self._evictable:
            b, _ = self._evictable.popitem(last=False)  # least recently released cached block
            del self._by_hash[self._hash[b]]
            self._hash[b] = None
            self.evictions += 1
        else:
            raise OutOfBlocks()
        self.ref[b] = 1
        return b

    def incref(self, b: int) -> None:
        if self.ref[b] == 0:
            self._evictable.pop(b)  # KeyError here would mean we shared a block that was not cached: a bug
        self.ref[b] += 1

    def decref(self, b: int) -> None:
        if self.ref[b] <= 0:
            raise ValueError(f"block {b} released more times than it was acquired")
        self.ref[b] -= 1
        if self.ref[b] == 0:
            if self._hash[b] is not None:
                self._evictable[b] = None  # keep its contents: another request may reuse this prefix
            else:
                self._free.append(b)

    # --- prefix cache -----------------------------------------------------------------------
    def lookup(self, h: bytes) -> Optional[int]:
        return self._by_hash.get(h) if self.enable_prefix_cache else None

    def register(self, b: int, h: bytes) -> None:
        """Mark a FULL block as holding the content identified by chain hash ``h``."""
        if not self.enable_prefix_cache or h in self._by_hash or self._hash[b] is not None:
            return
        self._hash[b] = h
        self._by_hash[h] = b

    def check_invariants(self) -> None:
        """Every block is exactly one of: free, evictable, or in use; counts are consistent."""
        free, evictable = set(self._free), set(self._evictable)
        assert not (free & evictable), "block both free and evictable"
        for b in range(1, self.num_blocks):
            assert self.ref[b] >= 0
            if self.ref[b] == 0:
                assert (b in free) != (b in evictable), f"idle block {b} must be in exactly one idle list"
            else:
                assert b not in free and b not in evictable, f"block {b} is in use but listed as idle"
        assert self.ref[NULL_BLOCK] == 0 and NULL_BLOCK not in free and NULL_BLOCK not in evictable
        assert all(self._by_hash[h] == b for h, b in self._by_hash.items() if self._hash[b] == h)


class PagedCache:
    """The K/V pool: per layer a tensor [num_blocks, kv_heads, block_size, head_dim] for K and one for V."""

    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        enable_prefix_cache: bool = True,
    ):
        self._cfg = cfg
        self.block_size = block_size
        self.num_blocks = num_blocks
        shape = (num_blocks, cfg.num_kv_heads, block_size, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.device = torch.device(device)
        self.allocator = BlockAllocator(num_blocks, block_size, enable_prefix_cache)

    def view(self, tables: Sequence[Sequence[int]], lengths: Sequence[int]) -> "PagedBatch":
        return PagedBatch(self, tables, lengths)

    def total_bytes(self) -> int:
        """Memory reserved for the pool (null block included)."""
        return bytes_per_token(self._cfg, self.k[0].dtype) * self.num_blocks * self.block_size

    def bytes_in_use(self) -> int:
        """Memory held by blocks that running requests currently own (block granularity)."""
        return bytes_per_token(self._cfg, self.k[0].dtype) * self.allocator.num_in_use() * self.block_size


class PagedBatch:
    """A batch of requests, each described by (block table, tokens already cached), for one forward pass."""

    def __init__(self, parent: PagedCache, tables: Sequence[Sequence[int]], lengths: Sequence[int]):
        self.parent = parent
        self.bs = parent.block_size
        self.tables = [list(t) for t in tables]
        self.lens = list(lengths)
        self.lengths = torch.tensor(self.lens, dtype=torch.long, device=parent.device)
        self.mask = None
        self.S = 0
        self.table: Optional[torch.Tensor] = None

    def prepare(self, T: int) -> None:
        for b, (tab, n) in enumerate(zip(self.tables, self.lens)):
            if len(tab) * self.bs < n + T:
                raise ValueError(f"row {b} has {len(tab)} blocks but needs {math.ceil((n + T) / self.bs)}")
        nb = math.ceil((max(self.lens) + T) / self.bs)  # blocks to gather per row (padded with the null block)
        self.S = nb * self.bs
        table = torch.full((len(self.tables), nb), NULL_BLOCK, dtype=torch.long)
        for b, tab in enumerate(self.tables):
            m = min(nb, len(tab))
            table[b, :m] = torch.tensor(tab[:m], dtype=torch.long)
        self.table = table.to(self.parent.device)
        dev = self.parent.device
        j = torch.arange(self.S, device=dev)[None, None, :]
        limit = (self.lengths[:, None] + torch.arange(T, device=dev)[None, :])[:, :, None]
        self.mask = (j <= limit)[:, None]  # [B, 1, T, S]; padding and unused block tails are masked

    def positions(self, B: int, T: int) -> torch.Tensor:
        return self.lengths[:, None] + torch.arange(T, device=self.parent.device)[None, :]

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        B, kvh, T, hd = k_new.shape
        pos = self.positions(B, T)  # [B, T] absolute positions of the new tokens
        blk = self.table.gather(1, pos // self.bs)  # physical block of each new token
        off = pos % self.bs  # offset inside that block
        self.parent.k[layer][blk, :, off] = k_new.permute(0, 2, 1, 3)
        self.parent.v[layer][blk, :, off] = v_new.permute(0, 2, 1, 3)
        # Gather every row's blocks into one dense [B, kvh, S, hd] tensor (real PagedAttention kernels read in place).
        nb = self.table.shape[1]
        K = self.parent.k[layer][self.table].permute(0, 2, 1, 3, 4).reshape(B, kvh, nb * self.bs, hd)
        V = self.parent.v[layer][self.table].permute(0, 2, 1, 3, 4).reshape(B, kvh, nb * self.bs, hd)
        return K, V

    def advance(self, T: int) -> None:
        pass  # lengths live on the requests (the engine updates them); nothing to do here
