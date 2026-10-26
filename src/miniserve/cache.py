"""KV caches: remember every past token's keys and values so decoding never recomputes them.

Why this is valid: the causal mask means a token's K and V depend only on tokens
before it, and RoPE rotates each token by its own absolute position, so once computed they
never change.

Two caches implement the same small interface that the model calls on every forward pass:

    cache.prepare(T)          once per forward: work out the attention mask for T new tokens
    cache.positions(B, T)     absolute positions of the new tokens, [B, T]
    cache.update(layer, k, v) write the new K/V, return the K/V to attend over
    cache.mask                "causal" | None | bool tensor (set by prepare)
    cache.advance(T)          once per forward, after all layers: grow by T tokens

``KVCache``  : one sequence (or a batch of equal-length sequences). Stage 2.
``SlotCache``: many independent sequences, each in its own slot, with its own length. Stage 3.
"""
from __future__ import annotations

from typing import Sequence

import torch

from miniserve.model import ModelConfig


def bytes_per_token(cfg: ModelConfig, dtype: torch.dtype = torch.bfloat16) -> int:
    """layers x 2 (K and V) x kv_heads x head_dim x bytes per number."""
    return cfg.num_layers * 2 * cfg.num_kv_heads * cfg.head_dim * torch.finfo(dtype).bits // 8


class KVCache:
    """Preallocated per-layer K/V buffers of shape [batch, kv_heads, max_len, head_dim].

    Preallocating (instead of ``torch.cat`` every step) avoids copying the whole cache on every
    new token, which would make decoding O(T^2) in memory traffic.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        self._cfg = cfg
        shape = (batch_size, cfg.num_kv_heads, max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.max_len = max_len
        self.device = torch.device(device)
        self.length = 0  # number of tokens currently stored (same for every layer)
        self.mask = "causal"

    # --- interface used by the model -------------------------------------------------------
    def prepare(self, T: int) -> None:
        """Choose the attention mask for T new tokens given what is already cached."""
        if self.length == 0:
            self.mask = "causal"  # fresh prompt: plain causal triangle
        elif T == 1:
            self.mask = None  # one new token may see everything cached
        else:
            # token i (absolute position length+i) may see keys 0 .. length+i
            device = self.device
            S = self.length + T
            self.mask = torch.arange(S, device=device)[None, :] <= (
                self.length + torch.arange(T, device=device)
            )[:, None]

    def positions(self, B: int, T: int) -> torch.Tensor:
        return (self.length + torch.arange(T, device=self.device)).expand(B, T)

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Write new keys/values at the current end; return views over everything stored so far."""
        t = k_new.shape[2]
        start = self.length
        if start + t > self.max_len:
            raise ValueError(f"KV cache full: {start} + {t} > max_len {self.max_len}")
        self.k[layer][:, :, start : start + t] = k_new
        self.v[layer][:, :, start : start + t] = v_new
        return self.k[layer][:, :, : start + t], self.v[layer][:, :, : start + t]

    def advance(self, T: int) -> None:
        self.length += T

    # --- helpers ---------------------------------------------------------------------------
    @staticmethod
    def bytes_per_token(cfg: ModelConfig, dtype: torch.dtype = torch.bfloat16) -> int:
        return bytes_per_token(cfg, dtype)

    def bytes_used(self) -> int:
        """Bytes of cache actually holding tokens (not the whole preallocated buffer)."""
        return bytes_per_token(self._cfg, self.k[0].dtype) * self.length


class SlotCache:
    """K/V storage for many independent sequences: buffers of [num_slots, kv_heads, max_len, head_dim].

    Each running request owns one *slot* and has its own length. A decode step over several requests
    gathers their slots into one batch (``view``); because every row can have a different length, the
    mask hides each row's unused positions. That mask is also what makes slot reuse safe: when a slot
    is given to a new request, the old occupant's stale K/V sit beyond the new length and are never attended.

    Memory note (Stage 4 fixes this): every slot reserves ``max_len`` positions whether it needs them or not.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_slots: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        self._cfg = cfg
        self.num_slots = num_slots
        self.max_len = max_len
        shape = (num_slots, cfg.num_kv_heads, max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.lengths = torch.zeros(num_slots, dtype=torch.long, device=device)
        self.device = torch.device(device)

    def view(self, slots: Sequence[int]) -> "SlotBatch":
        return SlotBatch(self, list(slots))

    def reset_slot(self, slot: int) -> None:
        self.lengths[slot] = 0

    def total_bytes(self) -> int:
        """Bytes reserved (all slots x max_len), not bytes holding real tokens."""
        return bytes_per_token(self._cfg, self.k[0].dtype) * self.num_slots * self.max_len

    def bytes_used(self) -> int:
        return bytes_per_token(self._cfg, self.k[0].dtype) * int(self.lengths.sum())


class SlotBatch:
    """A batch of slots seen as one cache for a single forward pass (same interface as KVCache)."""

    def __init__(self, parent: SlotCache, slots: list[int]):
        self.parent = parent
        self.slots = torch.tensor(slots, dtype=torch.long, device=parent.device)
        self.lengths = parent.lengths[self.slots].clone()  # [B] tokens already cached per row
        self.mask = None
        self.S = 0

    def prepare(self, T: int) -> None:
        self.S = int(self.lengths.max()) + T
        if self.S > self.parent.max_len:
            raise ValueError(f"slot cache full: need {self.S} > max_len {self.parent.max_len}")
        # row b, new token i sits at absolute position lengths[b]+i and may see keys 0 .. lengths[b]+i.
        # Everything beyond (padding up to S, or stale data of a previous occupant) is masked out.
        j = torch.arange(self.S, device=self.parent.device)[None, None, :]
        limit = (self.lengths[:, None] + torch.arange(T, device=self.parent.device)[None, :])[:, :, None]
        self.mask = (j <= limit)[:, None]  # [B, 1, T, S], broadcast over heads

    def positions(self, B: int, T: int) -> torch.Tensor:
        return self.lengths[:, None] + torch.arange(T, device=self.parent.device)[None, :]

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        B, _, T, _ = k_new.shape
        pos = self.positions(B, T)  # [B, T]
        sl = self.slots[:, None].expand(B, T)
        # buffer[slot_b, :, pos_bt, :] = k_new[b, :, t, :]  (advanced indices separated by a slice put B,T first)
        self.parent.k[layer][sl, :, pos] = k_new.permute(0, 2, 1, 3)
        self.parent.v[layer][sl, :, pos] = v_new.permute(0, 2, 1, 3)
        # Gather the rows' K/V: slice to the longest needed length first, then pick the slots (copies B x S).
        K = self.parent.k[layer][:, :, : self.S][self.slots]
        V = self.parent.v[layer][:, :, : self.S][self.slots]
        return K, V

    def advance(self, T: int) -> None:
        self.parent.lengths[self.slots] += T


class StaticSlotBatch(SlotBatch):
    """A ``SlotBatch`` whose every shape and index depends only on (batch size, max_len), never on a value on the device.

    The ordinary ``SlotBatch`` reads ``lengths.max()`` back to the CPU to slice the cache to the longest sequence, so the
    tensors it creates change shape from step to step. That is fine eagerly but impossible to record in a CUDA graph, which
    replays the same kernels on the same memory addresses. This view attends over the whole ``max_len`` window and relies on
    the mask to hide unused positions, and it takes the slot ids as a *tensor* (a static buffer the graph reads) rather than a
    Python list. Output is identical to ``SlotBatch`` up to floating-point rounding.
    """

    def __init__(self, parent: SlotCache, slots: torch.Tensor):
        self.parent = parent
        self.slots = slots  # [B] long tensor on the cache's device
        self.lengths = parent.lengths[slots]  # a device-side gather: no CPU round trip
        self.mask = None
        self.S = parent.max_len

    def prepare(self, T: int) -> None:
        dev = self.parent.device
        j = torch.arange(self.S, device=dev)[None, None, :]
        limit = (self.lengths[:, None] + torch.arange(T, device=dev)[None, :])[:, :, None]
        self.mask = (j <= limit)[:, None]  # [B, 1, T, max_len]
