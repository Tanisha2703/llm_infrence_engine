"""A from-scratch implementation of the Qwen2 transformer (decoder-only LLM).

Module and parameter names deliberately mirror the Hugging Face checkpoint
(``model.layers.0.self_attn.q_proj.weight`` ...) so the released weights load
with a plain ``load_state_dict`` and no key remapping.

The forward pass works with or without a ``KVCache`` (see cache.py). Without one,
every call recomputes attention over the full sequence (the Stage 1 behaviour, kept as
the correctness reference and benchmark baseline).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:  # cache.py imports ModelConfig from here, so avoid a runtime cycle
    from miniserve.cache import KVCache


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        c = json.loads(Path(path).read_text())
        rope = c.get("rope_theta")
        if rope is None:  # newer transformers nest it under rope_parameters
            rope = c.get("rope_parameters", {}).get("rope_theta", 10000.0)
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=float(rope),
            tie_word_embeddings=c.get("tie_word_embeddings", False),
        )


class RMSNorm(nn.Module):
    """x / rms(x) * weight. Cheaper than LayerNorm: no mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # do the reduction in fp32 to avoid bf16 precision loss
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary position embedding tables for ``positions`` of shape [B, T].

    Returns cos, sin of shape [B, 1, T, head_dim] (broadcast over heads).
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    )
    freqs = positions.float()[..., None] * inv_freq  # [B, T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [B, T, head_dim]
    return emb.cos().to(dtype)[:, None], emb.sin().to(dtype)[:, None]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (even, odd)-paired feature by an angle that depends on position."""
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    """Grouped-query attention: many query heads share a few key/value heads."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * d, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * d, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * d, bias=True)
        self.o_proj = nn.Linear(cfg.num_heads * d, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: "KVCache | None" = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        c = self.cfg
        q = self.q_proj(x).view(B, T, c.num_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, c.num_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.num_kv_heads, c.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # The cache stores only the 2 KV heads (GQA), already rotated by RoPE.
        mask = "causal"  # no cache: plain causal triangle over the whole input
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)  # k, v now cover all past + new tokens
            mask = cache.mask  # chosen once per forward by cache.prepare(T)

        # GQA: each KV head serves num_heads/num_kv_heads query heads.
        rep = c.num_heads // c.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        if isinstance(mask, str):
            # Fresh prompt: query and key lengths match, plain causal triangle.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif mask is None:
            # One new token may look at everything cached: no mask needed.
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # Explicit boolean mask (True = may attend): several new tokens after cached ones, or a batch
            # of sequences with different lengths.
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, c.num_heads * c.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: "KVCache | None" = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx)  # pre-norm residual
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Backbone(nn.Module):
    """Holds ``embed_tokens``, ``layers`` and ``norm`` under the checkpoint's ``model.`` prefix."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class Qwen2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight  # share one matrix

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: "KVCache | None" = None,
        last_only: bool = False,
    ) -> torch.Tensor:
        """input_ids: [B, T] -> logits [B, T, vocab] (or [B, 1, vocab] if ``last_only``).

        With a ``cache`` (KVCache or SlotBatch), ``input_ids`` holds only the NEW tokens; their positions
        continue from what the cache already holds (per sequence) and the cache grows by T.
        ``last_only`` skips computing logits for every prompt position: only the last one
        predicts the next token.
        """
        B, T = input_ids.shape
        if cache is not None:
            cache.prepare(T)  # the cache decides the mask for this forward pass
            if positions is None:
                positions = cache.positions(B, T)
        elif positions is None:
            positions = torch.arange(T, device=input_ids.device).expand(B, T)
        x = self.model.embed_tokens(input_ids)
        cos, sin = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_theta, x.dtype)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, i)
        if cache is not None:
            cache.advance(T)  # all layers wrote at the same offset; advance once, after the loop
        if last_only:
            x = x[:, -1:]
        return self.lm_head(self.model.norm(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


