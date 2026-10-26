"""Quantization from scratch: weights, activations, and the KV cache.

Why quantize: decode is memory-bound, so fewer bytes per weight means less data to stream per token and a smaller
model. The price is rounding error; the art is making that error invisible.

What is implemented here
------------------------
* ``QuantSpec``/``quantize_weight``/``dequantize_weight``: round-to-nearest (RTN) weight quantization with per-tensor, per-channel or
  group-wise scales, symmetric or asymmetric, any bit width (4-bit weights are really packed two per byte).
* ``QuantLinear``: a drop-in ``nn.Linear`` that stores the compact integer weights and dequantizes on the fly (a *reference* implementation;
  there is no fused kernel, so quantization gives no speedup here; see docs/RESULTS.md).
* ``gptq_quantize`` / ``quantize_model_gptq``: GPTQ (Frantar et al., 2022): quantize a layer's weights one column at a time and push each
  column's rounding error onto the not-yet-quantized columns, using second-order information from calibration data.
* ``ActQuantLinear`` / ``smooth_model``: simulated 8-bit activation quantization (W8A8) and SmoothQuant (Xiao et al., 2022), which moves activation
  outliers into the weights to make W8A8 work.
* ``Int8KVCache``: a KV cache stored as int8 with a per-token, per-head scale.

Everything here is exact arithmetic on stored integers + scales, so quality numbers are real; only speed is limited by the absence of fused kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from miniserve.cache import KVCache
from miniserve.model import ModelConfig, Qwen2, rope_cos_sin

# ---------------------------------------------------------------------------------------------------- the format


@dataclass(frozen=True)
class QuantSpec:
    """How a weight matrix [out_features, in_features] is quantized.

    group_size:  0  -> one scale per output channel (row);  -1 -> one scale for the whole tensor;  >0 -> one scale per ``group_size`` input features
    symmetric:   True -> codes in [-qmax, qmax], scale = absmax/qmax (no zero point);  False -> unsigned codes in [0, 2^bits-1] with a zero point
    """

    bits: int = 8
    group_size: int = 0
    symmetric: bool = True

    def __post_init__(self):
        if not 2 <= self.bits <= 8:
            raise ValueError("bits must be in 2..8")

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1 if self.symmetric else 2**self.bits - 1

    @property
    def name(self) -> str:
        g = "tensor" if self.group_size < 0 else "channel" if self.group_size == 0 else f"g{self.group_size}"
        return f"int{self.bits}-{g}-{'sym' if self.symmetric else 'asym'}"


def _group_view(t: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """View a [out, in] matrix as [G0, G1, group_len] so that scales reduce over the last dimension."""
    out, inn = t.shape
    if spec.group_size < 0:
        return t.reshape(1, 1, -1)
    if spec.group_size == 0:
        return t.reshape(out, 1, inn)
    if inn % spec.group_size:
        raise ValueError(f"in_features {inn} is not divisible by group_size {spec.group_size}")
    return t.reshape(out, inn // spec.group_size, spec.group_size)


def scale_zero(wg: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Scale (and zero point) for each group of ``wg`` (reduced over the last dim). Scales are rounded to fp16 because that is how they are
    stored, so the numbers used during quantization equal the numbers used at inference."""
    if spec.symmetric:
        scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-5) / spec.qmax
        zero = None
    else:
        # the representable range must contain 0 so that the zero point lies in [0, qmax] (otherwise it gets clamped and the range shifts)
        wmin, wmax = wg.amin(-1, keepdim=True).clamp(max=0), wg.amax(-1, keepdim=True).clamp(min=0)
        scale = ((wmax - wmin) / spec.qmax).clamp(min=1e-5)
        zero = torch.round(-wmin / scale).clamp(0, spec.qmax)
    return scale.half().float(), zero


def encode(wg: torch.Tensor, scale: torch.Tensor, zero: Optional[torch.Tensor], spec: QuantSpec) -> torch.Tensor:
    q = torch.round(wg / scale)
    return q.clamp(-spec.qmax, spec.qmax) if zero is None else (q + zero).clamp(0, spec.qmax)


def decode(q: torch.Tensor, scale: torch.Tensor, zero: Optional[torch.Tensor]) -> torch.Tensor:
    return q * scale if zero is None else (q - zero) * scale


@dataclass
class QuantizedWeight:
    codes: torch.Tensor  # [out, in] int8 (symmetric) or uint8 (asymmetric); the integers actually stored
    scale: torch.Tensor  # [G0, G1] float32 (fp16-representable)
    zero: Optional[torch.Tensor]  # [G0, G1] uint8 or None
    spec: QuantSpec


def quantize_weight(w: torch.Tensor, spec: QuantSpec) -> QuantizedWeight:
    """Round-to-nearest quantization of a [out, in] weight matrix."""
    wg = _group_view(w.float(), spec)
    scale, zero = scale_zero(wg, spec)
    q = encode(wg, scale, zero, spec)
    dtype = torch.int8 if spec.symmetric else torch.uint8
    return QuantizedWeight(q.reshape(w.shape).to(dtype), scale[..., 0], None if zero is None else zero[..., 0].to(torch.uint8), spec)


def dequantize_weight(qw: QuantizedWeight) -> torch.Tensor:
    g = _group_view(qw.codes.float(), qw.spec)
    zero = None if qw.zero is None else qw.zero.float()[..., None]
    return decode(g, qw.scale[..., None], zero).reshape(qw.codes.shape)


def pack4(codes: torch.Tensor, symmetric: bool) -> torch.Tensor:
    """Two 4-bit codes per byte: even columns in the low nibble, odd columns in the high nibble."""
    u = (codes.to(torch.int16) + (8 if symmetric else 0)).to(torch.uint8)  # symmetric -7..7 -> 1..15; asymmetric is already 0..15
    return u[:, 0::2] | (u[:, 1::2] << 4)


def unpack4(packed: torch.Tensor, symmetric: bool) -> torch.Tensor:
    u = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(packed.shape[0], -1)
    return (u.to(torch.int16) - (8 if symmetric else 0)).to(torch.int8 if symmetric else torch.uint8)


# ---------------------------------------------------------------------------------------------------- QuantLinear


class QuantLinear(nn.Module):
    """``nn.Linear`` with integer weights. Stores ``qweight`` (packed for 4 bit), fp16 ``scale`` and optional ``zero``; the forward pass
    dequantizes (every call, or once if ``cache_dequant``) and runs an ordinary matmul."""

    def __init__(self, qw: QuantizedWeight, bias: Optional[torch.Tensor] = None, cache_dequant: bool = False):
        super().__init__()
        self.spec = qw.spec
        self.out_features, self.in_features = qw.codes.shape
        self.register_buffer("qweight", pack4(qw.codes, qw.spec.symmetric) if qw.spec.bits == 4 else qw.codes.clone())
        self.register_buffer("scale", qw.scale.half())
        if qw.zero is not None:
            self.register_buffer("zero", qw.zero.clone())
        else:
            self.zero = None
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.cache_dequant = cache_dequant
        self._w: Optional[torch.Tensor] = None

    @classmethod
    def from_linear(cls, lin: nn.Linear, spec: QuantSpec, hessian: Optional[torch.Tensor] = None, method: str = "rtn",
                    cache_dequant: bool = False, **gptq_kw) -> "QuantLinear":
        w = lin.weight.data.float()
        qw = gptq_quantize(w, hessian, spec, **gptq_kw) if method == "gptq" else quantize_weight(w, spec)
        return cls(qw, lin.bias.data if lin.bias is not None else None, cache_dequant)

    def quantized_weight(self) -> QuantizedWeight:
        codes = unpack4(self.qweight, self.spec.symmetric) if self.spec.bits == 4 else self.qweight
        return QuantizedWeight(codes, self.scale.float(), self.zero, self.spec)

    def dequantize(self) -> torch.Tensor:
        return dequantize_weight(self.quantized_weight())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_dequant:
            if self._w is None:
                self._w = self.dequantize()
            w = self._w
        else:
            w = self.dequantize()
        return F.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def stored_bytes(self) -> int:
        """Bytes actually held by this module (packed weights + fp16 scales + zeros + bias)."""
        n = sum(b.numel() * b.element_size() for b in self.buffers() if b is not None)
        return n + (0 if self.bias is None else self.bias.numel() * self.bias.element_size())

    def ideal_bytes(self) -> float:
        """Bytes if every code were packed at exactly ``bits`` bits (what a real kernel format would use)."""
        n_scales = self.scale.numel()
        return self.out_features * self.in_features * self.spec.bits / 8 + n_scales * 2 + (0 if self.zero is None else n_scales * self.spec.bits / 8)


# ---------------------------------------------------------------------------------------------------- applying it to the model

LINEAR_SLOTS = (("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"), ("self_attn", "o_proj"),
                ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"))


def quantize_model_rtn(model: Qwen2, spec: QuantSpec, cache_dequant: bool = False) -> Qwen2:
    """Replace every transformer-layer Linear by a QuantLinear (round-to-nearest). Embeddings, norms and the (tied) output head stay in full precision."""
    for layer in model.model.layers:
        for parent_name, name in LINEAR_SLOTS:
            parent = getattr(layer, parent_name)
            setattr(parent, name, QuantLinear.from_linear(getattr(parent, name), spec, cache_dequant=cache_dequant))
    return model


def model_size_bytes(model: nn.Module) -> dict:
    """Memory held by the model's parameters and buffers (shared storage counted once), split into the quantizable part and the rest."""
    seen, linears, other = set(), 0, 0
    quantizable = {id(m) for layer in model.model.layers for p, n in LINEAR_SLOTS for m in [getattr(getattr(layer, p), n)]}
    for mod in model.modules():
        if id(mod) in quantizable:
            for t in list(mod.parameters(recurse=False)) + list(mod.buffers(recurse=False)):
                if t is not None and t.data_ptr() not in seen:
                    seen.add(t.data_ptr())
                    linears += t.numel() * t.element_size()
        else:
            for t in list(mod.parameters(recurse=False)) + list(mod.buffers(recurse=False)):
                if t is not None and t.data_ptr() not in seen:
                    seen.add(t.data_ptr())
                    other += t.numel() * t.element_size()
    return {"layer_linears": linears, "other": other, "total": linears + other}


# ---------------------------------------------------------------------------------------------------- GPTQ


@torch.no_grad()
def gptq_quantize(
    w: torch.Tensor,
    hessian: torch.Tensor,
    spec: QuantSpec,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> QuantizedWeight:
    """GPTQ (Frantar et al., 2022) for one weight matrix ``w`` [out, in].

    Round-to-nearest quantizes every weight independently. GPTQ instead processes the input columns left to right: after rounding column j it
    computes the resulting error in the layer's OUTPUT (using the Hessian H = X^T X of the layer's inputs X on calibration data) and spreads
    that error over the columns that are still unquantized, so later columns compensate for earlier rounding. Minimizes ||X W^T - X Q^T||^2.

    ``hessian``: [in, in] accumulated X^T X. Columns are handled in blocks (lazy batch updates) for speed; ``percdamp`` adds a fraction of the mean
    diagonal to H for numerical stability.
    """
    w = w.clone().float()
    out_f, in_f = w.shape
    h = hessian.clone().float()
    dead = torch.diag(h) == 0  # input features that never activated: their weights do not matter
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h += percdamp * torch.mean(torch.diag(h)) * torch.eye(in_f)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)  # upper Cholesky factor of H^-1

    gs = spec.group_size
    if gs > 0 and (blocksize % gs or in_f % gs):
        raise ValueError("blocksize must be a multiple of group_size and in_features divisible by group_size")
    n_groups = 1 if gs <= 0 else in_f // gs
    codes = torch.zeros(out_f, in_f)
    scales = torch.zeros(out_f, n_groups)
    zeros = None if spec.symmetric else torch.zeros(out_f, n_groups)

    if gs <= 0:  # per-channel: scales from the original rows, fixed up front
        s, z = scale_zero(w, spec)  # reduces over the last dim of [out, in]
        scales[:, 0] = s[:, 0]
        if zeros is not None:
            zeros[:, 0] = z[:, 0]

    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if gs > 0 and col % gs == 0:  # a new group starts: its scale comes from the CURRENT (error-compensated) weights
                s, z = scale_zero(w1[:, i : i + gs], spec)
                scales[:, col // gs] = s[:, 0]
                if zeros is not None:
                    zeros[:, col // gs] = z[:, 0]
            g = 0 if gs <= 0 else col // gs
            s_col = scales[:, g : g + 1]
            z_col = None if zeros is None else zeros[:, g : g + 1]
            wcol = w1[:, i : i + 1]
            q = encode(wcol, s_col, z_col, spec)
            codes[:, col : col + 1] = q
            err = (wcol - decode(q, s_col, z_col)) / hinv1[i, i]
            w1[:, i:] -= err @ hinv1[i : i + 1, i:]  # push the rounding error onto the remaining columns of this block
            err1[:, i : i + 1] = err
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]  # ...and onto all later blocks

    dtype = torch.int8 if spec.symmetric else torch.uint8
    return QuantizedWeight(codes.to(dtype), scales, None if zeros is None else zeros.to(torch.uint8), spec)


@torch.no_grad()
def quantize_model_gptq(
    model: Qwen2,
    calib_ids: torch.Tensor,
    spec: QuantSpec,
    blocksize: int = 128,
    percdamp: float = 0.01,
    batch: int = 8,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Qwen2:
    """Sequential GPTQ over all decoder layers. For each layer: run the calibration batch through it (with the previous layers already quantized)
    to collect the input Hessian of its linear layers, quantize them, then recompute the layer's outputs as the next layer's inputs."""
    cfg = model.cfg
    n, T = calib_ids.shape
    x = model.model.embed_tokens(calib_ids)
    cos, sin = rope_cos_sin(torch.arange(T)[None], cfg.head_dim, cfg.rope_theta, x.dtype)

    for li, layer in enumerate(model.model.layers):
        a, m = layer.self_attn, layer.mlp
        groups = {"qkv": ([("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj")], a.q_proj),  # these three share their input
                  "o": ([("self_attn", "o_proj")], a.o_proj),
                  "gate_up": ([("mlp", "gate_proj"), ("mlp", "up_proj")], m.gate_proj),  # these two share their input
                  "down": ([("mlp", "down_proj")], m.down_proj)}
        hs = {name: torch.zeros(mod.in_features, mod.in_features) for name, (_, mod) in groups.items()}
        counts = {name: 0 for name in groups}

        def make_hook(name):
            def hook(_mod, inp):
                xx = inp[0].reshape(-1, inp[0].shape[-1]).float()
                hs[name] += xx.T @ xx
                counts[name] += xx.shape[0]
            return hook

        handles = [mod.register_forward_pre_hook(make_hook(name)) for name, (_, mod) in groups.items()]
        for b in range(0, n, batch):
            layer(x[b : b + batch], cos, sin)
        for hnd in handles:
            hnd.remove()

        for name, (slots, _) in groups.items():
            h = 2.0 * hs[name] / max(counts[name], 1)
            for parent_name, lin_name in slots:
                parent = getattr(layer, parent_name)
                setattr(parent, lin_name, QuantLinear.from_linear(getattr(parent, lin_name), spec, hessian=h, method="gptq",
                                                                  blocksize=blocksize, percdamp=percdamp))
        x = torch.cat([layer(x[b : b + batch], cos, sin) for b in range(0, n, batch)])
        if progress:
            progress(li + 1, len(model.model.layers))
    return model


# ---------------------------------------------------------------------------------------------------- activations: W8A8 and SmoothQuant


def fake_quant_act(x: torch.Tensor, per: str = "tensor", bits: int = 8) -> torch.Tensor:
    """Simulated dynamic activation quantization (quantize then immediately dequantize), symmetric.
    per='tensor': one scale for the whole activation tensor;  per='token': one scale per token (row)."""
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax() if per == "tensor" else x.abs().amax(-1, keepdim=True)
    scale = amax.clamp(min=1e-5) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


class ActQuantLinear(nn.Module):
    """Wraps a linear layer so its INPUT is fake-quantized to int8 first (simulation of W8A8; the weight is expected to be quantized already)."""

    def __init__(self, inner: nn.Module, per: str = "tensor", bits: int = 8):
        super().__init__()
        self.inner, self.per, self.bits = inner, per, bits

    def forward(self, x):
        return self.inner(fake_quant_act(x, self.per, self.bits))


def apply_activation_quant(model: Qwen2, per: str = "tensor", bits: int = 8) -> Qwen2:
    for layer in model.model.layers:
        for parent_name, name in LINEAR_SLOTS:
            parent = getattr(layer, parent_name)
            setattr(parent, name, ActQuantLinear(getattr(parent, name), per, bits))
    return model


@torch.no_grad()
def collect_act_absmax(model: Qwen2, calib_ids: torch.Tensor, batch: int = 8) -> dict:
    """Per-input-channel max |activation| at the inputs of the attention (after input_layernorm) and MLP (after post_attention_layernorm)."""
    stats: dict = {}
    handles = []
    for li, layer in enumerate(model.model.layers):
        for key, norm in ((("attn", li), layer.input_layernorm), (("mlp", li), layer.post_attention_layernorm)):
            def hook(_m, _i, out, key=key):
                cur = out.abs().reshape(-1, out.shape[-1]).amax(0)
                stats[key] = cur if key not in stats else torch.maximum(stats[key], cur)
            handles.append(norm.register_forward_hook(hook))
    for b in range(0, calib_ids.shape[0], batch):
        model(calib_ids[b : b + batch], last_only=True)  # only the norm outputs matter; skip the huge logits tensor
    for h in handles:
        h.remove()
    return stats


@torch.no_grad()
def smooth_model(model: Qwen2, act_absmax: dict, alpha: float = 0.5) -> Qwen2:
    """SmoothQuant (Xiao et al., 2022): per input channel j choose s_j = max|X_j|^alpha / max|W_j|^(1-alpha); divide the activation channel by s_j
    (folded into the preceding RMSNorm weight) and multiply the weight column by s_j. The layer's output is mathematically unchanged, but the
    activation outliers shrink and the weights absorb them. Applied where a norm directly precedes the linears (q/k/v and gate/up)."""
    for li, layer in enumerate(model.model.layers):
        for key, norm, lins in (("attn", layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]),
                                ("mlp", layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj])):
            amax_x = act_absmax[(key, li)].clamp(min=1e-5)
            amax_w = torch.stack([m.weight.abs().amax(0) for m in lins]).amax(0).clamp(min=1e-5)
            s = (amax_x**alpha / amax_w ** (1 - alpha)).clamp(min=1e-5)
            norm.weight.div_(s)
            for m in lins:
                m.weight.mul_(s[None, :])
    return model


# ---------------------------------------------------------------------------------------------------- int8 KV cache


@torch.no_grad()
def calibrate_key_scales(model: Qwen2, calib_ids: torch.Tensor, batch: int = 8) -> torch.Tensor:
    """Static per-channel key scales [layers, kv_heads, head_dim] = max |key| over calibration tokens / 127.
    Keys have a few large channels (the q/k/v projections carry big biases), so one scale per CHANNEL (shared by all tokens) quantizes them
    far better than one scale per token."""
    cfg = model.cfg
    amax = torch.zeros(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim)
    dtype = next(model.parameters()).dtype
    for b in range(0, calib_ids.shape[0], batch):
        x = calib_ids[b : b + batch]
        cache = KVCache(cfg, x.shape[0], x.shape[1], dtype)
        model(x, cache=cache, last_only=True)
        for li in range(cfg.num_layers):
            amax[li] = torch.maximum(amax[li], cache.k[li][:, :, : x.shape[1]].abs().amax(dim=(0, 2)).float())
    return (amax / 127).clamp(min=1e-5)


class Int8KVCache(KVCache):
    """KV cache stored as int8. New K/V are quantized on write; ``update`` returns dequantized views for attention.
    Values: one fp16 scale per (token, kv head).  Keys: by default also per token; with ``key_scales`` (from ``calibrate_key_scales``) one
    STATIC scale per (layer, kv head, channel), which is much more accurate for keys and needs no per-token key scales in memory.
    Halves the cache of a bf16 model (about 6.3 KB vs 12.3 KB per token here) and quarters an fp32 one."""

    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: int, dtype: torch.dtype = torch.float32,
                 device: str | torch.device = "cpu",
                 key_scales: Optional[torch.Tensor] = None):  # deliberately not calling KVCache.__init__: it would allocate full-precision buffers
        self._cfg = cfg
        self.max_len = max_len
        self.device = torch.device(device)
        self.length = 0
        self.mask = "causal"
        self.out_dtype = dtype
        self.key_scales = None if key_scales is None else key_scales.to(device).float()
        shape, sshape = (batch_size, cfg.num_kv_heads, max_len, cfg.head_dim), (batch_size, cfg.num_kv_heads, max_len, 1)
        self.kq = [torch.zeros(shape, dtype=torch.int8, device=device) for _ in range(cfg.num_layers)]
        self.vq = [torch.zeros(shape, dtype=torch.int8, device=device) for _ in range(cfg.num_layers)]
        self.ks = [torch.zeros(sshape, dtype=torch.float16, device=device) for _ in range(cfg.num_layers)]
        self.vs = [torch.zeros(sshape, dtype=torch.float16, device=device) for _ in range(cfg.num_layers)]

    @staticmethod
    def _quant(x: torch.Tensor):
        scale = (x.float().abs().amax(-1, keepdim=True).clamp(min=1e-5) / 127).half()
        return torch.round(x.float() / scale.float()).clamp(-127, 127).to(torch.int8), scale

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        t = k_new.shape[2]
        start = self.length
        if start + t > self.max_len:
            raise ValueError(f"KV cache full: {start} + {t} > max_len {self.max_len}")
        end = start + t
        qv, sv = self._quant(v_new)
        self.vq[layer][:, :, start:end] = qv
        self.vs[layer][:, :, start:end] = sv
        if self.key_scales is not None:  # static per-channel key scales (clipped to +-127 for values larger than any seen in calibration)
            scale = self.key_scales[layer][None, :, None, :]
            self.kq[layer][:, :, start:end] = torch.round(k_new.float() / scale).clamp(-127, 127).to(torch.int8)
            k = self.kq[layer][:, :, :end].float() * scale
        else:  # dynamic per-token scales
            qk, sk = self._quant(k_new)
            self.kq[layer][:, :, start:end] = qk
            self.ks[layer][:, :, start:end] = sk
            k = self.kq[layer][:, :, :end].float() * self.ks[layer][:, :, :end].float()
        v = self.vq[layer][:, :, :end].float() * self.vs[layer][:, :, :end].float()
        return k.to(self.out_dtype), v.to(self.out_dtype)

    @staticmethod
    def bytes_per_token_int8(cfg: ModelConfig, static_keys: bool = False) -> int:
        """int8 values plus fp16 scales per head vector: keys and values both (dynamic), or values only when keys use static per-channel scales."""
        per_head = (cfg.head_dim + 2) * (1 if static_keys else 2) + (cfg.head_dim if static_keys else 0)
        return cfg.num_layers * cfg.num_kv_heads * per_head

    def bytes_used(self) -> int:
        return self.bytes_per_token_int8(self._cfg, self.key_scales is not None) * self.length
