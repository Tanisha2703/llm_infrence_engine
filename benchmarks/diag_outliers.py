"""Diagnostic: WHERE do activation / KV-cache quantization errors come from in Qwen2.5-0.5B?

  python benchmarks/diag_outliers.py

1. Per-tensor int8 activation quantization applied to ONE layer type at a time (weights int8 per-channel everywhere): which inputs hurt?
2. Are the activation outliers tied to specific TOKENS (e.g. the first token) or to specific CHANNELS?
3. Int8 KV cache: per-token key scales (what Int8KVCache does) vs per-channel key scales (the KIVI idea).
All numbers are perplexity on the first 10 windows of 512 tokens of WikiText-2 test (not comparable with the 40-window table).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from miniserve.evalutil import perplexity, wikitext_ids
from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.quant import ActQuantLinear, Int8KVCache, QuantSpec, quantize_model_rtn

SLOTS = {
    "q/k/v inputs": [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj")],
    "gate/up inputs": [("mlp", "gate_proj"), ("mlp", "up_proj")],
    "o_proj input": [("self_attn", "o_proj")],
    "down_proj input": [("mlp", "down_proj")],
}


class Int8KVCachePerChannelKeys(Int8KVCache):
    """Keys quantized with one scale per (head, channel) over the tokens of the call (the KIVI idea); values per token as before.
    Valid here because perplexity evaluation writes the whole window in one prefill call."""

    def update(self, layer, k_new, v_new):
        t, start = k_new.shape[2], self.length
        scale_k = (k_new.float().abs().amax(2, keepdim=True).clamp(min=1e-5) / 127).half()  # [B, kvh, 1, hd]
        qk = torch.round(k_new.float() / scale_k.float()).clamp(-127, 127)
        qv, sv = self._quant(v_new)
        self._k_deq = (qk * scale_k.float()).to(self.out_dtype)
        self.vq[layer][:, :, start : start + t] = qv
        self.vs[layer][:, :, start : start + t] = sv
        v = (self.vq[layer][:, :, : start + t].float() * self.vs[layer][:, :, : start + t].float()).to(self.out_dtype)
        return self._k_deq, v


def main():
    model, tok = load_model(DEFAULT_MODEL, dtype=torch.float32)
    ids = wikitext_ids(tok, "test", max_tokens=10 * 512 + 64)
    out = {}

    def ppl(m, cache_cls=None):
        return perplexity(m, ids, 512, 10, cache_cls)["ppl"]

    base = ppl(model)
    w8 = quantize_model_rtn(copy.deepcopy(model), QuantSpec(8, 0))
    out["fp32"], out["w8 weights only"] = base, ppl(w8)
    print(f"fp32 {base:.3f} | int8 weights only {out['w8 weights only']:.3f}\n")

    print("PER-TENSOR int8 activation quantization applied to ONE input type at a time (on top of int8 weights):")
    for label, slots in SLOTS.items():
        m = copy.deepcopy(w8)
        for layer in m.model.layers:
            for parent, name in slots:
                setattr(getattr(layer, parent), name, ActQuantLinear(getattr(getattr(layer, parent), name), "tensor"))
        out[f"per-tensor A8 only on {label}"] = ppl(m)
        print(f"  only {label:<16}: ppl {out[f'per-tensor A8 only on {label}']:8.3f}   (+{out[f'per-tensor A8 only on {label}'] - out['w8 weights only']:.3f})")
        del m

    print("\nWHERE are the outliers? down_proj inputs, per layer: the single largest |activation| overall, and which token and channel hold it")
    x = ids[:512][None]
    rows = []
    handles = []
    for li, layer in enumerate(model.model.layers):
        def hook(_m, inp, li=li):
            a = inp[0][0].abs()  # [T, 4864]
            tok_max = a.amax(1)
            rows.append((li, tok_max.max().item(), int(tok_max.argmax()), int(a[int(tok_max.argmax())].argmax()), tok_max.median().item()))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    with torch.inference_mode():
        model(x, last_only=True)
    for h in handles:
        h.remove()
    print(f"  {'layer':>5} {'max |act|':>10} {'at token':>9} {'channel':>8} {'median over tokens of per-token max':>38}")
    for li, mx, t, c, med in rows:
        if mx > 20 * med or li in (0, 12, 23):
            print(f"  {li:>5} {mx:>10.1f} {t:>9} {c:>8} {med:>38.2f}")
    out["down_proj_outliers"] = rows

    print("\nINT8 KV cache (fp32 weights): per-token key scales vs per-channel key scales")
    out["kv per-token (Int8KVCache)"] = ppl(model, Int8KVCache)
    out["kv per-channel keys"] = ppl(model, Int8KVCachePerChannelKeys)
    print(f"  fp32 cache {base:.3f} | per-token keys {out['kv per-token (Int8KVCache)']:.3f} | per-channel keys {out['kv per-channel keys']:.3f}")
    print("\nKEY vs VALUE magnitudes in layer 12 (largest |value| per channel / median channel):")
    from miniserve.cache import KVCache

    cache = KVCache(model.cfg, 1, 512)
    with torch.inference_mode():
        model(x, cache=cache)
    for name, buf in (("keys", cache.k[12]), ("values", cache.v[12])):
        a = buf[0, :, :512].abs().amax(1)  # [kvh, hd]: per-channel max over tokens
        print(f"  {name:<7}: largest channel {a.max():.2f}, median channel {a.median():.2f}, ratio {a.max() / a.median():.1f}")
        out[f"kv {name} channel ratio"] = (a.max() / a.median()).item()

    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path("results/raw/quant_diag.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote results/raw/quant_diag.json")


if __name__ == "__main__":
    main()
