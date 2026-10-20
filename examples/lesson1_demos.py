"""Tiny, hand-checkable demos for Lesson 1 segments. Run: python examples/lesson1_demos.py [segment]
Every number quoted in docs/lesson-01/*.md comes from this file."""
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch

from miniserve.model import ModelConfig, apply_rope, rope_cos_sin

torch.set_printoptions(precision=3, sci_mode=False)


def seg_1_1_tokens():
    from miniserve.loader import load_model

    _, tok = load_model()
    for text in ["The capital of France is", "unbelievable", "Hello, world!", "  indentation", "नमस्ते"]:
        ids = tok(text).input_ids
        print(f"{text!r:32} -> {len(ids)} tokens {[tok.decode(i) for i in ids]}")
    print("vocab size:", len(tok))


def seg_1_2_embeddings():
    from miniserve.loader import load_model

    m, tok = load_model()
    E = m.model.embed_tokens.weight
    print("embedding table shape:", tuple(E.shape), "->", E.numel() / 1e6, "M numbers")

    def vec(w):
        return E[tok(w).input_ids[0]]

    cos = torch.nn.functional.cosine_similarity
    for a, b in [(" Paris", " London"), (" Paris", " banana"), (" king", " queen"), (" king", " apple")]:
        print(f"cosine({a!r}, {b!r}) = {cos(vec(a), vec(b), dim=0).item():.3f}")


def seg_1_3_attention():
    print("3 tokens, 2 numbers each. Query of token 3 attends to all three keys.")
    q = torch.tensor([1.0, 0.0])
    K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
    raw = K @ q
    scaled = raw / math.sqrt(2)
    w = scaled.softmax(0)
    out = w @ V
    print("raw scores  q.k :", raw.tolist())
    print("scaled /sqrt(2) :", [round(x, 3) for x in scaled.tolist()])
    print("softmax weights :", [round(x, 3) for x in w.tolist()], "sum =", round(w.sum().item(), 3))
    print("output = sum(w*V):", [round(x, 3) for x in out.tolist()])
    print("\nsoftmax of [2,1,0]:", [round(x, 3) for x in torch.tensor([2.0, 1, 0]).softmax(0).tolist()])
    print("softmax of [20,10,0]:", [round(x, 5) for x in torch.tensor([20.0, 10, 0]).softmax(0).tolist()], "(saturates to one-hot)")
    print("why /sqrt(d): var of dot product of random d-dim vectors:")
    for d in (4, 64, 1024):
        a, b = torch.randn(20000, d), torch.randn(20000, d)
        print(f"  d={d:5}: std(q.k) = {(a * b).sum(1).std():.2f}   after /sqrt(d): {((a * b).sum(1) / math.sqrt(d)).std():.2f}")


def seg_1_4_mask():
    T = 4
    scores = torch.randn(T, T)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    print("mask (True = blocked):\n", mask.int())
    print("after masked softmax (rows sum to 1, zeros above diagonal):\n", scores.masked_fill(mask, float("-inf")).softmax(-1))
    print("\nWHY -inf: exp(-inf) = 0, so softmax gives exactly 0 weight to the future.")


def seg_1_5_gqa():
    cfg = ModelConfig.from_json(next(__import__("pathlib").Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/config.json")))
    per_tok = lambda kv_heads: cfg.num_layers * 2 * kv_heads * cfg.head_dim * 2  # bytes, bf16
    for name, h in [("MHA (14 kv heads)", cfg.num_heads), ("GQA (2 kv heads)", cfg.num_kv_heads), ("MQA (1 kv head)", 1)]:
        b = per_tok(h)
        print(f"{name:20}: {b:>7} bytes/token = {b/1024:6.1f} KB/token | 1000 tokens = {b*1000/2**20:6.1f} MB | 32 users x 2000 tok = {b*64000/2**30:.2f} GB")


def seg_1_6_rope():
    print("2D toy: rotate [1,0] by angle = position * 90 degrees")
    for pos in range(4):
        a = pos * math.pi / 2
        print(f"  pos {pos}: [{math.cos(a):+.0f}, {math.sin(a):+.0f}]")
    print("\nReal function, head_dim=2 (one rotating pair, angle = position radians):")
    q0 = torch.tensor([[[[0.7, 0.3]]]])  # [B,H,T,D]
    k0 = torch.tensor([[[[0.2, 0.9]]]])

    def rot(x, p):
        c, s = rope_cos_sin(torch.tensor([[p]]), 2, 10000.0, torch.float32)
        return apply_rope(x, c, s)

    for qp, kp in [(5, 3), (12, 10), (100, 98), (7, 3)]:
        dot = (rot(q0, qp) * rot(k0, kp)).sum().item()
        print(f"  q at pos {qp:3}, k at pos {kp:3} (distance {qp-kp}): q.k = {dot:.4f}")
    print("-> same distance => same score. Only RELATIVE position matters.")
    inv = 1.0 / (1_000_000.0 ** (torch.arange(0, 64, 2).float() / 64))
    print(f"\nfrequencies across 32 pairs: fastest {inv[0]:.2f} rad/token, slowest {inv[-1]:.2e} rad/token")


def seg_1_7_mlp_norm_residual():
    x = torch.tensor([3.0, 4.0])
    rms = x.pow(2).mean().sqrt()
    print(f"RMSNorm of [3,4]: rms = sqrt((9+16)/2) = {rms:.4f}; result = {(x / rms).tolist()}")
    print(f"  rms of the result = {(x / rms).pow(2).mean().sqrt():.4f} (always 1 before the learned scale)")
    from miniserve.loader import load_model

    m, _ = load_model()
    layer = m.model.layers[0]
    cnt = lambda mod: sum(p.numel() for p in mod.parameters())
    a, f = cnt(layer.self_attn), cnt(layer.mlp)
    total = m.num_params()
    emb = m.model.embed_tokens.weight.numel()
    print(f"\nper layer: attention {a/1e6:.2f}M, MLP {f/1e6:.2f}M, norms {(cnt(layer)-a-f)/1e3:.1f}K")
    print(f"x24 layers: {24*(a+f)/1e6:.0f}M; embedding/lm_head (tied, counted once): {emb/1e6:.0f}M; total {total/1e6:.0f}M")
    print(f"share: MLP {24*f/total:.0%}, attention {24*a/total:.0%}, embeddings {emb/total:.0%}")


def seg_1_8_sampling():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    for t in (0.5, 1.0, 2.0):
        print(f"temperature {t}: probs = {[round(x,3) for x in (logits/t).softmax(0).tolist()]}")
    p = logits.softmax(0)
    s, idx = p.sort(descending=True)
    cum = s.cumsum(0)
    print("\nprobs sorted:", [round(x, 3) for x in s.tolist()], "cumulative:", [round(x, 3) for x in cum.tolist()])
    print("top-p=0.9 keeps tokens until cumulative >= 0.9 ->", int((cum < 0.9).sum()) + 1, "tokens")
    print("greedy = argmax = temperature -> 0")


SEGMENTS = {k.split("_", 2)[1] + "." + k.split("_", 3)[2]: v for k, v in globals().items() if k.startswith("seg_")}

if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in SEGMENTS.items():
        if want in (None, name):
            print(f"\n######## segment {name}: {fn.__name__} ########")
            fn()
