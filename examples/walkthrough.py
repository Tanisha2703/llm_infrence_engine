"""Print what happens to a prompt, step by step, inside our model. Run: python examples/walkthrough.py"""
import torch

from miniserve.loader import load_model
from miniserve.model import rope_cos_sin

model, tok = load_model()
cfg = model.cfg
prompt = "The capital of France is"

print("== 1. TOKENIZE: text -> integers ==")
ids = tok(prompt, return_tensors="pt").input_ids
print("token ids   :", ids[0].tolist())
print("token pieces:", [tok.decode(i) for i in ids[0]])

print("\n== 2. EMBED: each integer -> a vector of numbers ==")
x = model.model.embed_tokens(ids)
print("shape:", tuple(x.shape), "= (1 prompt, 5 tokens, 896 numbers each)")
print("first 6 numbers of token 0:", [round(v, 3) for v in x[0, 0, :6].tolist()])

print("\n== 3. ONE ATTENTION BLOCK: what the shapes look like ==")
layer = model.model.layers[0]
h = layer.input_layernorm(x)
B, T, _ = h.shape
q = layer.self_attn.q_proj(h)
k = layer.self_attn.k_proj(h)
v = layer.self_attn.v_proj(h)
print("queries:", tuple(q.shape), "-> 14 heads x 64 numbers = 896")
print("keys   :", tuple(k.shape), "->  2 heads x 64 numbers = 128   (GQA: far fewer!)")
print("values :", tuple(v.shape), "->  2 heads x 64 numbers = 128")

print("\n== 4. ATTENTION WEIGHTS: who looks at whom (layer 0, head 0) ==")
qh = q.view(B, T, cfg.num_heads, cfg.head_dim).transpose(1, 2)
kh = k.view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
cos, sin = rope_cos_sin(torch.arange(T).expand(B, T), cfg.head_dim, cfg.rope_theta, x.dtype)
from miniserve.model import apply_rope

qh, kh = apply_rope(qh, cos, sin), apply_rope(kh, cos, sin)
scores = (qh[0, 0] @ kh[0, 0].T) / cfg.head_dim**0.5
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
w = scores.masked_fill(mask, float("-inf")).softmax(-1)
pieces = [tok.decode(i).strip() or "_" for i in ids[0]]
print(f"{'':>9}" + "".join(f"{p:>9}" for p in pieces))
for i, p in enumerate(pieces):
    print(f"{p:>9}" + "".join(f"{w[i, j].item():9.2f}" for j in range(T)))
print("(each ROW sums to 1; row 'is' can look at itself and everything before it, never after)")

print("\n== 5. FULL MODEL: scores for the NEXT token ==")
with torch.no_grad():
    logits = model(ids)
print("logits shape:", tuple(logits.shape), "= one score for each of 151,936 possible words, at each position")
last = logits[0, -1].softmax(-1)
top = last.topk(5)
for p, i in zip(top.values, top.indices):
    print(f"  {tok.decode(i)!r:>12}  probability {p.item():.1%}")

print("\n== 6. THE GENERATION LOOP: pick, append, repeat ==")
for _ in range(4):
    with torch.no_grad():
        nxt = model(ids)[:, -1].argmax(-1, keepdim=True)
    ids = torch.cat([ids, nxt], dim=1)
    print("  now:", repr(tok.decode(ids[0])))
