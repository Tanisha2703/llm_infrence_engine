"""Download a Hugging Face checkpoint and load it into our own Qwen2 model."""
from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from miniserve.model import ModelConfig, Qwen2

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def resolve_model_dir(name_or_path: str) -> Path:
    """Local directory as-is; otherwise a HF repo id fetched into the HF cache."""
    p = Path(name_or_path)
    if p.is_dir():
        return p
    return Path(
        snapshot_download(
            name_or_path,
            allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
        )
    )


def load_model(
    name_or_path: str = DEFAULT_MODEL,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Return (model, tokenizer). Model is our implementation, weights from the checkpoint."""
    d = resolve_model_dir(name_or_path)
    cfg = ModelConfig.from_json(d / "config.json")

    with torch.device("meta"):  # build without allocating memory, then fill from disk
        model = Qwen2(cfg)

    state = {}
    for f in sorted(d.glob("*.safetensors")):
        state.update(load_file(f))
    # Tied models ship no separate lm_head; the tie is re-established by assign below.
    model.load_state_dict(state, strict=False, assign=True)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model = model.to(device=device, dtype=dtype).eval()

    tok = AutoTokenizer.from_pretrained(d)
    return model, tok
