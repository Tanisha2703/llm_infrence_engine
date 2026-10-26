"""Our model must reproduce Hugging Face's logits. This is the foundation test:
if it fails, nothing built on top (cache, batching, speculation) can be trusted."""
import pytest
import torch
from transformers import AutoModelForCausalLM

from miniserve.loader import DEFAULT_MODEL, load_model, resolve_model_dir


@pytest.fixture(scope="module")
def ours():
    return load_model(DEFAULT_MODEL, dtype=torch.float32)


def test_logits_match_huggingface(ours):
    model, tok = ours
    hf = AutoModelForCausalLM.from_pretrained(
        resolve_model_dir(DEFAULT_MODEL), dtype=torch.float32
    ).eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        mine = model(ids)
        ref = hf(ids).logits
    assert mine.shape == ref.shape
    assert (mine - ref).abs().max().item() < 1e-3
    assert torch.equal(mine.argmax(-1), ref.argmax(-1))


def test_lm_head_is_tied(ours):
    model, _ = ours
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_param_count(ours):
    model, _ = ours
    assert 480e6 < model.num_params() < 500e6  # Qwen2.5-0.5B has ~494M params
