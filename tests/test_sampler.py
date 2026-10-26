import torch

from miniserve.sampler import sample


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 2.0, 0.5], [3.0, 0.0, 0.0]])
    assert sample(logits, temperature=0.0).tolist() == [1, 0]


def test_top_k_one_is_greedy():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[0.1, 2.0, 0.5]])
    for _ in range(20):
        assert sample(logits, temperature=1.0, top_k=1, generator=g).item() == 1


def test_top_p_keeps_only_the_nucleus():
    # probs ~ [0.64, 0.24, 0.09, 0.03]; top_p=0.7 must keep only tokens 0 and 1
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    g = torch.Generator().manual_seed(0)
    seen = {sample(logits, temperature=1.0, top_p=0.7, generator=g).item() for _ in range(300)}
    assert seen == {0, 1}


def test_top_p_always_keeps_most_likely_token():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    assert sample(logits, temperature=1.0, top_p=0.0001, generator=g).item() == 0


def test_sampling_follows_distribution():
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    g = torch.Generator().manual_seed(123)
    draws = torch.stack([sample(logits, temperature=1.0, generator=g) for _ in range(4000)]).flatten()
    freq = torch.bincount(draws, minlength=3).float() / 4000
    assert torch.allclose(freq, torch.tensor([0.7, 0.2, 0.1]), atol=0.03)


def test_seed_makes_sampling_reproducible():
    logits = torch.randn(1, 50)
    a = sample(logits, 1.0, generator=torch.Generator().manual_seed(7))
    b = sample(logits, 1.0, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
