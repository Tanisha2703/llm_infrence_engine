"""Speculative decoding must be EXACT: greedy output identical to the target's, sampled output distributed as the target's.

Two layers of evidence:
  * the acceptance rule, as a pure function, checked statistically on toy and real distributions;
  * the full decoding loop (caches, rollback, bonus tokens, budgets, stop tokens) checked against the plain target decoder.
"""
import copy
import itertools

import pytest
import torch

from miniserve.cache import KVCache
from miniserve.generate import generate
from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.sampler import logits_to_probs
from miniserve.speculative import (
    ModelDrafter,
    NGramDrafter,
    expected_tokens_per_round,
    generate_speculative,
    predicted_speedup,
    speculative_stream,
    verify_and_sample,
)

# ------------------------------------------------------------------------------------ formulas and sampler (no model)


def test_expected_tokens_formula():
    assert expected_tokens_per_round(0.0, 5) == pytest.approx(1.0)  # nothing ever accepted: just the correction token
    assert expected_tokens_per_round(1.0, 5) == pytest.approx(6.0)  # everything accepted: k + 1 tokens
    assert expected_tokens_per_round(0.5, 2) == pytest.approx(1.75)  # (1 - 0.5^3) / 0.5
    # brute force: P(accept first i) = alpha^i, emitted = accepted + 1
    a, k = 0.7, 4
    brute = sum((a**i) * (1 - a) * (i + 1) for i in range(k)) + (a**k) * (k + 1)
    assert expected_tokens_per_round(a, k) == pytest.approx(brute)


def test_predicted_speedup_limits():
    assert predicted_speedup(1.0, 4, draft_cost=0.0) == pytest.approx(5.0)  # free draft, perfect agreement
    assert predicted_speedup(0.0, 4, draft_cost=0.1) < 1.0  # never agreeing can only slow things down
    assert predicted_speedup(0.8, 4, draft_cost=0.05) > 2.0


def test_logits_to_probs_basics():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    g = logits_to_probs(logits, temperature=0.0)
    assert g.tolist() == [[1.0, 0.0, 0.0, 0.0]]  # greedy is a one-hot distribution
    p = logits_to_probs(logits, temperature=1.0)
    assert torch.allclose(p.sum(-1), torch.ones(1)) and torch.allclose(p, logits.softmax(-1))
    assert (logits_to_probs(logits, 1.0, top_k=2) > 0).sum().item() == 2
    assert (logits_to_probs(logits, 1.0, top_p=0.7) > 0).sum().item() == 2  # tokens 0 and 1 (mass 0.64 then 0.88)
    assert torch.allclose(logits_to_probs(logits, 1.0, top_k=2).sum(-1), torch.ones(1))


# ------------------------------------------------------------------------------------ the statistical proof (no model)


def _toy_tables(V, seed, noise):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(V, V, generator=g) * 1.5
    q = base.softmax(-1)  # target: next-token distribution given the last token (a Markov chain)
    p = (base + noise * torch.randn(V, V, generator=g)).softmax(-1)  # draft: a noisy version of it
    return q, p


def _spec_first_tokens(q, p, start, n_tokens, k, gen, deterministic):
    """Generate n_tokens with speculative rounds on toy Markov 'models'; return the tuple of tokens."""
    out, last = [], start
    while len(out) < n_tokens:
        draft, probs, prev = [], [], last
        for _ in range(k):
            pr = p[prev]
            x = int(torch.argmax(pr)) if deterministic else int(torch.multinomial(pr, 1, generator=gen))
            draft.append(x)
            probs.append(pr)
            prev = x
        target_probs = torch.stack([q[t] for t in [last] + draft])  # q_0 .. q_k
        emitted, n = verify_and_sample(draft, None if deterministic else torch.stack(probs), target_probs, gen)
        assert len(emitted) == n + 1
        out.extend(emitted)
        last = out[-1]
    return tuple(out[:n_tokens])


def _tv_to_target(q, start, V, counts, total):
    tv = 0.0
    for t1, t2 in itertools.product(range(V), repeat=2):
        exact = float(q[start, t1] * q[t1, t2])
        tv += abs(counts.get((t1, t2), 0) / total - exact)
    return tv / 2


@pytest.mark.parametrize("deterministic", [False, True], ids=["sampled-draft", "deterministic-draft (n-gram style)"])
def test_speculative_sampling_preserves_target_distribution(deterministic):
    """Joint distribution of the first two emitted tokens equals the target's, although the draft is quite different."""
    V, start, N, k = 3, 0, 9000, 2
    q, p = _toy_tables(V, seed=1, noise=1.5)
    gen = torch.Generator().manual_seed(123)
    counts = {}
    for _ in range(N):
        seq = _spec_first_tokens(q, p, start, 2, k, gen, deterministic)
        counts[seq] = counts.get(seq, 0) + 1
    # for reference: how far the DRAFT's own distribution is from the target's (so the test could fail if the rule were wrong)
    tv_draft = _tv_to_target(q, start, V, {(a, b): float(p[start, a] * p[a, b]) * N for a in range(V) for b in range(V)}, N)
    assert _tv_to_target(q, start, V, counts, N) < 0.035
    if not deterministic:
        assert tv_draft > 0.08  # the draft really differs, so a broken acceptance rule would be detected


def test_acceptance_probability_is_overlap_of_distributions():
    """P(accept first token) = sum_x min(p(x), q(x))  (Leviathan et al.'s alpha = 1 - TV(p, q))."""
    q = torch.tensor([0.5, 0.3, 0.2])
    p = torch.tensor([0.2, 0.3, 0.5])
    gen = torch.Generator().manual_seed(0)
    N, acc = 20000, 0
    for _ in range(N):
        x = int(torch.multinomial(p, 1, generator=gen))
        _, n = verify_and_sample([x], p[None], torch.stack([q, q]), gen)
        acc += n
    assert acc / N == pytest.approx(float(torch.minimum(p, q).sum()), abs=0.015)  # = 0.7


def test_greedy_rule_is_exact_comparison():
    one_hot = lambda i: torch.eye(4)[i]
    # target argmax 2; draft proposes 2 then 1
    emitted, n = verify_and_sample([2, 1], torch.stack([one_hot(2), one_hot(1)]),
                                   torch.stack([one_hot(2), one_hot(3), one_hot(0)]))
    assert (emitted, n) == ([2, 3], 1)  # first matches, second is rejected and replaced by the target's argmax
    emitted, n = verify_and_sample([2], torch.stack([one_hot(2)]), torch.stack([one_hot(2), one_hot(1)]))
    assert (emitted, n) == ([2, 1], 1)  # all accepted: bonus token is the target's argmax after the proposal


# ------------------------------------------------------------------------------------ the full loop, with the real model


@pytest.fixture(scope="module")
def ours():
    return load_model(DEFAULT_MODEL, dtype=torch.float32)


@pytest.fixture(scope="module")
def noisy_draft(ours):
    """An imperfect draft: the target with its weights perturbed, so some proposals are accepted and some rejected."""
    model, _ = ours
    d = copy.deepcopy(model)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for prm in d.parameters():
            prm.add_(torch.randn(prm.shape, generator=g) * prm.std() * 0.12)
    return d


PROMPTS = [
    "The three primary colors are",
    "Once upon a time in a small village by the sea there lived an old fisherman who",
    "def fibonacci(n):",
]
REPETITIVE = "one two three four five one two three four five one two three four five one two three"


@pytest.fixture(scope="module")
def reference(ours):
    model, tok = ours
    out = {}
    for text in PROMPTS + [REPETITIVE]:
        ids = tok(text).input_ids
        out[text] = (ids, generate(model, ids, 20).token_ids)
    return out


@pytest.mark.parametrize("k", [1, 3, 6])
def test_greedy_speculative_with_imperfect_draft_equals_target_greedy(ours, noisy_draft, reference, k):
    model, _ = ours
    rates = []
    for text in PROMPTS:
        ids, ref = reference[text]
        r = generate_speculative(model, ModelDrafter(noisy_draft, 128), ids, 20, k=k)
        assert r.token_ids == ref, f"speculative output differs from plain greedy for {text!r}"
        rates.append(r.acceptance_rate)
    assert 0.0 < sum(rates) / len(rates) < 1.0  # both the accept and the reject paths were exercised


def test_greedy_speculative_with_ngram_drafter_equals_target_greedy(ours, reference):
    model, _ = ours
    for text in [REPETITIVE] + PROMPTS:
        ids, ref = reference[text]
        r = generate_speculative(model, NGramDrafter(), ids, 20, k=5)
        assert r.token_ids == ref
    ids, _ = reference[REPETITIVE]
    r = generate_speculative(model, NGramDrafter(), ids, 20, k=5)
    assert r.proposed > 0 and r.accepted > 0  # the repetitive prompt really triggered proposals


def test_perfect_draft_accepts_everything_in_greedy_mode(ours, reference):
    model, _ = ours
    ids, ref = reference[PROMPTS[0]]
    k = 4
    r = generate_speculative(model, ModelDrafter(model, 128), ids, 20, k=k)
    assert r.token_ids == ref
    assert r.acceptance_rate == 1.0
    assert r.mean_emitted_per_round > k  # about k + 1 per round (the last round may be cut short by the budget)


def test_target_cache_always_holds_exactly_the_committed_context(ours, noisy_draft, reference):
    """The rollback invariant: after every round the cache holds ctx[:-1], and those K/V equal a fresh prefill's."""
    model, _ = ours
    ids, _ = reference[PROMPTS[1]]
    info = {}
    stream = speculative_stream(model, ModelDrafter(noisy_draft, 128), ids, 24, k=4, inspect=info)
    rejected_somewhere = False
    for toks, stat in stream:
        tc, ctx = info["cache"], info["ctx"]
        assert tc.length == len(ctx) - 1
        rejected_somewhere |= stat is not None and stat.accepted < stat.k
    assert rejected_somewhere
    fresh = KVCache(model.cfg, 1, len(ctx), torch.float32)
    with torch.inference_mode():
        model(torch.tensor([ctx[:-1]]), cache=fresh)
    for layer in (0, 11, 23):
        n = len(ctx) - 1
        assert (tc.k[layer][:, :, :n] - fresh.k[layer][:, :, :n]).abs().max().item() < 1e-3
        assert (tc.v[layer][:, :, :n] - fresh.v[layer][:, :, :n]).abs().max().item() < 1e-3


@pytest.mark.parametrize("max_new", [1, 2, 5, 7])
def test_token_budget_is_respected_for_any_k(ours, noisy_draft, reference, max_new):
    model, _ = ours
    ids, ref = reference[PROMPTS[0]]
    for k in (1, 4, 8):
        r = generate_speculative(model, ModelDrafter(noisy_draft, 128), ids, max_new, k=k)
        assert r.token_ids == ref[:max_new]


def test_stop_token_ends_generation_and_is_not_emitted(ours, noisy_draft, reference):
    model, _ = ours
    ids, ref = reference[PROMPTS[1]]
    stop = ref[6]
    expected = ref[: ref.index(stop)]
    r = generate_speculative(model, ModelDrafter(noisy_draft, 128), ids, 20, k=4, eos_ids=[stop])
    assert r.token_ids == expected and r.finish_reason == "stop"


def test_sampling_is_reproducible_with_a_seed(ours, noisy_draft, reference):
    model, _ = ours
    ids, _ = reference[PROMPTS[0]]
    run = lambda seed: generate_speculative(model, ModelDrafter(noisy_draft, 128), ids, 16, k=3, temperature=0.9, seed=seed).token_ids
    assert run(5) == run(5)
    assert run(5) != run(6)


def test_stats_are_consistent(ours, noisy_draft, reference):
    model, _ = ours
    ids, _ = reference[PROMPTS[0]]
    r = generate_speculative(model, ModelDrafter(noisy_draft, 128), ids, 20, k=4)
    assert sum(x.emitted for x in r.rounds) == len(r.token_ids) - 1  # the first token comes from the prefill
    assert all(x.emitted == x.accepted + 1 for x in r.rounds)
    assert len(r.token_times) == len(r.token_ids) and r.token_times == sorted(r.token_times)
    assert 0.0 <= r.acceptance_rate <= 1.0


def test_acceptance_rule_preserves_the_real_model_distribution(ours, noisy_draft, reference):
    """The same statistical check on REAL next-token distributions (top-10 restricted, so the support is small)."""
    model, _ = ours
    ids, _ = reference[PROMPTS[0]]
    x = torch.tensor([ids])
    with torch.inference_mode():
        q_full = logits_to_probs(model(x, last_only=True)[0, -1], 1.0, top_k=10)
        p_full = logits_to_probs(noisy_draft(x, last_only=True)[0, -1], 1.0, top_k=10)
    support = ((q_full > 0) | (p_full > 0)).nonzero().flatten()
    q, p = q_full[support], p_full[support]
    assert q.sum().item() == pytest.approx(1.0, abs=1e-5) and p.sum().item() == pytest.approx(1.0, abs=1e-5)
    gen = torch.Generator().manual_seed(7)
    N, counts = 15000, torch.zeros(len(support))
    for _ in range(N):
        draft_tok = int(torch.multinomial(p, 1, generator=gen))
        emitted, _ = verify_and_sample([draft_tok], p[None], torch.stack([q, q]), gen)
        counts[emitted[0]] += 1
    tv = 0.5 * (counts / N - q).abs().sum().item()
    tv_draft = 0.5 * (p - q).abs().sum().item()
    assert tv < 0.03
    assert tv_draft > 0.05  # the draft genuinely differs from the target here


def _has_local(name):
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(name, allow_patterns=["config.json"], local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.heavy  # loads the 1.5B target in fp32 (~6 GB of RAM); skip with: pytest -m "not heavy"
@pytest.mark.skipif(not _has_local("Qwen/Qwen2.5-1.5B-Instruct"), reason="Qwen2.5-1.5B-Instruct not downloaded")
def test_real_draft_target_pair_greedy_equality(ours):
    """The pair used in the benchmark: Qwen2.5-0.5B drafts for Qwen2.5-1.5B."""
    draft, tok = ours
    target, _ = load_model("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32)
    ids = tok("The capital of France is").input_ids
    ref = generate(target, ids, 12).token_ids
    r = generate_speculative(target, ModelDrafter(draft, 64), ids, 12, k=4)
    assert r.token_ids == ref
