"""Shared metrics utilities: ROUGE-L and Prometheus export helpers."""
from __future__ import annotations

from typing import Sequence


def rouge_l(hypothesis: str, reference: str) -> float:
    """Compute sentence-level ROUGE-L F1 score.

    Uses the longest common subsequence (LCS) of word tokens.
    Returns a float in [0, 1].
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs_len / len(hyp_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute the length of the longest common subsequence (DP, O(mn))."""
    m, n = len(a), len(b)
    # Use a rolling two-row DP table to keep memory O(n)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def print_table(rows: Sequence[tuple], headers: Sequence[str]) -> None:
    """Pretty-print a table to stdout."""
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in col_widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))
    print()
