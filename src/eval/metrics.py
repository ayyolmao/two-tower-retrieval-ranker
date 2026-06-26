"""Ranking metrics for top-K recommendation evaluation.

Pure functions over a single user's ranked predictions and ground-truth relevant
set, plus an ``evaluate`` helper that macro-averages each metric over many users.
These are the foundation for every reported number in the project (baselines,
two-tower retrieval, ranking, ablations), so correctness is verified by unit tests
on hand-computed toy examples in ``tests/test_metrics.py``.

Conventions:
  * ``ranked`` is an ordered sequence of item ids, best first (model output).
  * ``relevant`` is an unordered collection of ground-truth item ids.
  * ranks are 1-indexed: the item at index 0 sits at rank 1.
  * a user with an empty relevant set scores 0.0 here and is skipped by ``evaluate``.

Examples
--------
    from src.eval.metrics import recall_at_k, ndcg_at_k, mrr_at_k, evaluate

    ranked, relevant = [7, 3, 9, 1, 5], {3, 5}
    recall_at_k(ranked, relevant, 3)   # 0.5
    mrr_at_k(ranked, relevant, 3)      # 0.5
    ndcg_at_k(ranked, relevant, 3)     # 0.3869...

    evaluate([ranked], [relevant], ks=(3,))
    # {"recall@3": 0.5, "ndcg@3": 0.3869..., "mrr@3": 0.5}
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from statistics import fmean

_METRIC_FNS = {}  # populated below; maps metric name -> per-user function


def recall_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Fraction of relevant items present in the top-k predictions.

    Returns 0.0 if ``relevant`` is empty.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top_k = set(ranked[:k])
    return len(top_k & relevant) / len(relevant)


def mrr_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Reciprocal rank of the first relevant item within top-k (0.0 if none)."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked: Sequence[int],
    relevant: Collection[int],
    k: int,
    gains: Mapping[int, float] | None = None,
) -> float:
    """Normalized DCG@k.

    Binary relevance by default (gain 1 for relevant, 0 otherwise). Pass ``gains``
    (item id -> gain) for graded relevance; items absent from ``gains`` score 0.
    Returns 0.0 if there is no attainable gain (empty relevant set).
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0

    def gain(item: int) -> float:
        if gains is not None:
            return gains.get(item, 0.0)
        return 1.0 if item in relevant else 0.0

    dcg = sum(
        gain(item) / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
    )

    # Ideal DCG: the highest-gain relevant items packed into the top positions.
    ideal_gains = sorted((gain(item) for item in relevant), reverse=True)
    idcg = sum(
        g / math.log2(rank + 1)
        for rank, g in enumerate(ideal_gains[:k], start=1)
    )

    return dcg / idcg if idcg > 0 else 0.0


_METRIC_FNS.update(recall=recall_at_k, ndcg=ndcg_at_k, mrr=mrr_at_k)


def evaluate(
    ranked_lists: Sequence[Sequence[int]],
    relevant_sets: Sequence[Collection[int]],
    ks: Sequence[int] = (10, 100),
    metrics: Sequence[str] = ("recall", "ndcg", "mrr"),
) -> dict[str, float]:
    """Macro-average each metric@k over users.

    Users with an empty relevant set are skipped (they carry no signal and would
    bias the mean toward 0). Returns a flat dict keyed ``"{metric}@{k}"``.

    Raises:
        ValueError: if ``ranked_lists`` and ``relevant_sets`` differ in length,
            or an unknown metric name is requested.
    """
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError(
            f"ranked_lists ({len(ranked_lists)}) and relevant_sets "
            f"({len(relevant_sets)}) must have the same length."
        )
    unknown = set(metrics) - _METRIC_FNS.keys()
    if unknown:
        raise ValueError(f"Unknown metric(s): {sorted(unknown)}. Known: {sorted(_METRIC_FNS)}.")

    pairs = [
        (ranked, set(relevant))
        for ranked, relevant in zip(ranked_lists, relevant_sets)
        if relevant
    ]

    results: dict[str, float] = {}
    for metric in metrics:
        fn = _METRIC_FNS[metric]
        for k in ks:
            scores = [fn(ranked, relevant, k) for ranked, relevant in pairs]
            results[f"{metric}@{k}"] = fmean(scores) if scores else 0.0
    return results
