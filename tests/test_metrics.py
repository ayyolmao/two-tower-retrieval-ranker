"""Unit tests for ranking metrics — hand-computed toy cases.

Every expected value here is derivable by hand from the definitions, which is the
whole point: these tests pin down the off-by-one traps (1-indexed ranks, IDCG
sizing, empty relevant sets) that silently corrupt downstream numbers.
"""

import math

import pytest

from src.eval.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k

# Canonical toy example used across the suite.
#   ranked = [7, 3, 9, 1, 5], relevant = {3, 5}
#   top-3 = [7, 3, 9] → only item 3 is relevant (rank 2)
RANKED = [7, 3, 9, 1, 5]
RELEVANT = {3, 5}


# --- recall_at_k ----------------------------------------------------------------

def test_recall_toy():
    # 1 of 2 relevant items (item 3) in top-3
    assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.5)


def test_recall_all_found():
    # k large enough to include both relevant items
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(1.0)


def test_recall_no_hits():
    assert recall_at_k(RANKED, {42, 99}, 3) == 0.0


def test_recall_empty_relevant():
    assert recall_at_k(RANKED, set(), 3) == 0.0


def test_recall_k_larger_than_list():
    # k=100 but list has 5 items → no crash, both relevant found
    assert recall_at_k(RANKED, RELEVANT, 100) == pytest.approx(1.0)


# --- mrr_at_k -------------------------------------------------------------------

def test_mrr_toy():
    # first relevant (item 3) at rank 2 → 1/2
    assert mrr_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.5)


def test_mrr_first_position():
    assert mrr_at_k([5, 7, 3], RELEVANT, 3) == pytest.approx(1.0)


def test_mrr_no_hits_in_topk():
    # both relevant items sit beyond k=1
    assert mrr_at_k(RANKED, RELEVANT, 1) == 0.0


def test_mrr_empty_relevant():
    assert mrr_at_k(RANKED, set(), 3) == 0.0


# --- ndcg_at_k ------------------------------------------------------------------

def test_ndcg_toy():
    # DCG@3  = 1/log2(3)                 (item 3 at rank 2)
    # IDCG@3 = 1/log2(2) + 1/log2(3)     (2 relevant items, ideal ranks 1,2)
    dcg = 1 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(dcg / idcg)
    # ≈ 0.3869
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.3868528, abs=1e-6)


def test_ndcg_perfect_ranking():
    # all relevant items at the very top → NDCG = 1
    assert ndcg_at_k([3, 5, 7, 1, 9], RELEVANT, 5) == pytest.approx(1.0)


def test_ndcg_no_hits():
    assert ndcg_at_k(RANKED, {42, 99}, 3) == 0.0


def test_ndcg_empty_relevant():
    assert ndcg_at_k(RANKED, set(), 3) == 0.0


def test_ndcg_idcg_capped_by_k():
    # 3 relevant but k=1 → IDCG uses only the single best ideal position.
    # ranked puts a relevant item at rank 1 → perfect within the k=1 budget.
    assert ndcg_at_k([3, 0, 0], {3, 5, 7}, 1) == pytest.approx(1.0)


def test_ndcg_graded_gains():
    # graded relevance via gains map; item 5 worth more than item 3
    gains = {3: 1.0, 5: 3.0}
    relevant = {3, 5}
    ranked = [3, 5]  # lower-gain item ranked first → suboptimal
    dcg = 1.0 / math.log2(2) + 3.0 / math.log2(3)
    idcg = 3.0 / math.log2(2) + 1.0 / math.log2(3)  # ideal: high gain first
    assert ndcg_at_k(ranked, relevant, 2, gains=gains) == pytest.approx(dcg / idcg)


# --- evaluate -------------------------------------------------------------------

def test_evaluate_single_user():
    out = evaluate([RANKED], [RELEVANT], ks=(3,))
    assert out["recall@3"] == pytest.approx(0.5)
    assert out["mrr@3"] == pytest.approx(0.5)
    assert out["ndcg@3"] == pytest.approx(0.3868528, abs=1e-6)


def test_evaluate_macro_average():
    # user A: perfect recall@3 (1.0); user B: miss (0.0) → mean 0.5
    ranked_lists = [[3, 5, 9], [1, 2, 4]]
    relevant_sets = [{3, 5}, {7}]
    out = evaluate(ranked_lists, relevant_sets, ks=(3,), metrics=("recall",))
    assert out["recall@3"] == pytest.approx(0.5)


def test_evaluate_skips_empty_relevant():
    # second user has no relevant items → excluded from the mean, not counted as 0
    ranked_lists = [[3, 5, 9], [1, 2, 4]]
    relevant_sets = [{3, 5}, set()]
    out = evaluate(ranked_lists, relevant_sets, ks=(3,), metrics=("recall",))
    assert out["recall@3"] == pytest.approx(1.0)


def test_evaluate_all_empty_returns_zero():
    out = evaluate([[1, 2]], [set()], ks=(2,), metrics=("recall",))
    assert out["recall@2"] == 0.0


def test_evaluate_multiple_ks_and_metrics():
    out = evaluate([RANKED], [RELEVANT], ks=(3, 5), metrics=("recall", "ndcg", "mrr"))
    assert set(out) == {"recall@3", "recall@5", "ndcg@3", "ndcg@5", "mrr@3", "mrr@5"}


def test_evaluate_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate([[1, 2]], [{1}, {2}])


def test_evaluate_unknown_metric_raises():
    with pytest.raises(ValueError):
        evaluate([RANKED], [RELEVANT], metrics=("precision",))
