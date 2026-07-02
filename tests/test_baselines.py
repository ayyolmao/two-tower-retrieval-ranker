"""Unit tests for popularity and MF baselines.

All tests use a tiny synthetic DataFrame (5 users, 10 items) so they run in
milliseconds. The goal is to pin down the contracts that downstream code and
the evaluation loop depend on — not to verify that the models achieve good
Recall@100 (that's a data question, not a code question).

Key invariants being tested:
  * Popularity orders items by label=1 count, not by total interactions.
  * Both models always exclude seen_items from their output.
  * Output length is exactly min(n, available_candidates).
  * No duplicate items in any recommendation list.
  * The end-to-end eval loop (fit → recommend → evaluate) produces finite
    metrics in [0, 1] for both models.
"""

from __future__ import annotations

import pandas as pd
import pytest

import json
from pathlib import Path

from src.baselines.mf import MFBaseline
from src.baselines.popularity import PopularityBaseline
from src.baselines.run_baselines import _check_split_strategy
from src.eval.metrics import evaluate


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_train_df(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    """rows: (user_idx, item_idx, label)."""
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])
    df["timestamp"] = range(len(df))
    df["is_organic"] = 1
    df["label"] = df["label"].astype("int8")
    df["user_idx"] = df["user_idx"].astype("int32")
    df["item_idx"] = df["item_idx"].astype("int32")
    return df


N_USERS = 5
N_ITEMS = 10

# item 2 has 3 positives, item 5 has 2, item 7 has 1, item 3 has 0 positives
TRAIN_ROWS = [
    (0, 2, 1), (0, 5, 1), (0, 3, 0),
    (1, 2, 1), (1, 7, 1),
    (2, 2, 1), (2, 5, 1),
    (3, 7, 0), (3, 3, 0),
    (4, 5, 0),
]

TRAIN_DF = _make_train_df(TRAIN_ROWS)


# ---------------------------------------------------------------------------
# PopularityBaseline
# ---------------------------------------------------------------------------

class TestPopularityBaseline:
    def test_top_item_is_most_common_positive(self):
        model = PopularityBaseline()
        model.fit(TRAIN_DF)
        # item 2: 3 label=1 rows → most popular
        assert model.top_items[0] == 2

    def test_ranking_order(self):
        model = PopularityBaseline()
        model.fit(TRAIN_DF)
        # item 2 (3 hits) > item 5 (2 hits) > item 7 (1 hit)
        top3 = model.top_items[:3]
        assert top3 == [2, 5, 7]

    def test_zero_positive_items_excluded(self):
        model = PopularityBaseline()
        model.fit(TRAIN_DF)
        # items 3 and 4 have no label=1 rows
        assert 3 not in model.top_items
        assert 4 not in model.top_items

    def test_recommend_excludes_seen(self):
        model = PopularityBaseline()
        model.fit(TRAIN_DF)
        seen = frozenset([2])   # most popular item → must be skipped
        recs = model.recommend(0, n=3, seen_items=seen)
        assert 2 not in recs

    def test_recommend_returns_n_items(self):
        model = PopularityBaseline()
        model.fit(TRAIN_DF)
        recs = model.recommend(0, n=2)
        assert len(recs) == 2

    def test_recommend_no_duplicates(self):
        model = PopularityBaseline()
        model.fit(TRAIN_DF)
        recs = model.recommend(0, n=3)
        assert len(recs) == len(set(recs))

    def test_recommend_fewer_than_n_when_candidates_exhausted(self):
        model = PopularityBaseline()
        model.fit(TRAIN_DF)
        # only 3 items have label=1; requesting 10 should return ≤3
        recs = model.recommend(0, n=10, seen_items=frozenset())
        assert len(recs) <= 3

    def test_user_idx_unused(self):
        # popularity is non-personalised — different users get the same list
        model = PopularityBaseline()
        model.fit(TRAIN_DF)
        assert model.recommend(0, n=3) == model.recommend(99, n=3)


# ---------------------------------------------------------------------------
# MFBaseline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_mf() -> MFBaseline:
    model = MFBaseline(factors=4, iterations=2, alpha=10.0)
    model.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
    return model


class TestMFBaseline:
    def test_fit_completes_without_error(self):
        model = MFBaseline(factors=2, iterations=1)
        model.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        assert model.model is not None

    def test_recommend_returns_n_items(self, fitted_mf):
        recs = fitted_mf.recommend(0, n=3)
        assert len(recs) == 3

    def test_recommend_returns_valid_item_indices(self, fitted_mf):
        recs = fitted_mf.recommend(0, n=5)
        for item in recs:
            assert 0 <= item < N_ITEMS

    def test_recommend_excludes_seen(self, fitted_mf):
        seen = frozenset([0, 1, 2, 3])
        recs = fitted_mf.recommend(0, n=3, seen_items=seen)
        for item in recs:
            assert item not in seen

    def test_recommend_no_duplicates(self, fitted_mf):
        recs = fitted_mf.recommend(0, n=5)
        assert len(recs) == len(set(recs))

    def test_recommend_fewer_than_n_when_mostly_seen(self, fitted_mf):
        # N_ITEMS=10, seen=8 items → at most 2 candidates left
        seen = frozenset(range(8))
        recs = fitted_mf.recommend(0, n=5, seen_items=seen)
        assert len(recs) <= 2

    def test_unfitted_model_raises(self):
        model = MFBaseline()
        with pytest.raises(RuntimeError, match="fit()"):
            model.recommend(0, n=5)


# ---------------------------------------------------------------------------
# End-to-end: fit → recommend → evaluate
# ---------------------------------------------------------------------------

class TestEndToEndEval:
    """Smoke-test the full eval loop with both baselines on toy data.

    We don't assert specific metric values — those depend on model quality.
    We only assert that metrics are finite floats in [0, 1].
    """

    def _check_metrics(self, metrics: dict[str, float]) -> None:
        assert set(metrics.keys()) == {
            "recall@10", "recall@100", "ndcg@10", "ndcg@100", "mrr@10", "mrr@100"
        }
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def _eval(self, model: PopularityBaseline | MFBaseline) -> dict[str, float]:
        # val: user 0 should see item 5 as relevant (not seen in train mock)
        val_df = pd.DataFrame({
            "user_idx": [0, 1],
            "item_idx": [5, 7],
            "label":    [1,  1],
        })
        seen_by_user = {
            uid: frozenset(group["item_idx"].tolist())
            for uid, group in TRAIN_DF.groupby("user_idx")
        }
        user_ids = [0, 1]
        relevant_sets = [{5}, {7}]
        ranked_lists = [
            model.recommend(uid, n=100, seen_items=seen_by_user.get(uid, frozenset()))
            for uid in user_ids
        ]
        return evaluate(ranked_lists, relevant_sets, ks=(10, 100))

    def test_popularity_eval_loop(self):
        model = PopularityBaseline()
        model.fit(TRAIN_DF)
        self._check_metrics(self._eval(model))

    def test_mf_eval_loop(self):
        model = MFBaseline(factors=2, iterations=1)
        model.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        self._check_metrics(self._eval(model))


# ---------------------------------------------------------------------------
# Split strategy guard
# ---------------------------------------------------------------------------

class TestCheckSplitStrategy:
    def test_raises_if_meta_file_missing(self, tmp_path: Path):
        # Directory exists but no split_meta.json → must error immediately
        (tmp_path / "flat" / "50m").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="split_meta.json"):
            _check_split_strategy("likes", "50m", "flat", tmp_path)

    def test_raises_if_wrong_strategy(self, tmp_path: Path):
        # Meta file present but strategy is global_time → must error
        split_dir = tmp_path / "flat" / "50m"
        split_dir.mkdir(parents=True)
        (split_dir / "likes_split_meta.json").write_text(
            json.dumps({"strategy": "global_time", "rows": {}})
        )
        with pytest.raises(RuntimeError, match="global_time"):
            _check_split_strategy("likes", "50m", "flat", tmp_path)

    def test_passes_for_user_time_strategy(self, tmp_path: Path):
        split_dir = tmp_path / "flat" / "50m"
        split_dir.mkdir(parents=True)
        (split_dir / "likes_split_meta.json").write_text(
            json.dumps({"strategy": "user_time", "rows": {}})
        )
        _check_split_strategy("likes", "50m", "flat", tmp_path)  # must not raise
