"""Unit tests for the two-tower eval scoring path (src/training/train_two_tower.py).

Uses hand-set embedding weights so the brute-force dot-product ranking is
closed-form and hand-verifiable. Pins:
  * seen items are excluded from the ranked output (verified via set
    membership, not just shape, since exclusion uses -inf masking not removal)
  * ranking order matches the dot-product scores
  * evaluate_split wires ranking output into src.eval.metrics.evaluate correctly
"""

from __future__ import annotations

import torch

from src.models.two_tower import TwoTowerModel
from src.training.train_two_tower import _score_and_rank, evaluate_split


def _hand_set_model() -> TwoTowerModel:
    """1 user, 5 items, D=2. user0 . item_j hand-computed:
    item0=1.0, item1=0.5, item2=2.0, item3=-0.5, item4=-1.0
    """
    model = TwoTowerModel(n_users=1, n_items=5, embedding_dim=2)
    model.user_tower.embedding.weight.data = torch.tensor([[1.0, 0.0]])
    model.item_tower.embedding.weight.data = torch.tensor(
        [
            [1.0, 0.0],   # item0 -> 1.0
            [0.5, 0.0],   # item1 -> 0.5
            [2.0, 0.0],   # item2 -> 2.0
            [-0.5, 0.0],  # item3 -> -0.5
            [-1.0, 0.0],  # item4 -> -1.0
        ]
    )
    return model


class TestScoreAndRank:
    def test_ranks_by_descending_dot_product(self):
        model = _hand_set_model()
        ranked = _score_and_rank(
            model, user_ids=[0], seen_by_user={}, device=torch.device("cpu"), n=5
        )
        assert ranked == [[2, 0, 1, 3, 4]]

    def test_excludes_seen_items(self):
        model = _hand_set_model()
        ranked = _score_and_rank(
            model,
            user_ids=[0],
            seen_by_user={0: frozenset({0})},
            device=torch.device("cpu"),
            n=3,
        )
        # item0 excluded -> top 3 of the remainder: item2, item1, item3
        assert 0 not in ranked[0]
        assert ranked[0] == [2, 1, 3]

    def test_respects_n_cap(self):
        model = _hand_set_model()
        ranked = _score_and_rank(
            model, user_ids=[0], seen_by_user={}, device=torch.device("cpu"), n=2
        )
        assert ranked == [[2, 0]]


class TestEvaluateSplit:
    def test_returns_expected_metric_keys_in_unit_range(self):
        model = _hand_set_model()
        metrics = evaluate_split(
            model,
            user_ids=[0],
            relevant_sets=[{2}],
            seen_by_user={},
            device=torch.device("cpu"),
            n=5,
        )
        assert set(metrics.keys()) == {
            "recall@10", "recall@100", "ndcg@10", "ndcg@100", "mrr@10", "mrr@100",
        }
        for value in metrics.values():
            assert 0.0 <= value <= 1.0

    def test_top_ranked_relevant_item_scores_perfectly(self):
        # item2 is both the top-ranked item and the only relevant item
        model = _hand_set_model()
        metrics = evaluate_split(
            model,
            user_ids=[0],
            relevant_sets=[{2}],
            seen_by_user={},
            device=torch.device("cpu"),
            n=5,
        )
        assert metrics["recall@10"] == 1.0
        assert metrics["mrr@10"] == 1.0
        assert metrics["ndcg@10"] == 1.0
