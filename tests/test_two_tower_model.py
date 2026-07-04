"""Unit tests for the two-tower model (src/models/two_tower.py).

Pure tensor-shape/logic tests on tiny synthetic vocabs and hand-set weights —
no real data, no GPU required. The invariants pinned here:
  * Each tower is a plain embedding lookup: (B,) int64 indices -> (B, D) float32.
  * The in-batch logits matrix is the raw dot product of user/positive-item
    embeddings, with a false-negative mask so two anchors sharing the same
    true positive item don't score each other's positive as a hard negative.
  * Explicit negatives (from src/data/negative_sampling.py) are scored only
    against their own row, never cross-batched like in-batch negatives are.
  * The full cross-entropy loss pipeline produces a finite scalar and can
    actually drive the loss down on a tiny overfit-able batch.
"""

from __future__ import annotations

import torch

from src.models.two_tower import (
    ItemTower,
    TwoTowerModel,
    UserTower,
    in_batch_logits,
    sampled_softmax_loss,
)


class TestUserTower:
    def test_forward_returns_batch_by_dim_shape(self):
        tower = UserTower(n_users=5, embedding_dim=8)
        user_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        out = tower(user_idx)
        assert out.shape == (3, 8)
        assert out.dtype == torch.float32

    def test_distinct_indices_produce_distinct_rows(self):
        tower = UserTower(n_users=5, embedding_dim=8)
        out = tower(torch.tensor([0, 1], dtype=torch.long))
        assert not torch.allclose(out[0], out[1])

    def test_gradient_flows_to_embedding_weight(self):
        tower = UserTower(n_users=5, embedding_dim=8)
        out = tower(torch.tensor([0, 1], dtype=torch.long))
        out.sum().backward()
        assert tower.embedding.weight.grad is not None


class TestItemTower:
    def test_forward_returns_batch_by_dim_shape(self):
        tower = ItemTower(n_items=10, embedding_dim=8)
        item_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        out = tower(item_idx)
        assert out.shape == (3, 8)
        assert out.dtype == torch.float32


class TestTwoTowerModel:
    def test_holds_both_towers(self):
        model = TwoTowerModel(n_users=5, n_items=10, embedding_dim=8)
        assert isinstance(model.user_tower, UserTower)
        assert isinstance(model.item_tower, ItemTower)

    def test_all_item_embeddings_returns_full_catalog_matrix(self):
        model = TwoTowerModel(n_users=5, n_items=10, embedding_dim=8)
        embs = model.all_item_embeddings()
        assert embs.shape == (10, 8)

    def test_all_item_embeddings_matches_item_tower_forward(self):
        model = TwoTowerModel(n_users=5, n_items=10, embedding_dim=8)
        all_embs = model.all_item_embeddings()
        direct = model.item_tower(torch.arange(10, dtype=torch.long))
        torch.testing.assert_close(all_embs, direct)


# ---------------------------------------------------------------------------
# in_batch_logits: raw dot product + false-negative masking
# ---------------------------------------------------------------------------

# B=4, D=3 hand-set embeddings so every dot product is hand-computable.
_USER_EMB = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)
_POS_EMB = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 1.0],
    ]
)
# logits[i][j] = dot(user_i, pos_j) -- hand-computed:
_EXPECTED_LOGITS = torch.tensor(
    [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 2.0, 0.0, 1.0],
        [0.0, 0.0, 3.0, 1.0],
        [1.0, 2.0, 3.0, 3.0],
    ]
)


class TestInBatchLogits:
    def test_matches_hand_computed_dot_products(self):
        # distinct positive items -> no false-negative masking kicks in
        pos_item_idx = torch.tensor([10, 20, 30, 40])
        logits = in_batch_logits(_USER_EMB, _POS_EMB, pos_item_idx)
        torch.testing.assert_close(logits, _EXPECTED_LOGITS)

    def test_shared_positive_item_masks_off_diagonal_entries(self):
        # rows 0 and 2 share item 7 -> [0,2] and [2,0] must become -inf
        pos_item_idx = torch.tensor([7, 3, 7, 9])
        logits = in_batch_logits(_USER_EMB, _POS_EMB, pos_item_idx)
        assert logits[0, 2] == float("-inf")
        assert logits[2, 0] == float("-inf")

    def test_shared_positive_item_leaves_diagonal_untouched(self):
        pos_item_idx = torch.tensor([7, 3, 7, 9])
        logits = in_batch_logits(_USER_EMB, _POS_EMB, pos_item_idx)
        assert logits[0, 0] == _EXPECTED_LOGITS[0, 0]
        assert logits[2, 2] == _EXPECTED_LOGITS[2, 2]

    def test_shared_positive_item_leaves_unrelated_entries_untouched(self):
        pos_item_idx = torch.tensor([7, 3, 7, 9])
        logits = in_batch_logits(_USER_EMB, _POS_EMB, pos_item_idx)
        assert logits[0, 1] == _EXPECTED_LOGITS[0, 1]
        assert logits[1, 3] == _EXPECTED_LOGITS[1, 3]


# ---------------------------------------------------------------------------
# sampled_softmax_loss: concatenates in-batch + explicit-negative logits,
# cross-entropy against the diagonal as target.
# ---------------------------------------------------------------------------

class TestSampledSoftmaxLoss:
    def test_logits_shape_is_batch_by_batch_plus_k(self):
        torch.manual_seed(0)
        b, k, d = 4, 3, 8
        user_emb = torch.randn(b, d)
        pos_emb = torch.randn(b, d)
        neg_emb = torch.randn(b, k, d)
        pos_item_idx = torch.tensor([1, 2, 3, 4])

        loss, logits = sampled_softmax_loss(
            user_emb, pos_emb, pos_item_idx, neg_emb, return_logits=True
        )
        assert logits.shape == (b, b + k)

    def test_loss_is_finite_scalar(self):
        torch.manual_seed(0)
        b, k, d = 4, 3, 8
        user_emb = torch.randn(b, d)
        pos_emb = torch.randn(b, d)
        neg_emb = torch.randn(b, k, d)
        pos_item_idx = torch.tensor([1, 2, 3, 4])

        loss = sampled_softmax_loss(user_emb, pos_emb, pos_item_idx, neg_emb)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_explicit_negatives_scored_only_against_own_row(self):
        # user 0's explicit negatives should not appear anywhere in another
        # row's logits -- verify by checking the explicit-negative block
        # (columns B: B+K) equals einsum("bd,bkd->bk", user_emb, neg_emb),
        # not a cross-batched B x B x K product.
        torch.manual_seed(0)
        b, k, d = 3, 2, 4
        user_emb = torch.randn(b, d)
        pos_emb = torch.randn(b, d)
        neg_emb = torch.randn(b, k, d)
        pos_item_idx = torch.tensor([1, 2, 3])

        _, logits = sampled_softmax_loss(
            user_emb, pos_emb, pos_item_idx, neg_emb, return_logits=True
        )
        expected_explicit = torch.einsum("bd,bkd->bk", user_emb, neg_emb)
        torch.testing.assert_close(logits[:, b:], expected_explicit)


class TestOverfitSmokeTest:
    def test_loss_decreases_on_tiny_repeated_batch(self):
        torch.manual_seed(0)
        n_users, n_items, embedding_dim = 5, 10, 8
        model = TwoTowerModel(n_users=n_users, n_items=n_items, embedding_dim=embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        user_idx = torch.tensor([0, 1, 2, 3, 4])
        pos_item_idx = torch.tensor([0, 1, 2, 3, 4])

        losses = []
        for _ in range(200):
            optimizer.zero_grad()
            user_emb = model.user_tower(user_idx)
            pos_emb = model.item_tower(pos_item_idx)
            loss = sampled_softmax_loss(user_emb, pos_emb, pos_item_idx, neg_emb=None)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            assert torch.isfinite(loss)

        assert losses[-1] < 0.3 * losses[0]
