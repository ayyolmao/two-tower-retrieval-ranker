"""Two-tower retrieval model: user tower + item tower, dot-product scoring.

Bare ID-embedding lookup towers — no MLP, no side features. A dot product of
two learned embedding tables is architecturally equivalent to the MF baseline
(src/baselines/mf.py, implicit ALS) this model must beat, but trained with
in-batch softmax + explicit negative sampling (src/data/negative_sampling.py),
signal ALS's confidence-weighted least-squares doesn't use. Deeper towers
(MLP projection, history-aware sequence encoders) are future ablation work,
not v1 — CLAUDE.md's ablation list treats embedding_dim and history-length as
separate, later tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Embedding lookup: user_idx -> D-dim vector."""

    def __init__(self, n_users: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_users, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=embedding_dim**-0.5)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        """(B,) int64 -> (B, D) float32."""
        return self.embedding(user_idx)


class ItemTower(nn.Module):
    """Embedding lookup: item_idx -> D-dim vector. Same structure as
    UserTower, separate embedding table over items."""

    def __init__(self, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=embedding_dim**-0.5)

    def forward(self, item_idx: torch.Tensor) -> torch.Tensor:
        """(B,) int64 -> (B, D) float32."""
        return self.embedding(item_idx)


class TwoTowerModel(nn.Module):
    """Holds both towers. No fused forward — the training loop calls
    user_tower(...) / item_tower(...) directly so it can build the
    B x (B+K) in-batch-plus-explicit-negatives logits matrix itself.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_tower = UserTower(n_users, embedding_dim)
        self.item_tower = ItemTower(n_items, embedding_dim)
        self.n_items = n_items
        self.embedding_dim = embedding_dim

    def all_item_embeddings(self) -> torch.Tensor:
        """(n_items, D) full catalog embedding matrix, for brute-force eval
        scoring. Caller wraps in torch.no_grad()."""
        device = self.item_tower.embedding.weight.device
        return self.item_tower(torch.arange(self.n_items, device=device))


def in_batch_logits(
    user_emb: torch.Tensor, pos_emb: torch.Tensor, pos_item_idx: torch.Tensor
) -> torch.Tensor:
    """Raw dot-product similarity of every user against every batch positive.

    (B, D), (B, D), (B,) -> (B, B). Row i, col j = dot(user_i, pos_j); the
    diagonal is user i's true positive, every off-diagonal entry is an
    in-batch negative -- these fall out of batching itself, no separate
    construction needed.

    If two different anchor rows share the same true positive item (a real
    possibility with a large catalog and popular head items), the shared
    item is masked to -inf off the diagonal: otherwise a legitimately good
    recommendation for row j would be scored as a hard negative for row i
    just because it happens to be row j's target, too.
    """
    logits = user_emb @ pos_emb.T
    b = user_emb.shape[0]
    same_item = pos_item_idx.unsqueeze(0) == pos_item_idx.unsqueeze(1)
    off_diagonal = ~torch.eye(b, dtype=torch.bool, device=logits.device)
    return logits.masked_fill(same_item & off_diagonal, float("-inf"))


def sampled_softmax_loss(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    pos_item_idx: torch.Tensor,
    neg_emb: torch.Tensor | None,
    return_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy loss over in-batch negatives plus optional explicit
    negatives, target class = each row's own positive (the diagonal).

    neg_emb, if given, is (B, K, D) -- explicit negatives from
    src/data/negative_sampling.py's samplers, scored only against their own
    row (never cross-batched like the in-batch block is), then concatenated
    onto the in-batch logits to form a (B, B+K) matrix. Pass neg_emb=None to
    train on in-batch negatives alone.
    """
    logits = in_batch_logits(user_emb, pos_emb, pos_item_idx)
    if neg_emb is not None:
        logits_explicit = torch.einsum("bd,bkd->bk", user_emb, neg_emb)
        logits = torch.cat([logits, logits_explicit], dim=1)
    targets = torch.arange(user_emb.shape[0], device=user_emb.device)
    loss = F.cross_entropy(logits, targets)
    if return_logits:
        return loss, logits
    return loss
