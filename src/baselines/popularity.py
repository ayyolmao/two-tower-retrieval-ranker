"""Popularity baseline: recommend the globally most-liked items.

No personalization — every user gets the same ranked list (most label=1 items
across all training users), minus anything they already saw in training.

This is the floor every neural model must beat. It is surprisingly strong on
head-heavy distributions like Yambda (a small fraction of tracks dominate
most plays).

Usage
-----
    from src.baselines.popularity import PopularityBaseline
    import pandas as pd

    train_df = pd.read_parquet("data/processed/flat/50m/likes_train.parquet")

    model = PopularityBaseline()
    model.fit(train_df)

    seen = frozenset(train_df[train_df["user_idx"] == 0]["item_idx"].tolist())
    recs = model.recommend(user_idx=0, n=100, seen_items=seen)
"""

from __future__ import annotations

from itertools import islice

import pandas as pd


class PopularityBaseline:
    """Rank all items by global training popularity, filter seen, slice top-n.

    Attributes
    ----------
    top_items : list[int] | None
        Item indices sorted descending by number of label=1 training rows.
        None until :meth:`fit` is called.
    """

    def __init__(self) -> None:
        self.top_items: list[int] | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Build the global popularity ranking from training interactions.

        Only label=1 rows count — label=0 rows (low-play listens) are noise
        and would inflate the popularity of items users didn't actually enjoy.
        """
        counts = train_df[train_df["label"] == 1]["item_idx"].value_counts()
        self.top_items = counts.index.tolist()

    def recommend(
        self,
        user_idx: int,  # noqa: ARG002 — unused but kept for uniform interface
        n: int,
        seen_items: frozenset[int] = frozenset(),
    ) -> list[int]:
        """Return up to n items, skipping anything the user saw in training.

        Uses islice to stop iteration as soon as n unseen items are found,
        avoiding a full O(vocab_size) scan when only n << vocab_size items
        are needed.

        Excluding seen items matters for fair comparison: the neural two-tower
        also filters training positives from its candidate set.
        """
        if self.top_items is None:
            raise RuntimeError("PopularityBaseline.recommend() called before fit()")
        return list(islice(
            (item for item in self.top_items if item not in seen_items),
            n,
        ))
