"""Matrix factorisation baseline using Implicit ALS (Alternating Least Squares).

Implicit ALS (Hu, Koren & Volinsky 2008) factorises a *confidence*-weighted
user × item matrix rather than raw ratings. Confidence = 1 + α × interaction
count, so items a user interacted with many times get a higher training signal
than items they touched once — but we never treat absence of interaction as a
hard negative (the model just assigns lower confidence to unseen pairs).

This is the correct approach for Yambda's likes/listens data: there are no
explicit ratings, only implicit signals. Standard SVD or NMF would treat
unobserved (user, item) pairs as zero ratings, which is wrong.

Reference: Hu, Koren & Volinsky, "Collaborative Filtering for Implicit
Feedback Datasets", ICDM 2008.

Usage
-----
    import json, pandas as pd
    from src.baselines.mf import MFBaseline

    train_df = pd.read_parquet("data/processed/flat/50m/likes_train.parquet")
    vocab    = json.load(open("data/processed/flat/50m/likes_vocab.json"))

    model = MFBaseline(factors=64, iterations=20)
    model.fit(train_df, n_users=vocab["n_users"], n_items=vocab["n_items"])

    seen = frozenset(train_df[train_df["user_idx"] == 0]["item_idx"].tolist())
    recs = model.recommend(user_idx=0, n=100, seen_items=seen)
"""

from __future__ import annotations

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp


class MFBaseline:
    """Implicit ALS matrix factorisation baseline.

    Parameters
    ----------
    factors : int
        Embedding dimension for user and item latent vectors.
    iterations : int
        Number of ALS sweeps. 20 is a good default; more rarely helps.
    alpha : float
        Confidence scaling factor. Confidence = 1 + alpha * interaction_count.
        40.0 is the value from the original Hu et al. paper.
    """

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 20,
        alpha: float = 40.0,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.alpha = alpha
        self.model: implicit.als.AlternatingLeastSquares | None = None
        self.user_items: sp.csr_matrix | None = None

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int) -> None:
        """Build confidence matrix and train ALS.

        Confidence = 1 + alpha * count, where count is the number of label=1
        interactions for a given (user, item) pair. Only positive interactions
        contribute — label=0 rows have no signal and are excluded.

        The resulting user_items matrix is kept on self for use by recommend().
        """
        counts = (
            train_df[train_df["label"] == 1]
            .groupby(["user_idx", "item_idx"], sort=False)
            .size()
            .reset_index(name="count")
        )
        confidence = (1.0 + self.alpha * counts["count"]).astype("float32")
        user_items = sp.csr_matrix(
            (confidence.values, (counts["user_idx"].values, counts["item_idx"].values)),
            shape=(n_users, n_items),
            dtype="float32",
        )
        self.user_items = user_items

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            random_state=42,
        )
        self.model.fit(user_items)

    def recommend(
        self,
        user_idx: int,
        n: int,
        seen_items: frozenset[int] = frozenset(),
    ) -> list[int]:
        """Return up to n items ranked by ALS dot-product score.

        We overfetch by len(seen_items) so there are enough candidates left
        after filtering. We filter manually (rather than using implicit's
        filter_already_liked_items=True) because the training matrix only
        contains label=1 rows; seen_items is built from all train rows, which
        gives exact control over what gets excluded.
        """
        if self.model is None or self.user_items is None:
            raise RuntimeError("MFBaseline.recommend() called before fit()")

        n_fetch = n + len(seen_items)
        ids, _ = self.model.recommend(
            user_idx,
            self.user_items[user_idx],
            N=n_fetch,
            filter_already_liked_items=False,
        )
        return [int(i) for i in ids if int(i) not in seen_items][:n]
