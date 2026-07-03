"""Random and popularity-weighted negative sampling for the two-tower retriever.

Explicit negatives here are additive to in-batch negatives, not a replacement.
The future training loop draws K explicit negatives per (user, positive) row
from this module and concatenates them with the other positives already in
the minibatch (in-batch negatives) to form the full negative set for each
anchor user — batch construction and the loss itself live in the training
loop, not here.

Exclusion is built from ALL train rows regardless of label (see
build_user_positive_matrix), matching run_baselines.py's _build_seen_items
convention: a user having seen an item (even a label=0 low-play listen)
disqualifies it as a "negative" the user hasn't encountered.

Popularity weighting intentionally diverges from PopularityBaseline
(src/baselines/popularity.py): that class excludes zero-positive items
entirely from its ranking, which is correct for *recommending* (never
recommend something nobody liked). A negative sampler must do the opposite —
cold items still need occasional sampling as negatives, or their embeddings
never receive a gradient update and stay stuck near random init.

Note: run_baselines.py's _build_seen_items (dict of frozensets) and this
module's build_user_positive_matrix (sparse bool matrix) compute overlapping
information in different representations. This is known duplication, not
consolidated here — out of scope for this task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_user_positive_matrix(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
) -> sp.csr_matrix:
    """Boolean (n_users, n_items) sparse matrix, True at every (user, item)
    pair present in train_df, regardless of label.

    Duplicate (user, item) rows are harmless: scipy sums duplicate entries
    for numeric dtypes, but any nonzero entry reads as truthy in a bool
    matrix either way, so no groupby/dedup step is needed here (contrast
    with mf.py's confidence matrix, which needs exact counts).
    """
    rows = train_df[user_col].to_numpy()
    cols = train_df[item_col].to_numpy()
    data = np.ones(len(train_df), dtype=bool)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=bool)


def _sample_batch_with_exclusion(
    draw_fn: Callable[[int], np.ndarray],
    user_idx_batch: np.ndarray,
    n_per_user: int,
    exclude: sp.csr_matrix,
    positive_item_batch: np.ndarray | None,
    oversample_factor: float,
    max_resample_rounds: int,
) -> np.ndarray:
    """Draw n_per_user valid (unseen, non-duplicate) items per user in the batch.

    draw_fn(n) -> flat array of n raw item-index candidates from whatever
    distribution the caller wants (uniform for random sampling, weighted for
    popularity sampling) is the only thing that differs between samplers;
    everything else here is shared.

    Each round draws K = ceil(n_per_user * oversample_factor) candidates per
    still-deficient user, marks a candidate invalid if it's seen, is the
    batch's own positive, OR is a later repeat of a value already drawn
    earlier in the same row (draws are with replacement, so repeats are
    expected — especially under a skewed popularity distribution where the
    same head item can be drawn several times in one K-sized batch; without
    this dedup step, two valid-but-equal draws would occupy two of the top
    n_per_user slots and starve a genuinely-satisfiable row into needless
    retries). The first n_per_user valid candidates in draw order are kept
    via a stable argsort on the combined invalid mask, and rows still short
    retry — redrawing ONLY the still-deficient rows, never the whole batch —
    up to max_resample_rounds times. Each retry round draws (round_idx + 1)
    times the base K: a fixed-size retry has the SAME expected shortfall as
    the round that just failed (e.g. under a skewed popularity distribution,
    a user missing its two most-probable items as valid negatives may have
    an expected-unique-draws count below n_per_user in every fixed-size
    round), so retries only make progress if they draw more candidates each
    time. The retry loop is a Python loop, but it is bounded by pathological
    stragglers (expected empty in normal operation), never by batch or
    dataset size, so it does not become a per-sample or per-batch
    scalability bottleneck.
    """
    user_idx_batch = np.asarray(user_idx_batch)
    n_batch = len(user_idx_batch)
    k = max(n_per_user, int(np.ceil(n_per_user * oversample_factor)))

    selected = np.full((n_batch, n_per_user), -1, dtype=np.int64)
    active_rows = np.arange(n_batch)

    for _round in range(max_resample_rounds + 1):
        if len(active_rows) == 0:
            break

        n_active = len(active_rows)
        k_round = k * (_round + 1)
        candidates = draw_fn(n_active * k_round).reshape(n_active, k_round)
        active_users = user_idx_batch[active_rows]

        invalid_mask = np.asarray(
            exclude[np.repeat(active_users, k_round), candidates.ravel()]
        ).reshape(n_active, k_round)

        if positive_item_batch is not None:
            active_positives = np.asarray(positive_item_batch)[active_rows]
            invalid_mask = invalid_mask | (candidates == active_positives[:, None])

        # Flag later repeats of a value already drawn earlier in the row.
        # Sort by value (stable -> ties keep original draw order), flag
        # non-first entries in each equal-value run, then scatter back to
        # original positions — fully vectorized, no per-row python loop.
        value_order = np.argsort(candidates, axis=1, kind="stable")
        sorted_vals = np.take_along_axis(candidates, value_order, axis=1)
        dup_in_sorted = np.zeros_like(sorted_vals, dtype=bool)
        dup_in_sorted[:, 1:] = sorted_vals[:, 1:] == sorted_vals[:, :-1]
        dup_mask = np.empty_like(dup_in_sorted)
        np.put_along_axis(dup_mask, value_order, dup_in_sorted, axis=1)
        invalid_mask = invalid_mask | dup_mask

        # Stable sort pushes invalid candidates to the row-end while
        # preserving draw order among the valid ones.
        perm = np.argsort(invalid_mask, axis=1, kind="stable")
        ordered = np.take_along_axis(candidates, perm, axis=1)
        ordered_invalid = np.take_along_axis(invalid_mask, perm, axis=1)

        take = ordered[:, :n_per_user]
        take_invalid = ordered_invalid[:, :n_per_user]

        row_deficient = take_invalid.any(axis=1)

        good_local = np.where(~row_deficient)[0]
        selected[active_rows[good_local]] = take[good_local]

        active_rows = active_rows[row_deficient]

    if len(active_rows) > 0:
        raise RuntimeError(
            f"could not find {n_per_user} valid negatives for users "
            f"{user_idx_batch[active_rows].tolist()} after "
            f"{max_resample_rounds} resample rounds"
        )

    return selected.astype(np.int32)


class RandomNegativeSampler:
    """Draws negatives uniformly at random from the full item catalog.

    Usage
    -----
        train_df = pd.read_parquet("data/processed/flat/50m/likes_train.parquet")
        vocab = json.load(open("data/processed/flat/50m/likes_vocab.json"))

        sampler = RandomNegativeSampler(seed=42)
        sampler.fit(train_df, n_users=vocab["n_users"], n_items=vocab["n_items"])
        negs = sampler.sample_batch(user_idx_batch, n_per_user=5)
    """

    def __init__(
        self,
        seed: int | None = None,
        oversample_factor: float = 1.5,
        max_resample_rounds: int = 10,
    ) -> None:
        self.seed = seed
        self.oversample_factor = oversample_factor
        self.max_resample_rounds = max_resample_rounds
        self._rng = np.random.default_rng(seed)
        self.n_items: int | None = None
        self.exclude: sp.csr_matrix | None = None

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int) -> None:
        self.n_items = n_items
        self.exclude = build_user_positive_matrix(train_df, n_users, n_items)

    def sample_batch(
        self,
        user_idx_batch: np.ndarray,
        n_per_user: int,
        positive_item_batch: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.exclude is None:
            raise RuntimeError("RandomNegativeSampler.sample_batch() called before fit()")

        def draw_fn(n: int) -> np.ndarray:
            return self._rng.integers(0, self.n_items, size=n, dtype=np.int64)

        return _sample_batch_with_exclusion(
            draw_fn=draw_fn,
            user_idx_batch=np.asarray(user_idx_batch),
            n_per_user=n_per_user,
            exclude=self.exclude,
            positive_item_batch=(
                None if positive_item_batch is None else np.asarray(positive_item_batch)
            ),
            oversample_factor=self.oversample_factor,
            max_resample_rounds=self.max_resample_rounds,
        )

    def sample(self, user_idx: int, n: int, positive_item: int | None = None) -> np.ndarray:
        positive_batch = None if positive_item is None else np.array([positive_item])
        return self.sample_batch(np.array([user_idx]), n, positive_batch)[0]


class PopularityNegativeSampler:
    """Draws negatives weighted by global item popularity (dampened).

    Weighting is (count + 1) ** power then normalized, where count is the
    number of label=1 training rows for that item — same filter as
    PopularityBaseline.fit() (src/baselines/popularity.py), but with
    Laplace smoothing so zero-positive ("cold") items still get sampled
    occasionally. PopularityBaseline can afford to exclude cold items
    entirely because it's ranking things to *recommend*; this sampler
    cannot, because an item that never appears as a negative never gets a
    gradient update and its embedding stays stuck near random init.

    power=0.75 is the word2vec / YouTube-DNN convention for dampening head
    skew; power=0.0 gives a uniform distribution (Laplace-smoothed counts
    all become 1); power=1.0 samples proportional to raw popularity.

    Usage
    -----
        train_df = pd.read_parquet("data/processed/flat/50m/likes_train.parquet")
        vocab = json.load(open("data/processed/flat/50m/likes_vocab.json"))

        sampler = PopularityNegativeSampler(power=0.75, seed=42)
        sampler.fit(train_df, n_users=vocab["n_users"], n_items=vocab["n_items"])
        negs = sampler.sample_batch(user_idx_batch, n_per_user=5)
    """

    def __init__(
        self,
        power: float = 0.75,
        seed: int | None = None,
        oversample_factor: float = 1.5,
        max_resample_rounds: int = 10,
    ) -> None:
        self.power = power
        self.seed = seed
        self.oversample_factor = oversample_factor
        self.max_resample_rounds = max_resample_rounds
        self._rng = np.random.default_rng(seed)
        self.n_items: int | None = None
        self.exclude: sp.csr_matrix | None = None
        self.probs: np.ndarray | None = None
        self.cumprobs: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, n_users: int, n_items: int) -> None:
        self.n_items = n_items
        self.exclude = build_user_positive_matrix(train_df, n_users, n_items)

        counts = train_df[train_df["label"] == 1]["item_idx"].value_counts()
        freq = np.zeros(n_items, dtype=np.float64)
        freq[counts.index.to_numpy()] = counts.to_numpy()

        weights = (freq + 1.0) ** self.power
        self.probs = weights / weights.sum()
        self.cumprobs = np.cumsum(self.probs)

    def sample_batch(
        self,
        user_idx_batch: np.ndarray,
        n_per_user: int,
        positive_item_batch: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.exclude is None:
            raise RuntimeError("PopularityNegativeSampler.sample_batch() called before fit()")

        def draw_fn(n: int) -> np.ndarray:
            u = self._rng.random(n)
            # side="right" can return n_items when u lands past cumprobs[-1]
            # due to floating-point rounding of the cumulative sum to <1.0.
            idx = np.searchsorted(self.cumprobs, u, side="right")
            return np.clip(idx, 0, self.n_items - 1)

        return _sample_batch_with_exclusion(
            draw_fn=draw_fn,
            user_idx_batch=np.asarray(user_idx_batch),
            n_per_user=n_per_user,
            exclude=self.exclude,
            positive_item_batch=(
                None if positive_item_batch is None else np.asarray(positive_item_batch)
            ),
            oversample_factor=self.oversample_factor,
            max_resample_rounds=self.max_resample_rounds,
        )

    def sample(self, user_idx: int, n: int, positive_item: int | None = None) -> np.ndarray:
        positive_batch = None if positive_item is None else np.array([positive_item])
        return self.sample_batch(np.array([user_idx]), n, positive_batch)[0]


# ---------------------------------------------------------------------------
# Smoke test — samples a few negatives against real processed data if present
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _event, _size, _fmt = "likes", "50m", "flat"
    _root = Path("data/processed") / _fmt / _size
    _train_path = _root / f"{_event}_train.parquet"
    _vocab_path = _root / f"{_event}_vocab.json"

    if not _train_path.exists() or not _vocab_path.exists():
        print(
            f"{_train_path} or {_vocab_path} not found. Preprocess first:\n"
            f"    python -m src.data.preprocess --event {_event}"
        )
    else:
        _train_df = pd.read_parquet(_train_path)
        _vocab = json.load(open(_vocab_path))

        _random = RandomNegativeSampler(seed=42)
        _random.fit(_train_df, n_users=_vocab["n_users"], n_items=_vocab["n_items"])

        _popularity = PopularityNegativeSampler(power=0.75, seed=42)
        _popularity.fit(_train_df, n_users=_vocab["n_users"], n_items=_vocab["n_items"])

        for _uid in range(min(3, _vocab["n_users"])):
            print(f"user {_uid}:")
            print(f"  random negatives:     {_random.sample(_uid, n=5).tolist()}")
            print(f"  popularity negatives: {_popularity.sample(_uid, n=5).tolist()}")
