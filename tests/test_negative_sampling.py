"""Unit tests for random and popularity-weighted negative sampling.

Explicit negative samplers are additive to in-batch negatives (the future
training loop's job) — this module only needs to draw items a user has NOT
interacted with, optionally weighted by global popularity. The invariants
pinned here:
  * A sampled negative is never one of the user's train-set interactions
    (regardless of label — label=0 rows still mean "the user saw this").
  * A sampled negative is never the batch's own positive item, even if that
    item happens to be new to the user (defensive: explicit override).
  * No duplicate items within a single sample/sample_batch call.
  * Popularity weighting favours popular items but never zeroes out cold
    items (unlike PopularityBaseline, which must exclude them).
  * Everything is reproducible given a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.negative_sampling import (
    PopularityNegativeSampler,
    RandomNegativeSampler,
    _sample_batch_with_exclusion,
    build_user_positive_matrix,
)


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
N_ITEMS = 20

# item 2: 8 positives, item 5: 4 positives, item 7: 2 positives,
# items 8-19: 0 positives (cold). User 0 has interacted with {2, 5, 3}.
TRAIN_ROWS = (
    [(0, 2, 1)] * 3 + [(0, 5, 1)] * 2 + [(0, 3, 0)]
    + [(1, 2, 1)] * 3 + [(1, 7, 1)] * 2
    + [(2, 2, 1)] * 2 + [(2, 5, 1)] * 2
    + [(3, 7, 0)] + [(3, 3, 0)]
    + [(4, 5, 0)]
)

TRAIN_DF = _make_train_df(TRAIN_ROWS)


# ---------------------------------------------------------------------------
# build_user_positive_matrix
# ---------------------------------------------------------------------------

class TestBuildUserPositiveMatrix:
    def test_matrix_marks_all_train_rows_regardless_of_label(self):
        mat = build_user_positive_matrix(TRAIN_DF, N_USERS, N_ITEMS)
        # user 0, item 3 is label=0 but still an interaction -> must be marked seen
        assert mat[0, 3]

    def test_matrix_shape_is_n_users_by_n_items(self):
        mat = build_user_positive_matrix(TRAIN_DF, N_USERS, N_ITEMS)
        assert mat.shape == (N_USERS, N_ITEMS)

    def test_matrix_dtype_is_bool(self):
        mat = build_user_positive_matrix(TRAIN_DF, N_USERS, N_ITEMS)
        assert mat.dtype == bool

    def test_user_with_no_train_rows_has_all_false_row(self):
        # user index 4 is present but user index outside TRAIN_ROWS entirely,
        # e.g. pass n_users=6 so row 5 has zero interactions.
        mat = build_user_positive_matrix(TRAIN_DF, n_users=6, n_items=N_ITEMS)
        assert not mat[5].toarray().any()


# ---------------------------------------------------------------------------
# _sample_batch_with_exclusion (private vectorized core)
# ---------------------------------------------------------------------------

def _empty_exclude(n_users: int, n_items: int):
    return build_user_positive_matrix(
        _make_train_df([]), n_users=n_users, n_items=n_items
    )


class _FixedDraw:
    """draw_fn stub that ignores n and tiles a fixed pattern to length n.

    For a same-value pattern (e.g. [4, 4]) this guarantees "always a
    duplicate" regardless of how large a retry round's K grows to.
    """

    def __init__(self, values: list[int]):
        self.values = np.array(values, dtype=np.int64)
        self.calls: list[int] = []

    def __call__(self, n: int) -> np.ndarray:
        self.calls.append(n)
        return np.resize(self.values, n)


class _QueueDraw:
    """draw_fn stub returning successive preset responses, one per call."""

    def __init__(self, responses: list[list[int]]):
        self.responses = [np.array(r, dtype=np.int64) for r in responses]
        self.calls: list[int] = []

    def __call__(self, n: int) -> np.ndarray:
        self.calls.append(n)
        response = self.responses[len(self.calls) - 1]
        assert n == len(response), f"call {len(self.calls)}: expected n={len(response)}, got {n}"
        return response


class TestSampleBatchWithExclusion:
    def test_selects_first_n_valid_candidates_per_row(self):
        # 2 users, K=5 candidates each, n_per_user=2.
        draw = _FixedDraw([5, 3, 5, 7, 2, 9, 9, 1, 4, 6])
        exclude = build_user_positive_matrix(
            _make_train_df([(0, 5, 1), (1, 9, 1)]), n_users=2, n_items=10
        )
        result = _sample_batch_with_exclusion(
            draw_fn=draw,
            user_idx_batch=np.array([0, 1]),
            n_per_user=2,
            exclude=exclude,
            positive_item_batch=None,
            oversample_factor=2.5,
            max_resample_rounds=3,
        )
        # row0 candidates [5,3,5,7,2], item 5 seen -> first 2 valid in draw order: 3,7
        # row1 candidates [9,9,1,4,6], item 9 seen -> first 2 valid in draw order: 1,4
        np.testing.assert_array_equal(result, [[3, 7], [1, 4]])
        assert draw.calls == [10]  # exactly one round, no retry needed

    def test_preserves_draw_order_among_valid_candidates(self):
        draw = _FixedDraw([9, 2, 7, 4, 1])
        exclude = _empty_exclude(n_users=1, n_items=10)
        result = _sample_batch_with_exclusion(
            draw_fn=draw,
            user_idx_batch=np.array([0]),
            n_per_user=3,
            exclude=exclude,
            positive_item_batch=None,
            oversample_factor=5 / 3,
            max_resample_rounds=3,
        )
        # no exclusions -> must keep original draw order, not sort by value
        np.testing.assert_array_equal(result, [[9, 2, 7]])

    def test_tops_up_deficient_row_via_retry_without_redrawing_satisfied_rows(self):
        # row 0: [4, 4] duplicate -> deficient; row 1: [7, 8] fine -> satisfied.
        # retry round doubles K (round_idx+1) for the still-deficient row.
        draw = _QueueDraw(responses=[[4, 4, 7, 8], [5, 6, 10, 11]])
        exclude = _empty_exclude(n_users=2, n_items=20)
        result = _sample_batch_with_exclusion(
            draw_fn=draw,
            user_idx_batch=np.array([0, 1]),
            n_per_user=2,
            exclude=exclude,
            positive_item_batch=None,
            oversample_factor=1.0,
            max_resample_rounds=3,
        )
        np.testing.assert_array_equal(result, [[5, 6], [7, 8]])
        # round 1: both rows, K=2 -> 4 draws. round 2: only the deficient row
        # (row 1 is never redrawn), K=2*2=4 -> 4 draws.
        assert draw.calls == [4, 4]

    def test_raises_runtime_error_after_max_resample_rounds_exhausted(self):
        # always returns a duplicate -> row 7 can never be satisfied.
        draw = _FixedDraw([4, 4])
        exclude = _empty_exclude(n_users=8, n_items=10)
        with pytest.raises(RuntimeError, match="7"):
            _sample_batch_with_exclusion(
                draw_fn=draw,
                user_idx_batch=np.array([7]),
                n_per_user=2,
                exclude=exclude,
                positive_item_batch=None,
                oversample_factor=1.0,
                max_resample_rounds=2,
            )


# ---------------------------------------------------------------------------
# RandomNegativeSampler
# ---------------------------------------------------------------------------

class TestRandomNegativeSampler:
    def test_unfitted_sampler_raises(self):
        sampler = RandomNegativeSampler(seed=0)
        with pytest.raises(RuntimeError, match="fit()"):
            sampler.sample(0, n=3)

    def test_sample_returns_n_items(self):
        sampler = RandomNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        negs = sampler.sample(0, n=5)
        assert len(negs) == 5

    def test_sample_output_dtype_is_int32(self):
        sampler = RandomNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        negs = sampler.sample(0, n=5)
        assert negs.dtype == np.int32

    def test_sample_batch_shape_is_batch_by_n(self):
        sampler = RandomNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        negs = sampler.sample_batch(np.array([0, 1, 2]), n_per_user=4)
        assert negs.shape == (3, 4)

    def test_sample_never_includes_users_train_positives(self):
        sampler = RandomNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        # user 0 interacted with items {2, 3, 5} (all rows, any label)
        negs = sampler.sample(0, n=10)
        assert set(negs.tolist()).isdisjoint({2, 3, 5})

    def test_sample_excludes_explicit_positive_item_when_passed(self):
        sampler = RandomNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        # item 9 is not one of user 0's train interactions, but is passed as
        # the batch's positive -> must still be excluded defensively.
        negs = sampler.sample(0, n=10, positive_item=9)
        assert 9 not in negs.tolist()

    def test_sample_has_no_duplicates_within_one_call(self):
        sampler = RandomNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        negs = sampler.sample(0, n=10)
        assert len(set(negs.tolist())) == len(negs)

    def test_sample_reproducible_with_seed(self):
        a = RandomNegativeSampler(seed=42)
        a.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        b = RandomNegativeSampler(seed=42)
        b.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        np.testing.assert_array_equal(a.sample(0, n=5), b.sample(0, n=5))

    def test_sample_different_seeds_differ(self):
        a = RandomNegativeSampler(seed=1)
        a.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        b = RandomNegativeSampler(seed=2)
        b.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        assert not np.array_equal(a.sample(0, n=10), b.sample(0, n=10))

    def test_sample_matches_sample_batch_for_batch_size_one(self):
        a = RandomNegativeSampler(seed=7)
        a.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        single = a.sample(0, n=4)
        b = RandomNegativeSampler(seed=7)
        b.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        batch = b.sample_batch(np.array([0]), n_per_user=4)
        np.testing.assert_array_equal(single, batch[0])

    def test_raises_when_too_few_valid_items_remain_for_user(self):
        # 3-item catalog, user has seen 2 of them -> requesting 3 is impossible.
        tiny_train = _make_train_df([(0, 0, 1), (0, 1, 1)])
        sampler = RandomNegativeSampler(seed=0, max_resample_rounds=2)
        sampler.fit(tiny_train, n_users=1, n_items=3)
        with pytest.raises(RuntimeError):
            sampler.sample(0, n=3)

    def test_uniform_distribution_over_many_draws(self):
        # No exclusions, fixed seed: empirical per-item frequency over many
        # single-item draws should approach the uniform 1/n_items rate.
        train = _make_train_df([])
        n_items = 10
        sampler = RandomNegativeSampler(seed=123)
        sampler.fit(train, n_users=1, n_items=n_items)
        draws = sampler.sample(0, n=1)
        counts = np.zeros(n_items, dtype=np.int64)
        n_trials = 20_000
        for _ in range(n_trials):
            item = sampler.sample(0, n=1)[0]
            counts[item] += 1
        freqs = counts / n_trials
        for f in freqs:
            assert f == pytest.approx(1.0 / n_items, abs=0.02)


# ---------------------------------------------------------------------------
# PopularityNegativeSampler
# ---------------------------------------------------------------------------

class TestPopularityNegativeSampler:
    def test_unfitted_sampler_raises(self):
        sampler = PopularityNegativeSampler(seed=0)
        with pytest.raises(RuntimeError, match="fit()"):
            sampler.sample(0, n=3)

    def test_probs_sum_to_one_after_fit(self):
        sampler = PopularityNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        assert sampler.probs.sum() == pytest.approx(1.0)

    def test_probs_shape_matches_n_items(self):
        sampler = PopularityNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        assert sampler.probs.shape == (N_ITEMS,)

    def test_more_popular_item_gets_higher_probability(self):
        sampler = PopularityNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        # item 2: 8 positives, item 7: 2 positives
        assert sampler.probs[2] > sampler.probs[7]

    def test_zero_positive_items_still_have_nonzero_probability(self):
        sampler = PopularityNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        # items 8-19 have zero label=1 rows -> PopularityBaseline would
        # exclude them entirely; the sampler must NOT.
        assert sampler.probs[8] > 0

    def test_power_zero_gives_uniform_distribution(self):
        sampler = PopularityNegativeSampler(power=0.0, seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        expected = 1.0 / N_ITEMS
        for p in sampler.probs:
            assert p == pytest.approx(expected)

    def test_higher_power_increases_skew(self):
        def skew_ratio(power: float) -> float:
            sampler = PopularityNegativeSampler(power=power, seed=0)
            sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
            return sampler.probs[2] / sampler.probs[8]  # most popular / cold

        r0 = skew_ratio(0.0)
        r75 = skew_ratio(0.75)
        r1 = skew_ratio(1.0)
        assert r0 < r75 < r1

    def test_empirical_distribution_matches_fitted_probs_over_many_draws(self):
        sampler = PopularityNegativeSampler(power=0.75, seed=123)
        # user index N_USERS (out of range of TRAIN_ROWS) has zero train
        # interactions -> nothing excluded, draws track sampler.probs directly
        # (any exclusion would renormalize the conditional distribution and
        # invalidate a raw comparison against the unconditioned probs).
        sampler.fit(TRAIN_DF, n_users=N_USERS + 1, n_items=N_ITEMS)
        unexcluded_user = N_USERS
        n_trials = 20_000
        counts = np.zeros(N_ITEMS, dtype=np.int64)
        for _ in range(n_trials):
            item = sampler.sample(unexcluded_user, n=1)[0]
            counts[item] += 1
        freqs = counts / n_trials
        for item in range(N_ITEMS):
            assert freqs[item] == pytest.approx(sampler.probs[item], abs=0.02)

    def test_sample_excludes_user_positives(self):
        sampler = PopularityNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        negs = sampler.sample(0, n=10)
        assert set(negs.tolist()).isdisjoint({2, 3, 5})

    def test_sample_no_duplicates(self):
        sampler = PopularityNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        negs = sampler.sample(0, n=10)
        assert len(set(negs.tolist())) == len(negs)

    def test_sample_reproducible_with_seed(self):
        a = PopularityNegativeSampler(seed=42)
        a.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        b = PopularityNegativeSampler(seed=42)
        b.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        np.testing.assert_array_equal(a.sample(0, n=5), b.sample(0, n=5))

    def test_sample_batch_shape_and_dtype(self):
        sampler = PopularityNegativeSampler(seed=0)
        sampler.fit(TRAIN_DF, n_users=N_USERS, n_items=N_ITEMS)
        negs = sampler.sample_batch(np.array([0, 1, 2]), n_per_user=4)
        assert negs.shape == (3, 4)
        assert negs.dtype == np.int32
