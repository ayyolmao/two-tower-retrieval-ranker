"""Unit tests for time-based train/val/test splits — hand-built toy frames.

The per-user leave-one-out split is the leakage-sensitive piece: every assertion
here pins down that each user's newest interaction lands in test, the second-newest
in val, the rest in train, and that no train row ever post-dates that user's val/test
rows. Sparse-user fallbacks and the legacy global split are covered too.
"""

import pandas as pd
import pytest

from src.data.preprocess import _build_split_meta, time_split, user_time_split


def _frame(rows):
    """rows: list of (uid, item_id, timestamp). Other columns are filled with defaults."""
    df = pd.DataFrame(rows, columns=["uid", "item_id", "timestamp"])
    df["is_organic"] = 1
    df["label"] = 1
    return df


# --- user_time_split: core leave-one-out behaviour --------------------------------

def test_loo_assigns_newest_to_test_second_to_val():
    # user 1: ts 10,20,30,40 → train{10,20} val{30} test{40}
    # user 2: ts 5,15,25      → train{5}     val{15} test{25}
    df = _frame([
        (1, 100, 30), (1, 101, 10), (1, 102, 40), (1, 103, 20),
        (2, 200, 25), (2, 201, 5), (2, 202, 15),
    ])
    train, val, test = user_time_split(df)

    assert set(zip(test["uid"], test["timestamp"])) == {(1, 40), (2, 25)}
    assert set(zip(val["uid"], val["timestamp"])) == {(1, 30), (2, 15)}
    assert set(zip(train["uid"], train["timestamp"])) == {(1, 10), (1, 20), (2, 5)}


def test_loo_no_leakage_invariant():
    # For every user: max(train_ts) < val_ts < test_ts.
    df = _frame([
        (1, 100, 10), (1, 101, 20), (1, 102, 30), (1, 103, 40), (1, 104, 50),
        (2, 200, 1), (2, 201, 2), (2, 202, 3), (2, 203, 4),
    ])
    train, val, test = user_time_split(df)

    for uid in df["uid"].unique():
        train_max = train.loc[train["uid"] == uid, "timestamp"].max()
        val_ts = val.loc[val["uid"] == uid, "timestamp"].iloc[0]
        test_ts = test.loc[test["uid"] == uid, "timestamp"].iloc[0]
        assert train_max < val_ts < test_ts


def test_loo_one_row_per_user_in_val_and_test():
    df = _frame([
        (1, 100, 10), (1, 101, 20), (1, 102, 30),
        (2, 200, 11), (2, 201, 22), (2, 202, 33),
    ])
    _, val, test = user_time_split(df)
    assert not val["uid"].duplicated().any()
    assert not test["uid"].duplicated().any()


def test_loo_deterministic_on_tied_timestamps():
    # All ts equal: stable sort must give identical splits across runs.
    df = _frame([(1, 100, 7), (1, 101, 7), (1, 102, 7), (1, 103, 7)])
    a = user_time_split(df.copy())
    b = user_time_split(df.copy())
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


# --- user_time_split: sparse-user fallbacks ---------------------------------------

def test_loo_two_event_user_has_no_val_row():
    # 2 interactions → newest to test, older to train, nothing in val.
    df = _frame([(1, 100, 10), (1, 101, 20)])
    train, val, test = user_time_split(df)
    assert list(zip(test["uid"], test["timestamp"])) == [(1, 20)]
    assert list(zip(train["uid"], train["timestamp"])) == [(1, 10)]
    assert len(val) == 0


def test_loo_single_event_user_goes_to_train_only():
    df = _frame([(1, 100, 10)])
    train, val, test = user_time_split(df)
    assert list(zip(train["uid"], train["timestamp"])) == [(1, 10)]
    assert len(val) == 0
    assert len(test) == 0


# --- global_time split ------------------------------------------------------------

def test_global_time_split_sizes_and_order():
    df = _frame([(i, 1000 + i, i) for i in range(10)])
    train, val, test = time_split(df, train_frac=0.8, val_frac=0.1)
    assert (len(train), len(val), len(test)) == (8, 1, 1)
    # globally chronological: every train ts precedes val precedes test
    assert train["timestamp"].max() < val["timestamp"].min() <= val["timestamp"].max()
    assert val["timestamp"].max() < test["timestamp"].min()


# --- split metadata ---------------------------------------------------------------

def test_build_split_meta_user_time_records_per_user_cutoffs():
    # _build_split_meta consumes remapped frames keyed by user_idx.
    train = pd.DataFrame({"user_idx": [0, 0], "timestamp": [10, 20]})
    val = pd.DataFrame({"user_idx": [0], "timestamp": [30]})
    test = pd.DataFrame({"user_idx": [0], "timestamp": [40]})

    meta = _build_split_meta("user_time", train, val, test)

    assert meta["strategy"] == "user_time"
    assert meta["rows"] == {"train": 2, "val": 1, "test": 1}
    assert meta["users"] == {"train": 1, "val": 1, "test": 1}
    assert meta["user_holdout_ts"] == {"0": {"val_ts": 30, "test_ts": 40}}
    assert meta["timestamp_range"]["train"] == [10, 20]


def test_build_split_meta_global_time_records_boundaries():
    train = pd.DataFrame({"user_idx": [0, 1], "timestamp": [1, 2]})
    val = pd.DataFrame({"user_idx": [2], "timestamp": [3]})
    test = pd.DataFrame({"user_idx": [3], "timestamp": [4]})

    meta = _build_split_meta("global_time", train, val, test)

    assert meta["strategy"] == "global_time"
    assert "user_holdout_ts" not in meta
    assert meta["boundary_ts"] == {"train_max": 2, "val_max": 3}
