"""Preprocess Yambda interaction parquets into model-ready train/val/test splits.

Runs three sequential steps on a raw event parquet (output of src.data.download):
  1. k-core filtering: iteratively remove users/items with fewer than k interactions.
  2. Label assignment: likes -> label=1; listens -> label=1 if played_ratio_pct >= threshold.
  3. Time-based split (two strategies, selected by ``split_strategy``):
       * ``user_time`` (default): per-user leave-one-out. For each user, the newest
         interaction -> test, the 2nd-newest -> val, the rest -> train. Holds out each
         user's most recent activity, preventing temporal leakage within a user's history.
       * ``global_time``: sort the whole dataset by timestamp, cut at 80/10/10 by default.

IDs are remapped to contiguous [0, N) integers after k-core. A vocab JSON is written
alongside the split parquets so embedding tables can be sized correctly at model init.
A split-metadata JSON records the cutoffs (per-user holdout timestamps for ``user_time``,
global boundary timestamps for ``global_time``).

Output layout::

    data/processed/flat/50m/
        likes_train.parquet     # columns: user_idx, item_idx, timestamp, is_organic, label
        likes_val.parquet
        likes_test.parquet
        likes_vocab.json        # {"user2idx": {...}, "item2idx": {...}, "n_users": N, "n_items": M}
        likes_split_meta.json   # {"strategy": ..., "rows": {...}, "user_holdout_ts": {...}, ...}

Examples
--------
    # Preprocess likes with default k=5
    python -m src.data.preprocess --event likes

    # Preprocess listens with stricter k-core
    python -m src.data.preprocess --event listens --k 10 --train-frac 0.7 --val-frac 0.15

    # Programmatic use
    from src.data.preprocess import preprocess
    train_path, val_path, test_path, vocab_path = preprocess("likes", k=5)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.load import load_interactions


def _processed_path(event: str, split: str, size: str, fmt: str, root: str | Path) -> Path:
    return Path(root) / fmt / size / f"{event}_{split}.parquet"


def _vocab_path(event: str, size: str, fmt: str, root: str | Path) -> Path:
    return Path(root) / fmt / size / f"{event}_vocab.json"


def _split_meta_path(event: str, size: str, fmt: str, root: str | Path) -> Path:
    return Path(root) / fmt / size / f"{event}_split_meta.json"


def kcore_filter(
    df: pd.DataFrame,
    k: int = 5,
    user_col: str = "uid",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Iteratively remove users and items with fewer than k interactions until stable."""
    print(f"k-core filtering (k={k}):")
    print(f"  before:  rows={len(df):,}  users={df[user_col].nunique():,}  items={df[item_col].nunique():,}")

    prev_len = -1
    for round_ in range(1, 51):
        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= k].index)]
        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= k].index)]
        print(f"  round {round_}: rows={len(df):,}  users={df[user_col].nunique():,}  items={df[item_col].nunique():,}")
        if len(df) == prev_len:
            print(f"  converged in {round_} rounds.")
            break
        prev_len = len(df)

    return df.reset_index(drop=True)


def build_vocab(
    df: pd.DataFrame,
    user_col: str = "uid",
    item_col: str = "item_id",
) -> dict:
    """Build contiguous integer mappings for users and items.

    Sorts raw IDs before enumerating so the mapping is deterministic across runs.
    """
    user2idx = {int(uid): idx for idx, uid in enumerate(sorted(df[user_col].unique()))}
    item2idx = {int(iid): idx for idx, iid in enumerate(sorted(df[item_col].unique()))}
    return {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "n_users": len(user2idx),
        "n_items": len(item2idx),
    }


def assign_labels(
    df: pd.DataFrame,
    event: str,
    play_threshold: int = 50,
) -> pd.DataFrame:
    """Add a binary label column.

    Likes: every row is a positive (label=1).
    Listens: label=1 if played_ratio_pct >= play_threshold, else 0.
    """
    if event == "listens":
        df = df.copy()
        df["label"] = (df["played_ratio_pct"] >= play_threshold).astype("int8")
        pos_rate = df["label"].mean()
        print(f"Labels assigned (listens): {df['label'].sum():,} / {len(df):,} rows → label=1 ({pos_rate:.1%} positive)")
    else:
        df = df.copy()
        df["label"] = pd.array([1] * len(df), dtype="int8")
        print(f"Labels assigned ({event}): all {len(df):,} rows → label=1")
    return df


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by timestamp and cut into train/val/test chronologically."""
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    test_frac = 1.0 - train_frac - val_frac
    print(
        f"Time split ({train_frac:.0%}/{val_frac:.0%}/{test_frac:.0%}):\n"
        f"  train: {len(train):,} rows  (ts {train[timestamp_col].min()} → {train[timestamp_col].max()})\n"
        f"  val:   {len(val):,} rows  (ts {val[timestamp_col].min()} → {val[timestamp_col].max()})\n"
        f"  test:  {len(test):,} rows  (ts {test[timestamp_col].min()} → {test[timestamp_col].max()})"
    )
    return train, val, test


def user_time_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    user_col: str = "uid",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user leave-one-out temporal split.

    For each user, interactions are ordered by timestamp; the newest goes to test,
    the second-newest to val, and the remainder to train. This holds out every
    user's most recent activity for evaluation, avoiding temporal leakage within a
    user's own history (a user's train rows always precede their val/test rows).

    Sparse-user fallback (reachable only if k-core ``k`` is overridden below 3):
      * 1 interaction  -> train only (cannot hold the user's sole event out).
      * 2 interactions -> newest to test, older to train (no val row).
    """
    # Stable sort so equal timestamps (Yambda bins time into 5s buckets, so ties are
    # common) keep a deterministic order across runs.
    df = df.sort_values([user_col, timestamp_col], kind="stable").reset_index(drop=True)

    # Position from the end within each user (0 = newest), plus the user's row count.
    rank_from_end = df.groupby(user_col, sort=False).cumcount(ascending=False)
    user_count = df.groupby(user_col, sort=False)[user_col].transform("size")

    is_test = (rank_from_end == 0) & (user_count >= 2)
    is_val = (rank_from_end == 1) & (user_count >= 3)

    test = df[is_test].reset_index(drop=True)
    val = df[is_val].reset_index(drop=True)
    train = df[~(is_test | is_val)].reset_index(drop=True)

    def _ts_span(split_df: pd.DataFrame) -> str:
        if len(split_df) == 0:
            return "—"
        return f"ts {split_df[timestamp_col].min()} → {split_df[timestamp_col].max()}"

    print(
        "Per-user leave-one-out split (newest→test, 2nd-newest→val, rest→train):\n"
        f"  train: {len(train):,} rows  users={train[user_col].nunique():,}  ({_ts_span(train)})\n"
        f"  val:   {len(val):,} rows  users={val[user_col].nunique():,}  ({_ts_span(val)})\n"
        f"  test:  {len(test):,} rows  users={test[user_col].nunique():,}  ({_ts_span(test)})"
    )
    return train, val, test


def _build_split_meta(
    strategy: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str = "user_idx",
    timestamp_col: str = "timestamp",
) -> dict:
    """Summarise the split and record its cutoffs for documentation/reproducibility."""

    def _range(split_df: pd.DataFrame) -> list[int] | None:
        if len(split_df) == 0:
            return None
        return [int(split_df[timestamp_col].min()), int(split_df[timestamp_col].max())]

    meta: dict = {
        "strategy": strategy,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "users": {
            "train": int(train[user_col].nunique()),
            "val": int(val[user_col].nunique()),
            "test": int(test[user_col].nunique()),
        },
        "timestamp_range": {
            "train": _range(train),
            "val": _range(val),
            "test": _range(test),
        },
    }

    if strategy == "user_time":
        meta["rule"] = "per-user leave-one-out: newest->test, 2nd-newest->val, rest->train"
        val_ts = dict(zip(val[user_col].astype(int), val[timestamp_col].astype(int)))
        test_ts = dict(zip(test[user_col].astype(int), test[timestamp_col].astype(int)))
        holdout: dict = {}
        for uid in sorted(set(val_ts) | set(test_ts)):
            holdout[str(uid)] = {
                "val_ts": val_ts.get(uid),
                "test_ts": test_ts.get(uid),
            }
        meta["user_holdout_ts"] = holdout
    else:
        meta["rule"] = "global chronological cut by row fraction"
        train_range = _range(train)
        val_range = _range(val)
        meta["boundary_ts"] = {
            "train_max": train_range[1] if train_range else None,
            "val_max": val_range[1] if val_range else None,
        }

    return meta


def preprocess(
    event: str = "likes",
    size: str = "50m",
    fmt: str = "flat",
    root: str | Path = "data/raw",
    out: str | Path = "data/processed",
    k: int = 5,
    split_strategy: str = "user_time",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    play_threshold: int = 50,
) -> tuple[Path, Path, Path, Path]:
    """Full preprocessing pipeline: load → k-core → label → split → remap IDs → save.

    ``split_strategy`` selects ``user_time`` (per-user leave-one-out, default) or
    ``global_time`` (global chronological cut at ``train_frac``/``val_frac``).
    ``train_frac``/``val_frac`` are ignored when ``split_strategy == "user_time"``.

    Returns (train_path, val_path, test_path, vocab_path).
    """
    df = load_interactions(event, size=size, fmt=fmt, root=root)

    df = kcore_filter(df, k=k)
    df = assign_labels(df, event, play_threshold=play_threshold)

    vocab = build_vocab(df)
    print(f"\nVocab: n_users={vocab['n_users']:,}  n_items={vocab['n_items']:,}")

    if split_strategy == "user_time":
        train, val, test = user_time_split(df)
    elif split_strategy == "global_time":
        train, val, test = time_split(df, train_frac=train_frac, val_frac=val_frac)
    else:
        raise ValueError(
            f"unknown split_strategy {split_strategy!r}; expected 'user_time' or 'global_time'"
        )

    keep_cols = ["user_idx", "item_idx", "timestamp", "is_organic", "label"]

    def _remap_and_select(split_df: pd.DataFrame) -> pd.DataFrame:
        split_df = split_df.copy()
        split_df["user_idx"] = split_df["uid"].map(vocab["user2idx"]).astype("int32")
        split_df["item_idx"] = split_df["item_id"].map(vocab["item2idx"]).astype("int32")
        return split_df[keep_cols]

    train = _remap_and_select(train)
    val = _remap_and_select(val)
    test = _remap_and_select(test)

    train_path = _processed_path(event, "train", size, fmt, out)
    val_path = _processed_path(event, "val", size, fmt, out)
    test_path = _processed_path(event, "test", size, fmt, out)
    v_path = _vocab_path(event, size, fmt, out)
    meta_path = _split_meta_path(event, size, fmt, out)

    train_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    with open(v_path, "w") as f:
        json.dump(vocab, f, indent=2)

    meta = _build_split_meta(split_strategy, train, val, test)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"\nWritten:\n"
        f"  {train_path}\n"
        f"  {val_path}\n"
        f"  {test_path}\n"
        f"  {v_path}\n"
        f"  {meta_path}"
    )
    return train_path, val_path, test_path, v_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw Yambda parquet into train/val/test splits.")
    parser.add_argument("--event", default="likes")
    parser.add_argument("--size", default="50m")
    parser.add_argument("--type", dest="fmt", default="flat")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--split-strategy",
        choices=["user_time", "global_time"],
        default="user_time",
        help="user_time: per-user leave-one-out (default); global_time: global chronological cut.",
    )
    parser.add_argument("--train-frac", type=float, default=0.8, help="global_time only")
    parser.add_argument("--val-frac", type=float, default=0.1, help="global_time only")
    parser.add_argument("--play-threshold", type=int, default=50)
    args = parser.parse_args()

    preprocess(
        event=args.event,
        size=args.size,
        fmt=args.fmt,
        root=args.root,
        out=args.out,
        k=args.k,
        split_strategy=args.split_strategy,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        play_threshold=args.play_threshold,
    )


if __name__ == "__main__":
    main()
