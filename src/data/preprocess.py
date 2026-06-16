"""Preprocess Yambda interaction parquets into model-ready train/val/test splits.

Runs three sequential steps on a raw event parquet (output of src.data.download):
  1. k-core filtering: iteratively remove users/items with fewer than k interactions.
  2. Label assignment: likes -> label=1; listens -> label=1 if played_ratio_pct >= threshold.
  3. Chronological split: sort by timestamp, cut at 80/10/10 by default.

IDs are remapped to contiguous [0, N) integers after k-core. A vocab JSON is written
alongside the split parquets so embedding tables can be sized correctly at model init.

Output layout::

    data/processed/flat/50m/
        likes_train.parquet   # columns: user_idx, item_idx, timestamp, is_organic, label
        likes_val.parquet
        likes_test.parquet
        likes_vocab.json      # {"user2idx": {...}, "item2idx": {...}, "n_users": N, "n_items": M}

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


def preprocess(
    event: str = "likes",
    size: str = "50m",
    fmt: str = "flat",
    root: str | Path = "data/raw",
    out: str | Path = "data/processed",
    k: int = 5,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    play_threshold: int = 50,
) -> tuple[Path, Path, Path, Path]:
    """Full preprocessing pipeline: load → k-core → label → split → remap IDs → save.

    Returns (train_path, val_path, test_path, vocab_path).
    """
    df = load_interactions(event, size=size, fmt=fmt, root=root)

    df = kcore_filter(df, k=k)
    df = assign_labels(df, event, play_threshold=play_threshold)

    vocab = build_vocab(df)
    print(f"\nVocab: n_users={vocab['n_users']:,}  n_items={vocab['n_items']:,}")

    train, val, test = time_split(df, train_frac=train_frac, val_frac=val_frac)

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

    train_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    with open(v_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(
        f"\nWritten:\n"
        f"  {train_path}\n"
        f"  {val_path}\n"
        f"  {test_path}\n"
        f"  {v_path}"
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
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--play-threshold", type=int, default=50)
    args = parser.parse_args()

    preprocess(
        event=args.event,
        size=args.size,
        fmt=args.fmt,
        root=args.root,
        out=args.out,
        k=args.k,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        play_threshold=args.play_threshold,
    )


if __name__ == "__main__":
    main()
