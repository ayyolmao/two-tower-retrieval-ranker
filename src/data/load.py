"""Load Yambda interaction parquet files and build a tiny smoke-test sample.

The full dataset is multi-GB and gitignored. ``make_sample`` carves out a small
user-sampled subset that is committed to the repo so the data pipeline (and CI)
can be exercised without downloading anything.

Yambda flat schema (per https://huggingface.co/datasets/yandex/yambda):
    uid                  uint32   user id
    item_id              uint32   track id
    timestamp            uint32   delta time, 5s bins
    is_organic           uint8    1 = organic, 0 = recommendation-driven
    played_ratio_pct     uint16   listens only: percent played
    track_length_seconds uint32   listens only: track duration

Examples
--------
    # Load a downloaded subset
    python -c "from src.data.load import load_interactions; print(load_interactions('likes').head())"

    # Build the committed smoke-test sample from raw downloads
    python -m src.data.load --event likes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parquet_path(event: str, size: str, fmt: str, root: str | Path) -> Path:
    return Path(root) / fmt / size / f"{event}.parquet"


def load_interactions(
    event: str = "likes",
    size: str = "50m",
    fmt: str = "flat",
    root: str | Path = "data/raw",
) -> pd.DataFrame:
    """Load one interaction parquet file into a DataFrame.

    Raises FileNotFoundError with a hint to run the downloader if the file is
    missing.
    """
    path = _parquet_path(event, size, fmt, root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first:\n"
            f"    python -m src.data.download --size {size} --type {fmt} --events {event}"
        )
    return pd.read_parquet(path)


def make_sample(
    event: str = "likes",
    size: str = "50m",
    fmt: str = "flat",
    root: str | Path = "data/raw",
    out: str | Path = "data/sample",
    n_users: int = 200,
    seed: int = 42,
) -> Path:
    """Write a tiny user-sampled subset for smoke tests / CI.

    Samples ``n_users`` distinct uids (deterministic via ``seed``) and keeps all
    their interactions. The output mirrors the ``{fmt}/{size}/{event}.parquet``
    layout so the same loaders work against ``data/sample``.
    """
    df = load_interactions(event, size=size, fmt=fmt, root=root)

    rng = pd.Series(df["uid"].unique())
    sampled_uids = rng.sample(n=min(n_users, len(rng)), random_state=seed)
    sample = df[df["uid"].isin(sampled_uids)].reset_index(drop=True)

    out_path = _parquet_path(event, size, fmt, out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_path, index=False)

    print(
        f"Sample written: {out_path}\n"
        f"  users={sample['uid'].nunique():,}  rows={len(sample):,}  "
        f"cols={list(sample.columns)}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a smoke-test sample from raw Yambda parquet.")
    parser.add_argument("--event", default="likes")
    parser.add_argument("--size", default="50m")
    parser.add_argument("--type", dest="fmt", default="flat")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--out", default="data/sample")
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_sample(
        event=args.event,
        size=args.size,
        fmt=args.fmt,
        root=args.root,
        out=args.out,
        n_users=args.n_users,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
