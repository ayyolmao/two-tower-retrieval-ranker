"""Download a subset of the Yandex Yambda dataset from the Hugging Face Hub.

Yambda (https://huggingface.co/datasets/yandex/yambda) ships in three scales
(50m / 500m / 5b) and two layouts (flat / sequential), each as Parquet files per
event type (likes, listens, dislikes, ...). To avoid pulling all ~5.3B rows we
download only the requested ``{type}/{size}/{event}.parquet`` files.

Examples
--------
    # Default: flat/50m, likes + listens
    python -m src.data.download

    # Just likes, into a custom dir
    python -m src.data.download --events likes --out data/raw

    # A larger flavor
    python -m src.data.download --size 500m --events likes
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "yandex/yambda"
SIZES = ("50m", "500m", "5b")
FORMATS = ("flat", "sequential")
EVENTS = ("likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes")


def download_event(
    event: str,
    size: str = "50m",
    fmt: str = "flat",
    out: str | Path = "data/raw",
    repo_id: str = REPO_ID,
) -> Path:
    """Download one ``{fmt}/{size}/{event}.parquet`` file from the Hub.

    Returns the local path to the downloaded parquet file. Idempotent: if the
    file is already cached/present it is reused rather than re-downloaded.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    subfolder = f"{fmt}/{size}"
    filename = f"{event}.parquet"

    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        subfolder=subfolder,
        filename=filename,
        local_dir=out,
    )
    path = Path(local_path)
    size_mb = path.stat().st_size / 1e6
    print(f"  ✓ {subfolder}/{filename}  ({size_mb:,.1f} MB)  ->  {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Yambda subsets from HF Hub.")
    parser.add_argument("--size", default="50m", choices=SIZES)
    parser.add_argument("--type", dest="fmt", default="flat", choices=FORMATS)
    parser.add_argument(
        "--events",
        nargs="+",
        default=["likes", "listens"],
        choices=EVENTS,
        help="Interaction subsets to download.",
    )
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    print(f"Downloading Yambda {args.fmt}/{args.size}: {', '.join(args.events)}")
    for event in args.events:
        download_event(event, size=args.size, fmt=args.fmt, out=args.out, repo_id=args.repo_id)
    print("Done.")


if __name__ == "__main__":
    main()
