"""Run popularity and MF baselines end-to-end and log results to MLflow.

Loads preprocessed parquets, fits both models on training data, evaluates on
val and test, logs to MLflow (nested runs under a "baselines" parent), and
prints a side-by-side comparison table.

Usage
-----
    python -m src.baselines.run_baselines
    python -m src.baselines.run_baselines --event likes --mf-factors 64
    python -m src.baselines.run_baselines --no-mf   # skip MF (fast sanity check)

Prerequisites
-------------
    data/processed/flat/50m/likes_{train,val,test}.parquet must exist AND must
    have been generated with ``--split-strategy user_time`` (the default on main).
    Run ``python -m src.data.preprocess --event likes`` first if they don't exist
    or if you get a RuntimeError about the split strategy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.baselines.mf import MFBaseline
from src.baselines.popularity import PopularityBaseline
from src.data.preprocess import _processed_path, _split_meta_path, _vocab_path
from src.eval.metrics import evaluate
from src.training.experiment import get_experiment, log_metrics, start_run


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _check_split_strategy(event: str, size: str, fmt: str, root: str | Path) -> None:
    """Raise if parquets weren't generated with the user_time (LOO) strategy.

    Baselines must use LOO splits so metrics are comparable to the two-tower,
    which is also evaluated under LOO. Global-time splits produce different
    relevant-set sizes per user, making Recall@100 incomparable across models.
    """
    meta_path = _split_meta_path(event, size, fmt, root)
    if not meta_path.exists():
        raise RuntimeError(
            f"No split_meta.json found at {meta_path}. "
            f"Parquets were generated with old preprocess (before PR #5). "
            f"Re-run: python -m src.data.preprocess --event {event}"
        )
    meta = json.loads(meta_path.read_text())
    strategy = meta.get("strategy")
    if strategy != "user_time":
        raise RuntimeError(
            f"Parquets use '{strategy}' split strategy. "
            f"Baselines require 'user_time' (per-user leave-one-out) so metrics "
            f"are comparable to the two-tower. "
            f"Re-run: python -m src.data.preprocess --event {event}"
        )


def _load_splits(
    event: str = "likes",
    size: str = "50m",
    fmt: str = "flat",
    root: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load train/val/test parquets and vocab JSON."""
    train = pd.read_parquet(_processed_path(event, "train", size, fmt, root))
    val   = pd.read_parquet(_processed_path(event, "val",   size, fmt, root))
    test  = pd.read_parquet(_processed_path(event, "test",  size, fmt, root))
    vocab = json.loads(_vocab_path(event, size, fmt, root).read_text())
    return train, val, test, vocab


def _build_seen_items(train_df: pd.DataFrame) -> dict[int, frozenset[int]]:
    """Map user_idx → frozenset of item indices they interacted with in train.

    Uses all train rows (not just label=1) so we never recommend something the
    user has already heard, regardless of whether they enjoyed it.
    """
    return {
        uid: frozenset(group["item_idx"].tolist())
        for uid, group in train_df.groupby("user_idx")
    }


def _build_eval_data(
    split_df: pd.DataFrame,
) -> tuple[list[int], list[set[int]]]:
    """Return (user_ids, relevant_sets) for users with at least one label=1 row."""
    user_ids: list[int] = []
    relevant_sets: list[set[int]] = []
    for uid, group in split_df.groupby("user_idx"):
        relevant = set(group.loc[group["label"] == 1, "item_idx"].tolist())
        if relevant:
            user_ids.append(int(uid))
            relevant_sets.append(relevant)
    return user_ids, relevant_sets


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _evaluate_model(
    model: PopularityBaseline | MFBaseline,
    user_ids: list[int],
    relevant_sets: list[set[int]],
    seen_by_user: dict[int, frozenset[int]],
    n: int = 100,
) -> dict[str, float]:
    """Recommend n items per user, compute macro-averaged metrics."""
    ranked_lists: list[list[int]] = []
    for uid in user_ids:
        seen = seen_by_user.get(uid, frozenset())
        ranked_lists.append(model.recommend(uid, n=n, seen_items=seen))
    return evaluate(
        ranked_lists,
        relevant_sets,
        ks=(10, 100),
        metrics=("recall", "ndcg", "mrr"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    event: str = "likes",
    size: str = "50m",
    fmt: str = "flat",
    root: str | Path = "data/processed",
    mf_factors: int = 64,
    mf_iterations: int = 20,
    mf_alpha: float = 40.0,
    include_mf: bool = True,
) -> dict[str, dict[str, float]]:
    """Fit baselines, evaluate on val+test, log to MLflow.

    Returns a nested dict: {model_name: {split/metric: value}}.
    """
    print(f"Loading {event} splits from {Path(root) / fmt / size} …")
    _check_split_strategy(event, size, fmt, root)
    train_df, val_df, test_df, vocab = _load_splits(event, size, fmt, root)
    n_users, n_items = vocab["n_users"], vocab["n_items"]
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print(f"  n_users={n_users:,}  n_items={n_items:,}")

    seen_by_user = _build_seen_items(train_df)

    val_users,  val_relevant  = _build_eval_data(val_df)
    test_users, test_relevant = _build_eval_data(test_df)
    print(f"  eval users: val={len(val_users):,}  test={len(test_users):,}")

    models: list[tuple[str, PopularityBaseline | MFBaseline, dict]] = []

    # --- Popularity ---
    pop = PopularityBaseline()
    print("\nFitting PopularityBaseline …")
    pop.fit(train_df)
    models.append(("popularity", pop, {"model": "popularity"}))

    # --- MF / ALS ---
    if include_mf:
        mf_name = f"mf-als-{mf_factors}d"
        mf = MFBaseline(factors=mf_factors, iterations=mf_iterations, alpha=mf_alpha)
        print(f"\nFitting MFBaseline (factors={mf_factors}, iters={mf_iterations}) …")
        mf.fit(train_df, n_users=n_users, n_items=n_items)
        models.append((
            mf_name,
            mf,
            {"model": "mf-als", "factors": mf_factors, "iterations": mf_iterations, "alpha": mf_alpha},
        ))

    # --- MLflow + eval ---
    get_experiment()
    all_results: dict[str, dict[str, float]] = {}

    with start_run("baselines", tags={"event": event, "size": size}):
        for model_name, model, params in models:
            print(f"\nEvaluating {model_name} …")
            with start_run(model_name, nested=True, params=params):
                val_metrics  = _evaluate_model(model, val_users,  val_relevant,  seen_by_user)
                test_metrics = _evaluate_model(model, test_users, test_relevant, seen_by_user)

                display_metrics = {
                    **{f"val/{k}": v  for k, v in val_metrics.items()},
                    **{f"test/{k}": v for k, v in test_metrics.items()},
                }
                log_metrics(display_metrics, step=0)  # log_metrics sanitises '@' internally
                all_results[model_name] = display_metrics

    # --- Print table ---
    col_order = [
        "val/recall@100", "val/ndcg@10", "val/mrr@10",
        "test/recall@100", "test/ndcg@10", "test/mrr@10",
    ]
    header = f"{'Model':<20}" + "".join(f"{c:>18}" for c in col_order)
    print("\n" + header)
    print("-" * len(header))
    for model_name, results in all_results.items():
        row = f"{model_name:<20}" + "".join(f"{results.get(c, float('nan')):>18.4f}" for c in col_order)
        print(row)
    print()

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run popularity + MF baselines.")
    parser.add_argument("--event",        default="likes")
    parser.add_argument("--size",         default="50m")
    parser.add_argument("--type",         dest="fmt", default="flat")
    parser.add_argument("--root",         default="data/processed")
    parser.add_argument("--mf-factors",   type=int,   default=64)
    parser.add_argument("--mf-iterations",type=int,   default=20)
    parser.add_argument("--mf-alpha",     type=float, default=40.0)
    parser.add_argument("--no-mf",        dest="include_mf", action="store_false",
                        help="Skip MF baseline (faster for quick sanity checks)")
    args = parser.parse_args()

    run(
        event=args.event,
        size=args.size,
        fmt=args.fmt,
        root=args.root,
        mf_factors=args.mf_factors,
        mf_iterations=args.mf_iterations,
        mf_alpha=args.mf_alpha,
        include_mf=args.include_mf,
    )


if __name__ == "__main__":
    main()
