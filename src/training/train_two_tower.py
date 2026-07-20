"""Train the two-tower retrieval model: in-batch negatives + explicit
negative sampling, sampled-softmax loss, brute-force eval against the
popularity baseline.

Known limitation: in-batch negatives are implicitly popularity-biased
(popular items appear more often as positives, hence more often as
negatives too). The standard fix is a logQ correction subtracted from the
in-batch logits before softmax. This is deliberately deferred here -- the
acceptance bar is "beats the popularity baseline on Recall@100," not
"correctly debiased sampled softmax," and the false-negative mask in
src/models/two_tower.py's in_batch_logits already fixes the one in-batch
issue that's an outright correctness bug rather than a training-dynamics
refinement.

Usage
-----
    python -m src.training.train_two_tower
    python -m src.training.train_two_tower --embedding-dim 32 --epochs 10

Prerequisites
-------------
    data/processed/flat/50m/likes_{train,val,test}.parquet must exist AND
    have been generated with --split-strategy user_time (the default).
    Run python -m src.data.preprocess --event likes first if not.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baselines.popularity import PopularityBaseline
from src.baselines.run_baselines import (
    _build_eval_data,
    _build_seen_items,
    _check_split_strategy,
    _load_splits,
)
from src.data.negative_sampling import PopularityNegativeSampler, RandomNegativeSampler
from src.eval.metrics import evaluate
from src.models.two_tower import TwoTowerModel, sampled_softmax_loss
from src.training.experiment import get_experiment, log_metrics, start_run
from src.training.two_tower_dataset import InteractionDataset


def _score_and_rank(
    model,
    user_ids: list[int],
    seen_by_user: dict[int, frozenset[int]],
    device: torch.device,
    n: int = 100,
    eval_batch_size: int = 512,
) -> list[list[int]]:
    """Brute-force dot-product ranking against the full item catalog.

    Vectorized over the user axis (batched matmul against the full item
    embedding matrix), not a per-item/per-user Python loop. Seen-item
    exclusion is a small per-eval-batch loop over a dict (<= eval_batch_size
    iterations), matching run_baselines.py's dict-based seen_items access.
    No FAISS -- brute force over ~34K items is fine for this task.
    """
    model.eval()
    ranked_lists: list[list[int]] = []
    with torch.no_grad():
        item_embs = model.all_item_embeddings()  # (n_items, D)
        for start in range(0, len(user_ids), eval_batch_size):
            batch_uids = user_ids[start : start + eval_batch_size]
            uid_t = torch.tensor(batch_uids, dtype=torch.long, device=device)
            user_embs = model.user_tower(uid_t)  # (U, D)
            scores = user_embs @ item_embs.T  # (U, n_items)
            for row, uid in enumerate(batch_uids):
                seen = seen_by_user.get(uid, frozenset())
                if seen:
                    idx = torch.tensor(list(seen), dtype=torch.long, device=device)
                    scores[row, idx] = float("-inf")
            k = min(n, scores.shape[1])
            _, top_idx = torch.topk(scores, k=k, dim=1)
            ranked_lists.extend(top_idx.cpu().tolist())
    model.train()
    return ranked_lists


def evaluate_split(
    model,
    user_ids: list[int],
    relevant_sets: list[set[int]],
    seen_by_user: dict[int, frozenset[int]],
    device: torch.device,
    n: int = 100,
) -> dict[str, float]:
    """Mirrors run_baselines.py's _evaluate_model, vectorized scoring
    instead of a per-user model.recommend() loop."""
    ranked_lists = _score_and_rank(model, user_ids, seen_by_user, device, n=n)
    return evaluate(ranked_lists, relevant_sets, ks=(10, 100), metrics=("recall", "ndcg", "mrr"))


def _select_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _sample_explicit_negatives(
    neg_sampler: str,
    random_sampler: RandomNegativeSampler | None,
    popularity_sampler: PopularityNegativeSampler | None,
    user_idx_batch,
    n_negatives: int,
    pos_item_idx_batch,
    device: torch.device,
) -> torch.Tensor:
    """(B, K) or (B, 2K) int64 explicit-negative indices, K = n_negatives per
    sampler. 'both' concatenates random + popularity negatives per row."""
    user_np = user_idx_batch.cpu().numpy()
    pos_np = pos_item_idx_batch.cpu().numpy()

    parts = []
    if neg_sampler in ("random", "both"):
        parts.append(random_sampler.sample_batch(user_np, n_negatives, pos_np))
    if neg_sampler in ("popularity", "both"):
        parts.append(popularity_sampler.sample_batch(user_np, n_negatives, pos_np))

    neg_idx_np = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    return torch.from_numpy(neg_idx_np).to(device=device, dtype=torch.long)


def run(
    event: str = "likes",
    size: str = "50m",
    fmt: str = "flat",
    root: str | Path = "data/processed",
    embedding_dim: int = 64,
    batch_size: int = 512,
    epochs: int = 20,
    lr: float = 1e-3,
    neg_sampler: str = "popularity",
    n_negatives: int = 5,
    eval_every: int = 1,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Train the two-tower model, evaluate on val+test, log to MLflow, and
    print a two-tower-vs-popularity-baseline comparison table.

    Returns a nested dict: {model_name: {split/metric: value}}.
    """
    torch.manual_seed(seed)
    device = _select_device()

    print(f"Loading {event} splits from {Path(root) / fmt / size} … (device={device})")
    _check_split_strategy(event, size, fmt, root)
    train_df, val_df, test_df, vocab = _load_splits(event, size, fmt, root)
    n_users, n_items = vocab["n_users"], vocab["n_items"]
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print(f"  n_users={n_users:,}  n_items={n_items:,}")

    seen_by_user = _build_seen_items(train_df)
    val_users, val_relevant = _build_eval_data(val_df)
    test_users, test_relevant = _build_eval_data(test_df)
    print(f"  eval users: val={len(val_users):,}  test={len(test_users):,}")

    random_sampler = None
    popularity_sampler = None
    if neg_sampler in ("random", "both"):
        random_sampler = RandomNegativeSampler(seed=seed)
        random_sampler.fit(train_df, n_users=n_users, n_items=n_items)
    if neg_sampler in ("popularity", "both"):
        popularity_sampler = PopularityNegativeSampler(seed=seed)
        popularity_sampler.fit(train_df, n_users=n_users, n_items=n_items)

    dataset = InteractionDataset(train_df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TwoTowerModel(n_users=n_users, n_items=n_items, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    get_experiment()
    all_results: dict[str, dict[str, float]] = {}

    with start_run("two-tower", tags={"event": event, "size": size}):
        with start_run(
            f"two-tower-{embedding_dim}d-{neg_sampler}",
            nested=True,
            params={
                "embedding_dim": embedding_dim,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": epochs,
                "neg_sampler": neg_sampler,
                "n_negatives": n_negatives,
                "seed": seed,
            },
        ):
            for epoch in range(epochs):
                step_losses = []
                for user_idx_batch, pos_item_idx_batch in loader:
                    user_idx_batch = user_idx_batch.to(device)
                    pos_item_idx_batch = pos_item_idx_batch.to(device)

                    neg_idx = _sample_explicit_negatives(
                        neg_sampler, random_sampler, popularity_sampler,
                        user_idx_batch, n_negatives, pos_item_idx_batch, device,
                    )

                    user_emb = model.user_tower(user_idx_batch)
                    pos_emb = model.item_tower(pos_item_idx_batch)
                    b, k = neg_idx.shape
                    neg_emb = model.item_tower(neg_idx.reshape(-1)).reshape(b, k, -1)

                    loss = sampled_softmax_loss(user_emb, pos_emb, pos_item_idx_batch, neg_emb)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step_losses.append(loss.item())

                epoch_loss = fmean(step_losses)
                print(f"  epoch {epoch + 1}/{epochs}  train/loss={epoch_loss:.4f}")
                log_metrics({"train/loss": epoch_loss}, step=epoch)

                if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                    val_metrics = evaluate_split(model, val_users, val_relevant, seen_by_user, device)
                    log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
                    print(f"    val/recall@100={val_metrics['recall@100']:.4f}")

            test_metrics = evaluate_split(model, test_users, test_relevant, seen_by_user, device)
            log_metrics({f"test/{k}": v for k, v in test_metrics.items()}, step=epochs - 1)

            all_results["two-tower"] = {
                **{f"val/{k}": v for k, v in val_metrics.items()},
                **{f"test/{k}": v for k, v in test_metrics.items()},
            }

        # --- Popularity baseline, same splits, for an apples-to-apples comparison ---
        pop = PopularityBaseline()
        pop.fit(train_df)
        pop_val = evaluate(
            [pop.recommend(uid, n=100, seen_items=seen_by_user.get(uid, frozenset())) for uid in val_users],
            val_relevant, ks=(10, 100), metrics=("recall", "ndcg", "mrr"),
        )
        pop_test = evaluate(
            [pop.recommend(uid, n=100, seen_items=seen_by_user.get(uid, frozenset())) for uid in test_users],
            test_relevant, ks=(10, 100), metrics=("recall", "ndcg", "mrr"),
        )
        all_results["popularity"] = {
            **{f"val/{k}": v for k, v in pop_val.items()},
            **{f"test/{k}": v for k, v in pop_test.items()},
        }

    # --- Print comparison table ---
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
    parser = argparse.ArgumentParser(description="Train the two-tower retrieval model.")
    parser.add_argument("--event", default="likes")
    parser.add_argument("--size", default="50m")
    parser.add_argument("--type", dest="fmt", default="flat")
    parser.add_argument("--root", default="data/processed")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--neg-sampler", choices=["random", "popularity", "both"], default="popularity",
        help="'both' draws --n-negatives from EACH sampler, for an effective K = 2 * n_negatives.",
    )
    parser.add_argument("--n-negatives", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        event=args.event,
        size=args.size,
        fmt=args.fmt,
        root=args.root,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        neg_sampler=args.neg_sampler,
        n_negatives=args.n_negatives,
        eval_every=args.eval_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
