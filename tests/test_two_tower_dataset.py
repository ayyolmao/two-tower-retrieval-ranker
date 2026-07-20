"""Unit tests for InteractionDataset (src/training/two_tower_dataset.py).

Pins the invariants a DataLoader depends on:
  * Only label==1 rows are used as training positives (matches preprocess.py's
    convention that label==0 rows carry no positive-affinity signal).
  * Rows are NOT deduplicated -- repeat interactions get more training weight,
    same philosophy as MFBaseline's confidence counts.
  * DataLoader batching with drop_last=True yields fixed-size (B,) int64 pairs.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training.two_tower_dataset import InteractionDataset


def _make_train_df(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    """rows: (user_idx, item_idx, label)."""
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])
    df["timestamp"] = range(len(df))
    df["is_organic"] = 1
    df["label"] = df["label"].astype("int8")
    df["user_idx"] = df["user_idx"].astype("int32")
    df["item_idx"] = df["item_idx"].astype("int32")
    return df


class TestInteractionDataset:
    def test_length_counts_only_label_one_rows(self):
        df = _make_train_df([(0, 1, 1), (0, 2, 0), (1, 3, 1), (1, 4, 0)])
        dataset = InteractionDataset(df)
        assert len(dataset) == 2

    def test_length_does_not_dedupe_repeat_positives(self):
        df = _make_train_df([(0, 1, 1), (0, 1, 1), (0, 1, 1)])
        dataset = InteractionDataset(df)
        assert len(dataset) == 3

    def test_getitem_returns_exact_pair_in_order(self):
        df = _make_train_df([(0, 5, 1), (2, 9, 0), (3, 7, 1)])
        dataset = InteractionDataset(df)
        # label==1 rows only, in original order: (0,5), (3,7)
        assert dataset[0] == (0, 5)
        assert dataset[1] == (3, 7)


class TestDataLoaderIntegration:
    def test_batch_shapes_and_dtypes(self):
        df = _make_train_df([(i % 3, i % 5, 1) for i in range(20)])
        dataset = InteractionDataset(df)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
        user_idx_batch, pos_item_idx_batch = next(iter(loader))
        assert user_idx_batch.shape == (8,)
        assert pos_item_idx_batch.shape == (8,)
        assert user_idx_batch.dtype == torch.int64
        assert pos_item_idx_batch.dtype == torch.int64

    def test_drop_last_discards_ragged_final_batch(self):
        df = _make_train_df([(i % 3, i % 5, 1) for i in range(10)])
        dataset = InteractionDataset(df)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
        batches = list(loader)
        assert len(batches) == 1  # 10 rows / batch_size 8 -> 1 full batch, remainder dropped
