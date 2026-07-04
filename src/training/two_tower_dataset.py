"""PyTorch Dataset over (user_idx, item_idx) positive interaction pairs.

One row per label==1 interaction, matching src/data/preprocess.py's
one-row-per-interaction, non-deduplicated convention -- repeat interactions
naturally get more training weight, the same signal-density philosophy
MFBaseline's confidence-count weighting and PopularityBaseline's raw counts
already use. Only label==1 rows are used as positives (label==0 rows carry
no positive-affinity signal for the softmax target, unlike the seen-item
exclusion sets built elsewhere from ALL rows regardless of label).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
    ) -> None:
        positives = df[df["label"] == 1]
        self.user_idx = positives[user_col].to_numpy(dtype=np.int64)
        self.item_idx = positives[item_col].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return int(self.user_idx[idx]), int(self.item_idx[idx])
