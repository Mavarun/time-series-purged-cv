"""Purged K-Fold splitter for time series with overlapping labels."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


def _purge_and_embargo(train_indices, test_indices, starts, ends, embargo_td):
    """Inline purge (+ optional embargo) used before helpers are extracted."""
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if len(train_indices) == 0 or len(test_indices) == 0:
        return train_indices
    test_starts = starts.iloc[test_indices]
    test_ends = ends.iloc[test_indices]
    test_start_min = test_starts.min()
    test_end_max = test_ends.max()
    kept = []
    for idx in train_indices:
        t0, t1 = starts.iloc[idx], ends.iloc[idx]
        if t1 < test_start_min or t0 > test_end_max:
            kept.append(int(idx))
            continue
        if not bool(((t0 <= test_ends) & (test_starts <= t1)).any()):
            kept.append(int(idx))
    purged = np.asarray(kept, dtype=int)
    if embargo_td is None or embargo_td == 0:
        return purged
    if isinstance(embargo_td, (int, float, np.integer, np.floating)):
        embargo_stop = test_end_max + embargo_td
    else:
        embargo_stop = test_end_max + pd.Timedelta(embargo_td)
    starts_tr = starts.iloc[purged]
    mask = ~((starts_tr > test_end_max) & (starts_tr <= embargo_stop))
    return np.asarray(purged[np.asarray(mask)], dtype=int)


class PurgedKFold(BaseCrossValidator):
    """K-Fold over contiguous time folds with purge (+ optional embargo)."""

    def __init__(
        self,
        n_splits: int = 5,
        label_start_times: pd.Series | np.ndarray | None = None,
        label_end_times: pd.Series | np.ndarray | None = None,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_splits = n_splits
        self.label_start_times = label_start_times
        self.label_end_times = label_end_times
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _resolve_times(self, n_samples: int) -> tuple[pd.Series, pd.Series]:
        idx = pd.RangeIndex(n_samples)
        if self.label_start_times is None:
            starts = pd.Series(np.arange(n_samples), index=idx)
        else:
            starts = pd.Series(np.asarray(self.label_start_times), index=idx)
        if self.label_end_times is None:
            ends = starts.copy()
        else:
            ends = pd.Series(np.asarray(self.label_end_times), index=idx)
        if len(starts) != n_samples or len(ends) != n_samples:
            raise ValueError("label start/end times must match n_samples")
        return starts, ends

    def _embargo_td(self, starts: pd.Series, ends: pd.Series, n_samples: int):
        if self.embargo_pct <= 0:
            return 0
        if np.issubdtype(starts.dtype, np.datetime64) or isinstance(
            starts.iloc[0], (pd.Timestamp, np.datetime64)
        ):
            span = ends.max() - starts.min()
            return pd.Timedelta(span) * float(self.embargo_pct)
        return int(np.ceil(self.embargo_pct * n_samples))

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        if n_samples < self.n_splits:
            raise ValueError("Not enough samples for the requested n_splits")
        starts, ends = self._resolve_times(n_samples)
        embargo_td = self._embargo_td(starts, ends, n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        boundaries = np.cumsum(fold_sizes)
        all_idx = np.arange(n_samples)
        prev = 0
        for end in boundaries:
            test_indices = all_idx[prev:end]
            train_indices = np.concatenate([all_idx[:prev], all_idx[end:]])
            train_indices = _purge_and_embargo(
                train_indices, test_indices, starts, ends, embargo_td
            )
            yield train_indices, test_indices
            prev = end
