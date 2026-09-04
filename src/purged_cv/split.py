"""Purged K-Fold splitter for time series with overlapping labels."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from purged_cv.embargo import apply_purge_and_embargo


class PurgedKFold(BaseCrossValidator):
    """K-Fold over contiguous time folds with purge (+ optional embargo).

    Test folds are contiguous blocks along the sample order (time order).
    Training indices whose label intervals overlap the test fold are purged.
    An optional embargo window after each test fold is also removed from train.

    Parameters
    ----------
    n_splits :
        Number of folds.
    label_start_times, label_end_times :
        Per-sample label interval. Length must match n_samples at split time.
        May be integer bar indices or datetimes.
    embargo_pct :
        Fraction of the full sample span used as embargo length after each
        test fold. 0 disables embargo. For integer times, embargo bars =
        ceil(embargo_pct * n_samples). For datetime, embargo =
        embargo_pct * (max_end - min_start).
    """

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
        # ordinal / integer bars
        return int(np.ceil(self.embargo_pct * n_samples))

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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
            train_indices = apply_purge_and_embargo(
                train_indices,
                test_indices,
                starts,
                ends,
                embargo_td=embargo_td,
            )
            yield train_indices, test_indices
            prev = end
