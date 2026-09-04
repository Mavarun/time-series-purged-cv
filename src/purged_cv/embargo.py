"""Purge and embargo helpers for overlapping financial labels.

Follows the Lopez de Prado (Advances in Financial Machine Learning) construction:
- Purge: drop train samples whose label interval overlaps the test label interval.
- Embargo: after the test period, drop a further window of samples from train
  to damp serial correlation that would otherwise leak across the boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(times: pd.Series | np.ndarray | list, index: pd.Index | None = None) -> pd.Series:
    if isinstance(times, pd.Series):
        return times.reset_index(drop=True) if index is None else times
    if index is None:
        index = pd.RangeIndex(len(times))
    return pd.Series(times, index=index)


def purge_train_indices(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_start_times: pd.Series,
    label_end_times: pd.Series,
) -> np.ndarray:
    """Remove train indices whose label interval overlaps any test label interval.

    Intervals are closed: train [t0, t1] overlaps test [u0, u1] when
    t0 <= u1 and u0 <= t1.
    """
    label_start_times = pd.Series(label_start_times).reset_index(drop=True)
    label_end_times = pd.Series(label_end_times).reset_index(drop=True)

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if len(train_indices) == 0 or len(test_indices) == 0:
        return train_indices

    test_starts = label_start_times.iloc[test_indices]
    test_ends = label_end_times.iloc[test_indices]
    test_start_min = test_starts.min()
    test_end_max = test_ends.max()

    kept: list[int] = []
    for idx in train_indices:
        t0 = label_start_times.iloc[idx]
        t1 = label_end_times.iloc[idx]
        if t1 < test_start_min or t0 > test_end_max:
            kept.append(int(idx))
            continue
        overlaps = bool(((t0 <= test_ends) & (test_starts <= t1)).any())
        if not overlaps:
            kept.append(int(idx))
    return np.asarray(kept, dtype=int)


def embargo_train_indices(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_start_times: pd.Series,
    label_end_times: pd.Series,
    embargo_td: pd.Timedelta | float | int,
) -> np.ndarray:
    """Drop train samples whose label start falls inside the post-test embargo.

    Embargo window: (max_test_label_end, max_test_label_end + embargo_td].
    ``embargo_td`` may be a Timedelta (datetime index) or a numeric bar count.
    """
    return embargo_train_indices_with_starts(
        train_indices, test_indices, label_start_times, label_end_times, embargo_td
    )


def embargo_train_indices_with_starts(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_start_times: pd.Series,
    label_end_times: pd.Series,
    embargo_td: pd.Timedelta | float | int,
) -> np.ndarray:
    """Embargo using explicit label start and end times."""
    label_start_times = pd.Series(label_start_times).reset_index(drop=True)
    label_end_times = pd.Series(label_end_times).reset_index(drop=True)

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if len(train_indices) == 0 or len(test_indices) == 0:
        return train_indices

    test_end_max = label_end_times.iloc[test_indices].max()
    if isinstance(embargo_td, (int, float, np.integer, np.floating)):
        embargo_stop = test_end_max + embargo_td
    else:
        embargo_stop = test_end_max + pd.Timedelta(embargo_td)

    starts = label_start_times.iloc[train_indices]
    mask = ~((starts > test_end_max) & (starts <= embargo_stop))
    return np.asarray(train_indices[np.asarray(mask)], dtype=int)


def apply_purge_and_embargo(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_start_times: pd.Series,
    label_end_times: pd.Series,
    embargo_td: pd.Timedelta | float | int | None = None,
) -> np.ndarray:
    """Purge overlapping label intervals, then optionally apply embargo."""
    purged = purge_train_indices(
        train_indices, test_indices, label_start_times, label_end_times
    )
    if embargo_td is None or embargo_td == 0:
        return purged
    return embargo_train_indices_with_starts(
        purged, test_indices, label_start_times, label_end_times, embargo_td
    )
