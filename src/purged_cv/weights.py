"""Optional sample weights for overlapping labels (AFML-style uniqueness)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_uniqueness(
    label_start_times: pd.Series,
    label_end_times: pd.Series,
) -> pd.Series:
    """Estimate average label uniqueness over each sample's label interval.

    For each timestamp t covered by at least one label, concurrency(t) =
    number of labels covering t. A sample's average uniqueness is the mean of
    1/concurrency(t) over t in its [start, end] interval.

    Integer / ordinal times are treated as discrete bars. Datetime times are
    mapped onto a sorted union of endpoints (coarse but sufficient for this
    research slice - not a production event-time engine).
    """
    starts = pd.Series(label_start_times).reset_index(drop=True)
    ends = pd.Series(label_end_times).reset_index(drop=True)
    n = len(starts)
    if n == 0:
        return pd.Series(dtype=float)

    if np.issubdtype(starts.dtype, np.datetime64) or isinstance(
        starts.iloc[0], (pd.Timestamp, np.datetime64)
    ):
        # Coarse grid from sorted unique timestamps.
        grid = pd.Index(sorted(set(starts).union(set(ends))))
        positions = {ts: i for i, ts in enumerate(grid)}
        start_pos = starts.map(positions).to_numpy()
        end_pos = ends.map(positions).to_numpy()
        length = len(grid)
    else:
        start_pos = starts.astype(int).to_numpy()
        end_pos = ends.astype(int).to_numpy()
        length = int(end_pos.max()) + 1

    concurrency = np.zeros(length, dtype=float)
    for s, e in zip(start_pos, end_pos):
        concurrency[s : e + 1] += 1.0

    inv = np.zeros_like(concurrency)
    positive = concurrency > 0
    inv[positive] = 1.0 / concurrency[positive]

    out = np.empty(n, dtype=float)
    for i, (s, e) in enumerate(zip(start_pos, end_pos)):
        segment = inv[s : e + 1]
        out[i] = float(segment.mean()) if len(segment) else 0.0
    return pd.Series(out, name="avg_uniqueness")


def sample_weights_from_returns(
    returns: np.ndarray | pd.Series,
    label_start_times: pd.Series,
    label_end_times: pd.Series,
    normalize: bool = True,
) -> np.ndarray:
    """Weight samples by |outcome| scaled by average uniqueness (optional).

    ``returns`` should align with samples (e.g. forward return used for the
    label). Weights are non-negative. When normalize=True they sum to n.
    """
    rets = np.asarray(returns, dtype=float)
    uniq = average_uniqueness(label_start_times, label_end_times).to_numpy()
    w = np.abs(rets) * uniq
    w = np.clip(w, 0.0, None)
    if w.sum() == 0:
        w = np.ones_like(w)
    if normalize:
        w = w * (len(w) / w.sum())
    return w
