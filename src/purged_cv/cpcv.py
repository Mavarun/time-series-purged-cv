"""Combinatorial purged cross-validation (CPCV) with per-block embargo.

Reference: López de Prado, M. (2018), *Advances in Financial Machine
Learning*, ch. 12 ("Backtesting through cross-validation").

The sample (in time order) is cut into ``N = n_groups`` contiguous groups.
Every combination of ``k = n_test_groups`` groups is used once as the test
set, giving ``C(N, k)`` train/test splits. For each split:

* **Purge** - drop train samples whose label interval ``[start, end]``
  overlaps the label interval of any test sample.
* **Embargo** - after *each* contiguous test block (adjacent test groups are
  merged into one block), drop train samples whose label start falls in
  ``(block_max_end, block_max_end + embargo]``. Embargoing only after the
  last test block (as a single-fold splitter would) under-protects interior
  blocks, so it is applied block by block here.

Purge uses the block envelope ``[min start, max end]``. With label starts
sorted in time (validated) this is exactly equivalent to the per-sample
interval test in :func:`purged_cv.embargo.purge_train_indices` for train
samples outside the block, and it is vectorised.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator

from purged_cv.embargo import embargo_size


def _group_bounds(n_samples: int, n_groups: int) -> list[tuple[int, int]]:
    sizes = np.full(n_groups, n_samples // n_groups, dtype=int)
    sizes[: n_samples % n_groups] += 1
    stops = np.cumsum(sizes)
    starts = np.concatenate([[0], stops[:-1]])
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def _merge_adjacent(groups: tuple[int, ...]) -> list[list[int]]:
    blocks: list[list[int]] = []
    for g in sorted(groups):
        if blocks and blocks[-1][-1] == g - 1:
            blocks[-1].append(g)
        else:
            blocks.append([g])
    return blocks


class CombinatorialPurgedKFold(BaseCrossValidator):
    """CPCV splitter: all ``C(n_groups, n_test_groups)`` purged/embargoed splits.

    Parameters
    ----------
    n_groups :
        Number of contiguous time groups ``N`` (>= 2).
    n_test_groups :
        Groups per test set ``k`` (1 <= k < N). ``k = 1`` reduces to a purged
        K-fold with per-fold embargo.
    label_start_times, label_end_times :
        Per-sample label interval (integer bars or datetimes). Starts must be
        non-decreasing (time-ordered samples).
    embargo_pct :
        Embargo length as in :class:`purged_cv.split.PurgedKFold`.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        label_start_times: pd.Series | np.ndarray | None = None,
        label_end_times: pd.Series | np.ndarray | None = None,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("n_test_groups must satisfy 1 <= k < n_groups")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.label_start_times = label_start_times
        self.label_end_times = label_end_times
        self.embargo_pct = embargo_pct

    # ------------------------------------------------------------------ sizes
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of full OOS backtest paths: ``k/N * C(N, k) = C(N-1, k-1)``."""
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def test_group_combinations(self) -> list[tuple[int, ...]]:
        """Test-group tuples in split order."""
        return list(combinations(range(self.n_groups), self.n_test_groups))

    def group_bounds(self, n_samples: int) -> list[tuple[int, int]]:
        """Half-open ``(start, stop)`` sample bounds of each group."""
        if n_samples < self.n_groups:
            raise ValueError("Not enough samples for the requested n_groups")
        return _group_bounds(n_samples, self.n_groups)

    # ------------------------------------------------------------------ times
    def _resolve_times(self, n_samples: int) -> tuple[pd.Series, pd.Series]:
        if self.label_start_times is None:
            starts = pd.Series(np.arange(n_samples))
        else:
            starts = pd.Series(np.asarray(self.label_start_times))
        if self.label_end_times is None:
            ends = starts.copy()
        else:
            ends = pd.Series(np.asarray(self.label_end_times))
        if len(starts) != n_samples or len(ends) != n_samples:
            raise ValueError("label start/end times must match n_samples")
        if not starts.is_monotonic_increasing:
            raise ValueError("label_start_times must be non-decreasing (time order)")
        if (ends < starts).any():
            raise ValueError("label_end_times must be >= label_start_times")
        return starts, ends

    # ------------------------------------------------------------------ split
    def split_with_groups(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
        """Like :meth:`split` but also yields the test-group tuple."""
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        bounds = self.group_bounds(n_samples)
        starts, ends = self._resolve_times(n_samples)
        embargo = embargo_size(starts, ends, n_samples, self.embargo_pct)
        s_arr = starts.to_numpy()
        e_arr = ends.to_numpy()
        all_idx = np.arange(n_samples)

        for test_groups in self.test_group_combinations():
            test_mask = np.zeros(n_samples, dtype=bool)
            for g in test_groups:
                a, b = bounds[g]
                test_mask[a:b] = True
            keep = ~test_mask
            for block in _merge_adjacent(test_groups):
                a = bounds[block[0]][0]
                b = bounds[block[-1]][1]
                env_lo = s_arr[a:b].min()
                env_hi = e_arr[a:b].max()
                # purge: closed-interval overlap with the block envelope
                keep &= ~((s_arr <= env_hi) & (e_arr >= env_lo))
                # embargo: label starts just after this block's last label end
                if not (isinstance(embargo, (int, np.integer)) and embargo == 0):
                    keep &= ~((s_arr > env_hi) & (s_arr <= env_hi + embargo))
            yield all_idx[keep], all_idx[test_mask], test_groups

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for train_idx, test_idx, _ in self.split_with_groups(X, y, groups):
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Backtest-path assembly
# ---------------------------------------------------------------------------


def cpcv_path_map(cv: CombinatorialPurgedKFold) -> np.ndarray:
    """Assign splits to backtest paths.

    Returns an integer array of shape ``(n_paths, n_groups)`` whose entry
    ``[p, g]`` is the index of the split whose OOS predictions fill group
    ``g`` on path ``p``. Group ``g`` is tested in exactly ``n_paths`` splits;
    the ``j``-th of them (in split order) is used on path ``j`` (AFML 12.4).
    """
    combos = cv.test_group_combinations()
    out = np.full((cv.n_paths, cv.n_groups), -1, dtype=int)
    for g in range(cv.n_groups):
        using = [i for i, tg in enumerate(combos) if g in tg]
        out[:, g] = using
    return out


@dataclass
class CPCVPaths:
    """OOS predictions on every CPCV backtest path.

    Attributes
    ----------
    predictions :
        ``(n_paths, n_samples)``; row ``p`` is a complete out-of-sample
        prediction series assembled from the splits in ``path_map[p]``.
    path_map :
        ``(n_paths, n_groups)`` split index used for each (path, group).
    group_bounds :
        Half-open sample bounds of each group.
    """

    predictions: np.ndarray
    path_map: np.ndarray
    group_bounds: list[tuple[int, int]]

    @property
    def n_paths(self) -> int:
        return int(self.predictions.shape[0])


def cpcv_oos_predictions(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: CombinatorialPurgedKFold,
    *,
    method: str = "predict",
    sample_weight: np.ndarray | None = None,
    sample_weight_param: str = "clf__sample_weight",
) -> CPCVPaths:
    """Fit one clone per CPCV split and assemble ``n_paths`` OOS series.

    ``method`` may be ``"predict"``, ``"predict_proba"`` (positive-class
    column) or ``"decision_function"``. Every value on every path comes from
    a model whose (purged, embargoed) training set excluded that sample.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    bounds = cv.group_bounds(n)
    path_map = cpcv_path_map(cv)
    preds = np.full((cv.n_paths, n), np.nan, dtype=float)

    for split_id, (train, test, groups) in enumerate(cv.split_with_groups(X, y)):
        if len(train) == 0:
            raise ValueError(f"split {split_id} has an empty training set")
        model = clone(estimator)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs[sample_weight_param] = np.asarray(sample_weight)[train]
        model.fit(X[train], y[train], **fit_kwargs)
        out = getattr(model, method)(X[test])
        if method == "predict_proba":
            out = out[:, 1]
        values = np.full(n, np.nan)
        values[test] = np.asarray(out, dtype=float)
        for g in groups:
            a, b = bounds[g]
            path = int(np.flatnonzero(path_map[:, g] == split_id)[0])
            preds[path, a:b] = values[a:b]

    if np.isnan(preds).any():  # pragma: no cover - guarded by path_map design
        raise RuntimeError("incomplete CPCV path assembly")
    return CPCVPaths(predictions=preds, path_map=path_map, group_bounds=bounds)


def path_accuracies(paths: CPCVPaths, y: np.ndarray) -> np.ndarray:
    """Accuracy of each assembled path (hard-label predictions)."""
    y = np.asarray(y)
    return (paths.predictions == y[None, :]).mean(axis=1)
