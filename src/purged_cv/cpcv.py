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

from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np
import pandas as pd
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
