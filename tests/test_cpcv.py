"""CPCV split counts, purge/embargo correctness, and group coverage."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from purged_cv.cpcv import CombinatorialPurgedKFold
from purged_cv.embargo import purge_train_indices
from purged_cv.split import PurgedKFold


def _times(n: int, horizon: int) -> tuple[pd.Series, pd.Series]:
    return pd.Series(np.arange(n)), pd.Series(np.arange(n) + horizon)


@pytest.mark.parametrize("n_groups,k", [(6, 2), (5, 1), (8, 3)])
def test_split_count_and_paths(n_groups, k):
    starts, ends = _times(240, 5)
    cv = CombinatorialPurgedKFold(n_groups, k, starts, ends, embargo_pct=0.0)
    splits = list(cv.split(np.zeros((240, 1))))
    assert len(splits) == cv.get_n_splits() == comb(n_groups, k)
    assert cv.n_paths == comb(n_groups - 1, k - 1)
    # each group is a test group in exactly n_paths splits
    counts = np.zeros(n_groups, dtype=int)
    for tg in cv.test_group_combinations():
        counts[list(tg)] += 1
    assert (counts == cv.n_paths).all()


def test_no_overlap_between_train_and_test_labels():
    n, h = 300, 10
    starts, ends = _times(n, h)
    cv = CombinatorialPurgedKFold(6, 2, starts, ends, embargo_pct=0.02)
    s, e = starts.to_numpy(), ends.to_numpy()
    for train, test in cv.split(np.zeros((n, 1))):
        assert len(np.intersect1d(train, test)) == 0
        # brute force: no train label interval intersects any test label interval
        for tr in train:
            assert not np.any((s[tr] <= e[test]) & (s[test] <= e[tr]))


def test_envelope_purge_matches_bruteforce_purge():
    n, h = 180, 7
    starts, ends = _times(n, h)
    cv = CombinatorialPurgedKFold(6, 2, starts, ends, embargo_pct=0.0)
    for train, test in cv.split(np.zeros((n, 1))):
        candidates = np.setdiff1d(np.arange(n), test)
        brute = purge_train_indices(candidates, test, starts, ends)
        np.testing.assert_array_equal(train, brute)


def test_embargo_applied_after_every_test_block():
    n, h, emb = 120, 0, 0.05  # point labels; embargo = ceil(0.05*120) = 6 bars
    starts, ends = _times(n, h)
    cv = CombinatorialPurgedKFold(6, 2, starts, ends, embargo_pct=emb)
    bounds = cv.group_bounds(n)
    for train, test, groups in cv.split_with_groups(np.zeros((n, 1))):
        for g in groups:
            stop = bounds[g][1]
            if g + 1 in groups or stop >= n:
                continue
            # the 6 bars after each non-terminal test block are not in train
            assert not np.isin(np.arange(stop, min(stop + 6, n)), train).any()
            # bar stop+6 is outside the embargo and (point labels) not purged
            if stop + 6 < n and not any(bounds[x][0] <= stop + 6 < bounds[x][1] for x in groups):
                assert stop + 6 in train


def test_k1_matches_purged_kfold_when_single_block():
    # With k=1 every test set is one block, so CPCV == PurgedKFold.
    n, h = 200, 8
    starts, ends = _times(n, h)
    cpcv = CombinatorialPurgedKFold(5, 1, starts, ends, embargo_pct=0.03)
    pkf = PurgedKFold(5, starts, ends, embargo_pct=0.03)
    X = np.zeros((n, 1))
    for (tr_a, te_a), (tr_b, te_b) in zip(cpcv.split(X), pkf.split(X)):
        np.testing.assert_array_equal(te_a, te_b)
        np.testing.assert_array_equal(tr_a, tr_b)


def test_datetime_labels_supported():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    starts = pd.Series(idx)
    ends = pd.Series(idx + pd.Timedelta(days=3))
    cv = CombinatorialPurgedKFold(4, 2, starts, ends, embargo_pct=0.02)
    splits = list(cv.split(np.zeros((120, 1))))
    assert len(splits) == 6
    for train, test in splits:
        assert len(train) > 0 and len(np.intersect1d(train, test)) == 0


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(1, 1)
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(4, 4)
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(4, 2, embargo_pct=1.0)
    cv = CombinatorialPurgedKFold(3, 1, pd.Series([2, 1, 0]), pd.Series([2, 1, 0]))
    with pytest.raises(ValueError):
        list(cv.split(np.zeros((3, 1))))
