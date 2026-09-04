"""Unit tests for purge / embargo boundaries and PurgedKFold index logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purged_cv.embargo import (
    apply_purge_and_embargo,
    embargo_train_indices_with_starts,
    purge_train_indices,
)
from purged_cv.split import PurgedKFold


def test_purge_removes_overlapping_label_intervals():
    # 10 samples; each label spans 3 bars: [i, i+2]
    starts = pd.Series(np.arange(10))
    ends = pd.Series(np.arange(10) + 2)
    train = np.array([0, 1, 2, 3, 7, 8, 9])
    test = np.array([4, 5, 6])
    purged = purge_train_indices(train, test, starts, ends)
    assert 3 not in purged
    assert 7 not in purged
    assert 2 not in purged
    assert 0 in purged
    assert 1 in purged  # [1,3] ends before test starts at 4
    assert 8 not in purged  # [8,10] overlaps test end 8
    assert 9 in purged  # [9,11] starts after test_end_max 8


def test_purge_keeps_fully_outside_samples():
    # test sample 2 occupies [10,12]; sample 3 at [13,14] is fully after (no overlap)
    starts = pd.Series([0, 1, 10, 13])
    ends = pd.Series([1, 2, 12, 14])
    train = np.array([0, 1, 2, 3])
    test = np.array([2])
    purged = purge_train_indices(train, test, starts, ends)
    np.testing.assert_array_equal(purged, np.array([0, 1, 3]))


def test_embargo_drops_post_test_window():
    starts = pd.Series(np.arange(20))
    ends = pd.Series(np.arange(20))  # point labels
    train = np.arange(0, 20)
    test = np.array([5, 6, 7])
    # test_end_max = 7; embargo_td = 3 -> drop starts in (7, 10]
    out = embargo_train_indices_with_starts(train, test, starts, ends, embargo_td=3)
    for bad in (8, 9, 10):
        assert bad not in out
    assert 7 in out  # start == test_end_max is not strictly inside embargo
    assert 11 in out


def test_apply_purge_and_embargo_combined():
    starts = pd.Series(np.arange(30))
    ends = pd.Series(np.arange(30) + 5)
    train = np.arange(30)
    test = np.array([10, 11, 12])
    out = apply_purge_and_embargo(train, test, starts, ends, embargo_td=2)
    test_start_min = starts.iloc[test].min()
    test_end_max = ends.iloc[test].max()
    for idx in out:
        t0, t1 = starts.iloc[idx], ends.iloc[idx]
        assert not (t0 <= test_end_max and test_start_min <= t1)
        assert not (t0 > test_end_max and t0 <= test_end_max + 2)


def test_purged_kfold_no_train_test_overlap_and_purge():
    n, horizon, n_splits = 100, 10, 5
    X = np.random.randn(n, 3)
    starts = pd.Series(np.arange(n))
    ends = pd.Series(np.arange(n) + horizon)
    cv = PurgedKFold(
        n_splits=n_splits,
        label_start_times=starts,
        label_end_times=ends,
        embargo_pct=0.02,
    )
    seen_test = []
    for train_idx, test_idx in cv.split(X):
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        seen_test.extend(test_idx.tolist())
        for tr in train_idx:
            for te in test_idx:
                t0, t1 = starts.iloc[tr], ends.iloc[tr]
                u0, u1 = starts.iloc[te], ends.iloc[te]
                assert not (t0 <= u1 and u0 <= t1)
    assert sorted(seen_test) == list(range(n))


def test_purged_kfold_rejects_bad_args():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, embargo_pct=1.5)
