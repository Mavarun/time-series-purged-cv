"""Label-ACF embargo length and the two-sided embargo (slice 7)."""

import numpy as np
import pytest

from purged_cv.embargo import acf_embargo_bars, label_acf, pre_embargo_train_indices
from purged_cv.leakage import make_autocorrelated_null_dataset as make
from purged_cv.split import PurgedKFold


def test_acf_is_one_at_lag_zero_and_handles_constant_labels():
    a = label_acf(np.array([0, 1] * 50), 3)
    assert a[0] == 1.0 and a[1] < -0.9
    assert np.all(label_acf(np.ones(50), 2)[1:] == 0)


def test_iid_shocks_need_little_or_no_embargo():
    bars = [acf_embargo_bars(make(n_samples=1500, horizon=10, seed=s).y, 10) for s in range(5)]
    assert max(bars) <= 5


def test_persistent_shocks_need_a_longer_embargo():
    lo = np.median([acf_embargo_bars(make(n_samples=1500, horizon=10, seed=s, shock_phi=0.5).y, 10) for s in range(5)])
    hi = np.median([acf_embargo_bars(make(n_samples=1500, horizon=10, seed=s, shock_phi=0.9).y, 10) for s in range(5)])
    assert hi > lo and hi >= 10


def test_never_inside_band_returns_cap():
    y = np.repeat([0, 1], 200)  # one switch: ACF stays high
    assert acf_embargo_bars(y, 5, max_lag=50) == 45


def test_acf_embargo_validation():
    with pytest.raises(ValueError):
        acf_embargo_bars(np.zeros(10), 5)
    with pytest.raises(ValueError):
        acf_embargo_bars(np.zeros(100), 0)


def test_pre_embargo_drops_rows_ending_just_before_the_test_fold():
    starts = np.arange(100)
    ends = starts + 2
    train = np.r_[np.arange(0, 40), np.arange(60, 100)]
    test = np.arange(40, 60)
    kept = pre_embargo_train_indices(train, test, starts, ends, 5)
    # label ends in [35, 40) -> starts 33..37 dropped
    assert set(train) - set(kept) == {33, 34, 35, 36, 37}


def test_purged_kfold_defaults_unchanged_and_both_sides_drop_more():
    d = make(n_samples=600, horizon=5, seed=0, shock_phi=0.8)
    s, e = d.label_start_times, d.label_end_times
    base = list(PurgedKFold(5, s, e).split(d.X, d.y))
    same = list(PurgedKFold(5, s, e, 0.01, None, "after").split(d.X, d.y))
    for (a, _), (b, _) in zip(base, same):
        np.testing.assert_array_equal(a, b)
    after = list(PurgedKFold(5, s, e, embargo_bars=20).split(d.X, d.y))
    both = list(PurgedKFold(5, s, e, embargo_bars=20, embargo_side="both").split(d.X, d.y))
    assert sum(len(a) for a, _ in both) < sum(len(a) for a, _ in after)
    # no train row's label ends within 20 bars before a later test fold
    for tr, te in both:
        lo = te.min()
        assert not np.any((e.values[tr] >= lo - 20) & (e.values[tr] < lo))


def test_acf_mode_records_the_length_and_needs_y():
    d = make(n_samples=600, horizon=5, seed=0, shock_phi=0.9)
    cv = PurgedKFold(5, d.label_start_times, d.label_end_times, embargo_bars="acf")
    list(cv.split(d.X, d.y))
    assert cv.last_embargo_ == acf_embargo_bars(d.y, 5)
    with pytest.raises(ValueError):
        list(cv.split(d.X))


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        PurgedKFold(5, embargo_bars=-1)
    with pytest.raises(ValueError):
        PurgedKFold(5, embargo_bars="auto")
    with pytest.raises(ValueError):
        PurgedKFold(5, embargo_side="before")
