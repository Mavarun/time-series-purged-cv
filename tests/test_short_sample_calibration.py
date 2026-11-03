"""Calibration at short n and with regime-switching returns: wiring, determinism, guards."""

import numpy as np
import pytest

from purged_cv.calibration import INNOVATIONS, MIN_SCORED_BARS, selection_tests_one_seed


def test_regime_innovations_are_registered_and_run_end_to_end():
    assert "regime" in INNOVATIONS
    out = selection_tests_one_seed(0.0, 3, n=400, n_sims=200, innovations="regime", with_rc=True, n_boot=49)
    for key in ("dsr_raw", "dsr_simulated", "max_p", "rc_p"):
        assert np.isfinite(out[key])
    assert 0.0 < out["max_p"] <= 1.0 and 0.0 < out["rc_p"] <= 1.0


def test_same_seed_same_numbers_for_every_innovation_type():
    for innov in INNOVATIONS:
        a = selection_tests_one_seed(0.1, 5, n=300, n_sims=100, innovations=innov)
        b = selection_tests_one_seed(0.1, 5, n=300, n_sims=100, innovations=innov)
        assert a == b


def test_default_path_is_unchanged_by_the_new_option():
    # slice-4 default (normal innovations, long n) must not move: compare against the explicit option
    a = selection_tests_one_seed(0.0, 7, n=500, n_sims=100)
    b = selection_tests_one_seed(0.0, 7, n=500, n_sims=100, innovations="normal")
    assert a == b


def test_too_short_a_sample_is_rejected_rather_than_scored():
    warm = 90  # longest lookback 89 + 1
    with pytest.raises(ValueError, match="scored bars"):
        selection_tests_one_seed(0.0, 0, n=warm + MIN_SCORED_BARS - 1, n_sims=50)
    out = selection_tests_one_seed(0.0, 0, n=warm + MIN_SCORED_BARS, n_sims=50)
    assert np.isfinite(out["max_p"])


def test_unknown_innovation_still_rejected():
    with pytest.raises(ValueError):
        selection_tests_one_seed(0.0, 0, n=300, n_sims=50, innovations="stable")
