"""Slice 8b frozen protocol and the signal generator."""

import numpy as np

from purged_cv import edge_cost as P
from purged_cv import fold_count_calibration as S8


def test_constants_frozen_and_seeds_disjoint():
    assert (P.N, P.FRESH, P.FOLDS, P.SHOCK_PHIS, P.HORIZON, P.BETA, P.MAX_COST) == (500, 5000, (5, 20), (0.0, 0.9), 20, 0.15, 0.005)
    assert not set(P.SEEDS) & set(S8.SEEDS) and len(P.SEEDS) == 100


def test_beta_zero_has_no_edge_and_beta_gives_one():
    acc0, acc1 = [], []
    for s in range(3):
        for beta, acc in ((0.0, acc0), (P.BETA, acc1)):
            d = P.make_signal_dataset(20000, beta=beta, seed=s)
            acc.append(((d.X[:, 0] > 0).astype(int) == d.y).mean())
    assert abs(np.mean(acc0) - 0.5) < 0.03
    assert np.mean(acc1) > 0.58


def test_labels_are_overlapping_sign_labels():
    d = P.make_signal_dataset(300, seed=1)
    assert (d.label_end_times - d.label_start_times == P.HORIZON).all()
    np.testing.assert_array_equal(d.y, (d.forward_returns > 0).astype(int))


def test_one_seed_and_call_shapes():
    r = P.one_seed(5, 0.9, 8100)
    assert 0.4 < r["truth"] < 0.8 and set(r["accuracy"]) == set(S8.ARMS)
    s = P.summarize([r, P.one_seed(5, 0.9, 8101)])
    assert s["purge_only"]["cost_vs_purge_only"] == 0.0
    c = P.call({(5, 0.9): s})
    assert set(c) == {"max_cost", "cheap", "costs"}
