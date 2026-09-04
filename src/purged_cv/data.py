"""Data helpers: synthetic AR returns and optional yfinance daily series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LabeledDataset:
    """Feature matrix, binary sign labels, and label intervals."""

    X: np.ndarray
    y: np.ndarray
    forward_returns: np.ndarray
    label_start_times: pd.Series
    label_end_times: pd.Series
    feature_names: list[str]
    meta: dict


def make_horizon_labels(
    returns: np.ndarray | pd.Series,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary sign labels from the next ``horizon``-bar simple return sum.

    Returns (y, forward_returns) each of length len(returns) - horizon.
    y = 1 if forward sum > 0 else 0. Zero-sum bars are labeled 0.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    r = np.asarray(returns, dtype=float)
    n = len(r) - horizon
    if n <= 0:
        raise ValueError("series too short for horizon")
    fwd = np.array([r[i + 1 : i + 1 + horizon].sum() for i in range(n)])
    y = (fwd > 0).astype(int)
    return y, fwd


def make_synthetic_ar_dataset(
    n_bars: int = 800,
    horizon: int = 10,
    n_lags: int = 5,
    ar_coef: float = 0.25,
    shock_scale: float = 0.01,
    leak_strength: float = 0.0,
    seed: int = 42,
) -> LabeledDataset:
    """Synthetic AR(1) returns with horizon sign labels and lag features.

    Parameters
    ----------
    leak_strength :
        If > 0, appends scaled interior-horizon residual features that sit
        inside the label path (for stress tests). Leave at 0 for the honest
        research slice demo.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, shock_scale, size=n_bars)
    returns = np.zeros(n_bars)
    for t in range(1, n_bars):
        returns[t] = ar_coef * returns[t - 1] + shocks[t]

    y, fwd = make_horizon_labels(returns, horizon)
    n = len(y)

    lags = np.zeros((n, n_lags), dtype=float)
    for i in range(n):
        for k in range(n_lags):
            src = i - k
            lags[i, k] = returns[src] if src >= 0 else 0.0

    feature_names = [f"lag_{k+1}" for k in range(n_lags)]
    X = lags

    if leak_strength > 0:
        interior = []
        names = []
        for frac, name in [
            (0.2, "leak_q1"),
            (0.4, "leak_q2"),
            (0.6, "leak_q3"),
            (0.8, "leak_q4"),
        ]:
            j = max(1, int(horizon * frac))
            interior.append(
                leak_strength * np.array([shocks[i + j] for i in range(n)], dtype=float)
            )
            names.append(name)
        X = np.column_stack([X] + interior)
        feature_names = feature_names + names

    starts = pd.Series(np.arange(n), name="label_start")
    ends = pd.Series(np.arange(n) + horizon, name="label_end")

    return LabeledDataset(
        X=X,
        y=y,
        forward_returns=fwd,
        label_start_times=starts,
        label_end_times=ends,
        feature_names=feature_names,
        meta={
            "source": "synthetic_ar1",
            "n_bars": n_bars,
            "horizon": horizon,
            "ar_coef": ar_coef,
            "shock_scale": shock_scale,
            "leak_strength": leak_strength,
            "seed": seed,
            "citation": "Synthetic AR(1) returns generated in-process; not market data.",
        },
    )


def make_engineered_leakage_dataset(
    n_samples: int = 1000,
    horizon: int = 25,
    seed: int = 0,
) -> LabeledDataset:
    """Fixture where naive shuffled KFold is optimistically biased vs purged CV.

    Construction
    ------------
    A latent bar-level factor ``f[t]`` drives overlapping horizon labels::

        y[i] = 1{ sum_{j=0..h-1} f[i+j] > 0 }

    Features include several *interior* factor values from inside that same
    label window (plus a time index). Those features are legitimate same-row
    functions of the label path, so a linear model gets real signal - but
    because labels overlap, shuffled KFold also benefits from train/test
    mixing of shared path information. Purged + embargoed contiguous folds
    strip overlapping train samples and typically score lower.

    This is a stress fixture for CV diagnostics, not a trading signal.
    """
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n_samples + horizon)
    y = np.array([1 if f[i : i + horizon].sum() > 0 else 0 for i in range(n_samples)])
    fwd = np.array([f[i : i + horizon].sum() for i in range(n_samples)])

    fracs = (0.2, 0.4, 0.6, 0.8)
    interior = []
    names = []
    for frac in fracs:
        j = max(1, int(horizon * frac))
        interior.append(np.array([f[i + j] for i in range(n_samples)], dtype=float))
        names.append(f"interior_f_{frac:.1f}")
    time_frac = np.arange(n_samples, dtype=float) / max(n_samples - 1, 1)
    X = np.column_stack(interior + [time_frac])

    starts = pd.Series(np.arange(n_samples), name="label_start")
    ends = pd.Series(np.arange(n_samples) + horizon, name="label_end")

    return LabeledDataset(
        X=X,
        y=y,
        forward_returns=fwd,
        label_start_times=starts,
        label_end_times=ends,
        feature_names=names + ["time_frac"],
        meta={
            "source": "engineered_leakage",
            "horizon": horizon,
            "seed": seed,
            "citation": "Synthetic engineered-leakage fixture for CV stress tests only.",
        },
    )


def load_yfinance_daily_dataset(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str | None = None,
    horizon: int = 5,
    n_lags: int = 5,
) -> LabeledDataset:
    """Download daily adjusted closes via yfinance and build lag / sign labels.

    Cite: data from Yahoo Finance via the ``yfinance`` library. Network
    required. Prefer ``make_synthetic_ar_dataset`` for offline reproducibility.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    rets = close.pct_change().dropna()
    y, fwd = make_horizon_labels(rets.to_numpy(), horizon)
    n = len(y)
    r = rets.to_numpy()
    lags = np.zeros((n, n_lags), dtype=float)
    for i in range(n):
        for k in range(n_lags):
            src = i - k
            lags[i, k] = r[src] if src >= 0 else 0.0

    idx = rets.index[:n]
    starts = pd.Series(idx, index=np.arange(n), name="label_start")
    end_idx = rets.index[horizon : horizon + n]
    if len(end_idx) < n:
        end_vals = list(end_idx) + [rets.index[-1]] * (n - len(end_idx))
        ends = pd.Series(end_vals, index=np.arange(n), name="label_end")
    else:
        ends = pd.Series(end_idx[:n], index=np.arange(n), name="label_end")

    return LabeledDataset(
        X=lags,
        y=y,
        forward_returns=fwd,
        label_start_times=starts,
        label_end_times=ends,
        feature_names=[f"lag_{k+1}" for k in range(n_lags)],
        meta={
            "source": "yfinance",
            "ticker": ticker,
            "start": start,
            "end": end,
            "horizon": horizon,
            "citation": "Yahoo Finance daily bars via yfinance; research only, not a trading signal.",
        },
    )
