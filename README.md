# time-series-purged-cv

Purged and embargoed time-series cross-validation so ML on overlapping financial labels does not silently leak.

## Hypothesis

1. Standard KFold on overlapping financial labels leaks future info into train and inflates OOS scores.
2. Purged + embargoed time-series CV (Lopez de Prado style) removes that leakage and lowers the optimistic gap.
3. A simple ridge/logistic return-sign model will show higher naive-CV score than purged-CV score on the same data; report both honestly.

## Method

- **Labels:** sign of the next `horizon`-bar return sum (overlapping when `horizon > 1`).
- **Naive CV:** `sklearn.model_selection.KFold(shuffle=True)` — mixes time and ignores label overlap.
- **Purged CV:** contiguous folds; train samples whose label interval overlaps the test fold are **purged**; an **embargo** window after each test fold is also dropped from train (AFML / Lopez de Prado).
- **Model:** `StandardScaler` + `LogisticRegression` or `RidgeClassifier` (no PyTorch).
- **Optional weights:** `|forward_return| * average_uniqueness` over the label interval.
- **Data (default):** synthetic AR(1) returns generated in-process (reproducible, offline). Optional `--yfinance` pulls Yahoo Finance daily bars via `yfinance`.

This repository is a **research slice**. It does **not** claim production accuracy or live PnL.

## Metrics (reproducible synthetic run)

Command: `python scripts/run_purged_cv_slice.py`  
Settings: `n_bars=800`, `horizon=10`, `n_lags=5`, `n_splits=5`, `embargo_pct=0.01`, model=`logistic`, seed=`42`.

| CV | Mean accuracy | Std | Fold scores |
|----|---------------|-----|-------------|
| Naive shuffled KFold | **0.5139** | 0.0399 | 0.576, 0.513, 0.456, 0.532, 0.494 |
| Purged + embargoed | **0.5013** | 0.0281 | 0.462, 0.481, 0.506, 0.544, 0.513 |
| Gap (naive − purged) | **0.0127** | | |

On lag-only AR features the edge is small (near chance), but the naive score is still slightly optimistic vs purged — consistent with hypothesis (3).

**Engineered-leakage stress test** (`make_engineered_leakage_dataset`, interior label-path factors as features): naive **0.666** vs purged **0.601** (gap **0.065**). Pytest asserts naive exceeds purged by > 0.04 on this fixture.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
python scripts/run_purged_cv_slice.py
python scripts/run_purged_cv_slice.py --json
# optional market data (network; cite Yahoo Finance / yfinance):
python scripts/run_purged_cv_slice.py --yfinance --ticker SPY
```

## Package layout

```
src/purged_cv/
  split.py      # PurgedKFold
  embargo.py    # purge / embargo helpers
  weights.py    # optional average-uniqueness sample weights
  data.py       # synthetic AR + optional yfinance helpers
  model.py      # logistic / ridge return-sign classifiers
  metrics.py    # naive vs purged comparison
scripts/run_purged_cv_slice.py
tests/
```

## Limits / what is still weak

- Single linear return-sign model; no hyperparameter search, no transaction costs, no portfolio construction.
- Embargo length is a crude fraction of the sample span, not an asset-specific correlation time.
- Average-uniqueness weights use a coarse discrete grid (research-grade, not a production event-time engine).
- Synthetic AR(1) is a citation-clear null-ish demo; yfinance path needs network and is still not a trading signal.
- Gap on honest lag features is small; the large gap is easiest to see on the engineered-leakage fixture.
- No claim of production accuracy or live PnL.

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Purged CV / embargo.)
- Synthetic data: in-process AR(1). Optional bars: Yahoo Finance via `yfinance`.
