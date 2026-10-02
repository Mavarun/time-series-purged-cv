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

## Slice 2 (2026-10-03): CPCV, PBO, deflated Sharpe, leakage tests

### What was added

- **`CombinatorialPurgedKFold`** (`cpcv.py`): all `C(N,k)` splits over `N` contiguous groups, purge by label-interval overlap, **embargo after every test block** (not just the last one). The vectorised envelope purge is tested equal to the brute-force per-sample purge; `k=1` is tested equal to `PurgedKFold`.
- **CPCV backtest paths**: `cpcv_oos_predictions` stitches `C(N-1,k-1)` complete out-of-sample prediction series. A memoriser estimator proves no value on any path comes from a model trained on that row.
- **PBO via CSCV** (`overfitting.py`, Bailey-Borwein-López de Prado-Zhu): `C(S,S/2)` block splits, vectorised per-block Sharpe statistics; reports PBO, logits, P(OOS loss), IS->OOS degradation slope.
- **PSR / DSR / min track record** (`sharpe.py`): reproduces the Bailey & López de Prado (2014) worked example (`E[max SR]=0.1132`, `DSR=0.9004`).
- **Leakage experiment** (`leakage.py`): autocorrelated null labels, features independent of labels by construction.
- **Selection diagnostics with costs** (`selection.py`): best-of-40 causal sign rules on AR(1) returns, net of per-trade costs; DSR, PBO and walk-forward OOS.

### Test design

| Experiment | Ground truth | Data | What is compared |
|---|---|---|---|
| Leakage | accuracy = 0.5 (features independent of labels) | 1500 samples, labels = sign of next-20 iid shocks (lag-1 label corr > 0.8), 3 AR(1) features with phi=0.98, 3 seeds | k-NN scored by shuffled 5-fold, purged 5-fold (1% embargo), CPCV(6,2) paths, and a fresh independent draw |
| Leakage controls | same | horizon=1 (no overlap) or feature phi=0 (no persistence) | naive - purged gap should vanish |
| Selection (null) | no rule has an edge (phi=0) | 2520 daily bars, vol 1%, 40 rules = 10 lookbacks x long/short x {0, 0.5} z-threshold | full-sample best net Sharpe vs DSR, PBO (CSCV, S=16), walk-forward (504-bar train, 63-bar step) OOS net Sharpe |
| Selection (edge) | lag-1 momentum edge (phi=0.1) | same | same, at 2 and 5 bps per unit position change |

Costs: `cost_bps / 1e4 * |delta position|` on every trade, including walk-forward switches. All Sharpe numbers below are annualised (sqrt(252)) and **net of costs** unless marked gross.

### Results (`python scripts/run_overfitting_diagnostics.py`, seeds fixed)

Leakage (k-NN, h=20, mean of 3 seeds; truth 0.5):

| naive shuffled KFold | purged KFold | CPCV path mean | fresh holdout |
|---|---|---|---|
| **0.705** | 0.505 | 0.501 | 0.496 |

Controls: naive - purged gap is +0.001 with horizon 1 and +0.019 with iid features. Leakage needs **both** overlapping labels and serially correlated features. Across seeds the gap grows with label overlap (h = 1 < 5 < 20, about 0.00 / 0.12 / 0.20).

Best-of-40 rule selection (seed 1):

| phi | cost bps | best trial (L, dir, z) | IS best net SR | PSR(0) | DSR (null var) | DSR (x-sec var) | PBO | WF OOS net SR | WF OOS gross SR |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 2 | (89, +1, 0.0) | 0.46 | 0.925 | 0.227 | 0.384 | 0.583 | -0.12 | 0.01 |
| 0.00 | 5 | (89, +1, 0.0) | 0.43 | 0.910 | 0.199 | 0.185 | 0.459 | -0.46 | -0.27 |
| 0.10 | 2 | (1, +1, 0.5) | 1.34 | 1.000 | 0.977 | 0.045 | 0.044 | 1.32 | 1.55 |
| 0.10 | 5 | (1, +1, 0.0) | 0.88 | 0.997 | 0.703 | 0.001 | 0.101 | 0.63 | 1.07 |

Reading: on the null, the full-sample winner shows an attractive 0.4-0.5 Sharpe, but DSR stays far below 0.95, PBO is about 0.5, and walk-forward OOS net Sharpe is <= 0. With a planted edge and low costs, all three diagnostics agree it is real. At 5 bps the same edge's walk-forward net Sharpe roughly halves (1.32 -> 0.63), and DSR no longer clears 0.95.

### Weaknesses (honest)

- Everything is synthetic. No market data is used in tests, so these are method checks, not evidence of any tradable edge.
- **Cross-sectional DSR over-deflates mirrored grids.** Long/short mirror images of a real edge inflate the cross-trial Sharpe variance, so the paper's default variance rejects a genuine strategy (0.045 above). The null-variance DSR is reported alongside, and a test pins this behaviour. Neither variance choice models trial correlation, so the effective number of trials is overstated (conservative).
- The PBO metric is per-period Sharpe on equal contiguous blocks. Leftover rows are dropped, and S=16 means 12,870 splits (~2 s).
- Leakage results average 3 seeds because overlapping labels leave only ~n/h independent observations. Single-seed fold scores swing by about ±0.05.
- The walk-forward selector uses trailing net Sharpe only, with no shrinkage or turnover penalty. Costs are linear bps with no impact model.
- Z-thresholded rules scale by the full-sample return std (a mild global input; zero-threshold rules are strictly causal and tested as such).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
python scripts/run_purged_cv_slice.py
python scripts/run_purged_cv_slice.py --json
python scripts/run_overfitting_diagnostics.py          # CPCV / PBO / DSR / leakage tables
python scripts/run_overfitting_diagnostics.py --json
# optional market data (network; cite Yahoo Finance / yfinance):
python scripts/run_purged_cv_slice.py --yfinance --ticker SPY
```

## Package layout

```
src/purged_cv/
  split.py      # PurgedKFold
  cpcv.py       # CombinatorialPurgedKFold + OOS backtest-path assembly
  overfitting.py# PBO via CSCV
  sharpe.py     # PSR, deflated Sharpe, min track record length
  leakage.py    # naive vs purged vs CPCV on autocorrelated null labels
  selection.py  # best-of-N rule selection with costs: DSR, PBO, walk-forward
  embargo.py    # purge / embargo helpers
  weights.py    # optional average-uniqueness sample weights
  data.py       # synthetic AR + optional yfinance helpers
  model.py      # logistic / ridge return-sign classifiers
  metrics.py    # naive vs purged comparison
scripts/run_purged_cv_slice.py
scripts/run_overfitting_diagnostics.py
tests/
.github/workflows/tests.yml   # offline pytest on 3.11 / 3.12
```

## Limits / what is still weak

- Single linear return-sign model in the original slice; no hyperparameter search or portfolio construction. Costs are modelled only in the selection diagnostics (linear bps).
- Embargo length is a crude fraction of the sample span, not an asset-specific correlation time.
- Average-uniqueness weights use a coarse discrete grid (research-grade, not a production event-time engine).
- Synthetic AR(1) is a citation-clear null-ish demo; yfinance path needs network and is still not a trading signal.
- Gap on honest lag features is small; the large gap is easiest to see on the engineered-leakage fixture.
- No claim of production accuracy or live PnL.

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Purged CV / embargo, ch. 7; CPCV, ch. 12.)
- Bailey, D. H., Borwein, J. M., López de Prado, M., Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4).
- Bailey, D. H., López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5).
- Bailey, D. H., López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk* 15(2).
- Synthetic data: in-process AR(1). Optional bars: Yahoo Finance via `yfinance`.
