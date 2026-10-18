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

## Slice 3 (2026-10-12): causal vol, walk-forward shrinkage / turnover, sqrt impact

### What was added

- **Causal expanding vol** for z-thresholded rules (`expanding_vol`, default). The old full-sample return std is kept as `vol_mode="full_sample"` for ablation; it is a mild look-ahead and is no longer the default.
- **Walk-forward score modes** (`trailing_scores`, `walk_forward_select`): plain trailing net Sharpe (status quo), James-Stein shrinkage toward the cross-sectional mean, Sharpe minus a turnover penalty, and both. Default mode remains `"sharpe"` so existing diagnostics are unchanged.
- **Square-root impact costs** (`net_returns_with_impact`): `cost_bps/1e4 * |delta| + impact_coef * sqrt(|delta|)` on the same bar.
- **Ablation harness** (`run_selector_ablation`, `scripts/run_selector_ablation.py` → `artifacts/selector_ablation.{md,json}`).

### Results (mean of 4 seeds; n=2520, train=504, step=63, shrink=0.5, turnover_penalty=5)

OOS net Sharpe (annualised), linear costs, no impact:

| phi | cost bps | sharpe | shrunk | sharpe − turnover | shrunk − turnover |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 2 | -0.22 | -0.22 | **0.13** | **0.13** |
| 0.00 | 5 | -0.30 | -0.30 | **0.08** | **0.08** |
| 0.10 | 2 | **0.90** | **0.90** | 0.21 | 0.20 |
| 0.10 | 5 | **0.33** | **0.33** | 0.17 | 0.16 |

Switches roughly halve under the turnover penalty (null: 17.8 → 8.5; edge at 2 bps: 12.0 → 6.8). Shrinkage alone at intensity 0.5 does not move the needle on this grid. Adding square-root impact (`0.0005` / `0.001`) lowers every OOS Sharpe; under high impact the turnover penalty is less harmful on the edge because plain Sharpe starts churning more.

**Reading (honest):** the turnover penalty is a good *null* regulariser — it stops the walk-forward from chasing noisy high-turnover winners and flips null OOS Sharpe from negative to near zero — but a planted lag-1 momentum edge *is* high turnover, so the same penalty cuts a real edge's OOS Sharpe from ~0.9 to ~0.2. Do not turn it on by default when the candidate strategies are themselves high-frequency. Causal expanding vol vs full-sample vol changes WF OOS Sharpe only slightly on this synthetic AR (edge 0.90 vs 0.93).

### Test design

| Experiment | Ground truth | What is compared |
|---|---|---|
| Expanding vol causality | past vols unchanged when the future is rewritten | `expanding_vol` |
| Z-threshold path | expanding ≠ full-sample positions | `rule_positions(..., vol_mode=)` |
| Impact | impact_coef > 0 raises costs | `net_returns_with_impact` |
| Turnover penalty | fewer (or equal) switches vs plain Sharpe | walk-forward modes |
| Ablation script | all four modes × three impact levels | `run_selector_ablation` |

Reproduce: `python scripts/run_selector_ablation.py` (offline, seeded).

### Weaknesses (honest)

- Everything is synthetic. No market data is used in tests, so these are method checks, not evidence of any tradable edge.
- **Cross-sectional DSR over-deflates mirrored grids.** Long/short mirror images of a real edge inflate the cross-trial Sharpe variance, so the paper's default variance rejects a genuine strategy (0.045 above). The null-variance DSR is reported alongside, and a test pins this behaviour. Slice 4 adds correlation-aware trial counts. The simulated null still assumes Gaussian, homoskedastic Sharpe errors, and it estimates the trial correlation from the same sample it tests.
- The max-statistic test checks whether the *best* trial beats luck. It does not control the false-discovery rate across several "winners" (no stepwise / Romano-Wolf procedure yet).
- The PBO metric is per-period Sharpe on equal contiguous blocks. Leftover rows are dropped, and S=16 means 12,870 splits (~2 s).
- Leakage results average 3 seeds because overlapping labels leave only ~n/h independent observations. Single-seed fold scores swing by about ±0.05.
- **Turnover penalty hurts short-horizon edges, and its null benefit is not significant.** Over 20 seeds (slice 4) the null gain is within one s.e. It is available but not the walk-forward default.
- Uniform James-Stein shrinkage is exactly ranking-invariant (slice 4 pins this). Per-trial empirical-Bayes shrinkage re-ranks only ~1-3% of windows with no OOS gain. Shrinkage toward rule families (e.g. by lookback) is untested.
- Square-root impact is a one-parameter proxy, not a calibrated Almgren-Chriss model, and is not fit to any tape.
- Expanding-vol warm-up still uses the first `min_periods` bars as a floor, so the very start of the sample is only approximately causal.

## Slice 4 (2026-10-18): correlation-aware trial counts, max-statistic test, EB shrinkage

### What was added

- **Effective number of trials** (`sharpe.py`). The 40-rule grid is highly correlated (shared lookbacks, long/short mirrors, two thresholds), so "40 independent trials" overstates the selection hurdle. Two estimates:
  - eigenvalue **participation ratio** of the trial correlation (`effective_trials_participation`): counts directions; anti-conservative on mirrored grids because `max(x, -x) = |x|` is a bigger draw than one trial;
  - **simulated**: draw zero-skill Sharpes `N(0, C / (T-1))` with the grid's own correlation `C`, take the mean of the max as the DSR benchmark (`null_max_sharpe_draws`), and invert the false-strategy formula for the implied N (`implied_effective_trials`).
- `deflated_sharpe_for_selection(..., n_trials_mode="raw" | "participation" | "simulated")`. Default `raw` keeps all older numbers.
- **Max-statistic p-value** (`max_sharpe_pvalue`): P(best of the correlated zero-skill Sharpes >= the observed best), a Gaussian reality-check style test against the whole null distribution of the max.
- **Calibration study** (`calibration.py`, `scripts/run_dsr_calibration.py` → `artifacts/dsr_calibration.{md,json}`): 200 fresh seeds per cell, null false-positive rate and edge power of each hurdle, gross and net of costs.
- **Empirical-Bayes shrinkage** walk-forward modes (`eb_sharpe`, `eb_sharpe_minus_turnover`): per-trial standard error from the window's skew/kurtosis, prior variance from the window's own cross-section. A test pins that the slice-3 uniform shrinkage **cannot** change walk-forward picks (it is an affine map of the Sharpes), which is why its column always equalled plain Sharpe.
- `run_selection_diagnostics` / `run_overfitting_diagnostics.py` also report simulated-N DSR, the implied N, and the max-stat p-value.

### Calibration results (`python scripts/run_dsr_calibration.py`, 200 seeds, n=2520)

Pass = DSR > 0.95 (null Sharpe variance) or max-stat p < 0.05. Rows with phi = 0 are false-positive rates. Rows with phi > 0 are power. Costs are linear bps per unit position change.

| phi | cost bps | DSR raw N=40 | DSR participation N (~4.5) | DSR simulated N (~21) | max-stat p<0.05 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0 | 0.000 | 0.060 | 0.010 | **0.050** |
| 0.00 | 2 | 0.000 | 0.040 | 0.005 | 0.035 |
| 0.00 | 5 | 0.000 | 0.020 | 0.000 | 0.010 |
| 0.05 | 2 | 0.015 | 0.130 | 0.035 | 0.110 |
| 0.10 | 0 | 0.775 | 0.980 | 0.845 | 0.975 |
| 0.10 | 2 | 0.430 | 0.840 | 0.545 | **0.785** |
| 0.10 | 5 | 0.070 | 0.365 | 0.100 | 0.255 |

(Full table, including phi = 0.03 and binomial s.e. of about 0.015-0.035, is in `artifacts/dsr_calibration.md`.)

**Reading (honest):** raw-N DSR never fires on the null, but it also misses more than half of a real phi = 0.1 edge at 2 bps. The max-statistic test hits its nominal 5% false-positive rate on the gross null (0.050 ± 0.015). Costs make it conservative on net nulls. At 2 bps it detects the phi = 0.1 edge 78.5% of the time vs 43% for raw DSR. Simulated-N DSR is a smaller step (power 0.545, null FPR 0.010) because DSR benchmarks against the *mean* of the max, not its tail. The participation-ratio DSR has the most power but runs slightly above 5% on the gross null (0.060), as expected from the mirror argument, so it is not recommended as a hurdle. At 5 bps every hurdle loses most of its power because the edge mostly goes to costs.

Seed-1 diagnostics (`run_overfitting_diagnostics.py`): null at 2 bps has DSR 0.227 → 0.318 (simulated N = 20.6) and max-stat p = 0.80, so it is still rejected. The phi = 0.1 edge at 2 bps has DSR 0.978 → 0.989 and p = 0.0005. At 5 bps it has DSR 0.703 → 0.793 and p = 0.078, so it is still not significant. These seed-1 numbers already include slice 3's causal vol, which is why some cells differ from the slice-2 table by about 0.01.

### EB shrinkage and a re-check of slice 3 (`run_selector_ablation.py`, 4 and 20 seeds)

- EB picks a different trial than plain Sharpe in only **0.6-3%** of walk-forward windows. With 20 seeds, OOS net Sharpe moves by at most ~0.02 (paired s.e. ~0.01). **No gain.** EB stays opt-in, and the default is still plain trailing Sharpe. With 504-bar windows, the per-trial standard errors barely differ, so there is little for EB to work with.
- **Correction to slice 3:** the 4-seed claim that the turnover penalty "flips null OOS Sharpe from negative to near zero" does not hold over 20 seeds. On the null at 2 bps with no impact, the change is -0.01 ± 0.09 (-0.06 → -0.07). Under impact or at 5 bps it gains about +0.04 to +0.06, which is still within one s.e. The cost to a real edge does hold up: phi = 0.1 at 2 bps goes 0.59 → 0.02 (-0.57 ± 0.14). Tables: `artifacts/selector_ablation.md` (seeds 0-3) and `artifacts/selector_ablation_20seeds.md`.

### What ships

- On by default: nothing changes in existing outputs. The new DSR/p-value fields are extra columns.
- Opt-in: `n_trials_mode="simulated"` / `"participation"`, `max_sharpe_pvalue`, and `eb_sharpe*` walk-forward modes.
- Recommended hurdle for a correlated grid: max-stat p-value or simulated-N DSR. Not recommended: participation-N DSR (slightly anti-conservative), and the turnover penalty when candidates are high-turnover.

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
python scripts/run_dsr_calibration.py                  # DSR effective-N / max-stat calibration (~1.5 min)
python scripts/run_selector_ablation.py                # walk-forward selector ablation (incl. EB)
# optional market data (network; cite Yahoo Finance / yfinance):
python scripts/run_purged_cv_slice.py --yfinance --ticker SPY
```

## Package layout

```
src/purged_cv/
  split.py      # PurgedKFold
  cpcv.py       # CombinatorialPurgedKFold + OOS backtest-path assembly
  overfitting.py# PBO via CSCV
  sharpe.py     # PSR, deflated Sharpe, min track record length,
                # effective N of correlated trials, max-statistic p-value
  calibration.py# null false-positive / edge power of selection hurdles
  leakage.py    # naive vs purged vs CPCV on autocorrelated null labels
  selection.py  # best-of-N rule selection with costs: DSR, PBO, walk-forward,
                # causal vol, shrinkage / EB / turnover scores, sqrt impact
  embargo.py    # purge / embargo helpers
  weights.py    # optional average-uniqueness sample weights
  data.py       # synthetic AR + optional yfinance helpers
  model.py      # logistic / ridge return-sign classifiers
  metrics.py    # naive vs purged comparison
scripts/run_purged_cv_slice.py
scripts/run_overfitting_diagnostics.py
scripts/run_selector_ablation.py
scripts/run_dsr_calibration.py
tests/
.github/workflows/tests.yml   # offline pytest on 3.11 / 3.12
```

## Limits / what is still weak

- Single linear return-sign model in the original slice; no hyperparameter search or portfolio construction. Selection diagnostics now include linear bps plus an optional square-root impact term.
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
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica* 68(5). (Max-statistic idea; here a Gaussian approximation, not the bootstrap.)
- Efron, B., Morris, C. (1973). Stein's estimation rule and its competitors: an empirical Bayes approach. *JASA* 68(341).
- Synthetic data: in-process AR(1). Optional bars: Yahoo Finance via `yfinance`.
