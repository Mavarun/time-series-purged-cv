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

## Slice 5 (2026-10-24): stationary-bootstrap reality check, GARCH-t returns

The slice-4 max-statistic test draws the null Sharpes from `N(0, C / (T-1))`, which assumes iid-normal bars. Real daily returns cluster in volatility and have fat tails. This slice asks whether that assumption miscalibrates the recommended hurdle, and whether a resampling null does better.

### What was added

- **Stationary bootstrap** (`bootstrap.py`, Politis & Romano 1994): resample whole rows of the `T x 40` trial-return matrix in wrap-around blocks with geometric lengths (mean 10 bars, fixed a priori). This keeps the serial dependence and the cross-trial correlation.
- **White's reality check** (`reality_check_pvalue`): statistic `max_j sqrt(T) SR_j`, with bootstrap draws re-centred on each trial's own Sharpe (least-favourable null). `p = (1 + #{V* >= V}) / (1 + B)`.
- **GARCH(1,1) Student-t(5) returns** (`make_garch_ar_returns`): alpha 0.08, beta 0.90, the same unconditional vol and AR(1) coefficient as `make_ar_returns`, with parameters fixed a priori.
- `calibration_table(..., innovations="garch", with_rc=True)`. Both are opt-in, and the defaults reproduce every slice-4 number (a test checks the normal rows cell by cell).

### Results (`python scripts/run_rc_calibration.py`, 200 seeds, n=2520, 499 bootstrap draws, ~3 min on 8 cores)

| returns | phi | cost bps | DSR raw N=40 | DSR simulated N | max-stat p<0.05 | reality check p<0.05 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 0.00 | 0 | 0.000 | 0.010 | 0.050 | 0.055 |
| normal | 0.00 | 2 | 0.000 | 0.005 | 0.035 | 0.025 |
| normal | 0.10 | 2 | 0.430 | 0.545 | **0.785** | 0.730 |
| garch-t | 0.00 | 0 | 0.005 | 0.005 | 0.060 | 0.055 |
| garch-t | 0.00 | 2 | 0.000 | 0.005 | 0.020 | 0.015 |
| garch-t | 0.10 | 2 | 0.200 | 0.315 | **0.645** | 0.625 |

(Full table, with participation-N DSR, gross edge rows and binomial s.e. of about 0.01-0.035, is in `artifacts/rc_calibration.md`.)

**Reading (honest):**

- **No gain from the bootstrap.** On the gross null the reality check rejects 5.5% under both return types, while the Gaussian max-stat rejects 5.0% (normal) and 6.0% (GARCH-t). The differences are within one s.e. (about 0.016). With 2,520 bars and these rule returns, the iid-normal Sharpe null is already adequate.
- **The bootstrap has slightly less power** on the phi = 0.1 edge at 2 bps: 0.730 vs 0.785 (normal) and 0.625 vs 0.645 (GARCH-t). These are within about 1-2 s.e. but the same sign in every edge cell, so the max-stat p-value stays the recommended hurdle.
- **GARCH-t costs every hurdle power.** At 2 bps the edge is detected 0.645 vs 0.785 of the time by max-stat and 0.200 vs 0.430 by raw DSR. Volatility clustering makes the momentum rule's realised Sharpe noisier at the same unconditional vol.
- Participation-N DSR becomes more anti-conservative under GARCH-t: 0.085 false positives on the gross null, against 0.060 under normal returns. That is one more reason not to use it as a hurdle.
- The block length (10) was not tuned, and Hansen's SPA refinement is not implemented. Neither changes the conclusion that the bootstrap null adds nothing measurable here.

### What ships

- Opt-in only: `reality_check_pvalue`, `stationary_bootstrap_indices`, `make_garch_ar_returns` and the calibration flags. The default outputs are unchanged.
- Recommended hurdle unchanged: the Gaussian max-statistic p-value (or simulated-N DSR). The bootstrap check is a robustness cross-check when the returns are far from the GARCH-t case tested here, for example at much shorter samples. That case is not tested.

## Slice 6 (2026-11-03): the hurdles at one to ten years of data, and under volatility regimes

Slices 4 and 5 calibrated every hurdle at n = 2520 bars only, and listed shorter samples and regime switching
as untested. This slice runs the same 40-rule grid, 2 bps cost and hurdles (DSR raw / participation-N /
simulated-N, Gaussian max-statistic p, White's reality check with the stationary bootstrap, mean block 10,
499 draws, all fixed a priori) at n in {252, 504, 1260, 2520} on iid-normal, GARCH(1,1)-t and a new
Markov-volatility-regime AR(1) (`make_regime_ar_returns`: two states with shock-std multipliers 0.5 / 1.5
rescaled to the same unconditional vol, mean regime length 100 bars). 300 fresh seeds per cell (5000-5299,
disjoint from the 0-199 of slices 4 and 5). `phi = 0` rows are false-positive rates (nominal 5%, +-1.3 points
of binomial s.e. at 300 seeds); `phi = 0.1` rows are power. Reproduce with
`python scripts/run_short_sample_calibration.py` (~5-9 min on 8 cores; `--n-seeds`, `--n-bars` to shrink);
artifacts in `artifacts/short_sample_calibration.{json,md}`. `calibration.selection_tests_one_seed` now accepts
`innovations="regime"` and refuses a sample with fewer than 60 scored bars after the 90-bar warm-up.

### Results (share of series passing, 300 seeds per cell)

False positives (`phi = 0`, truth: no rule has an edge):

| returns | bars | DSR participation | max-stat p<0.05 | reality check p<0.05 | DSR simulated |
| :--- | ---: | ---: | ---: | ---: | ---: |
| normal | 252 | 0.077 | 0.053 | 0.063 | 0.017 |
| normal | 504 | 0.070 | 0.040 | 0.053 | 0.003 |
| normal | 1260 | 0.063 | 0.057 | 0.050 | 0.007 |
| normal | 2520 | 0.030 | 0.020 | 0.023 | 0.003 |
| GARCH-t | 252 | **0.103** | 0.043 | **0.070** | 0.007 |
| GARCH-t | 504 | 0.063 | 0.020 | 0.023 | 0.000 |
| GARCH-t | 1260 | 0.050 | 0.033 | 0.030 | 0.003 |
| GARCH-t | 2520 | 0.043 | 0.040 | 0.037 | 0.010 |
| vol regime | 252 | 0.070 | 0.033 | 0.037 | 0.010 |
| vol regime | 504 | 0.050 | 0.027 | 0.020 | 0.000 |
| vol regime | 1260 | 0.037 | 0.023 | 0.020 | 0.003 |
| vol regime | 2520 | 0.040 | 0.013 | 0.017 | 0.000 |

Power (`phi = 0.1`, a real lag-1 momentum edge in the grid):

| returns | bars | DSR participation | max-stat p<0.05 | reality check p<0.05 | DSR simulated |
| :--- | ---: | ---: | ---: | ---: | ---: |
| normal | 252 | 0.137 | 0.103 | 0.077 | 0.033 |
| normal | 504 | 0.233 | 0.187 | 0.117 | 0.043 |
| normal | 1260 | 0.557 | 0.463 | 0.420 | 0.217 |
| normal | 2520 | 0.843 | 0.813 | 0.763 | 0.527 |
| GARCH-t | 252 | 0.147 | 0.093 | 0.077 | 0.023 |
| GARCH-t | 504 | 0.203 | 0.123 | 0.077 | 0.033 |
| GARCH-t | 1260 | 0.407 | 0.330 | 0.260 | 0.120 |
| GARCH-t | 2520 | 0.657 | 0.607 | 0.557 | 0.317 |
| vol regime | 252 | 0.163 | 0.110 | 0.070 | 0.023 |
| vol regime | 504 | 0.213 | 0.127 | 0.090 | 0.020 |
| vol regime | 1260 | 0.360 | 0.240 | 0.207 | 0.097 |
| vol regime | 2520 | 0.707 | 0.637 | 0.593 | 0.343 |

Reading:

- **Short samples do not inflate the max-statistic test or the reality check.** Their false-positive rates
  are 1.3-7.0% across all 12 null cells; they run higher at 252 bars (3.3-7.0%) than at 2520 (1.3-4.0%),
  where the grid's 40 rules make both tests conservative, but never past 7%. The worry that hurdles tuned at
  10 years break at 1 year did not materialise for these two.
- **The participation-N DSR is the liberal hurdle at short n**: 7.7% (normal) and 10.3% (GARCH-t) at 252 bars
  and 7.0% at 504 bars on normal returns, against 3-4% at 2520. It is the only hurdle above 7.0%.
  Raw N = 40 and simulated-N DSR stay conservative (0-1.7% at 252 bars).
- **The price of the clean null is power, and it is small at one year.** On a real phi = 0.1 edge the
  max-statistic test finds it in 9-11% of 252-bar series (null: 3-5%), 12-19% at 504 bars, and only
  reaches 46% (normal) / 33% (GARCH-t) / 24% (regime) at 1260 bars. One to two years of daily bars cannot
  tell a phi = 0.1 lag-1 edge from the best of 40 noise rules.
- **The reality check does not buy power here.** It is below the Gaussian max-statistic at every
  phi = 0.1 cell with n >= 252 (for example 0.420 vs 0.463 at 1260 normal, 0.260 vs 0.330 at 1260 GARCH-t),
  and at 252 bars GARCH-t its null rate (7.0%) is higher than the max-statistic's (4.3%). Slice 5's "matches
  the null rate with slightly less power" holds at every length; the bootstrap's value is the tail model, not
  extra detection.
- **Volatility regimes cost power but do not break the nulls.** At 2520 bars max-stat power falls from 0.813
  (normal) to 0.637 (regime) and 0.607 (GARCH-t), with null rates 1.3-4.0%. A regime-switching vol is as
  hard on a Sharpe test as clustering plus fat tails, even with Gaussian shocks.

### What the tests check

- `make_regime_ar_returns`: unconditional std equals `vol`, variance-mixture kurtosis > 3.3, volatility
  clusters while returns are uncorrelated at phi = 0, the AR coefficient is kept, equal multipliers reduce to
  constant-vol moments, regime lengths follow `p_stay`, seeded, validated.
- Calibration: `"regime"` is registered and runs end to end with `rc_p`, same seed gives the same numbers for
  every innovation type, the default path is unchanged, a sample under the scored-bar floor is rejected.
- Artifact: 24 cells, seeds disjoint from slices 4-5, null control for max-stat and reality check at every n
  (< 9%), max-stat power strictly rising in n and < 15% at 252 bars, participation DSR above max-stat on the
  252-bar nulls.

### Weaknesses of this slice

- 300 seeds give a +-1.3 point s.e. on a 5% rate and +-2.9 points on a 50% rate, so the 0.01-0.03
  differences between hurdles in single cells are noise; only the pattern across cells is evidence.
- One edge (phi = 0.1 lag-1 momentum), one grid, one cost (2 bps), one regime shape (two vol states, no
  change in phi). Regime-dependent autocorrelation, vol shocks that move with the signal and cross-asset
  panels are untested.
- Power is the full-sample hurdle's; walk-forward selection (slice 3) at these lengths is not evaluated.
- Real daily return panels were not used (offline, seeded).

## Slice 7 (2026-11-10): a data-driven, two-sided embargo, and whether it buys anything

The limits list said the embargo is "a crude fraction of the sample span, not an asset-specific correlation
time". This slice adds the correlation-time version and tests it where it should matter most: bar shocks that
are themselves persistent, so labels stay correlated *past* the label horizon and purging the overlapping
windows no longer removes every shared-information channel.

### What was added

- `make_autocorrelated_null_dataset(..., shock_phi=)`: unit-variance AR(1) bar shocks (stationary start);
  `shock_phi = 0` reproduces the old draws exactly. Features stay independent of labels, so true accuracy is 0.5.
- `embargo.acf_embargo_bars(y, horizon)`: the embargo is read off the label autocorrelation: the first lag
  past the horizon where the ACF drops inside the `2/sqrt(n)` band, minus the horizon (labels only).
- `embargo.pre_embargo_train_indices` and `PurgedKFold(embargo_bars=int | "acf", embargo_side="after" | "both")`:
  a two-sided embargo, because persistent shocks leak symmetrically, while the AFML embargo only covers the rows
  after the test fold. Defaults are unchanged (`embargo_pct = 0.01`, after).
- Frozen protocol `purged_cv.embargo_calibration` (written before the grid): shock_phi 0 / 0.5 / 0.8 / 0.9 x
  horizon 5 / 20, n = 1500, k-NN, 5 folds, 100 fresh seeds (7000-7099). Five arms: purge only, 1% after
  (default), 5% after, ACF after, ACF both sides. A *leak* is CV accuracy more than 2 s.e. above 0.5. The ACF
  two-sided embargo is recommended only if it is leak-free everywhere, the default leaks in at least 2 of the
  6 persistent cells, and it keeps at least 95% of the default's train rows when shocks are iid.

### Results (`python scripts/run_embargo_calibration.py`, ~30 s on 8 cores; `artifacts/embargo_calibration.md`)

CV accuracy - 0.5, 100 seeds per cell (s.e. 0.002-0.005). The last two columns are a post-hoc positive control
on the same seeds (`scripts/run_embargo_positive_control.py`; it selects nothing).

| shock phi | horizon | purge only | 1% after (default) | ACF both sides | median ACF embargo (bars) | ACF-both train rows vs default | contiguous, no purge | shuffled K-fold |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 5 | +0.0018 | +0.0022 | +0.0018 | 0 | 1.010 | +0.0025 | **+0.1207** |
| 0.0 | 20 | +0.0047 | +0.0039 | +0.0046 | 0 | 1.006 | +0.0066 | **+0.2073** |
| 0.5 | 5 | +0.0002 | +0.0007 | +0.0002 | 1 | 1.008 | +0.0015 | **+0.1488** |
| 0.5 | 20 | +0.0039 | +0.0036 | +0.0038 | 0 | 1.005 | +0.0065 | **+0.2232** |
| 0.8 | 5 | +0.0001 | +0.0001 | +0.0004 | 7 | 0.998 | +0.0015 | **+0.1858** |
| 0.8 | 20 | +0.0019 | +0.0021 | +0.0014 | 0 | 0.999 | +0.0051 | **+0.2393** |
| 0.9 | 5 | -0.0019 | -0.0022 | -0.0027 | 17 | 0.981 | -0.0006 | **+0.2155** |
| 0.9 | 20 | +0.0006 | +0.0016 | +0.0012 | 7 | 0.988 | +0.0059 | **+0.2529** |

Reading:

- **No arm leaks in any cell, not even purge only**, so the default never fails and the pre-stated rule does
  **not** recommend the ACF two-sided embargo. It stays available as an option; the default is unchanged.
- The positive control shows the measurement works: shuffled K-fold is inflated by 12-25 points in every cell,
  and more as shocks get more persistent. But contiguous folds **with no purge at all** stay within 2 s.e. of
  0.5 too. At 5 folds of 300 rows only the few dozen rows at each fold boundary can leak, which is below what
  100 seeds can resolve here (s.e. 0.002-0.005). The first-order protection is keeping folds contiguous; purge
  and embargo are second-order at this scale.
- The ACF length behaves as intended at horizon 5 (median 0 / 1 / 7 / 17 bars as shock_phi goes 0 -> 0.9) and
  costs at most 2% of the default's train rows. At horizon 20 it is unstable (median 0-7 bars; exploratory seeds
  0-5 gave 1 to 85 at shock_phi 0.9): overlapping 20-bar labels have few effective observations, so the white-noise band is too narrow.

### What the tests check

- `shock_phi`: old draws unchanged at 0, features independent of it, label ACF past the horizon only with
  persistent shocks, unit shock variance, validation.
- ACF embargo and two-sided embargo: zero for iid shocks, longer with persistence, the cap when the ACF never
  falls inside the band, the exact rows the pre-test embargo drops, defaults reproduce the old splits, `"acf"`
  needs `y`, bad options rejected.
- Artifacts: full grid and seeds, no leak in any arm, the call reproduces from the record, the ACF length is
  monotone in persistence at horizon 5; the control's shuffled leak is > 10 points everywhere and monotone in
  persistence, contiguous folds without purge show no measurable leak.

### Weaknesses of this slice

- One learner (k-NN on 3 slow AR features), one sample size, 5 folds. The boundary leak scales with the number
  of fold boundaries times the correlation length over n; many folds (CPCV with small groups) or short samples
  would make it visible and are not run here.
- The ACF band ignores that the labels themselves are autocorrelated (Bartlett's formula would widen it), which
  is why the estimate is noisy at horizon 20.
- A null tests only optimism. What an embargo costs on a real edge (fewer train rows near the test fold) is not
  measured; at <= 2% of rows it is unlikely to matter here.

## Slice 8 (2026-11-19): many folds, short samples, and what the embargo costs on a real edge

Slice 7 left two things open: whether purging alone still holds when there are many fold boundaries relative
to n (the residual boundary leak should grow with the number of boundaries), and what an embargo costs when
there is a real, persistent edge (it drops the train rows closest to each test fold). Both protocols were
frozen in code before any score was computed; a 2-seed smoke run of each script was only used to check it runs.

### What was added

- `purged_cv.fold_count_calibration` (frozen): the slice 7 null at n 500 / 1500 x 5 / 10 / 20 folds x shock phi
  0 / 0.9, horizon 20, k-NN, 100 fresh seeds (8000-8099). Arms: purge only, 1% after (default), ACF both sides,
  plus contiguous K-fold with **no purge** as an in-grid positive control. The grid only counts if the control
  leaks in at least half the cells.
- `purged_cv.edge_cost` (frozen) and `make_signal_dataset`: the same persistent features and AR(1) shocks plus a
  real edge, `beta * X[t-1, 0]` added to every bar shock (beta 0.15). Truth is the full-sample model scored on the
  next 5000 bars. n 500 x 5 / 20 folds x shock phi 0 / 0.9, seeds 8100-8199. The embargo is called *cheap* only if
  both embargo arms cost at most 0.005 CV accuracy against purge only in every cell.

### Results: optimism on the null (`python scripts/run_fold_count_calibration.py`, ~1 min on 8 cores)

CV accuracy - 0.5, 100 seeds per cell (s.e. 0.004-0.009); `*` = more than 2 s.e. above zero.

| n | folds | shock phi | purge only | 1% after (default) | ACF both sides | no purge (control) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 5 | 0.0 | -0.0034 | -0.0046 | -0.0036 | +0.0105 |
| 500 | 5 | 0.9 | -0.0035 | -0.0054 | -0.0071 | +0.0187 |
| 500 | 10 | 0.0 | -0.0024 | -0.0045 | -0.0037 | +0.0270* |
| 500 | 10 | 0.9 | +0.0009 | -0.0005 | -0.0099 | +0.0474* |
| 500 | 20 | 0.0 | -0.0033 | -0.0036 | -0.0050 | +0.0637* |
| 500 | 20 | 0.9 | -0.0022 | -0.0030 | -0.0129 | +0.0997* |
| 1500 | 5 | 0.0 | +0.0015 | +0.0012 | +0.0011 | +0.0057 |
| 1500 | 5 | 0.9 | -0.0036 | -0.0033 | -0.0035 | +0.0020 |
| 1500 | 10 | 0.0 | -0.0007 | +0.0002 | -0.0005 | +0.0081* |
| 1500 | 10 | 0.9 | -0.0031 | -0.0035 | -0.0033 | +0.0069 |
| 1500 | 20 | 0.0 | -0.0001 | -0.0002 | -0.0002 | +0.0140* |
| 1500 | 20 | 0.9 | -0.0042 | -0.0043 | -0.0051 | +0.0186* |

- The control works: it leaks in 7 of 12 cells, and its optimism grows with the fold count in every
  (n, shock phi) pair, up to +0.100 at n 500, 20 folds.
- **Purge alone removes all of it**, in every cell, including 20 folds at n 500 (25-row test folds).
  The default leaks in 0 of 8 many-fold cells, so it is documented as sufficient here and the ACF two-sided
  embargo is **not** recommended (pre-stated call).

### Results: cost on a real edge (`python scripts/run_edge_cost.py`, ~20 s on 8 cores)

| folds | shock phi | truth | purge-only bias vs truth | 1% after cost (s.e.) | ACF both sides cost (s.e.) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.0 | 0.562 | -0.0001 | +0.0020 (0.0016) | +0.0064 (0.0034) |
| 5 | 0.9 | 0.506 | -0.0080 | +0.0020 (0.0014) | +0.0036 (0.0024) |
| 20 | 0.0 | 0.562 | +0.0083 | +0.0008 (0.0014) | +0.0083 (0.0027) |
| 20 | 0.9 | 0.506 | +0.0043 | +0.0024 (0.0012) | +0.0111 (0.0030) |

- The 1% default costs at most 0.0024 CV accuracy against purge only; it is cheap.
- The two-sided ACF embargo costs up to 0.0111 (20 folds, shock phi 0.9), about
  3.7 s.e. It makes CV pessimistic, which is the other half of why it is not recommended.
  The pre-stated "embargo is cheap" call is therefore **no** (it needed both arms under 0.005).
- No arm's bias against truth is more than about 1 s.e. from zero, so purged CV is roughly unbiased here.

### Weaknesses of this slice

- One learner (k-NN) and one feature process again. A learner that uses the nearest-in-time rows less directly
  could leak less, and one that memorises more could leak more.
- With shock phi 0.9 the persistent noise swamps the edge (truth 0.562 -> 0.506), so those two
  edge-cost cells say little about a usable edge.
- Truth is one 5000-bar continuation per seed, which adds its own noise (bias s.e. about 0.01).
- The control leaks in only 7 of 12 cells: it is not detectable at 5 folds or at n 1500 / 10 folds / shock phi 0.9, so
  "no leak" in those cells is weaker evidence than at 20 folds.

## Slice 9 (2026-11-19): does purging stay sufficient beyond k-NN?

Slices 7-8 tested only a k-NN. This slice re-runs the slice-8 null (n 1500, horizon 20, feature phi 0.98) with
three learners: logistic regression, the k-NN again, and a new seeded random forest (`make_sign_classifier("forest")`,
50 trees, min leaf 5). Protocol frozen in `purged_cv/learner_calibration.py` before any score: folds {5, 20} x shock
phi {0, 0.9}, 60 fresh seeds (9000-9059), arms purge only and the 1% default, the no-purge control per learner. A
learner is **testable** only if its control leaks in >= 2 of its 4 cells; the k-NN-only caveat is dropped only if
the default is leak-free for every testable learner and >= 2 learners are testable.

Run: `python scripts/run_learner_calibration.py` (~6 min, 8 cores); `artifacts/learner_calibration.{json,md}`.
Cells are CV accuracy - 0.5 (s.e.); `*` = more than 2 s.e. above zero.

| learner | folds | shock phi | purge only | 1% after (default) | no purge (control) |
| :--- | ---: | ---: | :--- | :--- | :--- |
| logistic | 5 | 0.0 | +0.0010 (0.0070) | -0.0012 (0.0071) | +0.0009 (0.0071) |
| logistic | 5 | 0.9 | +0.0001 (0.0097) | -0.0000 (0.0096) | +0.0034 (0.0099) |
| logistic | 20 | 0.0 | +0.0012 (0.0081) | +0.0003 (0.0080) | +0.0088 (0.0080) |
| logistic | 20 | 0.9 | +0.0074 (0.0106) | +0.0059 (0.0107) | +0.0189 (0.0107) |
| knn | 5 | 0.0 | -0.0041 (0.0046) | -0.0054 (0.0045) | -0.0008 (0.0045) |
| knn | 5 | 0.9 | -0.0016 (0.0057) | -0.0014 (0.0057) | +0.0055 (0.0053) |
| knn | 20 | 0.0 | -0.0046 (0.0050) | -0.0053 (0.0048) | +0.0118 (0.0051)* |
| knn | 20 | 0.9 | -0.0009 (0.0056) | -0.0019 (0.0056) | +0.0247 (0.0055)* |
| forest | 5 | 0.0 | -0.0065 (0.0053) | -0.0083 (0.0051) | -0.0035 (0.0051) |
| forest | 5 | 0.9 | -0.0012 (0.0063) | -0.0006 (0.0062) | +0.0069 (0.0062) |
| forest | 20 | 0.0 | -0.0058 (0.0053) | -0.0045 (0.0050) | +0.0108 (0.0051)* |
| forest | 20 | 0.9 | +0.0001 (0.0063) | -0.0003 (0.0064) | +0.0258 (0.0062)* |

**Calls.** The control leaks for the k-NN and the forest at 20 folds only (2 of 4 cells each), so both are
testable; purge only and the default are leak-free in all of their cells. Logistic's control never leaks
(0 of 4), so logistic is **not testable** here. Two testable learners pass, so the k-NN-only caveat is dropped for
flexible learners. The package default does not change.

Reading:

- The forest leaks about as much as the k-NN when nothing is purged (+0.026 vs +0.025 at 20 folds, shock phi 0.9):
  its leaves isolate runs of neighbouring bars much as nearest neighbours do.
- The k-NN rows reproduce slice 8 on fresh seeds: no leak at 5 folds, a leak at 20, none once purged.
- At 5 folds the purged k-NN and forest arms sit slightly *below* 0.5 (down to -0.008, about 1.6 s.e.), the
  known small pessimism from training on less data next to the test fold. It is not significant here.

### Weaknesses of this slice

- Logistic is untestable, not cleared: its control's s.e. is about twice the k-NN's (0.0107 vs 0.0055 at 20 folds,
  shock phi 0.9) and its +0.019 point estimate there is under 2 s.e. A linear learner may leak a little in ways
  60 seeds cannot see.
- One forest size and one feature process; n 1500 only (slice 8 found the leak larger at n 500).
- "No leak" means "not more than 2 s.e. above zero" in each cell; with 4 cells per learner, a small leak in one
  cell could be missed.

## Slice 10 (2026-11-20): logistic regression leaks too, and purging still removes it

Slice 9 could not test logistic regression: its no-purge control never leaked at n 1500 with 60 seeds. This slice
gives that control its best chance: n 500, 20 folds (25-row test folds), and 240 fresh seeds (9100-9339), four times
slice 9's. The k-NN cells repeat slice 8's n 500 / 20-fold cells as a reproduction check. Protocol frozen in
`purged_cv/logistic_power.py` before any score: logistic is **testable** iff its control leaks in >= 1 of 2 cells,
and the default **holds** iff it is then leak-free in both. No logistic call is made unless the k-NN control leaks
in both cells and the k-NN default in neither.

Run: `python scripts/run_logistic_power.py` (~5 min, 8 cores); `artifacts/logistic_power.{json,md}`.
Cells are CV accuracy - 0.5 (s.e.); `*` = more than 2 s.e. above zero.

| learner | shock phi | purge only | 1% after (default) | no purge (control) |
| :--- | ---: | :--- | :--- | :--- |
| logistic | 0.0 | -0.0028 (0.0065) | -0.0022 (0.0064) | +0.0279 (0.0061)* |
| logistic | 0.9 | +0.0008 (0.0083) | -0.0000 (0.0083) | +0.0541 (0.0073)* |
| knn | 0.0 | +0.0074 (0.0045) | +0.0073 (0.0045) | +0.0675 (0.0040)* |
| knn | 0.9 | +0.0078 (0.0061) | +0.0063 (0.0061) | +0.1011 (0.0051)* |

**Calls.** The k-NN reproduction passes: its control leaks in both cells (+0.068 / +0.101 against slice 8's
+0.064 / +0.100) and its default in neither. Logistic's control leaks in **both** cells, so logistic is testable,
and purge only and the default are leak-free in both. **The default holds for a linear learner**, so slice 9's
"logistic untestable" caveat is replaced by a pass at n 500. The package default does not change.

Reading:

- A linear model does leak without purging: +0.054 at shock phi 0.9, about half the k-NN's +0.101. Slice 9 missed
  it because at n 1500 the leak is smaller and 60 seeds gave an s.e. of about 0.011.
- The purged k-NN cells sit at +0.007 / +0.008, 1.6 and 1.3 s.e. above zero. Under the rule that is not a leak,
  but it is the largest purged k-NN optimism in slices 8-10, so a small residual at 25-row test folds cannot be
  ruled out.

### Weaknesses of this slice

- One cell size for logistic (n 500, 20 folds). At n 1500 the logistic leak is still unmeasured, only bounded
  (slice 9: +0.019, s.e. 0.011).
- Still one feature process (3 AR(1) features, phi 0.98, horizon 20).
- 2-s.e. tests in 4 cells with no multiplicity correction. The purged k-NN cells are close enough to the line that a
  re-run on other seeds could flip one of them.

## Slice 11 (2026-11-20): the logistic leak at n 1500 is measured, and it is about 4x smaller than at n 500

Slice 10 measured a logistic leak at n 500. At n 1500 the leak was still only bounded (slice 9: +0.019, s.e. 0.011,
60 seeds). This slice measures it, with logistic only, shock phi 0 and 0.9, 20 folds, and 480 fresh seeds
(9400-9879), eight times slice 9's. The protocol was frozen in `purged_cv/logistic_n1500.py` before any score of the
grid: the leak counts as **measured** iff the control leaks in ≥ 1 cell, and the default **holds** iff it is then
leak-free in both. The expectation written before the run was a leak at phi 0.9 only (about +0.015), the default
holding, and the leak shrinking with n.

Run: `python scripts/run_logistic_n1500.py` (~6 min, 8 cores); `artifacts/logistic_n1500.{json,md}` (per-seed scores
included). Cells are CV accuracy - 0.5 (s.e.); `*` = more than 2 s.e. above zero.

| shock phi | purge only | 1% after (default) | no purge (control) | slice 10 control (n 500) |
| ---: | :--- | :--- | :--- | ---: |
| 0.0 | +0.0001 (0.0028) | +0.0002 (0.0027) | +0.0073 (0.0028)* | +0.0279 |
| 0.9 | +0.0010 (0.0036) | +0.0004 (0.0036) | +0.0128 (0.0035)* | +0.0541 |

**Calls.** The control leaks in 2 of 2 cells, so the leak is measured: +0.0073 (s.e. 0.0028)
at phi 0 and +0.0128 (s.e. 0.0035) at phi 0.9. Purge only and the 1% default are leak-free in both cells
(|optimism| ≤ 0.0010). **The default holds at n 1500.** The expectation was wrong at phi 0, which leaks too, at
2.6 s.e.

### Post-hoc reading (decides nothing; `logistic_n1500.leak_scaling`)

From n 500 to n 1500, the control leak falls 3.8x (phi 0) and 4.2x (phi 0.9).
As a power law in n, the exponents are 1.23 ± 0.40 and 1.31 ± 0.28.
Both are above 1 but within 2 s.e. of it. Leak ~ 1/n is what a fixed-width boundary band predicts: the rows whose
labels overlap the test fold are a fixed number per fold, so their share falls as 1/n. The data cannot tell 1/n
from a slightly faster fall.

### Weaknesses of this slice

- Logistic only. The k-NN and forest leaks at n 1500 come from slices 8-9 on other seeds.
- Two 2-s.e. tests per arm with no multiplicity correction. The phi-0 control leak (2.6 s.e.) is the closest call.
- Still one feature process (3 AR(1) features, phi 0.98, horizon 20).

## Slice 12 (2026-11-20): the purged k-NN residual at 25-row test folds is real at shock phi 0.9

Slice 10's purged k-NN cells (n 500, 20 folds) sat 1.6 and 1.3 s.e. above zero, so a small residual leak could not
be ruled out. This slice re-runs those two cells on 480 fresh seeds (9900-10379), twice slice 10's. Protocol frozen
in `purged_cv/knn_residual.py` before any score: the control must leak in both cells, and the residual is **real**
iff the default (`pct1_after`) then leaks in >= 1 cell. Expectation written before the run: real at phi 0.9, about
+0.007. A 4-seed smoke run of the script used seeds 9900-9903 before the full run; nothing changed after it.

Run: `python scripts/run_knn_residual.py` (~5 min, 8 cores); `artifacts/knn_residual.{json,md}`.
Cells are CV accuracy - 0.5 (s.e.); `*` = more than 2 s.e. above zero.

| shock phi | purge only | 1% after (default) | no purge (control) | slice 10 default |
| ---: | :--- | :--- | :--- | ---: |
| 0.0 | +0.0024 (0.0034) | +0.0029 (0.0034) | +0.0648 (0.0030)* | +0.0073 |
| 0.9 | +0.0131 (0.0042)* | +0.0110 (0.0042)* | +0.1048 (0.0037)* | +0.0063 |

**Calls.** The control leaks in both cells, so the call is made. The default leaks in 1 of 2 cells: +0.0110
(s.e. 0.0042) at shock phi 0.9, 2.6 s.e. above zero; at phi 0 it is +0.0029 (s.e. 0.0034). Purge only gives the same
picture. **The residual is real**: with 25-row test folds and a strongly persistent shock, the package default leaves
about one point of CV accuracy of optimism for a k-NN. The expectation held in direction; the size is larger than
the +0.007 guessed. The package default does not change in this slice; a longer embargo needs its own protocol.

Reading:

- The leak left after purging is about a tenth of the no-purge control's (+0.011 against +0.105).
- Post hoc, pooling with slice 10's independent seeds by inverse variance: +0.0095 (s.e. 0.0034) at phi 0.9 and
  +0.0045 (s.e. 0.0027) at phi 0. This is not a pre-stated test.
- Slices 7-8 found purging removed every measurable leak up to 20 folds at n 500 with fewer seeds. Their bound is
  replaced by this measurement for the k-NN at phi 0.9.

### Weaknesses of this slice

- One learner, one sample size and one fold count; whether a two-sided or ACF embargo removes the residual is
  untested.
- Two 2-s.e. tests, no multiplicity correction. The phi 0.9 cell clears the line at 2.6 s.e., so a Bonferroni
  correction over the two cells (2.28 s.e.) would not change the call.
- Still one feature process (3 AR(1) features, phi 0.98, horizon 20).

## Slice 13 (2026-11-20): slice 12's k-NN residual does not reappear on fresh seeds

Slice 13 was meant to test whether the two-sided ACF embargo (`acf_both`) removes slice 12's residual. The protocol
(frozen in `purged_cv/acf_residual.py`) first required that residual to reappear: in the same cell (k-NN, n 500,
20 folds, shock phi 0.9), on 480 fresh seeds (10400-10879), both the control and the default had to leak. The
candidate would then remove it iff it was leak-free **and** its paired drop from the default was > 2 s.e., using
>= 90% of the default's training rows. The smoke run used seeds 99990-99991, outside the evaluation range.

Run: `python scripts/run_acf_residual.py` (~30 s, 8 cores); `artifacts/acf_residual.{json,md}`.

| arm | optimism | mean train rows |
| :--- | :--- | ---: |
| purge_only | +0.0018 (0.0041) | 8740 |
| pct1_after | -0.0005 (0.0041) | 8645 |
| acf_both | -0.0109 (0.0041) | 8393 |
| contiguous_no_purge | +0.0904 (0.0037)* | 9500 |

**Calls.** The control still leaks (+0.090), but the default does not: -0.0005 (s.e. 0.0041). **Slice 12's
residual is not reproduced**, so, as pre-stated, no call is made on `acf_both`. A test scores a slice 12 seed with
both modules and gets identical numbers, so the difference is not in the code. The package default does not change.

Reading:

- `acf_both` reads -0.0109 (s.e. 0.0041), 2.6 s.e. **below** zero: on this null it is pessimistic, not neutral. Its
  paired drop from the default is +0.0104 (s.e. 0.0016) at 97% of the default's training rows. The expectation
  (leak-free, drop under 2 s.e.) was wrong on the drop.
- Post hoc (`purged_cv/residual_pool.py`, not pre-stated): the default cell pooled over slices 10, 12 and 13 (1200
  disjoint seeds) is +0.0054 (s.e. 0.0026), z 2.05. Cochran's Q gives p 0.14, and slices 12 and 13 differ by
  1.96 s.e. A residual of about half a point is plausible but not established; +0.011 (mean + 2 s.e.) bounds it.
- Slice 12's "the residual is real" was the pre-stated call on its own seeds, and it stays in the record. It did not
  replicate, and it was one of many 2-s.e. tests run across slices 7-12.

### Weaknesses of this slice

- One cell. The pooled estimate sits on the 2-s.e. line, so another 480 seeds could move it either way.
- The pessimism of `acf_both` was measured on a null only. On a real edge, slice 8b put its cost at up to 0.011.

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
python scripts/run_rc_calibration.py                   # reality check vs max-stat, normal vs GARCH-t (~3 min, 8 cores)
python scripts/run_embargo_calibration.py              # slice 7 embargo grid (~30 s, 8 cores)
python scripts/run_embargo_positive_control.py         # slice 7 post-hoc positive control
python scripts/run_fold_count_calibration.py           # slice 8 many folds / short samples (~1 min, 8 cores)
python scripts/run_edge_cost.py                        # slice 8b embargo cost on a real edge (~20 s, 8 cores)
python scripts/run_learner_calibration.py              # slice 9 logistic / k-NN / forest leak check (~6 min, 8 cores)
python scripts/run_logistic_power.py                   # slice 10 logistic leak at n 500 / 20 folds / 240 seeds (~5 min, 8 cores)
python scripts/run_logistic_n1500.py                   # slice 11 logistic leak at n 1500 / 480 seeds (~6 min, 8 cores)
python scripts/run_knn_residual.py                     # slice 12 purged k-NN residual at n 500 / 20 folds / 480 seeds (~5 min, 8 cores)
python scripts/run_acf_residual.py                     # slice 13 replication + two-sided ACF embargo, one cell / 480 seeds (~30 s, 8 cores)
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
  bootstrap.py  # stationary bootstrap, White's reality check p-value
  leakage.py    # naive vs purged vs CPCV on autocorrelated null labels
  selection.py  # best-of-N rule selection with costs: DSR, PBO, walk-forward,
                # causal vol, shrinkage / EB / turnover scores, sqrt impact
  embargo.py    # purge / embargo helpers
  weights.py    # optional average-uniqueness sample weights
  data.py       # synthetic AR + optional yfinance helpers
  model.py      # logistic / ridge / k-NN / random-forest return-sign classifiers
  learner_calibration.py # slice 9 frozen protocol: purge sufficiency per learner
  logistic_power.py # slice 10 frozen protocol: logistic leak at n 500 / 20 folds
  metrics.py    # naive vs purged comparison
scripts/run_purged_cv_slice.py
scripts/run_overfitting_diagnostics.py
scripts/run_selector_ablation.py
scripts/run_dsr_calibration.py
scripts/run_rc_calibration.py
tests/
.github/workflows/tests.yml   # offline pytest on 3.11 / 3.12
```

## Limits / what is still weak

- Single linear return-sign model in the original slice; no hyperparameter search or portfolio construction. Selection diagnostics now include linear bps plus an optional square-root impact term.
- Embargo length defaults to a fraction of the sample span; a label-ACF length and a two-sided embargo are options (slice 7). On the slice 7-8 nulls purging alone removed every measurable leak up to 20 folds at n = 500, and the two-sided ACF embargo cost up to 0.011 CV accuracy on a real edge (slice 8), so it stays opt-in. Slice 9 found the same for a random forest; logistic regression was not testable there (its no-purge control never leaked at n 1500), but at n 500 / 20 folds / 240 seeds its control leaks (+0.028 / +0.054) and purging removes it (slice 10). With 480 seeds, the purged k-NN at n 500 / 20 folds leaked +0.011 (s.e. 0.0042) at shock phi 0.9 (slice 12), but 480 fresh seeds gave -0.0005 (slice 13). Pooled over 1200 seeds it is +0.0054 (s.e. 0.0026), post hoc, so a residual of about half a point at 25-row test folds is plausible but not established. Only one feature process was tested.
- Average-uniqueness weights use a coarse discrete grid (research-grade, not a production event-time engine).
- Hurdle calibration covers iid-normal, GARCH(1,1)-t and two-state volatility-regime AR(1) returns at n = 252-2520 (slice 6). Regime-dependent autocorrelation, walk-forward selection at short n and real return panels are untested; the participation-N DSR runs liberal (7-10% null) at one to two years.
- Synthetic AR(1) is a citation-clear null-ish demo; yfinance path needs network and is still not a trading signal.
- Gap on honest lag features is small; the large gap is easiest to see on the engineered-leakage fixture.
- No claim of production accuracy or live PnL.

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Purged CV / embargo, ch. 7; CPCV, ch. 12.)
- Bailey, D. H., Borwein, J. M., López de Prado, M., Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4).
- Bailey, D. H., López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5).
- Bailey, D. H., López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk* 15(2).
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica* 68(5). (Max-statistic test: Gaussian approximation in slice 4, stationary bootstrap in slice 5.)
- Politis, D. N., Romano, J. P. (1994). The Stationary Bootstrap. *JASA* 89(428).
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics* 31(3).
- Efron, B., Morris, C. (1973). Stein's estimation rule and its competitors: an empirical Bayes approach. *JASA* 68(341).
- Synthetic data: in-process AR(1). Optional bars: Yahoo Finance via `yfinance`.
