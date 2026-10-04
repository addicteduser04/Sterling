# BAM endogenous predictability — Phase 2 report

## 1. Phase 1 audit and corrected DM/HAC methodology

Forecast windows are treated as half-open intervals `(origin, target]`. The
monthly overlap-derived HAC bandwidth is 0 for J+5 and J+10 and 1 for J+22.
The earlier mechanical h−1 choice was therefore too large. Forecasts were not
changed. Corrected inference still finds no significant aggregate improvement
over persistence among Phase 1 models.

## 2. Sample accounting

The machine-generated `sample_accounting.json` records 5,328 unique daily BAM
publication dates from 2004-09-06 through 2026-06-18, no duplicates, median gap
one day and maximum gap ten days. The 4,203 figure is the number of daily
feature-origin rows usable to train J+22 at the earliest origin—not
date×maturity observations. J+5 and J+10 have 4,220 and 4,215 rows.

## 3. Direct maturity forecasting

`DIRECT_RIDGE` uses a separate standardized, purged-CV model for every maturity
and horizon. `DIRECT_AR` selects among publication-step lag orders (1, 2, 5, 10).
`DIRECT_ARIMA` is rejected: Ordinary ARIMA assumes equally spaced observations, but BAM publication dates are irregular; integrated candidates also duplicate persistence. No forecasts were silently dropped.

## 4. Nonlinear factor forecasting

`NS_XGBOOST` and `PCA_XGBOOST` use the same 46-feature information set and
origin-local transformations as Phase 1. Exactly 3 configurations
were evaluated per origin/horizon/representation, with depth ≤2, 100–180 trees,
subsample and column fractions 0.8, minimum child weight 5, L1=0.1, L2=10 and
seed 42. This is 954
configuration fits in the monthly selection and
2214
in the weekly selection, before final refits. Importance is averaged across
historical origin-specific models.

## 5. Monthly results and economic magnitude

The strict-common master table is `phase2_master_strict.csv`. Its best point
estimate by horizon is:

```
 horizon        model  rmse_improvement_pct
       5 DIRECT_RIDGE              0.608270
      10    PCA_RIDGE              1.460102
      22  PCA_XGBOOST              3.670384
```

DIRECT_RIDGE's all-available J+5 improvement is about 0.54% and statistically
insignificant. PCA_XGBOOST's all-available J+22 improvement is about 2.31% and
also insignificant. No model meets the strong-evidence rule. Point estimates
below one without robust significance are classified as weak evidence.

## 6. Weekly robustness and dependence

The separate robustness experiment contains 123
weekly origins from 2024-01-05 through
2026-05-08. Its date-derived overlap bandwidth is
reported in `weekly_overlap_audit.csv`; denser origins are not treated as
independent. Weekly best point estimates are:

```
 horizon     model  rmse_improvement_pct
       5 DIRECT_AR              9.372934
      10 DIRECT_AR              5.720766
      22 DIRECT_AR              3.937656
```

The relatively large direct-AR weekly gains are statistically insignificant
after overlap-robust inference and are therefore weak/episodic evidence, not a
general win. Daily origins were not run: they would multiply highly overlapping
forecasts without materially increasing effective information and would require
thousands of repeated NS/PCA/XGBoost fits. This is a documented reliability and
compute decision, not selective omission.

## 7. Stability and interpretation

Cumulative and rolling loss plots show whether numerical gains are persistent
or episodic. Maturity winners are reported without promoting isolated cells to
a general forecasting claim. The evidence does not establish that direct
maturity dynamics or nonlinear mappings reliably beat the random walk.

## 8. Explicit research decisions

1. **Direct maturities:** no robust monthly win. DIRECT_RIDGE has a tiny J+5
   point gain; other aggregate direct results lose. Some maturities win, but
   the pattern is not statistically stable.
2. **Nonlinearity:** NS_XGBOOST does not outperform NS_RIDGE consistently.
   PCA_XGBOOST improves on PCA_RIDGE at J+22 and has a 2.31% all-available gain
   over persistence, but its robust DM p-value is about 0.47.
3. **Any robust winner:** none. All below-one ratios are weak evidence.
4. **Denser origins:** weekly direct AR point estimates improve by 3.9–9.4%,
   but robust p-values remain above 0.22. This suggests episodic recent-sample
   structure, not a stable general law.
5. **Endogenous predictability:** the combined monthly and weekly evidence does
   not establish reliable predictability from the curve's own history.

## 9. Decision and Phase 3 recommendation

Historical BAM curve information alone does not reliably forecast J+5/J+10/J+22
changes beyond persistence. Increasing endogenous model complexity is not the
recommended next step. Phase 3 should audit Moroccan policy rates, interbank
rates, CPI vintages, Treasury auctions, liquidity, financing conditions and FX,
recording source, frequency, history, true publication date, revisions, expected
yield link and leakage risk before any macro-augmented backtest.
