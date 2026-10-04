# BAM yield-curve predictability: Phase 1 report

## Motivation and persistence benchmark

The experiment asks whether information available at origin *t* predicts the
J+5, J+10 or J+22 yield change better than a zero-change random walk. Rates and
errors are stored as decimals. Persistence is the primary benchmark.

## Direct change forecasting and leakage controls

All models use the same monthly origins and nine observed BAM maturities. NS and
PCA transformations, feature scaling and hyperparameter selection are refitted
at every origin. A direct training row *s* is admitted only when *s+h <= t*.
Inner validation is expanding and purged so a training target cannot extend
beyond the first validation origin. The feature set has 46 economically
interpretable level, lagged-change, slope, curvature, momentum and volatility
variables; the earliest origin has 4155
usable rows.

## Nelson–Siegel regularized dynamics

`NS_RIDGE` and `NS_ELASTICNET` predict the three factor changes separately for
each horizon and reconstruct yields with the origin-local NS loadings. This
tests factor dynamics without abandoning the NS cross-sectional restriction.

## Leakage-safe PCA and PCA dynamics

At the latest origin PC1, PC2 and PC3 explain
93.24%,
6.34%, and
0.34%; cumulatively the
first three explain 99.92%.
The loadings support a level interpretation for PC1, a short-versus-long slope
for PC2, and a curvature/twist interpretation for PC3 (component signs are
arbitrary). `PCA_RIDGE` forecasts direct component changes. `PCA_RIDGE_VAR` is a
small Ridge VAR with lag order in (1, 2, 5), selected using past validation and
recursed to the requested horizon.

## Results

Best aggregate result by horizon:

```
 horizon         model     rmse  rmse_improvement_pct
       5   PERSISTENCE 0.001198              0.000000
      10   PERSISTENCE 0.001784              0.000000
      22   PERSISTENCE 0.002621              0.000000
      44 PCA_RIDGE_VAR 0.003535              3.217107
      66 PCA_RIDGE_VAR 0.004282              5.527511
     132 PCA_RIDGE_VAR 0.006306              9.606523
     252 PCA_RIDGE_VAR 0.010283              7.500311
```

Full master comparison is in `phase1_master_comparison.csv`; maturity-level
winners are in `phase1_best_by_horizon_maturity.csv`. Robust DM tests derive
their Bartlett/Newey–West bandwidth from actual origin-to-target window overlap
and aggregate the nine maturity losses to one curve loss per origin before
inference. At monthly frequency the bandwidth is 0 for J+5/J+10 and 1 for J+22.

Models with a statistically significant aggregate improvement over persistence
at 5%: none.

## Scientific interpretation

The Phase 1 evidence does not show robust short-horizon predictability. Any
numerical gain must be read together with its DM result and economic magnitude;
negative findings are retained. Differences between NS and PCA indicate whether
the cross-sectional restriction matters, while differences among Ridge,
Elastic Net and regularized VAR isolate the dynamics assumption.

## Limitations and deferred experiments

The forecast-origin count is 54, so
power is limited, especially for overlapping J+22 errors. Existing DNS/blend
rows are included only where their origin, target date, horizon and maturity
exactly match the new experiment. Macro data are not used in Phase 1 because
publication/revision timing needs a separate audit. XGBoost, direct maturity
models, regimes and neural networks remain deferred until Phase 1 validation,
as required; no Phase 2 tuning informed these results.

## Conclusion

Persistence remains the standard to beat. Phase 1 should be considered evidence
of predictability only where both the RMSE ratio is below one and the robust DM
comparison supports the gain; otherwise the honest conclusion is that the
available historical curve information does not reliably improve the random
walk forecast.
