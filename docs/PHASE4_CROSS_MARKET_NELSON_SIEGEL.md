# Phase 4 — Cross-market Nelson–Siegel forecasting

## Decision

**PHASE 4D NOT YET JUSTIFIED**

The decision is based only on the pre-specified independent-market forecasts. A Phase 4D spillover model is justified only if a non-persistence specification improves RMSE in every primary market and horizon; that robustness condition is not met. No Phase 4D model was run.

## Scope and safeguards

Phase 3B remains formally deferred. Existing BAM Phase 1/2 outputs were read as the unchanged domestic benchmark. Europe and the United States were modelled independently; no foreign curve entered another market's predictors. All source rates were explicitly converted from percent per annum to decimal. Horizons J+5, J+10 and J+22 denote subsequent observations on each market's own calendar. Selection did not use macro data, neural networks, boosted trees, or forecasting outcomes to alter the input panels.

## Data audit

| market | definition | source_unit | internal_unit | first_observation | last_observation | observations | maturity_count | maturities | median_gap_days | maximum_gap_days | gaps_over_3_days | duplicate_dates | missing_cells | negative_cells | minimum_decimal_yield | maximum_decimal_yield |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | BAM endogenous interpolated sovereign curve | percent in existing derived panel | decimal | 2004-09-06 | 2026-06-18 | 5328 | 23 | 3M|6M|1Y|2Y|3Y|4Y|5Y|6Y|7Y|8Y|9Y|10Y|11Y|12Y|13Y|14Y|15Y|16Y|17Y|18Y|19Y|20Y|30Y | 1.000000 | 10 | 146 | 0 | 0 | 0 | 0.013070 | 0.061390 |
| EUROPE | ECB AAA euro-area central-government Svensson zero-coupon spot curve | percent per annum | decimal | 2004-09-06 | 2026-06-24 | 5572 | 23 | 3M|6M|1Y|2Y|3Y|4Y|5Y|6Y|7Y|8Y|9Y|10Y|11Y|12Y|13Y|14Y|15Y|16Y|17Y|18Y|19Y|20Y|30Y | 1.000000 | 6 | 51 | 0 | 0 | 24056 | -0.010091 | 0.051750 |
| US_CORE | US Treasury constant-maturity par yields; fixed 3M-10Y core | percent per annum | decimal | 1990-01-02 | 2026-08-10 | 9154 | 8 | 3M|6M|1Y|2Y|3Y|5Y|7Y|10Y | 1.000000 | 4 | 287 | 0 | 0 | 0 | 0.000000 | 0.091200 |
| US_EXTENDED | US Treasury constant-maturity par yields; fixed core plus 20Y/30Y | percent per annum | decimal | 2006-02-09 | 2026-08-10 | 5126 | 10 | 3M|6M|1Y|2Y|3Y|5Y|7Y|10Y|20Y|30Y | 1.000000 | 4 | 160 | 0 | 0 | 0 | 0.000000 | 0.056300 |

Europe is the ECB AAA-rated euro-area central-government nominal zero-coupon spot curve estimated with the Svensson method. Its changing euro-area composition is a definitional caveat, not a missing-data defect. Negative observations were retained. The U.S. primary panel is the fixed complete 3M–10Y Treasury par-yield panel. The 20Y/30Y panel is secondary and begins after the documented long 30-year publication/issuance discontinuity; no maturity was interpolated. BAM remains the existing endogenous derived panel.

## Nelson–Siegel representation

| market | lambda | fit_rmse | pc1_variance | pc3_cumulative |
| --- | --- | --- | --- | --- |
| BAM | 8.000000 | 0.000710 | 0.922457 | 0.994779 |
| EUROPE | 3.000000 | 0.000766 | 0.932425 | 0.999201 |
| US_CORE | 2.000000 | 0.000563 | 0.954473 | 0.999553 |

Decay was calibrated separately by market over the same fixed grid. This preserves economically comparable level/slope/curvature representations without imposing a common lambda on structurally different markets. Daily factor dynamics are:

| market | factor | level_std | change_std | lag1_autocorrelation |
| --- | --- | --- | --- | --- |
| BAM | beta0 | 0.011002 | 0.001294 | 0.993082 |
| BAM | beta1 | 0.013657 | 0.001227 | 0.995963 |
| BAM | beta2 | 0.025017 | 0.002569 | 0.994726 |
| EUROPE | beta0 | 0.015582 | 0.000520 | 0.999442 |
| EUROPE | beta1 | 0.015184 | 0.000553 | 0.999337 |
| EUROPE | beta2 | 0.013566 | 0.001388 | 0.994769 |
| US_CORE | beta0 | 0.018324 | 0.000708 | 0.999253 |
| US_CORE | beta1 | 0.019260 | 0.000749 | 0.999245 |
| US_CORE | beta2 | 0.022773 | 0.002094 | 0.995774 |

Maturity-specific fit errors, lambda-grid diagnostics, PCA loadings and common-date factor correlations are stored as machine-readable tables in `outputs_phase4/`.

## Forecast design

The model set is Persistence, DNS-Kalman, NS-Ridge, NS-ElasticNet, PCA-Ridge and PCA-Ridge-VAR. Every tuning sample is expanding-window and target-purged. Europe uses its longest reliable history from 2004; U.S. full-history training begins in 1990. A second U.S. run begins at the common 2004-09-06 boundary and ends at BAM's 2026-06-18 boundary. Evaluation origins begin in 2022, matching the retained BAM experiment. Overlap-aware HAC bandwidths are derived from actual origin/target dates.

## Primary forecast results

| market | model | horizon | rmse | rmse_ratio_vs_persistence | mae | bias | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EUROPE | DNS_KALMAN_OU | 5 | 0.001350 | 1.120057 | 0.001064 | -0.000276 | 1219.000000 |
| EUROPE | DNS_KALMAN_OU | 10 | 0.001932 | 1.073954 | 0.001502 | -0.000513 | 1219.000000 |
| EUROPE | DNS_KALMAN_OU | 22 | 0.002793 | 1.065701 | 0.002117 | -0.000905 | 1219.000000 |
| EUROPE | NS_ELASTICNET | 5 | 0.001330 | 1.103414 | 0.001054 | -0.000289 | 1219.000000 |
| EUROPE | NS_ELASTICNET | 10 | 0.001898 | 1.054592 | 0.001481 | -0.000487 | 1219.000000 |
| EUROPE | NS_ELASTICNET | 22 | 0.002709 | 1.033837 | 0.002067 | -0.000760 | 1219.000000 |
| EUROPE | NS_RIDGE | 5 | 0.001315 | 1.091070 | 0.001042 | -0.000179 | 1219.000000 |
| EUROPE | NS_RIDGE | 10 | 0.001928 | 1.071567 | 0.001501 | -0.000256 | 1219.000000 |
| EUROPE | NS_RIDGE | 22 | 0.002686 | 1.025009 | 0.002056 | -0.000326 | 1219.000000 |
| EUROPE | PCA_RIDGE | 5 | 0.001365 | 1.132996 | 0.001056 | -0.000171 | 1219.000000 |
| EUROPE | PCA_RIDGE | 10 | 0.001947 | 1.082053 | 0.001507 | -0.000246 | 1219.000000 |
| EUROPE | PCA_RIDGE | 22 | 0.002705 | 1.032187 | 0.002062 | -0.000295 | 1219.000000 |
| EUROPE | PCA_RIDGE_VAR | 5 | 0.001371 | 1.137692 | 0.001061 | -0.000235 | 1219.000000 |
| EUROPE | PCA_RIDGE_VAR | 10 | 0.001898 | 1.055107 | 0.001469 | -0.000381 | 1219.000000 |
| EUROPE | PCA_RIDGE_VAR | 22 | 0.002625 | 1.001641 | 0.002005 | -0.000557 | 1219.000000 |
| EUROPE | PERSISTENCE | 5 | 0.001205 | 1.000000 | 0.000934 | -0.000273 | 1219.000000 |
| EUROPE | PERSISTENCE | 10 | 0.001799 | 1.000000 | 0.001399 | -0.000461 | 1219.000000 |
| EUROPE | PERSISTENCE | 22 | 0.002621 | 1.000000 | 0.001970 | -0.000735 | 1219.000000 |
| US_CORE | DNS_KALMAN_OU | 5 | 0.001493 | 1.079161 | 0.001215 | -0.000394 | 440.000000 |
| US_CORE | DNS_KALMAN_OU | 10 | 0.002501 | 1.036013 | 0.001896 | -0.000762 | 440.000000 |
| US_CORE | DNS_KALMAN_OU | 22 | 0.003245 | 1.062217 | 0.002523 | -0.001234 | 440.000000 |
| US_CORE | NS_ELASTICNET | 5 | 0.001476 | 1.067165 | 0.001178 | -0.000292 | 440.000000 |
| US_CORE | NS_ELASTICNET | 10 | 0.002475 | 1.025185 | 0.001842 | -0.000566 | 440.000000 |
| US_CORE | NS_ELASTICNET | 22 | 0.003119 | 1.021063 | 0.002366 | -0.000823 | 440.000000 |
| US_CORE | NS_RIDGE | 5 | 0.001451 | 1.048801 | 0.001158 | -0.000313 | 440.000000 |
| US_CORE | NS_RIDGE | 10 | 0.002407 | 0.997236 | 0.001818 | -0.000603 | 440.000000 |
| US_CORE | NS_RIDGE | 22 | 0.003049 | 0.997969 | 0.002334 | -0.000897 | 440.000000 |
| US_CORE | PCA_RIDGE | 5 | 0.001436 | 1.037798 | 0.001160 | -0.000337 | 440.000000 |
| US_CORE | PCA_RIDGE | 10 | 0.002417 | 1.001242 | 0.001840 | -0.000648 | 440.000000 |
| US_CORE | PCA_RIDGE | 22 | 0.003035 | 0.993434 | 0.002344 | -0.001030 | 440.000000 |
| US_CORE | PCA_RIDGE_VAR | 5 | 0.001419 | 1.025412 | 0.001147 | -0.000328 | 440.000000 |
| US_CORE | PCA_RIDGE_VAR | 10 | 0.002355 | 0.975582 | 0.001803 | -0.000640 | 440.000000 |
| US_CORE | PCA_RIDGE_VAR | 22 | 0.002957 | 0.968077 | 0.002290 | -0.000979 | 440.000000 |
| US_CORE | PERSISTENCE | 5 | 0.001384 | 1.000000 | 0.001115 | -0.000265 | 440.000000 |
| US_CORE | PERSISTENCE | 10 | 0.002414 | 1.000000 | 0.001798 | -0.000516 | 440.000000 |
| US_CORE | PERSISTENCE | 22 | 0.003055 | 1.000000 | 0.002313 | -0.000710 | 440.000000 |
| BAM | PERSISTENCE | 5 | 0.001478 | 1.000000 | nan | nan | nan |
| BAM | PERSISTENCE | 10 | 0.001551 | 1.000000 | nan | nan | nan |
| BAM | PERSISTENCE | 22 | 0.002149 | 1.000000 | nan | nan | nan |
| BAM | DNS_KALMAN_OU | 5 | 0.001598 | 1.081322 | nan | nan | nan |
| BAM | DNS_KALMAN_OU | 10 | 0.001628 | 1.049889 | nan | nan | nan |
| BAM | DNS_KALMAN_OU | 22 | 0.002100 | 0.977161 | nan | nan | nan |
| BAM | NS_RIDGE | 5 | 0.001519 | 1.027677 | nan | nan | nan |
| BAM | NS_RIDGE | 10 | 0.001560 | 1.006056 | nan | nan | nan |
| BAM | NS_RIDGE | 22 | 0.002756 | 1.282580 | nan | nan | nan |
| BAM | NS_ELASTICNET | 5 | 0.001574 | 1.064810 | nan | nan | nan |
| BAM | NS_ELASTICNET | 10 | 0.001622 | 1.046048 | nan | nan | nan |
| BAM | NS_ELASTICNET | 22 | 0.002191 | 1.019813 | nan | nan | nan |
| BAM | PCA_RIDGE | 5 | 0.001492 | 1.009238 | nan | nan | nan |
| BAM | PCA_RIDGE | 10 | 0.001528 | 0.985399 | nan | nan | nan |
| BAM | PCA_RIDGE | 22 | 0.002179 | 1.013986 | nan | nan | nan |
| BAM | PCA_RIDGE_VAR | 5 | 0.001558 | 1.054289 | nan | nan | nan |
| BAM | PCA_RIDGE_VAR | 10 | 0.001605 | 1.034737 | nan | nan | nan |
| BAM | PCA_RIDGE_VAR | 22 | 0.002121 | 0.986978 | nan | nan | nan |

Best model by market and horizon:

| market | model | horizon | rmse | rmse_ratio_vs_persistence | mae | bias | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | PERSISTENCE | 5 | 0.001478 | 1.000000 | nan | nan | nan |
| BAM | PCA_RIDGE | 10 | 0.001528 | 0.985399 | nan | nan | nan |
| BAM | DNS_KALMAN_OU | 22 | 0.002100 | 0.977161 | nan | nan | nan |
| EUROPE | PERSISTENCE | 5 | 0.001205 | 1.000000 | 0.000934 | -0.000273 | 1219.000000 |
| EUROPE | PERSISTENCE | 10 | 0.001799 | 1.000000 | 0.001399 | -0.000461 | 1219.000000 |
| EUROPE | PERSISTENCE | 22 | 0.002621 | 1.000000 | 0.001970 | -0.000735 | 1219.000000 |
| US_CORE | PERSISTENCE | 5 | 0.001384 | 1.000000 | 0.001115 | -0.000265 | 440.000000 |
| US_CORE | PCA_RIDGE_VAR | 10 | 0.002355 | 0.975582 | 0.001803 | -0.000640 | 440.000000 |
| US_CORE | PCA_RIDGE_VAR | 22 | 0.002957 | 0.968077 | 0.002290 | -0.000979 | 440.000000 |

Aggregate DM rejections at 5%:

| market | model | horizon | maturity | statistic | p_value | mean_loss_difference | n | hac_bandwidth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | DNS_KALMAN_OU | 5 | aggregate | 2.540281 | 0.014682 | 0.000000 | 45 | 0 |
| BAM | NS_ELASTICNET | 5 | aggregate | 3.873947 | 0.000301 | 0.000000 | 53 | 0 |
| BAM | NS_ELASTICNET | 10 | aggregate | 3.312181 | 0.001689 | 0.000000 | 53 | 0 |
| BAM | NS_ELASTICNET | 22 | aggregate | 2.013800 | 0.049218 | 0.000000 | 53 | 1 |
| BAM | PCA_RIDGE_VAR | 5 | aggregate | 4.033277 | 0.000181 | 0.000000 | 53 | 0 |
| BAM | PCA_RIDGE_VAR | 10 | aggregate | 2.780674 | 0.007537 | 0.000000 | 53 | 0 |
| EUROPE | DNS_KALMAN_OU | 5 | aggregate | 6.452830 | 0.000000 | 0.000000 | 53 | 0 |
| EUROPE | DNS_KALMAN_OU | 10 | aggregate | 3.994356 | 0.000205 | 0.000000 | 53 | 0 |
| EUROPE | DNS_KALMAN_OU | 22 | aggregate | 3.242764 | 0.002070 | 0.000001 | 53 | 1 |
| EUROPE | NS_ELASTICNET | 5 | aggregate | 9.089625 | 0.000000 | 0.000000 | 53 | 0 |
| EUROPE | NS_ELASTICNET | 10 | aggregate | 10.075114 | 0.000000 | 0.000000 | 53 | 0 |
| EUROPE | NS_ELASTICNET | 22 | aggregate | 5.355826 | 0.000002 | 0.000000 | 53 | 1 |
| EUROPE | NS_RIDGE | 5 | aggregate | 4.028707 | 0.000183 | 0.000000 | 53 | 0 |
| EUROPE | NS_RIDGE | 10 | aggregate | 2.469756 | 0.016840 | 0.000000 | 53 | 0 |
| EUROPE | PCA_RIDGE | 5 | aggregate | 5.741862 | 0.000000 | 0.000000 | 53 | 0 |
| EUROPE | PCA_RIDGE | 10 | aggregate | 3.292066 | 0.001792 | 0.000001 | 53 | 0 |
| EUROPE | PCA_RIDGE_VAR | 5 | aggregate | 6.522104 | 0.000000 | 0.000000 | 53 | 0 |
| EUROPE | PCA_RIDGE_VAR | 10 | aggregate | 2.556993 | 0.013516 | 0.000000 | 53 | 0 |
| US_CORE | DNS_KALMAN_OU | 5 | aggregate | 4.589275 | 0.000027 | 0.000000 | 55 | 0 |
| US_CORE | DNS_KALMAN_OU | 10 | aggregate | 2.487483 | 0.015985 | 0.000000 | 55 | 0 |
| US_CORE | DNS_KALMAN_OU | 22 | aggregate | 2.672371 | 0.009938 | 0.000001 | 55 | 1 |
| US_CORE | NS_ELASTICNET | 5 | aggregate | 6.415346 | 0.000000 | 0.000000 | 55 | 0 |
| US_CORE | NS_ELASTICNET | 10 | aggregate | 5.200418 | 0.000003 | 0.000000 | 55 | 0 |
| US_CORE | NS_ELASTICNET | 22 | aggregate | 3.442124 | 0.001122 | 0.000000 | 55 | 1 |
| US_CORE | NS_RIDGE | 5 | aggregate | 2.350973 | 0.022405 | 0.000000 | 55 | 0 |
| US_CORE | PCA_RIDGE | 5 | aggregate | 2.085654 | 0.041750 | 0.000000 | 55 | 0 |
| US_COMMON | DNS_KALMAN_OU | 5 | aggregate | 3.716698 | 0.000495 | 0.000000 | 53 | 0 |
| US_COMMON | NS_ELASTICNET | 5 | aggregate | 6.013153 | 0.000000 | 0.000000 | 53 | 0 |
| US_COMMON | NS_ELASTICNET | 10 | aggregate | 4.648015 | 0.000023 | 0.000000 | 53 | 0 |
| US_COMMON | NS_RIDGE | 5 | aggregate | 2.028590 | 0.047635 | 0.000000 | 53 | 0 |
| US_COMMON | PCA_RIDGE | 5 | aggregate | 3.794207 | 0.000388 | 0.000000 | 53 | 0 |
| US_COMMON | PCA_RIDGE_VAR | 5 | aggregate | 3.632028 | 0.000643 | 0.000000 | 53 | 0 |

In the DM table, a positive mean loss difference means the candidate model is worse than persistence. Most significant rejections therefore reinforce—not overturn—the persistence benchmark.

The common-window U.S. results are included in `phase4_master_comparison.csv`; they diagnose sensitivity to the much longer U.S. estimation history and are not substituted selectively for the primary result.

## Interpretation and limits

Persistence is the correct hard benchmark for highly persistent yield curves. A model's isolated win is not evidence of transferable cross-market structure. Differences among BAM, ECB spot yields, and Treasury par yields also limit literal RMSE-level comparisons; ratios against each market's own persistence benchmark are the principal cross-market statistic. Market-specific holiday calendars mean J+h is comparable in information steps, not exact calendar days.

The experiment supports comparison of representation and predictability, but spillover modelling should wait unless improvements are robust across markets/horizons and survive overlap-aware inference. The saved cumulative-loss plots reveal whether average results depend on a short episode rather than stable gains.

## Reproducibility inventory

- `data/metadata/us_treasury_source_manifest.csv`: annual official-file URLs, retrieval date, sizes and SHA-256 hashes.
- `data/metadata/ecb_yield_curve_source_manifest.csv`: official ECB dataset definition and methodology provenance.
- `outputs_phase4/phase4_data_metadata.json`: panels, unit rule and horizon semantics.
- `outputs_phase4/*_full/` and `us_core_common/`: forecasts, metrics, DM tests, PCA and sample diagnostics.
- `outputs_phase4/phase4_forecast_metrics.csv`, `phase4_dm_tests.csv`, `phase4_factor_correlations.csv`: cross-market tables.
- Thirteen `fig_*.png` diagnostics cover representative fits, factor histories, correlations, horizon RMSE ratios and cumulative losses.

## Final status

**PHASE 4D NOT YET JUSTIFIED**
