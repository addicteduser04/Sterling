# Phase 5C0 — Long-horizon forecast validation and mean-reversion benchmarking

## 1. Motivation and decisions

Phase 5 established large relative gains at long horizons. This validation asks whether those forecasts are absolutely precise and whether sophisticated dynamics add value beyond simple training-only mean reversion. Frozen Phase 5 forecasts were not modified or rerun.

**MEDIUM-HORIZON FORECASTS ARE ABSOLUTELY INFORMATIVE; J+252 GAINS ARE MAINLY RELATIVE AND IMPRECISE**

**DNS GAINS ARE PRIMARILY EXPLAINED BY SIMPLE MEAN REVERSION**

**PHASE 5C SHOULD REMAIN DEFERRED**

BAM DNS has RMSE 26.6 bp at J+44 and 33.4 bp at J+66, with median absolute errors 13.7 and 16.8 bp. This is economically interpretable precision, not merely a ratio. At J+252 BAM DNS RMSE rises to 61.8 bp, while Europe and U.S. best-model RMSEs exceed 100 bp; those annual forecasts are directionally interesting but not precise point predictions. A simple NS-factor AR(1) is 0.6–1.0% better than BAM DNS at J+44/J+66/J+132 and 0.8% better at J+252. The expanding historical mean is better still at BAM J+252 (56.7 versus 61.8 bp). Conditional modelling should wait because the incremental sophisticated-model signal is absent, equilibrium-distance variation is weak, and annual inference remains descriptive-only.

## 2. Why persistence weakens with horizon

Persistence imposes zero change. As realized changes accumulate, its BAM RMSE grows from 13.8 bp at J+5 to 103.5 bp at J+252; Europe grows from 12.0 to 111.2 bp and the U.S. from 13.8 to 134.6 bp. The best models also deteriorate materially—BAM DNS rises from 15.1 bp at J+5 to 61.8 bp at J+252—just more slowly. A falling ratio therefore does not mean constant absolute accuracy.

## 3. Central validation table

| market | horizon | persistence_rmse_bp | simple_mr_model | simple_mr_rmse_bp | best_model | best_model_rmse_bp | best_vs_rw_ratio | best_vs_simple_mr_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | 5 | 13.799 | DIRECT_AR1_MEAN_REVERSION | 13.763 | PERSISTENCE | 13.799 | 1.000 | 1.003 |
| BAM | 10 | 14.421 | DIRECT_AR1_MEAN_REVERSION | 14.334 | PERSISTENCE | 14.421 | 1.000 | 1.006 |
| BAM | 22 | 18.878 | DIRECT_AR1_MEAN_REVERSION | 18.566 | PCA_RIDGE_VAR | 18.811 | 0.996 | 1.013 |
| BAM | 44 | 28.368 | NS_AR1_MEAN_REVERSION | 26.365 | DNS_KALMAN_OU | 26.624 | 0.939 | 1.010 |
| BAM | 66 | 38.528 | NS_AR1_MEAN_REVERSION | 33.165 | DNS_KALMAN_OU | 33.373 | 0.866 | 1.006 |
| BAM | 132 | 67.957 | NS_AR1_MEAN_REVERSION | 49.456 | DNS_KALMAN_OU | 49.936 | 0.735 | 1.010 |
| BAM | 252 | 103.488 | HISTORICAL_MEAN | 56.654 | DNS_KALMAN_OU | 61.802 | 0.597 | 1.091 |
| EUROPE | 5 | 11.982 | DIRECT_AR1_MEAN_REVERSION | 11.986 | PERSISTENCE | 11.982 | 1.000 | 1.000 |
| EUROPE | 10 | 17.842 | DIRECT_AR1_MEAN_REVERSION | 17.737 | PERSISTENCE | 17.842 | 1.000 | 1.006 |
| EUROPE | 22 | 26.206 | DIRECT_AR1_MEAN_REVERSION | 25.950 | PERSISTENCE | 26.206 | 1.000 | 1.010 |
| EUROPE | 44 | 36.529 | DIRECT_AR1_MEAN_REVERSION | 35.729 | PCA_RIDGE_VAR | 35.354 | 0.968 | 0.989 |
| EUROPE | 66 | 45.323 | DIRECT_AR1_MEAN_REVERSION | 43.883 | PCA_RIDGE_VAR | 42.817 | 0.945 | 0.976 |
| EUROPE | 132 | 69.759 | DIRECT_AR1_MEAN_REVERSION | 66.071 | PCA_RIDGE_VAR | 63.058 | 0.904 | 0.954 |
| EUROPE | 252 | 111.165 | PCA_AR1_MEAN_REVERSION | 100.248 | PCA_RIDGE_VAR | 102.827 | 0.925 | 1.026 |
| US_CORE | 5 | 13.754 | DIRECT_AR1_MEAN_REVERSION | 13.652 | PERSISTENCE | 13.754 | 1.000 | 1.008 |
| US_CORE | 10 | 24.140 | DIRECT_AR1_MEAN_REVERSION | 23.835 | PCA_RIDGE_VAR | 23.551 | 0.976 | 0.988 |
| US_CORE | 22 | 30.549 | DIRECT_AR1_MEAN_REVERSION | 30.002 | PCA_RIDGE_VAR | 29.574 | 0.968 | 0.986 |
| US_CORE | 44 | 48.535 | DIRECT_AR1_MEAN_REVERSION | 47.002 | PCA_RIDGE_VAR | 45.328 | 0.934 | 0.964 |
| US_CORE | 66 | 63.710 | DIRECT_AR1_MEAN_REVERSION | 60.869 | PCA_RIDGE_VAR | 57.960 | 0.910 | 0.952 |
| US_CORE | 132 | 93.640 | PCA_AR1_MEAN_REVERSION | 86.992 | PCA_RIDGE_VAR | 76.635 | 0.818 | 0.881 |
| US_CORE | 252 | 134.641 | PCA_AR1_MEAN_REVERSION | 118.867 | PCA_RIDGE_VAR | 104.277 | 0.774 | 0.877 |

## 4. Required market absolute-accuracy tables

### BAM

| Horizon | Persistence RMSE bp | Best Model | Best RMSE bp | Best MAE bp | Improvement % |
| --- | --- | --- | --- | --- | --- |
| 5 | 13.799 | PERSISTENCE | 13.799 | 4.949 | 0.000 |
| 10 | 14.421 | PERSISTENCE | 14.421 | 6.073 | 0.000 |
| 22 | 18.878 | PCA_RIDGE_VAR | 18.811 | 11.781 | 0.356 |
| 44 | 28.368 | DNS_KALMAN_OU | 26.624 | 18.144 | 6.146 |
| 66 | 38.528 | DNS_KALMAN_OU | 33.373 | 23.182 | 13.381 |
| 132 | 67.957 | DNS_KALMAN_OU | 49.936 | 35.402 | 26.518 |
| 252 | 103.488 | DNS_KALMAN_OU | 61.802 | 50.613 | 40.281 |
### EUROPE

| Horizon | Persistence RMSE bp | Best Model | Best RMSE bp | Best MAE bp | Improvement % |
| --- | --- | --- | --- | --- | --- |
| 5 | 11.982 | PERSISTENCE | 11.982 | 9.299 | 0.000 |
| 10 | 17.842 | PERSISTENCE | 17.842 | 13.837 | 0.000 |
| 22 | 26.206 | PERSISTENCE | 26.206 | 19.699 | 0.000 |
| 44 | 36.529 | PCA_RIDGE_VAR | 35.354 | 26.413 | 3.217 |
| 66 | 45.323 | PCA_RIDGE_VAR | 42.817 | 33.123 | 5.528 |
| 132 | 69.759 | PCA_RIDGE_VAR | 63.058 | 48.507 | 9.607 |
| 252 | 111.165 | PCA_RIDGE_VAR | 102.827 | 76.451 | 7.500 |
### US_CORE

| Horizon | Persistence RMSE bp | Best Model | Best RMSE bp | Best MAE bp | Improvement % |
| --- | --- | --- | --- | --- | --- |
| 5 | 13.754 | PERSISTENCE | 13.754 | 11.087 | 0.000 |
| 10 | 24.140 | PCA_RIDGE_VAR | 23.551 | 18.026 | 2.442 |
| 22 | 30.549 | PCA_RIDGE_VAR | 29.574 | 22.896 | 3.192 |
| 44 | 48.535 | PCA_RIDGE_VAR | 45.328 | 36.102 | 6.608 |
| 66 | 63.710 | PCA_RIDGE_VAR | 57.960 | 46.180 | 9.025 |
| 132 | 93.640 | PCA_RIDGE_VAR | 76.635 | 59.284 | 18.160 |
| 252 | 134.641 | PCA_RIDGE_VAR | 104.277 | 73.008 | 22.552 |

## 5. Absolute error distributions

| market | model | horizon | rmse_bp | mae_bp | median_absolute_error_bp | p75_absolute_error_bp | p90_absolute_error_bp | within_25bp | within_50bp | within_100bp | bias_bp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | DNS_KALMAN_OU | 44 | 26.624 | 18.144 | 13.667 | 22.794 | 36.355 | 0.776 | 0.957 | 0.987 | 0.811 |
| BAM | DNS_KALMAN_OU | 66 | 33.373 | 23.182 | 16.793 | 28.573 | 47.594 | 0.673 | 0.908 | 0.976 | -0.168 |
| BAM | DNS_KALMAN_OU | 132 | 49.936 | 35.402 | 25.936 | 44.555 | 88.330 | 0.492 | 0.804 | 0.908 | -3.103 |
| BAM | DNS_KALMAN_OU | 252 | 61.802 | 50.613 | 43.501 | 71.370 | 103.395 | 0.275 | 0.577 | 0.886 | -2.588 |
| EUROPE | PCA_RIDGE_VAR | 44 | 35.354 | 26.413 | 20.299 | 36.528 | 56.331 | 0.594 | 0.855 | 0.981 | -9.779 |
| EUROPE | PCA_RIDGE_VAR | 66 | 42.817 | 33.123 | 26.143 | 43.790 | 69.797 | 0.482 | 0.796 | 0.967 | -14.133 |
| EUROPE | PCA_RIDGE_VAR | 132 | 63.058 | 48.507 | 36.856 | 66.581 | 101.575 | 0.308 | 0.647 | 0.898 | -22.766 |
| EUROPE | PCA_RIDGE_VAR | 252 | 102.827 | 76.451 | 57.965 | 95.475 | 176.121 | 0.200 | 0.419 | 0.762 | -26.929 |
| US_CORE | PCA_RIDGE_VAR | 44 | 45.328 | 36.102 | 30.040 | 49.499 | 70.886 | 0.419 | 0.752 | 0.956 | -18.367 |
| US_CORE | PCA_RIDGE_VAR | 66 | 57.960 | 46.180 | 38.629 | 66.862 | 97.797 | 0.333 | 0.625 | 0.906 | -24.690 |
| US_CORE | PCA_RIDGE_VAR | 132 | 76.635 | 59.284 | 48.046 | 85.945 | 132.260 | 0.315 | 0.517 | 0.810 | -44.132 |
| US_CORE | PCA_RIDGE_VAR | 252 | 104.277 | 73.008 | 46.923 | 94.088 | 174.534 | 0.284 | 0.514 | 0.784 | -55.315 |

The fixed 10/25/50/100 bp thresholds are retained for cross-market comparability. At BAM J+44, 77.6% of maturity forecasts are within 25 bp and 95.7% within 50 bp. At J+252 those proportions fall to 27.5% and 57.7%. Europe and U.S. annual 90th-percentile absolute errors are roughly 176 and 175 bp, respectively. Thus a large relative gain at one year coexists with wide economically relevant errors.

## 6. BAM DNS versus simple mean reversion

| horizon | PERSISTENCE | HISTORICAL_MEAN | NS_AR1_MEAN_REVERSION | DNS_KALMAN_OU | dns_vs_rw_pct | dns_vs_simple_mr_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 22 | 18.878 | 69.964 | 18.572 | 18.946 | -0.362 | -2.016 |
| 44 | 28.368 | 69.145 | 26.365 | 26.624 | 6.146 | -0.982 |
| 66 | 38.528 | 66.834 | 33.165 | 33.373 | 13.381 | -0.627 |
| 132 | 67.957 | 61.385 | 49.456 | 49.936 | 26.518 | -0.970 |
| 252 | 103.488 | 56.654 | 61.323 | 61.802 | 40.281 | -0.780 |

The DNS-versus-simple-MR percentages are negative at every reported horizon: DNS never beats NS-AR1. At J+44/J+66, both models outperform persistence by moving factors toward an origin-local long-run level. At J+252, complete historical-mean reversion performs best. The evidence supports mean reversion in BAM yields/factors, but not incremental state-space predictive value.

## 7. Europe and U.S. simple mean reversion

Europe PCA-Ridge-VAR beats the best simple benchmark at J+44/J+66/J+132 by about 1.1%, 2.4%, and 4.6%, but loses to PCA-AR1 at J+252 by 2.6%. U.S. PCA-Ridge-VAR beats the best simple benchmark by about 3.6%, 4.8%, 11.9%, and 12.3% at J+44/J+66/J+132/J+252. These are incremental numerical gains beyond generic mean reversion, especially in the U.S., but pairwise DM and moving-block intervals do not establish significance.

## 8. Factor half-lives

| factor | median_ar1_phi | ar1_half_life_observation_steps | ou_kappa_per_calendar_day | ou_half_life_calendar_days | frozen_dns_transition_increment |
| --- | --- | --- | --- | --- | --- |
| beta0 | 0.995 | 141.995 | 0.006 | 107.440 | delta=1 per publication step |
| beta1 | 0.997 | 232.425 | 0.004 | 180.155 | delta=1 per publication step |
| beta2 | 0.996 | 186.373 | 0.005 | 144.419 | delta=1 per publication step |

AR(1) median half-lives are roughly 142, 232 and 186 publication observations for BAM level, slope and curvature. DNS/OU estimates 107, 180 and 144 calendar days. These are comparable in broad economic scale but not identical units. Importantly, the frozen DNS estimator fits kappa from calendar-day gaps while its forecast loop advances `delta=1` per publication step. Phase 5 forecasts are preserved, so this unit convention is disclosed rather than silently changed. A future methodological revision would need a separately versioned rerun.

## 9. Distance from equilibrium

| horizon | distance_category | origins | mean_distance | mean_dns_loss_gain_bp2 | median_dns_loss_gain_bp2 |
| --- | --- | --- | --- | --- | --- |
| 5 | LOW | 53 | 0.634 | -38.372 | -33.760 |
| 5 | MEDIUM | 1 | 1.005 | -28.035 | -28.035 |
| 10 | LOW | 53 | 0.634 | -29.080 | -32.269 |
| 10 | MEDIUM | 1 | 1.005 | -3.657 | -3.657 |
| 22 | LOW | 52 | 0.639 | -5.339 | -28.379 |
| 22 | MEDIUM | 1 | 1.005 | 140.579 | 140.579 |
| 44 | LOW | 51 | 0.646 | 91.913 | 6.177 |
| 44 | MEDIUM | 1 | 1.005 | 297.732 | 297.732 |
| 66 | LOW | 50 | 0.652 | 367.572 | 85.395 |
| 66 | MEDIUM | 1 | 1.005 | 525.549 | 525.549 |
| 132 | LOW | 46 | 0.678 | 2137.561 | 973.955 |
| 132 | MEDIUM | 1 | 1.005 | 1525.379 | 1525.379 |
| 252 | LOW | 41 | 0.694 | 6943.766 | 3475.895 |
| 252 | MEDIUM | 1 | 1.005 | 4695.945 | 4695.945 |

DNS gains turn positive from J+44 and grow with horizon, but nearly all origins fall in the fixed ex-ante `LOW` standardized-distance category; only one is `MEDIUM` and none is `HIGH`. This sample cannot validate a stable distance-conditioned effect. The scatter and automatically selected factor decompositions are descriptive bridges, not a conditional model.

## 10. Magnitude and directional calibration

| market | model | horizon | mean_absolute_predicted_change_bp | mean_absolute_realized_change_bp | direction_accuracy | change_correlation | calibration_slope_realized_on_predicted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | DNS_KALMAN_OU | 5 | 5.316 | 4.949 | 0.516 | 0.085 | 0.159 |
| BAM | DNS_KALMAN_OU | 10 | 5.823 | 6.073 | 0.568 | 0.146 | 0.265 |
| BAM | DNS_KALMAN_OU | 22 | 7.807 | 11.075 | 0.621 | 0.251 | 0.478 |
| BAM | DNS_KALMAN_OU | 44 | 12.293 | 17.779 | 0.675 | 0.371 | 0.715 |
| BAM | DNS_KALMAN_OU | 66 | 16.853 | 25.658 | 0.739 | 0.490 | 0.957 |
| BAM | DNS_KALMAN_OU | 132 | 28.447 | 50.546 | 0.846 | 0.705 | 1.448 |
| BAM | DNS_KALMAN_OU | 252 | 41.917 | 84.167 | 0.870 | 0.909 | 1.904 |
| EUROPE | PCA_RIDGE_VAR | 5 | 4.524 | 9.299 | 0.512 | 0.017 | 0.029 |
| EUROPE | PCA_RIDGE_VAR | 10 | 4.770 | 13.837 | 0.522 | 0.030 | 0.072 |
| EUROPE | PCA_RIDGE_VAR | 22 | 5.827 | 19.699 | 0.516 | 0.100 | 0.309 |
| EUROPE | PCA_RIDGE_VAR | 44 | 8.321 | 26.736 | 0.522 | 0.163 | 0.514 |
| EUROPE | PCA_RIDGE_VAR | 66 | 11.089 | 33.735 | 0.510 | 0.208 | 0.618 |
| EUROPE | PCA_RIDGE_VAR | 132 | 19.017 | 50.183 | 0.493 | 0.270 | 0.752 |
| EUROPE | PCA_RIDGE_VAR | 252 | 31.892 | 78.460 | 0.598 | 0.119 | 0.377 |
| US_CORE | PCA_RIDGE_VAR | 5 | 3.887 | 11.087 | 0.529 | 0.153 | 0.389 |
| US_CORE | PCA_RIDGE_VAR | 10 | 5.043 | 17.977 | 0.539 | 0.277 | 0.932 |
| US_CORE | PCA_RIDGE_VAR | 22 | 8.608 | 23.132 | 0.570 | 0.348 | 0.863 |
| US_CORE | PCA_RIDGE_VAR | 44 | 15.513 | 36.604 | 0.560 | 0.463 | 1.020 |
| US_CORE | PCA_RIDGE_VAR | 66 | 21.979 | 47.578 | 0.568 | 0.523 | 1.087 |
| US_CORE | PCA_RIDGE_VAR | 132 | 37.959 | 63.695 | 0.657 | 0.729 | 1.311 |
| US_CORE | PCA_RIDGE_VAR | 252 | 56.318 | 91.406 | 0.750 | 0.764 | 1.386 |

BAM DNS direction accuracy increases from 67.5% at J+44 to 87.0% at J+252, but it underpredicts movement magnitude: mean absolute predicted versus realized change is 12.3 versus 17.8 bp at J+44 and 41.9 versus 84.2 bp at J+252. The annual calibration slope near 1.90 likewise indicates under-dispersed predictions. Europe and U.S. PCA forecasts show analogous underprediction of large long-horizon moves.

## 11. Statistical inference and bootstrap uncertainty

Favorable pairwise DM results at 5%:

| market | model | benchmark | horizon | p_value | mean_loss_difference | n | hac_bandwidth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | DNS_KALMAN_OU | HISTORICAL_MEAN | 5 | 0.000 | -0.000 | 54 | 0 |
| BAM | DNS_KALMAN_OU | HISTORICAL_MEAN | 10 | 0.000 | -0.000 | 54 | 0 |
| BAM | DNS_KALMAN_OU | HISTORICAL_MEAN | 22 | 0.000 | -0.000 | 53 | 1 |
| BAM | DNS_KALMAN_OU | HISTORICAL_MEAN | 44 | 0.001 | -0.000 | 52 | 2 |
| BAM | DNS_KALMAN_OU | HISTORICAL_MEAN | 66 | 0.005 | -0.000 | 51 | 3 |
| EUROPE | DNS_KALMAN_OU | HISTORICAL_MEAN | 5 | 0.000 | -0.000 | 54 | 0 |
| EUROPE | DNS_KALMAN_OU | HISTORICAL_MEAN | 10 | 0.000 | -0.000 | 54 | 0 |
| EUROPE | DNS_KALMAN_OU | HISTORICAL_MEAN | 22 | 0.000 | -0.000 | 53 | 1 |
| EUROPE | DNS_KALMAN_OU | HISTORICAL_MEAN | 44 | 0.000 | -0.000 | 52 | 2 |
| EUROPE | DNS_KALMAN_OU | HISTORICAL_MEAN | 66 | 0.000 | -0.000 | 51 | 3 |
| US_CORE | DNS_KALMAN_OU | HISTORICAL_MEAN | 5 | 0.000 | -0.000 | 56 | 0 |
| US_CORE | DNS_KALMAN_OU | HISTORICAL_MEAN | 10 | 0.000 | -0.000 | 55 | 0 |
| US_CORE | DNS_KALMAN_OU | HISTORICAL_MEAN | 22 | 0.000 | -0.000 | 55 | 1 |
| US_CORE | DNS_KALMAN_OU | HISTORICAL_MEAN | 44 | 0.000 | -0.000 | 54 | 2 |
| US_CORE | DNS_KALMAN_OU | HISTORICAL_MEAN | 66 | 0.002 | -0.000 | 53 | 3 |

DNS versus NS-AR1 has positive loss differences at every BAM horizon, so the comparison favors the simple model; at J+22 this disadvantage is significant, while long-horizon differences are not. Moving-block bootstrap intervals at J+44/J+66 all include zero:

| market | model | benchmark | horizon | status | block_length | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | DNS_KALMAN_OU | NS_AR1_MEAN_REVERSION | 44 | MOVING_BLOCK_BOOTSTRAP_DESCRIPTIVE | 3 | -0.000 | 0.000 |
| BAM | DNS_KALMAN_OU | NS_AR1_MEAN_REVERSION | 66 | MOVING_BLOCK_BOOTSTRAP_DESCRIPTIVE | 4 | -0.000 | 0.000 |
| BAM | DNS_KALMAN_OU | NS_AR1_MEAN_REVERSION | 132 | REJECTED_TOO_LITTLE_EFFECTIVE_INFORMATION | 7 | nan | nan |
| BAM | DNS_KALMAN_OU | NS_AR1_MEAN_REVERSION | 252 | REJECTED_TOO_LITTLE_EFFECTIVE_INFORMATION | 13 | nan | nan |
| EUROPE | PCA_RIDGE_VAR | PCA_AR1_MEAN_REVERSION | 44 | MOVING_BLOCK_BOOTSTRAP_DESCRIPTIVE | 3 | -0.000 | 0.000 |
| EUROPE | PCA_RIDGE_VAR | PCA_AR1_MEAN_REVERSION | 66 | MOVING_BLOCK_BOOTSTRAP_DESCRIPTIVE | 4 | -0.000 | 0.000 |
| EUROPE | PCA_RIDGE_VAR | PCA_AR1_MEAN_REVERSION | 132 | REJECTED_TOO_LITTLE_EFFECTIVE_INFORMATION | 7 | nan | nan |
| EUROPE | PCA_RIDGE_VAR | PCA_AR1_MEAN_REVERSION | 252 | REJECTED_TOO_LITTLE_EFFECTIVE_INFORMATION | 13 | nan | nan |
| US_CORE | PCA_RIDGE_VAR | PCA_AR1_MEAN_REVERSION | 44 | MOVING_BLOCK_BOOTSTRAP_DESCRIPTIVE | 3 | -0.000 | 0.000 |
| US_CORE | PCA_RIDGE_VAR | PCA_AR1_MEAN_REVERSION | 66 | MOVING_BLOCK_BOOTSTRAP_DESCRIPTIVE | 4 | -0.000 | 0.000 |
| US_CORE | PCA_RIDGE_VAR | PCA_AR1_MEAN_REVERSION | 132 | REJECTED_TOO_LITTLE_EFFECTIVE_INFORMATION | 7 | nan | nan |
| US_CORE | PCA_RIDGE_VAR | PCA_AR1_MEAN_REVERSION | 252 | REJECTED_TOO_LITTLE_EFFECTIVE_INFORMATION | 13 | nan | nan |

The bootstrap is explicitly rejected at J+132/J+252 because block lengths of 7/13 leave too little effective information. Reporting a narrow IID interval would be misleading.

## 12. Forecast uncertainty

Valid predictive intervals were not generated. The frozen backtest stores point forecasts but not origin-level propagated state covariance, and `forecast_states` returns states without covariance paths. Constructing 50/80/95% intervals would require retaining Kalman posterior covariance, propagating it consistently through the OU transition and measurement equation, and resolving the calendar-day/publication-step convention. Intervals were therefore not fabricated.

## 13. Maturity-level accuracy and stability

Machine-readable maturity-level RMSE/MAE tables show BAM DNS gains are broad rather than isolated, consistent with Phase 5 segment results. Error-distribution and cumulative evidence still show substantial tail risk. No crisis-specific model or retrospectively selected regime was fitted.

## 14. Interpretation discipline

- “DNS beats persistence by 40%” is numerically true at BAM J+252.
- It does not mean one-year BAM yields are predicted precisely: RMSE is 61.8 bp and the 90th-percentile absolute error is 103.4 bp.
- BAM yields exhibit long-horizon mean-reverting forecastability because simple NS-AR1 and even the historical mean reproduce or exceed DNS gains.
- DNS does not add demonstrated value beyond simple mean reversion.

Medium horizons offer the better practical compromise: BAM J+44/J+66 errors are materially smaller and most forecasts remain inside 50 bp, whereas annual forecasts are better treated as directional/mean-reversion scenarios.

## 15. Limitations

Long-horizon origins overlap heavily; J+132/J+252 are descriptive-only. ECB spot and Treasury par curves differ structurally. Simple AR(1)s deliberately constrain phi to `[0,0.999]` and use no lag search. Historical means may shift over structural regimes. Threshold coverage aggregates maturities. No valid forecast intervals are available. No macro, cross-market, nonlinear, conditional or regime model was introduced.

## 16. Reproducibility and conclusion

Outputs contain frozen-plus-benchmark forecasts, decimal and bp metrics, quantiles, calibration, maturity tables, pairwise overlap-aware DM tests, conservative bootstrap diagnostics, equilibrium distances, representative decompositions, half-life comparisons and twelve figures. All benchmark parameters are fitted using data through the forecast origin only.

**MEDIUM-HORIZON FORECASTS ARE ABSOLUTELY INFORMATIVE; J+252 GAINS ARE MAINLY RELATIVE AND IMPRECISE**

**DNS GAINS ARE PRIMARILY EXPLAINED BY SIMPLE MEAN REVERSION**

**PHASE 5C SHOULD REMAIN DEFERRED**
