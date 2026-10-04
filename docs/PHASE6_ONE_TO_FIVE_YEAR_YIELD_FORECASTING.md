# Phase 6 — One- to five-year BAM yield-curve forecasting

## 1. Motivation and final decisions

Phase 6 asks how far BAM curve forecasting can be extended before a point forecast becomes chiefly an equilibrium scenario. It creates new versioned DNS models and leaves all historical Phase 5/5C0 artifacts untouched.

**1Y FORECAST PRACTICALLY INFORMATIVE**

**2Y–3Y FORECASTS PRIMARILY EQUILIBRIUM ESTIMATES**

**4Y–5Y OUTPUTS SHOULD BE TREATED AS LONG-RUN SCENARIOS**

**SIMPLE NS MEAN REVERSION REMAINS SUFFICIENT**

The one-year experiment has 188 monthly origins but only 14.5 non-overlapping equivalents. Preferred-model RMSE is about 64.6 bp, so “informative” means a broad directional/curve scenario—not high-precision pricing. By two years the median NS-AR1 forecast is only 20 bp from its equilibrium curve, falling below 10 bp at three years. Four/five-year estimates have only 3.0/2.2 effective episodes and persistence becomes competitive or best.

## 2. DNS time-unit correction

`DNS_CALENDAR_TIME_V2` estimates OU κ using actual calendar-day gaps and propagates from origin t to realized target T using the exact `(T−t)` calendar days. `DNS_PUBLICATION_TIME_V2` estimates dynamics on unit-spaced synthetic observation time and propagates h observation steps. `DNS_FROZEN_CONVENTION` reproduces the old mixed convention solely for sensitivity analysis; it does not replace or overwrite frozen results.

Calendar V2 is scientifically preferred because BAM publication steps do not map exactly to years. Publication V2 is a coherent sensitivity model. Both select lambda and estimate states using origin-local history only.

## 3. Horizon feasibility frozen before results

| market | horizon | available_observations | minimum_training_observations | realized_historical_forecasts | monthly_forecast_origins | non_overlapping_equivalent_windows | overlap_bandwidth | effective_inferential_information | earliest_possible_origin | latest_origin_with_realized_target | remaining_usable_historical_span_days | classification | classification_basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | 252 | 5328 | 1260 | 188 | 188 | 14.460 | 12 | 14.460 | 2009-10-30 | 2025-05-30 | 5691 | LIMITED_POWER | sample structure only; frozen before model results |
| BAM | 504 | 5328 | 1260 | 175 | 175 | 6.730 | 25 | 6.730 | 2009-10-30 | 2024-04-30 | 5296 | DESCRIPTIVE_ONLY | sample structure only; frozen before model results |
| BAM | 756 | 5328 | 1260 | 162 | 162 | 4.260 | 37 | 4.260 | 2009-10-30 | 2023-03-31 | 4900 | DESCRIPTIVE_ONLY | sample structure only; frozen before model results |
| BAM | 1008 | 5328 | 1260 | 150 | 150 | 3.000 | 49 | 3.000 | 2009-10-30 | 2022-03-31 | 4535 | SCENARIO_ONLY | sample structure only; frozen before model results |
| BAM | 1260 | 5328 | 1260 | 138 | 138 | 2.190 | 62 | 2.190 | 2009-10-30 | 2021-03-31 | 4170 | SCENARIO_ONLY | sample structure only; frozen before model results |

J+252 is `LIMITED_POWER`; J+504/J+756 are `DESCRIPTIVE_ONLY`; J+1008/J+1260 are `SCENARIO_ONLY`. Conventional DM inference is suppressed outside J+252. Thousands of maturity rows are never treated as independent forecast episodes.

## 4. Observation steps versus calendar duration

| horizon | median_calendar_days | p10_calendar_days | p90_calendar_days | minimum_calendar_days | maximum_calendar_days | pairs |
| --- | --- | --- | --- | --- | --- | --- |
| 252 | 376.000 | 371.000 | 384.000 | 365 | 398 | 5076 |
| 504 | 753.000 | 743.000 | 763.000 | 737 | 780 | 4824 |
| 756 | 1129.000 | 1117.000 | 1141.000 | 1109 | 1153 | 4572 |
| 1008 | 1506.000 | 1495.000 | 1517.000 | 1487 | 1526 | 4320 |
| 1260 | 1883.000 | 1870.000 | 1894.000 | 1866 | 1902 | 4068 |

The median durations are 376, 753, 1,129, 1,506 and 1,883 calendar days—approximately 1.03, 2.06, 3.09, 4.12 and 5.16 years. Labels 1Y–5Y are therefore approximate.

## 5. Main historical accuracy

| horizon | classification | rw_rmse_bp | historical_mean_rmse_bp | ns_ar1_rmse_bp | dns_calendar_v2_rmse_bp | dns_publication_v2_rmse_bp | best_model | best_rmse_bp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 252 | LIMITED_POWER | 67.138 | 86.619 | 64.794 | 67.983 | 64.559 | DNS_PUBLICATION_TIME_V2 | 64.559 |
| 504 | DESCRIPTIVE_ONLY | 96.002 | 93.087 | 82.496 | 85.194 | 82.947 | NS_AR1_MEAN_REVERSION | 82.496 |
| 756 | DESCRIPTIVE_ONLY | 99.106 | 97.942 | 92.330 | 94.753 | 93.987 | DIRECT_AR1_MEAN_REVERSION | 89.506 |
| 1008 | SCENARIO_ONLY | 96.435 | 101.887 | 98.748 | 101.988 | 101.841 | DIRECT_AR1_MEAN_REVERSION | 92.269 |
| 1260 | SCENARIO_ONLY | 99.972 | 107.830 | 105.949 | 109.580 | 109.945 | PERSISTENCE | 99.972 |

The broad 2009–2025 evaluation differs from Phase 5C0's 2022-era sample. At J+252 the coherent publication-time DNS is 64.6 bp, NS-AR1 is 64.8 bp and calendar DNS is 68.0 bp. At J+504 NS-AR1 is best among these at 82.5 bp. Direct yield AR(1) is numerically best at J+756/J+1008 (89.5/92.3 bp), while persistence wins J+1260 at 100.0 bp. Thus simple equilibrium-oriented dynamics do not deliver monotonically improving five-year accuracy.

## 6. Absolute accuracy and error distributions

| model | horizon | rmse_bp | mae_bp | median_absolute_error_bp | p75_absolute_error_bp | p90_absolute_error_bp | bias_bp | direction_accuracy | within_25bp | within_50bp | within_100bp | within_150bp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIRECT_AR1_MEAN_REVERSION | 252 | 65.165 | 51.807 | 41.642 | 75.758 | 104.007 | 17.864 | 0.505 | 0.322 | 0.559 | 0.888 | 0.968 |
| DIRECT_AR1_MEAN_REVERSION | 504 | 85.992 | 74.990 | 74.868 | 107.523 | 131.054 | 31.697 | 0.577 | 0.135 | 0.331 | 0.688 | 0.966 |
| DIRECT_AR1_MEAN_REVERSION | 756 | 89.506 | 79.117 | 77.636 | 111.249 | 134.942 | 43.093 | 0.560 | 0.114 | 0.266 | 0.675 | 0.947 |
| DIRECT_AR1_MEAN_REVERSION | 1008 | 92.269 | 79.172 | 83.360 | 118.114 | 144.155 | 61.855 | 0.594 | 0.179 | 0.323 | 0.633 | 0.939 |
| DIRECT_AR1_MEAN_REVERSION | 1260 | 101.104 | 88.130 | 89.381 | 128.191 | 159.600 | 80.695 | 0.600 | 0.122 | 0.257 | 0.614 | 0.854 |
| DNS_CALENDAR_TIME_V2 | 252 | 67.983 | 55.045 | 45.574 | 85.364 | 114.582 | 32.694 | 0.576 | 0.260 | 0.548 | 0.828 | 0.987 |
| DNS_CALENDAR_TIME_V2 | 504 | 85.194 | 73.219 | 66.779 | 108.190 | 136.889 | 49.987 | 0.601 | 0.146 | 0.330 | 0.711 | 0.959 |
| DNS_CALENDAR_TIME_V2 | 756 | 94.753 | 82.192 | 79.179 | 120.633 | 150.077 | 62.097 | 0.540 | 0.139 | 0.286 | 0.652 | 0.899 |
| DNS_CALENDAR_TIME_V2 | 1008 | 101.988 | 90.140 | 89.641 | 131.101 | 155.421 | 77.518 | 0.490 | 0.119 | 0.238 | 0.580 | 0.874 |
| DNS_CALENDAR_TIME_V2 | 1260 | 109.580 | 97.641 | 103.509 | 136.862 | 162.480 | 91.779 | 0.475 | 0.097 | 0.212 | 0.478 | 0.826 |
| DNS_PUBLICATION_TIME_V2 | 252 | 64.559 | 50.869 | 42.575 | 79.049 | 106.171 | 25.608 | 0.576 | 0.329 | 0.570 | 0.870 | 0.982 |
| DNS_PUBLICATION_TIME_V2 | 504 | 82.947 | 71.060 | 61.183 | 108.163 | 132.886 | 43.998 | 0.570 | 0.149 | 0.364 | 0.709 | 0.973 |
| DNS_PUBLICATION_TIME_V2 | 756 | 93.987 | 81.384 | 74.524 | 118.604 | 149.341 | 58.112 | 0.514 | 0.128 | 0.274 | 0.663 | 0.903 |
| DNS_PUBLICATION_TIME_V2 | 1008 | 101.841 | 89.701 | 89.543 | 132.427 | 156.259 | 75.212 | 0.484 | 0.113 | 0.247 | 0.586 | 0.857 |
| DNS_PUBLICATION_TIME_V2 | 1260 | 109.945 | 97.315 | 102.576 | 139.698 | 164.719 | 90.859 | 0.506 | 0.107 | 0.218 | 0.487 | 0.812 |
| HISTORICAL_MEAN | 252 | 86.619 | 74.315 | 73.693 | 104.120 | 141.219 | 48.986 | 0.505 | 0.164 | 0.349 | 0.728 | 0.939 |
| HISTORICAL_MEAN | 504 | 93.087 | 80.398 | 79.773 | 111.092 | 150.002 | 55.504 | 0.577 | 0.156 | 0.309 | 0.665 | 0.900 |
| HISTORICAL_MEAN | 756 | 97.942 | 85.249 | 87.217 | 116.007 | 155.653 | 63.444 | 0.560 | 0.151 | 0.266 | 0.606 | 0.879 |
| HISTORICAL_MEAN | 1008 | 101.887 | 88.486 | 91.219 | 118.609 | 163.823 | 77.125 | 0.594 | 0.139 | 0.255 | 0.588 | 0.852 |
| HISTORICAL_MEAN | 1260 | 107.830 | 94.650 | 92.643 | 126.834 | 172.365 | 90.136 | 0.600 | 0.097 | 0.200 | 0.596 | 0.808 |
| NS_AR1_MEAN_REVERSION | 252 | 64.794 | 52.237 | 43.593 | 82.105 | 106.075 | 27.122 | 0.550 | 0.306 | 0.559 | 0.866 | 0.987 |
| NS_AR1_MEAN_REVERSION | 504 | 82.496 | 71.195 | 63.176 | 107.652 | 130.705 | 44.699 | 0.589 | 0.147 | 0.352 | 0.720 | 0.978 |
| NS_AR1_MEAN_REVERSION | 756 | 92.330 | 80.827 | 75.191 | 115.860 | 143.756 | 57.881 | 0.556 | 0.130 | 0.261 | 0.676 | 0.931 |
| NS_AR1_MEAN_REVERSION | 1008 | 98.748 | 86.476 | 84.772 | 120.094 | 153.423 | 74.055 | 0.584 | 0.137 | 0.253 | 0.603 | 0.876 |
| NS_AR1_MEAN_REVERSION | 1260 | 105.949 | 93.509 | 94.195 | 126.211 | 166.119 | 88.516 | 0.589 | 0.102 | 0.196 | 0.584 | 0.814 |
| PERSISTENCE | 252 | 67.138 | 47.341 | 32.568 | 71.096 | 102.056 | 7.179 | nan | 0.440 | 0.617 | 0.897 | 0.944 |
| PERSISTENCE | 504 | 96.002 | 75.782 | 64.486 | 110.329 | 167.324 | 13.264 | nan | 0.253 | 0.415 | 0.711 | 0.855 |
| PERSISTENCE | 756 | 99.106 | 81.701 | 77.361 | 114.304 | 159.753 | 14.908 | nan | 0.178 | 0.352 | 0.665 | 0.870 |
| PERSISTENCE | 1008 | 96.435 | 84.644 | 79.010 | 108.387 | 147.873 | 28.617 | nan | 0.087 | 0.234 | 0.692 | 0.903 |
| PERSISTENCE | 1260 | 99.972 | 87.289 | 84.410 | 114.804 | 153.758 | 49.639 | nan | 0.078 | 0.249 | 0.654 | 0.890 |

At one year, publication DNS places 57.0% of maturity forecasts within 50 bp and 87.0% within 100 bp; its median error is 42.6 bp. At two years the corresponding rates fall to 36.4% and 70.9%. Beyond three years RMSE is around 90–110 bp for most models, and positive biases become large. Point precision deteriorates even when forecasts converge smoothly.

## 7. Relative accuracy versus persistence

At one year NS-AR1 and publication DNS improve RMSE by only about 3.5–3.8% over persistence in the expanded historical sample. NS-AR1 improves by 14.1% at two years and 6.8% at three years, but loses at four/five years. The relative horizon plots distinguish this from absolute error, which rises sharply through two years and then stays near an economically large plateau.

## 8. Historical mean and simple mean reversion

The expanding historical mean is leakage-safe but not universally strong: RMSE is 86.6, 93.1, 97.9, 101.9 and 107.8 bp. Its earlier J+252 success was evaluation-period dependent. NS-AR1 is materially better through three years, then becomes indistinguishable from an equilibrium scenario and eventually loses to persistence.

## 9. Corrected DNS versus NS-AR1

Calendar DNS is worse than NS-AR1 at every horizon. Publication DNS is only 0.24 bp better at J+252 and is worse thereafter. The limited-power J+252 DM p-values are 0.146 for calendar DNS versus NS-AR1 and 0.829 for publication DNS versus NS-AR1. There is no evidence that corrected DNS adds value beyond simple NS mean reversion.

## 10. Time-unit sensitivity

| horizon | frozen_vs_v2_curve_rms_difference_bp | frozen_dns_rmse_bp | calendar_v2_rmse_bp |
| --- | --- | --- | --- |
| 252 | 10.674 | 64.487 | 67.983 |
| 504 | 9.996 | 81.937 | 85.194 |
| 756 | 7.224 | 92.241 | 94.753 |
| 1008 | 4.561 | 100.203 | 101.988 |
| 1260 | 2.699 | 108.453 | 109.580 |

Correcting the mixed convention changes historical curves by about 10.7 bp RMS at J+252 and 10.0 bp at J+504. The difference shrinks at longer horizons because both specifications collapse toward their long-run mean. Calendar V2 is slightly less accurate in this sample; coherence, not ex-post RMSE, is why it is preferred scientifically.

## 11. Forecast convergence

| horizon | ns_ar1_equilibrium_distance_bp | calendar_dns_equilibrium_distance_bp |
| --- | --- | --- |
| 252 | 40.813 | 25.877 |
| 504 | 20.017 | 8.176 |
| 756 | 9.778 | 3.269 |
| 1008 | 5.753 | 1.302 |
| 1260 | 3.401 | 0.532 |

Median NS-AR1 distance from equilibrium falls from 40.8 bp at one year to 20.0, 9.8, 5.8 and 3.4 bp from two through five years. Calendar DNS converges even faster: 25.9 bp at one year, 8.2 bp at two, and only 0.5 bp at five. Consequently, the 4Y/5Y outputs are primarily long-run equilibrium estimates rather than dynamically informative forecasts.

## 12. Historical yield-curve visualizations

For every horizon, three origins are selected algorithmically using NS-AR1 curve-error quantiles: nearest Q25 (`successful`), Q50 (`typical`) and Q75 (`difficult`). Each presentation-quality figure shows the origin/persistence curve, NS-AR1 forecast, realized curve, training-only equilibrium, and a maturity-level residual panel. No example is hand-picked.

## 13. Latest prospective BAM forecast

The latest prospective curves have no realized targets and are explicitly stored separately. Human-facing percentage forecasts for NS-AR1 and calendar DNS are:

| model | maturity | current_percent | equilibrium_percent | 1Y_percent | 2Y_percent | 3Y_percent | 4Y_percent | 5Y_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNS_CALENDAR_TIME_V2 | 0.250 | 2.235 | 2.721 | 2.636 | 2.670 | 2.673 | 2.673 | 2.673 |
| DNS_CALENDAR_TIME_V2 | 0.500 | 2.277 | 2.760 | 2.675 | 2.708 | 2.711 | 2.711 | 2.711 |
| DNS_CALENDAR_TIME_V2 | 1.000 | 2.309 | 2.835 | 2.749 | 2.781 | 2.783 | 2.783 | 2.783 |
| DNS_CALENDAR_TIME_V2 | 2.000 | 2.365 | 2.975 | 2.888 | 2.917 | 2.919 | 2.919 | 2.919 |
| DNS_CALENDAR_TIME_V2 | 5.000 | 2.888 | 3.328 | 3.241 | 3.265 | 3.266 | 3.266 | 3.266 |
| DNS_CALENDAR_TIME_V2 | 10.000 | 3.340 | 3.753 | 3.666 | 3.686 | 3.687 | 3.687 | 3.687 |
| DNS_CALENDAR_TIME_V2 | 15.000 | 3.672 | 4.039 | 3.955 | 3.975 | 3.975 | 3.975 | 3.975 |
| DNS_CALENDAR_TIME_V2 | 20.000 | 3.753 | 4.238 | 4.157 | 4.177 | 4.177 | 4.177 | 4.177 |
| DNS_CALENDAR_TIME_V2 | 30.000 | 4.100 | 4.484 | 4.408 | 4.430 | 4.431 | 4.431 | 4.431 |
| NS_AR1_MEAN_REVERSION | 0.250 | 2.235 | 2.721 | 2.628 | 2.702 | 2.717 | 2.720 | 2.721 |
| NS_AR1_MEAN_REVERSION | 0.500 | 2.277 | 2.760 | 2.668 | 2.741 | 2.755 | 2.759 | 2.760 |
| NS_AR1_MEAN_REVERSION | 1.000 | 2.309 | 2.835 | 2.745 | 2.816 | 2.830 | 2.833 | 2.834 |
| NS_AR1_MEAN_REVERSION | 2.000 | 2.365 | 2.975 | 2.889 | 2.957 | 2.970 | 2.973 | 2.974 |
| NS_AR1_MEAN_REVERSION | 5.000 | 2.888 | 3.328 | 3.253 | 3.314 | 3.325 | 3.328 | 3.328 |
| NS_AR1_MEAN_REVERSION | 10.000 | 3.340 | 3.753 | 3.687 | 3.742 | 3.751 | 3.752 | 3.753 |
| NS_AR1_MEAN_REVERSION | 15.000 | 3.672 | 4.039 | 3.979 | 4.030 | 4.038 | 4.039 | 4.039 |
| NS_AR1_MEAN_REVERSION | 20.000 | 3.753 | 4.238 | 4.180 | 4.229 | 4.236 | 4.238 | 4.238 |
| NS_AR1_MEAN_REVERSION | 30.000 | 4.100 | 4.484 | 4.427 | 4.476 | 4.483 | 4.484 | 4.484 |

The latest NS-AR1 curve moves from current short yields near 2.23–2.37% toward an equilibrium short end near 2.72–2.97%, while the 30-year point moves from 4.10% toward 4.48%. By three years, most of the convergence has occurred. These are model scenarios, not guaranteed future rates.

## 14. Forecast uncertainty

Predictive covariance bands are not shown. Although V2 is coherent in time units, the current pipeline has not validated joint state/measurement covariance coverage over multi-year, highly overlapping horizons. Historical out-of-sample quantiles are reported instead. At five years only about 2.2 independent episodes exist, so plotting narrow parametric bands would imply false precision.

## 15. Guarded inference

| model | benchmark | horizon | status | statistic | p_value | mean_loss_difference | n | hac_bandwidth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNS_CALENDAR_TIME_V2 | NS_AR1_MEAN_REVERSION | 252 | LIMITED_POWER_DM | 1.462 | 0.146 | 0.000 | 188 | 12 |
| DNS_CALENDAR_TIME_V2 | NS_AR1_MEAN_REVERSION | 504 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 175 | 25 |
| DNS_CALENDAR_TIME_V2 | NS_AR1_MEAN_REVERSION | 756 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 162 | 37 |
| DNS_CALENDAR_TIME_V2 | NS_AR1_MEAN_REVERSION | 1008 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 150 | 49 |
| DNS_CALENDAR_TIME_V2 | NS_AR1_MEAN_REVERSION | 1260 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 138 | 62 |
| DNS_PUBLICATION_TIME_V2 | NS_AR1_MEAN_REVERSION | 252 | LIMITED_POWER_DM | -0.216 | 0.829 | -0.000 | 188 | 12 |
| DNS_PUBLICATION_TIME_V2 | NS_AR1_MEAN_REVERSION | 504 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 175 | 25 |
| DNS_PUBLICATION_TIME_V2 | NS_AR1_MEAN_REVERSION | 756 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 162 | 37 |
| DNS_PUBLICATION_TIME_V2 | NS_AR1_MEAN_REVERSION | 1008 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 150 | 49 |
| DNS_PUBLICATION_TIME_V2 | NS_AR1_MEAN_REVERSION | 1260 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 138 | 62 |
| NS_AR1_MEAN_REVERSION | PERSISTENCE | 252 | LIMITED_POWER_DM | -0.225 | 0.822 | -0.000 | 188 | 12 |
| NS_AR1_MEAN_REVERSION | PERSISTENCE | 504 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 175 | 25 |
| NS_AR1_MEAN_REVERSION | PERSISTENCE | 756 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 162 | 37 |
| NS_AR1_MEAN_REVERSION | PERSISTENCE | 1008 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 150 | 49 |
| NS_AR1_MEAN_REVERSION | PERSISTENCE | 1260 | NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION | nan | nan | nan | 138 | 62 |

Only J+252 receives a limited-power DM calculation. J+504–J+1260 p-values are intentionally absent. Non-rejection at these horizons cannot be interpreted as equality.

## 16. Prediction, accuracy, relative improvement, and equilibrium

- **Prediction:** a model-generated future curve conditional on the latest observed history.
- **Accuracy:** historical distance between forecast and realized curves, reported in bp.
- **Relative improvement:** error reduction versus persistence; it can coexist with large absolute errors.
- **Equilibrium estimate:** the long-run curve implied by training-only factor means. At 4Y/5Y this is the correct interpretation of mean-reversion outputs.

## 17. Practical interpretation and limitations

One-year forecasts offer moderate scenario information but roughly 65 bp RMSE. Two-year forecasts are less precise and already close to equilibrium. Three years is descriptive; four/five years lack enough independent historical episodes and should not be used as precise point forecasts. Results depend on a long evaluation period containing structural yield regimes, the BAM curve itself is derived/interpolated outside the nine evaluated maturities, and no macro, international, nonlinear or conditional predictors were introduced.

## 18. Reproducibility and conclusion

Outputs include the ex-ante audit, calendar mapping, all seven model forecasts, absolute and maturity metrics, guarded DM results, equilibrium convergence, time-unit sensitivity, deterministic example selection, latest prospective tables, and 24 figures. Internal rates remain decimal; charts and human tables alone convert to percent or bp.

**1Y FORECAST PRACTICALLY INFORMATIVE**

**2Y–3Y FORECASTS PRIMARILY EQUILIBRIUM ESTIMATES**

**4Y–5Y OUTPUTS SHOULD BE TREATED AS LONG-RUN SCENARIOS**

**SIMPLE NS MEAN REVERSION REMAINS SUFFICIENT**
