# Phase 5 — Medium- and long-horizon yield-curve predictability

## 1. Motivation and Phase 4 baseline

Phase 4 found that persistence was extremely difficult to beat at J+5/J+10/J+22. Phase 5 tests, without assuming, whether direct factor changes or OU mean reversion become useful at J+44, J+66, J+132 and J+252. J+h always means the h-th subsequent published market observation on that market's own business/publication calendar—not h calendar days.

## 2. Decisions

**LONG-HORIZON PREDICTABILITY SUPPORTED**

**PHASE 5C CONDITIONAL PREDICTABILITY JUSTIFIED**

**PHASE 4D INTERNATIONAL TRANSMISSION REMAINS DEFERRED**

The first decision requires economically nontrivial improvement across at least two markets and neighboring long horizons, rather than an isolated winning cell. The Phase 5C decision uses only the completed horizon, maturity and temporal-stability evidence; no Phase 5C model was implemented. No international curve was used as another market's predictor.

The central result is a genuine horizon response, but not a universal model response. At J+44/J+66, BAM DNS reaches 0.939/0.866 of persistence RMSE, Europe PCA-Ridge-VAR reaches 0.968/0.945, and U.S. PCA-Ridge-VAR reaches 0.934/0.910. At J+132/J+252 the numerical gains become larger—BAM DNS 0.735/0.597, Europe PCA-Ridge-VAR 0.904/0.925, and U.S. PCA-Ridge-VAR 0.818/0.774—but those horizons are descriptive-only. None of the favorable aggregate DM tests reaches 5%; BAM DNS approaches conventional significance at J+66/J+132/J+252 (p≈0.098/0.069/0.065), while Europe and U.S. gains remain less precisely estimated. Thus long-horizon predictability is supported by economic magnitude, neighboring-horizon consistency and maturity breadth, not by decisive conventional inference.

## 3. Ex-ante horizon feasibility and sample audit

Classification was frozen before forecasts were run. `FULLY_FEASIBLE` requires adequate raw and overlap-adjusted information; `LIMITED_POWER` permits inference with caution; `DESCRIPTIVE_ONLY` retains the forecast but rejects strong inferential claims; `NOT_FEASIBLE` would suppress modelling. No classification used forecast performance.

| market | horizon | total_observations | monthly_evaluation_origins | weekly_evaluation_origins | training_rows_earliest_origin | training_rows_latest_origin | monthly_maximum_overlap_lag | monthly_effective_loss_length | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | 5 | 5328 | 54 | 232 | 4242 | 5310 | 0 | 54.00 | FULLY_FEASIBLE |
| BAM | 10 | 5328 | 54 | 231 | 4237 | 5305 | 0 | 54.00 | FULLY_FEASIBLE |
| BAM | 22 | 5328 | 53 | 228 | 4225 | 5276 | 1 | 26.50 | FULLY_FEASIBLE |
| BAM | 44 | 5328 | 52 | 224 | 4203 | 5234 | 2 | 17.33 | LIMITED_POWER |
| BAM | 66 | 5328 | 51 | 218 | 4181 | 5192 | 3 | 12.75 | LIMITED_POWER |
| BAM | 132 | 5328 | 47 | 204 | 4115 | 5047 | 6 | 6.71 | DESCRIPTIVE_ONLY |
| BAM | 252 | 5328 | 42 | 180 | 3995 | 4818 | 12 | 3.23 | DESCRIPTIVE_ONLY |
| EUROPE | 5 | 5572 | 54 | 233 | 4424 | 5549 | 0 | 54.00 | FULLY_FEASIBLE |
| EUROPE | 10 | 5572 | 54 | 232 | 4419 | 5544 | 0 | 54.00 | FULLY_FEASIBLE |
| EUROPE | 22 | 5572 | 53 | 230 | 4407 | 5512 | 1 | 26.50 | FULLY_FEASIBLE |
| EUROPE | 44 | 5572 | 52 | 225 | 4385 | 5470 | 2 | 17.33 | LIMITED_POWER |
| EUROPE | 66 | 5572 | 51 | 220 | 4363 | 5426 | 3 | 12.75 | LIMITED_POWER |
| EUROPE | 132 | 5572 | 48 | 207 | 4297 | 5298 | 6 | 6.86 | DESCRIPTIVE_ONLY |
| EUROPE | 252 | 5572 | 42 | 183 | 4177 | 5048 | 11 | 3.50 | DESCRIPTIVE_ONLY |
| US_CORE | 5 | 9154 | 56 | 240 | 7999 | 9143 | 0 | 56.00 | FULLY_FEASIBLE |
| US_CORE | 10 | 9154 | 55 | 239 | 7994 | 9116 | 0 | 55.00 | FULLY_FEASIBLE |
| US_CORE | 22 | 9154 | 55 | 236 | 7982 | 9104 | 1 | 27.50 | FULLY_FEASIBLE |
| US_CORE | 44 | 9154 | 54 | 232 | 7960 | 9061 | 2 | 18.00 | LIMITED_POWER |
| US_CORE | 66 | 9154 | 53 | 227 | 7938 | 9019 | 3 | 13.25 | LIMITED_POWER |
| US_CORE | 132 | 9154 | 50 | 214 | 7872 | 8890 | 6 | 7.14 | DESCRIPTIVE_ONLY |
| US_CORE | 252 | 9154 | 44 | 188 | 7752 | 8646 | 12 | 3.38 | DESCRIPTIVE_ONLY |

J+132 and J+252 are retained for all markets but are explicitly descriptive-only. Their many overlapping raw forecasts correspond to very little independent loss-series information. Weekly counts are audit diagnostics only; monthly origins remain primary and daily origins were not introduced.

## 4. Methodology and leakage controls

The unchanged model family is Persistence, DNS-Kalman/OU, NS-Ridge, NS-ElasticNet, PCA-Ridge and PCA-Ridge-VAR. Ridge/ElasticNet estimate each Δfactor(t,h) directly. PCA is refitted at every origin. PCA-Ridge-VAR remains a recursively iterated regularized factor VAR. DNS parameters, lambda, Kalman state and OU dynamics are estimated using origin-local history, then propagated directly to h. Direct training origin s is accepted only when s+h≤t; recent unrealized outcomes are excluded. Hyperparameter validation is expanding and target-purged. Rates remain decimal.

All seven horizons use actual half-open forecast intervals `(origin,target]`. HAC bandwidth is the maximum overlapping lag measured in the monthly loss sequence, not h−1. Aggregate DM tests average maturity losses within origin before inference. Raw maturity-test p-values and Benjamini–Hochberg FDR diagnostics are both retained.

## 5. Phase 5A master horizon table — RMSE ratio versus persistence

Values below one favor the model.

| market | model | 5 | 10 | 22 | 44 | 66 | 132 | 252 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | DNS_KALMAN_OU | 1.0957 | 1.0666 | 1.0036 | 0.9385 | 0.8662 | 0.7348 | 0.5972 |
| BAM | NS_ELASTICNET | 1.0711 | 1.0516 | 1.0423 | 1.0194 | 1.0452 | 1.0383 | 1.0841 |
| BAM | NS_RIDGE | 1.0364 | 1.0152 | 1.2653 | 1.2968 | 1.1932 | 0.9290 | 1.0767 |
| BAM | PCA_RIDGE | 1.0228 | 1.0150 | 1.0755 | 0.9676 | 0.9344 | 0.9733 | 1.1785 |
| BAM | PCA_RIDGE_VAR | 1.0612 | 1.0409 | 0.9964 | 0.9714 | 0.9564 | 0.9546 | 0.9761 |
| BAM | PERSISTENCE | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| EUROPE | DNS_KALMAN_OU | 1.1207 | 1.0746 | 1.0657 | 1.0911 | 1.1168 | 1.1704 | 1.1889 |
| EUROPE | NS_ELASTICNET | 1.1041 | 1.0551 | 1.0338 | 1.0361 | 1.0477 | 1.0619 | 1.0810 |
| EUROPE | NS_RIDGE | 1.0919 | 1.0718 | 1.0250 | 1.0360 | 1.0358 | 1.1574 | 1.2210 |
| EUROPE | PCA_RIDGE | 1.1337 | 1.0823 | 1.0322 | 1.0482 | 1.0528 | 1.1272 | 1.1179 |
| EUROPE | PCA_RIDGE_VAR | 1.1382 | 1.0557 | 1.0016 | 0.9678 | 0.9447 | 0.9039 | 0.9250 |
| EUROPE | PERSISTENCE | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| US_CORE | DNS_KALMAN_OU | 1.0776 | 1.0360 | 1.0622 | 1.0823 | 1.0885 | 1.1351 | 1.1352 |
| US_CORE | NS_ELASTICNET | 1.0670 | 1.0252 | 1.0211 | 1.0246 | 1.0286 | 1.0477 | 1.0729 |
| US_CORE | NS_RIDGE | 1.0486 | 0.9972 | 0.9980 | 0.9964 | 1.0046 | 1.0063 | 1.0771 |
| US_CORE | PCA_RIDGE | 1.0380 | 1.0012 | 0.9934 | 0.9904 | 0.9850 | 0.9946 | 1.0220 |
| US_CORE | PCA_RIDGE_VAR | 1.0254 | 0.9756 | 0.9681 | 0.9339 | 0.9097 | 0.8184 | 0.7745 |
| US_CORE | PERSISTENCE | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 6. Persistence degradation

| market | horizon | rmse | mae |
| --- | --- | --- | --- |
| BAM | 5 | 0.0014 | 0.0005 |
| BAM | 10 | 0.0014 | 0.0006 |
| BAM | 22 | 0.0019 | 0.0011 |
| BAM | 44 | 0.0028 | 0.0018 |
| BAM | 66 | 0.0039 | 0.0026 |
| BAM | 132 | 0.0068 | 0.0051 |
| BAM | 252 | 0.0103 | 0.0084 |
| EUROPE | 5 | 0.0012 | 0.0009 |
| EUROPE | 10 | 0.0018 | 0.0014 |
| EUROPE | 22 | 0.0026 | 0.0020 |
| EUROPE | 44 | 0.0037 | 0.0027 |
| EUROPE | 66 | 0.0045 | 0.0034 |
| EUROPE | 132 | 0.0070 | 0.0050 |
| EUROPE | 252 | 0.0111 | 0.0078 |
| US_CORE | 5 | 0.0014 | 0.0011 |
| US_CORE | 10 | 0.0024 | 0.0018 |
| US_CORE | 22 | 0.0031 | 0.0023 |
| US_CORE | 44 | 0.0049 | 0.0037 |
| US_CORE | 66 | 0.0064 | 0.0048 |
| US_CORE | 132 | 0.0094 | 0.0064 |
| US_CORE | 252 | 0.0135 | 0.0091 |

Persistence absolute error generally grows with h, but this alone is not model value. A factor model adds value only when its errors grow more slowly, which is captured by the ratios above.

## 7. Best model by market and horizon

| market | horizon | model | rmse | rmse_ratio_vs_persistence | rmse_improvement_pct | p_value | hac_bandwidth | classification | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | 5 | PERSISTENCE | 0.0014 | 1.0000 | 0.0000 | nan | nan | FULLY_FEASIBLE | NEUTRAL_OR_INCONCLUSIVE |
| BAM | 10 | PERSISTENCE | 0.0014 | 1.0000 | 0.0000 | nan | nan | FULLY_FEASIBLE | NEUTRAL_OR_INCONCLUSIVE |
| BAM | 22 | PCA_RIDGE_VAR | 0.0019 | 0.9964 | 0.3555 | 0.8219 | 1.0000 | FULLY_FEASIBLE | WEAK_POSITIVE |
| BAM | 44 | DNS_KALMAN_OU | 0.0027 | 0.9385 | 6.1457 | 0.2785 | 2.0000 | LIMITED_POWER | WEAK_POSITIVE |
| BAM | 66 | DNS_KALMAN_OU | 0.0033 | 0.8662 | 13.3807 | 0.0975 | 3.0000 | LIMITED_POWER | WEAK_POSITIVE |
| BAM | 132 | DNS_KALMAN_OU | 0.0050 | 0.7348 | 26.5182 | 0.0687 | 6.0000 | DESCRIPTIVE_ONLY | WEAK_POSITIVE |
| BAM | 252 | DNS_KALMAN_OU | 0.0062 | 0.5972 | 40.2810 | 0.0646 | 12.0000 | DESCRIPTIVE_ONLY | WEAK_POSITIVE |
| EUROPE | 5 | PERSISTENCE | 0.0012 | 1.0000 | 0.0000 | nan | nan | FULLY_FEASIBLE | NEUTRAL_OR_INCONCLUSIVE |
| EUROPE | 10 | PERSISTENCE | 0.0018 | 1.0000 | 0.0000 | nan | nan | FULLY_FEASIBLE | NEUTRAL_OR_INCONCLUSIVE |
| EUROPE | 22 | PERSISTENCE | 0.0026 | 1.0000 | 0.0000 | nan | nan | FULLY_FEASIBLE | NEUTRAL_OR_INCONCLUSIVE |
| EUROPE | 44 | PCA_RIDGE_VAR | 0.0035 | 0.9678 | 3.2171 | 0.4193 | 2.0000 | LIMITED_POWER | WEAK_POSITIVE |
| EUROPE | 66 | PCA_RIDGE_VAR | 0.0043 | 0.9447 | 5.5275 | 0.2581 | 3.0000 | LIMITED_POWER | WEAK_POSITIVE |
| EUROPE | 132 | PCA_RIDGE_VAR | 0.0063 | 0.9039 | 9.6065 | 0.2956 | 6.0000 | DESCRIPTIVE_ONLY | WEAK_POSITIVE |
| EUROPE | 252 | PCA_RIDGE_VAR | 0.0103 | 0.9250 | 7.5003 | 0.4287 | 11.0000 | DESCRIPTIVE_ONLY | WEAK_POSITIVE |
| US_CORE | 5 | PERSISTENCE | 0.0014 | 1.0000 | 0.0000 | nan | nan | FULLY_FEASIBLE | NEUTRAL_OR_INCONCLUSIVE |
| US_CORE | 10 | PCA_RIDGE_VAR | 0.0024 | 0.9756 | 2.4418 | 0.3099 | 0.0000 | FULLY_FEASIBLE | WEAK_POSITIVE |
| US_CORE | 22 | PCA_RIDGE_VAR | 0.0030 | 0.9681 | 3.1923 | 0.4457 | 1.0000 | FULLY_FEASIBLE | WEAK_POSITIVE |
| US_CORE | 44 | PCA_RIDGE_VAR | 0.0045 | 0.9339 | 6.6082 | 0.3336 | 2.0000 | LIMITED_POWER | WEAK_POSITIVE |
| US_CORE | 66 | PCA_RIDGE_VAR | 0.0058 | 0.9097 | 9.0251 | 0.3311 | 3.0000 | LIMITED_POWER | WEAK_POSITIVE |
| US_CORE | 132 | PCA_RIDGE_VAR | 0.0077 | 0.8184 | 18.1600 | 0.3321 | 6.0000 | DESCRIPTIVE_ONLY | WEAK_POSITIVE |
| US_CORE | 252 | PCA_RIDGE_VAR | 0.0104 | 0.7745 | 22.5520 | 0.2097 | 12.0000 | DESCRIPTIVE_ONLY | WEAK_POSITIVE |

## 8. Cross-market horizon comparison

| horizon | BAM_best_gain | Europe_best_gain | US_best_gain | consistent_winner |
| --- | --- | --- | --- | --- |
| 5 | 0.0000 | 0.0000 | 0.0000 | PERSISTENCE |
| 10 | 0.0000 | 0.0000 | 2.4418 | NO |
| 22 | 0.3555 | 0.0000 | 3.1923 | NO |
| 44 | 6.1457 | 3.2171 | 6.6082 | NO |
| 66 | 13.3807 | 5.5275 | 9.0251 | NO |
| 132 | 26.5182 | 9.6065 | 18.1600 | NO |
| 252 | 40.2810 | 7.5003 | 22.5520 | NO |

The absence of a universal winner is central: an isolated gain cannot establish sovereign-curve predictability as a cross-market regularity.

Persistence stops being the numerical winner at J+44 in all three markets, but the mechanism differs: OU/DNS mean reversion drives BAM, whereas recursive PCA dynamics drive Europe and the U.S. This is a descriptive forecastability threshold, not a structural law selected in advance.

## 9. DNS and mean-reversion horizon response

| market | 5 | 10 | 22 | 44 | 66 | 132 | 252 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | 1.0957 | 1.0666 | 1.0036 | 0.9385 | 0.8662 | 0.7348 | 0.5972 |
| EUROPE | 1.1207 | 1.0746 | 1.0657 | 1.0911 | 1.1168 | 1.1704 | 1.1889 |
| US_CORE | 1.0776 | 1.0360 | 1.0622 | 1.0823 | 1.0885 | 1.1351 | 1.1352 |

Estimated OU diagnostics:

| market | factor | kappa_per_day | half_life_days | half_life_observations_approx |
| --- | --- | --- | --- | --- |
| BAM | beta0 | 0.0065 | 107.4397 | 76.7426 |
| BAM | beta1 | 0.0038 | 180.1551 | 128.6822 |
| BAM | beta2 | 0.0048 | 144.4188 | 103.1563 |
| EUROPE | beta0 | 0.0004 | 1854.9336 | 1324.9526 |
| EUROPE | beta1 | 0.0001 | 5217.3782 | 3726.6987 |
| EUROPE | beta2 | 0.0043 | 160.2532 | 114.4666 |
| US_CORE | beta0 | 0.0009 | 797.8887 | 569.9205 |
| US_CORE | beta1 | 0.0006 | 1097.4695 | 783.9068 |
| US_CORE | beta2 | 0.0021 | 325.0435 | 232.1740 |

The half-life comparison is diagnostic only. A forecast horizon approaching an estimated half-life does not guarantee that noisy state estimates or a fixed OU law will beat persistence.

## 10. NS-Ridge horizon response

| market | 5 | 10 | 22 | 44 | 66 | 132 | 252 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | 1.0364 | 1.0152 | 1.2653 | 1.2968 | 1.1932 | 0.9290 | 1.0767 |
| EUROPE | 1.0919 | 1.0718 | 1.0250 | 1.0360 | 1.0358 | 1.1574 | 1.2210 |
| US_CORE | 1.0486 | 0.9972 | 0.9980 | 0.9964 | 1.0046 | 1.0063 | 1.0771 |

This distinguishes factor representation from factor dynamics: improvement by NS-Ridge without DNS would point toward supervised dynamics, whereas joint failure indicates that merely changing the dynamics is insufficient.

## 11. PCA-Ridge-VAR horizon response

| market | 5 | 10 | 22 | 44 | 66 | 132 | 252 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAM | 1.0612 | 1.0409 | 0.9964 | 0.9714 | 0.9564 | 0.9546 | 0.9761 |
| EUROPE | 1.1382 | 1.0557 | 1.0016 | 0.9678 | 0.9447 | 0.9039 | 0.9250 |
| US_CORE | 1.0254 | 0.9756 | 0.9681 | 0.9339 | 0.9097 | 0.8184 | 0.7745 |

This directly tests whether the modest U.S. J+10/J+22 Phase 4 pattern strengthens with horizon rather than extrapolating it.

## 12. Factor predictability

| market | model | factor | horizon | rmse | sign_accuracy |
| --- | --- | --- | --- | --- | --- |
| BAM | NS_ELASTICNET | beta0 | 5 | 0.0026 | 0.4630 |
| BAM | NS_ELASTICNET | beta0 | 10 | 0.0030 | 0.4815 |
| BAM | NS_ELASTICNET | beta0 | 22 | 0.0038 | 0.5094 |
| BAM | NS_ELASTICNET | beta0 | 44 | 0.0047 | 0.5385 |
| BAM | NS_ELASTICNET | beta0 | 66 | 0.0053 | 0.4510 |
| BAM | NS_ELASTICNET | beta0 | 132 | 0.0090 | 0.5319 |
| BAM | NS_ELASTICNET | beta0 | 252 | 0.0180 | 0.5476 |
| BAM | NS_ELASTICNET | beta1 | 5 | 0.0024 | 0.5185 |
| BAM | NS_ELASTICNET | beta1 | 10 | 0.0027 | 0.5370 |
| BAM | NS_ELASTICNET | beta1 | 22 | 0.0037 | 0.5472 |
| BAM | NS_ELASTICNET | beta1 | 44 | 0.0046 | 0.5192 |
| BAM | NS_ELASTICNET | beta1 | 66 | 0.0049 | 0.4706 |
| BAM | NS_ELASTICNET | beta1 | 132 | 0.0065 | 0.5532 |
| BAM | NS_ELASTICNET | beta1 | 252 | 0.0158 | 0.5714 |
| BAM | NS_ELASTICNET | beta2 | 5 | 0.0043 | 0.4815 |
| BAM | NS_ELASTICNET | beta2 | 10 | 0.0060 | 0.5185 |
| BAM | NS_ELASTICNET | beta2 | 22 | 0.0095 | 0.4528 |
| BAM | NS_ELASTICNET | beta2 | 44 | 0.0115 | 0.4615 |
| BAM | NS_ELASTICNET | beta2 | 66 | 0.0120 | 0.5882 |
| BAM | NS_ELASTICNET | beta2 | 132 | 0.0116 | 0.4894 |
| BAM | NS_ELASTICNET | beta2 | 252 | 0.0427 | 0.5714 |
| BAM | NS_RIDGE | beta0 | 5 | 0.0024 | 0.6852 |
| BAM | NS_RIDGE | beta0 | 10 | 0.0029 | 0.5741 |
| BAM | NS_RIDGE | beta0 | 22 | 0.0034 | 0.6226 |
| BAM | NS_RIDGE | beta0 | 44 | 0.0058 | 0.4808 |
| BAM | NS_RIDGE | beta0 | 66 | 0.0059 | 0.5490 |
| BAM | NS_RIDGE | beta0 | 132 | 0.0100 | 0.7021 |
| BAM | NS_RIDGE | beta0 | 252 | 0.0192 | 0.7857 |
| BAM | NS_RIDGE | beta1 | 5 | 0.0023 | 0.6296 |
| BAM | NS_RIDGE | beta1 | 10 | 0.0028 | 0.5000 |
| BAM | NS_RIDGE | beta1 | 22 | 0.0037 | 0.5472 |
| BAM | NS_RIDGE | beta1 | 44 | 0.0062 | 0.4231 |
| BAM | NS_RIDGE | beta1 | 66 | 0.0056 | 0.4706 |
| BAM | NS_RIDGE | beta1 | 132 | 0.0108 | 0.4894 |
| BAM | NS_RIDGE | beta1 | 252 | 0.0197 | 0.6667 |
| BAM | NS_RIDGE | beta2 | 5 | 0.0042 | 0.6481 |
| BAM | NS_RIDGE | beta2 | 10 | 0.0063 | 0.5556 |
| BAM | NS_RIDGE | beta2 | 22 | 0.0125 | 0.5283 |
| BAM | NS_RIDGE | beta2 | 44 | 0.0150 | 0.5000 |
| BAM | NS_RIDGE | beta2 | 66 | 0.0161 | 0.4706 |
| BAM | NS_RIDGE | beta2 | 132 | 0.0184 | 0.4681 |
| BAM | NS_RIDGE | beta2 | 252 | 0.0581 | 0.4048 |
| BAM | PCA_RIDGE | PC1 | 5 | 0.0033 | 0.5000 |
| BAM | PCA_RIDGE | PC1 | 10 | 0.0033 | 0.4630 |
| BAM | PCA_RIDGE | PC1 | 22 | 0.0050 | 0.5472 |
| BAM | PCA_RIDGE | PC1 | 44 | 0.0063 | 0.7500 |
| BAM | PCA_RIDGE | PC1 | 66 | 0.0093 | 0.7059 |
| BAM | PCA_RIDGE | PC1 | 132 | 0.0187 | 0.5957 |
| BAM | PCA_RIDGE | PC1 | 252 | 0.0347 | 0.7143 |
| BAM | PCA_RIDGE | PC2 | 5 | 0.0018 | 0.4815 |
| BAM | PCA_RIDGE | PC2 | 10 | 0.0020 | 0.4444 |
| BAM | PCA_RIDGE | PC2 | 22 | 0.0026 | 0.4906 |
| BAM | PCA_RIDGE | PC2 | 44 | 0.0045 | 0.4231 |
| BAM | PCA_RIDGE | PC2 | 66 | 0.0046 | 0.5490 |
| BAM | PCA_RIDGE | PC2 | 132 | 0.0052 | 0.6383 |
| BAM | PCA_RIDGE | PC2 | 252 | 0.0040 | 0.7619 |
| BAM | PCA_RIDGE | PC3 | 5 | 0.0006 | 0.6481 |
| BAM | PCA_RIDGE | PC3 | 10 | 0.0009 | 0.5185 |
| BAM | PCA_RIDGE | PC3 | 22 | 0.0013 | 0.5849 |
| BAM | PCA_RIDGE | PC3 | 44 | 0.0018 | 0.5962 |
| BAM | PCA_RIDGE | PC3 | 66 | 0.0023 | 0.5490 |
| BAM | PCA_RIDGE | PC3 | 132 | 0.0033 | 0.5745 |
| BAM | PCA_RIDGE | PC3 | 252 | 0.0108 | 0.6667 |
| BAM | PCA_RIDGE_VAR | PC1 | 5 | 0.0035 | 0.5000 |
| BAM | PCA_RIDGE_VAR | PC1 | 10 | 0.0036 | 0.5185 |
| BAM | PCA_RIDGE_VAR | PC1 | 22 | 0.0046 | 0.6792 |
| BAM | PCA_RIDGE_VAR | PC1 | 44 | 0.0070 | 0.6731 |
| BAM | PCA_RIDGE_VAR | PC1 | 66 | 0.0098 | 0.6275 |
| BAM | PCA_RIDGE_VAR | PC1 | 132 | 0.0185 | 0.7021 |
| BAM | PCA_RIDGE_VAR | PC1 | 252 | 0.0296 | 0.4762 |
| BAM | PCA_RIDGE_VAR | PC2 | 5 | 0.0017 | 0.4630 |
| BAM | PCA_RIDGE_VAR | PC2 | 10 | 0.0018 | 0.4815 |
| BAM | PCA_RIDGE_VAR | PC2 | 22 | 0.0024 | 0.4151 |
| BAM | PCA_RIDGE_VAR | PC2 | 44 | 0.0036 | 0.4615 |
| BAM | PCA_RIDGE_VAR | PC2 | 66 | 0.0043 | 0.5098 |
| BAM | PCA_RIDGE_VAR | PC2 | 132 | 0.0050 | 0.5957 |
| BAM | PCA_RIDGE_VAR | PC2 | 252 | 0.0049 | 0.8333 |
| BAM | PCA_RIDGE_VAR | PC3 | 5 | 0.0007 | 0.5556 |
| BAM | PCA_RIDGE_VAR | PC3 | 10 | 0.0009 | 0.4815 |
| BAM | PCA_RIDGE_VAR | PC3 | 22 | 0.0012 | 0.6415 |
| BAM | PCA_RIDGE_VAR | PC3 | 44 | 0.0016 | 0.6346 |
| BAM | PCA_RIDGE_VAR | PC3 | 66 | 0.0020 | 0.5490 |
| BAM | PCA_RIDGE_VAR | PC3 | 132 | 0.0025 | 0.6383 |
| BAM | PCA_RIDGE_VAR | PC3 | 252 | 0.0036 | 0.7381 |
| EUROPE | NS_ELASTICNET | beta0 | 5 | 0.0013 | 0.4630 |
| EUROPE | NS_ELASTICNET | beta0 | 10 | 0.0018 | 0.3889 |
| EUROPE | NS_ELASTICNET | beta0 | 22 | 0.0025 | 0.3962 |
| EUROPE | NS_ELASTICNET | beta0 | 44 | 0.0033 | 0.3269 |
| EUROPE | NS_ELASTICNET | beta0 | 66 | 0.0040 | 0.3137 |
| EUROPE | NS_ELASTICNET | beta0 | 132 | 0.0064 | 0.1875 |
| EUROPE | NS_ELASTICNET | beta0 | 252 | 0.0096 | 0.2381 |
| EUROPE | NS_ELASTICNET | beta1 | 5 | 0.0016 | 0.4630 |
| EUROPE | NS_ELASTICNET | beta1 | 10 | 0.0024 | 0.4444 |
| EUROPE | NS_ELASTICNET | beta1 | 22 | 0.0033 | 0.3774 |
| EUROPE | NS_ELASTICNET | beta1 | 44 | 0.0047 | 0.4038 |
| EUROPE | NS_ELASTICNET | beta1 | 66 | 0.0066 | 0.4706 |
| EUROPE | NS_ELASTICNET | beta1 | 132 | 0.0109 | 0.5625 |
| EUROPE | NS_ELASTICNET | beta1 | 252 | 0.0184 | 0.5952 |
| EUROPE | NS_ELASTICNET | beta2 | 5 | 0.0037 | 0.4444 |
| EUROPE | NS_ELASTICNET | beta2 | 10 | 0.0049 | 0.5185 |
| EUROPE | NS_ELASTICNET | beta2 | 22 | 0.0081 | 0.5094 |
| EUROPE | NS_ELASTICNET | beta2 | 44 | 0.0106 | 0.3846 |
| EUROPE | NS_ELASTICNET | beta2 | 66 | 0.0120 | 0.4314 |
| EUROPE | NS_ELASTICNET | beta2 | 132 | 0.0134 | 0.4583 |
| EUROPE | NS_ELASTICNET | beta2 | 252 | 0.0201 | 0.5238 |
| EUROPE | NS_RIDGE | beta0 | 5 | 0.0013 | 0.5370 |
| EUROPE | NS_RIDGE | beta0 | 10 | 0.0018 | 0.5741 |
| EUROPE | NS_RIDGE | beta0 | 22 | 0.0023 | 0.6226 |
| EUROPE | NS_RIDGE | beta0 | 44 | 0.0031 | 0.6538 |
| EUROPE | NS_RIDGE | beta0 | 66 | 0.0036 | 0.6471 |
| EUROPE | NS_RIDGE | beta0 | 132 | 0.0053 | 0.8333 |
| EUROPE | NS_RIDGE | beta0 | 252 | 0.0097 | 0.8333 |
| EUROPE | NS_RIDGE | beta1 | 5 | 0.0016 | 0.5185 |
| EUROPE | NS_RIDGE | beta1 | 10 | 0.0025 | 0.5741 |
| EUROPE | NS_RIDGE | beta1 | 22 | 0.0033 | 0.6604 |
| EUROPE | NS_RIDGE | beta1 | 44 | 0.0052 | 0.6923 |
| EUROPE | NS_RIDGE | beta1 | 66 | 0.0082 | 0.5686 |
| EUROPE | NS_RIDGE | beta1 | 132 | 0.0150 | 0.5000 |
| EUROPE | NS_RIDGE | beta1 | 252 | 0.0223 | 0.6190 |
| EUROPE | NS_RIDGE | beta2 | 5 | 0.0037 | 0.4815 |
| EUROPE | NS_RIDGE | beta2 | 10 | 0.0048 | 0.4630 |
| EUROPE | NS_RIDGE | beta2 | 22 | 0.0083 | 0.5660 |
| EUROPE | NS_RIDGE | beta2 | 44 | 0.0106 | 0.5769 |
| EUROPE | NS_RIDGE | beta2 | 66 | 0.0110 | 0.7451 |
| EUROPE | NS_RIDGE | beta2 | 132 | 0.0116 | 0.7292 |
| EUROPE | NS_RIDGE | beta2 | 252 | 0.0120 | 0.8333 |
| EUROPE | PCA_RIDGE | PC1 | 5 | 0.0052 | 0.4815 |
| EUROPE | PCA_RIDGE | PC1 | 10 | 0.0081 | 0.5000 |
| EUROPE | PCA_RIDGE | PC1 | 22 | 0.0117 | 0.4717 |
| EUROPE | PCA_RIDGE | PC1 | 44 | 0.0167 | 0.5192 |
| EUROPE | PCA_RIDGE | PC1 | 66 | 0.0199 | 0.4510 |
| EUROPE | PCA_RIDGE | PC1 | 132 | 0.0322 | 0.4375 |
| EUROPE | PCA_RIDGE | PC1 | 252 | 0.0529 | 0.3571 |
| EUROPE | PCA_RIDGE | PC2 | 5 | 0.0017 | 0.4815 |
| EUROPE | PCA_RIDGE | PC2 | 10 | 0.0024 | 0.6296 |
| EUROPE | PCA_RIDGE | PC2 | 22 | 0.0035 | 0.7170 |
| EUROPE | PCA_RIDGE | PC2 | 44 | 0.0058 | 0.7308 |
| EUROPE | PCA_RIDGE | PC2 | 66 | 0.0100 | 0.5882 |
| EUROPE | PCA_RIDGE | PC2 | 132 | 0.0185 | 0.5208 |
| EUROPE | PCA_RIDGE | PC2 | 252 | 0.0265 | 0.5714 |
| EUROPE | PCA_RIDGE | PC3 | 5 | 0.0012 | 0.4259 |
| EUROPE | PCA_RIDGE | PC3 | 10 | 0.0016 | 0.5370 |
| EUROPE | PCA_RIDGE | PC3 | 22 | 0.0027 | 0.5849 |
| EUROPE | PCA_RIDGE | PC3 | 44 | 0.0034 | 0.5192 |
| EUROPE | PCA_RIDGE | PC3 | 66 | 0.0038 | 0.6471 |
| EUROPE | PCA_RIDGE | PC3 | 132 | 0.0052 | 0.6458 |
| EUROPE | PCA_RIDGE | PC3 | 252 | 0.0055 | 0.7857 |
| EUROPE | PCA_RIDGE_VAR | PC1 | 5 | 0.0053 | 0.5000 |
| EUROPE | PCA_RIDGE_VAR | PC1 | 10 | 0.0078 | 0.5741 |
| EUROPE | PCA_RIDGE_VAR | PC1 | 22 | 0.0113 | 0.4528 |
| EUROPE | PCA_RIDGE_VAR | PC1 | 44 | 0.0153 | 0.5385 |
| EUROPE | PCA_RIDGE_VAR | PC1 | 66 | 0.0182 | 0.4902 |
| EUROPE | PCA_RIDGE_VAR | PC1 | 132 | 0.0261 | 0.5000 |
| EUROPE | PCA_RIDGE_VAR | PC1 | 252 | 0.0424 | 0.5714 |
| EUROPE | PCA_RIDGE_VAR | PC2 | 5 | 0.0017 | 0.5556 |
| EUROPE | PCA_RIDGE_VAR | PC2 | 10 | 0.0024 | 0.5185 |
| EUROPE | PCA_RIDGE_VAR | PC2 | 22 | 0.0034 | 0.6226 |
| EUROPE | PCA_RIDGE_VAR | PC2 | 44 | 0.0054 | 0.5000 |
| EUROPE | PCA_RIDGE_VAR | PC2 | 66 | 0.0079 | 0.4510 |
| EUROPE | PCA_RIDGE_VAR | PC2 | 132 | 0.0140 | 0.3958 |
| EUROPE | PCA_RIDGE_VAR | PC2 | 252 | 0.0240 | 0.3571 |
| EUROPE | PCA_RIDGE_VAR | PC3 | 5 | 0.0012 | 0.5185 |
| EUROPE | PCA_RIDGE_VAR | PC3 | 10 | 0.0016 | 0.4815 |
| EUROPE | PCA_RIDGE_VAR | PC3 | 22 | 0.0026 | 0.5849 |
| EUROPE | PCA_RIDGE_VAR | PC3 | 44 | 0.0034 | 0.5962 |
| EUROPE | PCA_RIDGE_VAR | PC3 | 66 | 0.0040 | 0.5882 |
| EUROPE | PCA_RIDGE_VAR | PC3 | 132 | 0.0046 | 0.7708 |
| EUROPE | PCA_RIDGE_VAR | PC3 | 252 | 0.0062 | 0.8095 |
| US_CORE | NS_ELASTICNET | beta0 | 5 | 0.0016 | 0.3393 |
| US_CORE | NS_ELASTICNET | beta0 | 10 | 0.0023 | 0.3818 |
| US_CORE | NS_ELASTICNET | beta0 | 22 | 0.0031 | 0.4182 |
| US_CORE | NS_ELASTICNET | beta0 | 44 | 0.0042 | 0.4259 |
| US_CORE | NS_ELASTICNET | beta0 | 66 | 0.0048 | 0.4717 |
| US_CORE | NS_ELASTICNET | beta0 | 132 | 0.0059 | 0.3000 |
| US_CORE | NS_ELASTICNET | beta0 | 252 | 0.0085 | 0.1818 |
| US_CORE | NS_ELASTICNET | beta1 | 5 | 0.0020 | 0.4464 |
| US_CORE | NS_ELASTICNET | beta1 | 10 | 0.0033 | 0.3636 |
| US_CORE | NS_ELASTICNET | beta1 | 22 | 0.0046 | 0.4727 |
| US_CORE | NS_ELASTICNET | beta1 | 44 | 0.0068 | 0.4259 |
| US_CORE | NS_ELASTICNET | beta1 | 66 | 0.0086 | 0.3585 |
| US_CORE | NS_ELASTICNET | beta1 | 132 | 0.0135 | 0.2400 |
| US_CORE | NS_ELASTICNET | beta1 | 252 | 0.0194 | 0.1136 |
| US_CORE | NS_ELASTICNET | beta2 | 5 | 0.0050 | 0.5357 |
| US_CORE | NS_ELASTICNET | beta2 | 10 | 0.0077 | 0.4545 |
| US_CORE | NS_ELASTICNET | beta2 | 22 | 0.0093 | 0.3455 |
| US_CORE | NS_ELASTICNET | beta2 | 44 | 0.0145 | 0.3704 |
| US_CORE | NS_ELASTICNET | beta2 | 66 | 0.0178 | 0.3396 |
| US_CORE | NS_ELASTICNET | beta2 | 132 | 0.0171 | 0.2800 |
| US_CORE | NS_ELASTICNET | beta2 | 252 | 0.0246 | 0.4545 |
| US_CORE | NS_RIDGE | beta0 | 5 | 0.0016 | 0.5357 |
| US_CORE | NS_RIDGE | beta0 | 10 | 0.0023 | 0.5636 |
| US_CORE | NS_RIDGE | beta0 | 22 | 0.0032 | 0.5091 |
| US_CORE | NS_RIDGE | beta0 | 44 | 0.0042 | 0.5370 |
| US_CORE | NS_RIDGE | beta0 | 66 | 0.0047 | 0.5094 |
| US_CORE | NS_RIDGE | beta0 | 132 | 0.0051 | 0.6800 |
| US_CORE | NS_RIDGE | beta0 | 252 | 0.0057 | 0.8182 |
| US_CORE | NS_RIDGE | beta1 | 5 | 0.0021 | 0.5357 |
| US_CORE | NS_RIDGE | beta1 | 10 | 0.0032 | 0.5636 |
| US_CORE | NS_RIDGE | beta1 | 22 | 0.0046 | 0.5818 |
| US_CORE | NS_RIDGE | beta1 | 44 | 0.0064 | 0.6296 |
| US_CORE | NS_RIDGE | beta1 | 66 | 0.0080 | 0.6792 |
| US_CORE | NS_RIDGE | beta1 | 132 | 0.0124 | 0.7800 |
| US_CORE | NS_RIDGE | beta1 | 252 | 0.0202 | 0.8182 |
| US_CORE | NS_RIDGE | beta2 | 5 | 0.0051 | 0.4464 |
| US_CORE | NS_RIDGE | beta2 | 10 | 0.0078 | 0.4364 |
| US_CORE | NS_RIDGE | beta2 | 22 | 0.0092 | 0.5455 |
| US_CORE | NS_RIDGE | beta2 | 44 | 0.0152 | 0.5000 |
| US_CORE | NS_RIDGE | beta2 | 66 | 0.0186 | 0.6038 |
| US_CORE | NS_RIDGE | beta2 | 132 | 0.0149 | 0.8000 |
| US_CORE | NS_RIDGE | beta2 | 252 | 0.0205 | 0.7727 |
| US_CORE | PCA_RIDGE | PC1 | 5 | 0.0032 | 0.6071 |
| US_CORE | PCA_RIDGE | PC1 | 10 | 0.0060 | 0.4727 |
| US_CORE | PCA_RIDGE | PC1 | 22 | 0.0075 | 0.5818 |
| US_CORE | PCA_RIDGE | PC1 | 44 | 0.0123 | 0.5556 |
| US_CORE | PCA_RIDGE | PC1 | 66 | 0.0162 | 0.6226 |
| US_CORE | PCA_RIDGE | PC1 | 132 | 0.0247 | 0.6800 |
| US_CORE | PCA_RIDGE | PC1 | 252 | 0.0361 | 0.6591 |
| US_CORE | PCA_RIDGE | PC2 | 5 | 0.0017 | 0.5000 |
| US_CORE | PCA_RIDGE | PC2 | 10 | 0.0025 | 0.5636 |
| US_CORE | PCA_RIDGE | PC2 | 22 | 0.0035 | 0.5818 |
| US_CORE | PCA_RIDGE | PC2 | 44 | 0.0049 | 0.6296 |
| US_CORE | PCA_RIDGE | PC2 | 66 | 0.0062 | 0.7170 |
| US_CORE | PCA_RIDGE | PC2 | 132 | 0.0084 | 0.8200 |
| US_CORE | PCA_RIDGE | PC2 | 252 | 0.0137 | 0.9091 |
| US_CORE | PCA_RIDGE | PC3 | 5 | 0.0010 | 0.4821 |
| US_CORE | PCA_RIDGE | PC3 | 10 | 0.0015 | 0.4909 |
| US_CORE | PCA_RIDGE | PC3 | 22 | 0.0020 | 0.5091 |
| US_CORE | PCA_RIDGE | PC3 | 44 | 0.0028 | 0.5556 |
| US_CORE | PCA_RIDGE | PC3 | 66 | 0.0034 | 0.6038 |
| US_CORE | PCA_RIDGE | PC3 | 132 | 0.0036 | 0.6800 |
| US_CORE | PCA_RIDGE | PC3 | 252 | 0.0050 | 0.6818 |
| US_CORE | PCA_RIDGE_VAR | PC1 | 5 | 0.0032 | 0.5893 |
| US_CORE | PCA_RIDGE_VAR | PC1 | 10 | 0.0058 | 0.4909 |
| US_CORE | PCA_RIDGE_VAR | PC1 | 22 | 0.0072 | 0.5818 |
| US_CORE | PCA_RIDGE_VAR | PC1 | 44 | 0.0114 | 0.5000 |
| US_CORE | PCA_RIDGE_VAR | PC1 | 66 | 0.0148 | 0.6038 |
| US_CORE | PCA_RIDGE_VAR | PC1 | 132 | 0.0202 | 0.7000 |
| US_CORE | PCA_RIDGE_VAR | PC1 | 252 | 0.0278 | 0.7500 |
| US_CORE | PCA_RIDGE_VAR | PC2 | 5 | 0.0017 | 0.4821 |
| US_CORE | PCA_RIDGE_VAR | PC2 | 10 | 0.0025 | 0.5636 |
| US_CORE | PCA_RIDGE_VAR | PC2 | 22 | 0.0036 | 0.6000 |
| US_CORE | PCA_RIDGE_VAR | PC2 | 44 | 0.0050 | 0.6296 |
| US_CORE | PCA_RIDGE_VAR | PC2 | 66 | 0.0060 | 0.6981 |
| US_CORE | PCA_RIDGE_VAR | PC2 | 132 | 0.0069 | 0.8000 |
| US_CORE | PCA_RIDGE_VAR | PC2 | 252 | 0.0086 | 0.9091 |
| US_CORE | PCA_RIDGE_VAR | PC3 | 5 | 0.0010 | 0.4643 |
| US_CORE | PCA_RIDGE_VAR | PC3 | 10 | 0.0016 | 0.4909 |
| US_CORE | PCA_RIDGE_VAR | PC3 | 22 | 0.0020 | 0.5455 |
| US_CORE | PCA_RIDGE_VAR | PC3 | 44 | 0.0029 | 0.5741 |
| US_CORE | PCA_RIDGE_VAR | PC3 | 66 | 0.0036 | 0.6038 |
| US_CORE | PCA_RIDGE_VAR | PC3 | 132 | 0.0036 | 0.6600 |
| US_CORE | PCA_RIDGE_VAR | PC3 | 252 | 0.0046 | 0.7045 |

Sign accuracy is secondary. Factor RMSE remains the primary factor-level diagnostic, and factor gains are not substituted for curve-yield accuracy.

## 13. Maturity and curve-segment evidence

| market | model | horizon | segment | rmse_ratio_mean | improving_maturities | maturities |
| --- | --- | --- | --- | --- | --- | --- |
| BAM | DNS_KALMAN_OU | 5 | long | 1.0666 | 0 | 3 |
| BAM | DNS_KALMAN_OU | 5 | medium | 0.9808 | 1 | 2 |
| BAM | DNS_KALMAN_OU | 5 | short | 1.4112 | 0 | 4 |
| BAM | DNS_KALMAN_OU | 10 | long | 1.0479 | 0 | 3 |
| BAM | DNS_KALMAN_OU | 10 | medium | 0.9762 | 1 | 2 |
| BAM | DNS_KALMAN_OU | 10 | short | 1.2771 | 0 | 4 |
| BAM | DNS_KALMAN_OU | 22 | long | 1.0102 | 1 | 3 |
| BAM | DNS_KALMAN_OU | 22 | medium | 0.9254 | 2 | 2 |
| BAM | DNS_KALMAN_OU | 22 | short | 1.0671 | 0 | 4 |
| BAM | DNS_KALMAN_OU | 44 | long | 0.9528 | 3 | 3 |
| BAM | DNS_KALMAN_OU | 44 | medium | 0.8817 | 2 | 2 |
| BAM | DNS_KALMAN_OU | 44 | short | 0.9594 | 4 | 4 |
| BAM | DNS_KALMAN_OU | 66 | long | 0.8862 | 3 | 3 |
| BAM | DNS_KALMAN_OU | 66 | medium | 0.8448 | 2 | 2 |
| BAM | DNS_KALMAN_OU | 66 | short | 0.8445 | 4 | 4 |
| BAM | DNS_KALMAN_OU | 132 | long | 0.7715 | 3 | 3 |
| BAM | DNS_KALMAN_OU | 132 | medium | 0.7313 | 2 | 2 |
| BAM | DNS_KALMAN_OU | 132 | short | 0.6756 | 4 | 4 |
| BAM | DNS_KALMAN_OU | 252 | long | 0.6200 | 3 | 3 |
| BAM | DNS_KALMAN_OU | 252 | medium | 0.6048 | 2 | 2 |
| BAM | DNS_KALMAN_OU | 252 | short | 0.5486 | 4 | 4 |
| BAM | NS_ELASTICNET | 5 | long | 1.0509 | 0 | 3 |
| BAM | NS_ELASTICNET | 5 | medium | 0.9844 | 1 | 2 |
| BAM | NS_ELASTICNET | 5 | short | 1.3540 | 0 | 4 |
| BAM | NS_ELASTICNET | 10 | long | 1.0406 | 1 | 3 |
| BAM | NS_ELASTICNET | 10 | medium | 0.9927 | 1 | 2 |
| BAM | NS_ELASTICNET | 10 | short | 1.2304 | 0 | 4 |
| BAM | NS_ELASTICNET | 22 | long | 1.0541 | 0 | 3 |
| BAM | NS_ELASTICNET | 22 | medium | 0.9782 | 1 | 2 |
| BAM | NS_ELASTICNET | 22 | short | 1.0966 | 0 | 4 |
| BAM | NS_ELASTICNET | 44 | long | 1.0286 | 0 | 3 |
| BAM | NS_ELASTICNET | 44 | medium | 0.9733 | 1 | 2 |
| BAM | NS_ELASTICNET | 44 | short | 1.0476 | 0 | 4 |
| BAM | NS_ELASTICNET | 66 | long | 1.0544 | 0 | 3 |
| BAM | NS_ELASTICNET | 66 | medium | 1.0214 | 0 | 2 |
| BAM | NS_ELASTICNET | 66 | short | 1.0482 | 0 | 4 |
| BAM | NS_ELASTICNET | 132 | long | 1.0464 | 0 | 3 |
| BAM | NS_ELASTICNET | 132 | medium | 1.0303 | 0 | 2 |
| BAM | NS_ELASTICNET | 132 | short | 1.0329 | 0 | 4 |
| BAM | NS_ELASTICNET | 252 | long | 1.0517 | 1 | 3 |
| BAM | NS_ELASTICNET | 252 | medium | 1.1745 | 0 | 2 |
| BAM | NS_ELASTICNET | 252 | short | 1.0592 | 0 | 4 |
| BAM | NS_RIDGE | 5 | long | 1.0318 | 1 | 3 |
| BAM | NS_RIDGE | 5 | medium | 0.9539 | 1 | 2 |
| BAM | NS_RIDGE | 5 | short | 1.2641 | 0 | 4 |
| BAM | NS_RIDGE | 10 | long | 1.0249 | 1 | 3 |
| BAM | NS_RIDGE | 10 | medium | 0.9806 | 1 | 2 |
| BAM | NS_RIDGE | 10 | short | 1.1096 | 2 | 4 |
| BAM | NS_RIDGE | 22 | long | 1.2492 | 0 | 3 |
| BAM | NS_RIDGE | 22 | medium | 1.3278 | 0 | 2 |
| BAM | NS_RIDGE | 22 | short | 1.2316 | 0 | 4 |
| BAM | NS_RIDGE | 44 | long | 1.4239 | 0 | 3 |
| BAM | NS_RIDGE | 44 | medium | 1.2670 | 0 | 2 |
| BAM | NS_RIDGE | 44 | short | 1.0756 | 0 | 4 |
| BAM | NS_RIDGE | 66 | long | 1.2169 | 0 | 3 |
| BAM | NS_RIDGE | 66 | medium | 1.2174 | 0 | 2 |
| BAM | NS_RIDGE | 66 | short | 1.1360 | 0 | 4 |
| BAM | NS_RIDGE | 132 | long | 0.8638 | 3 | 3 |
| BAM | NS_RIDGE | 132 | medium | 0.9428 | 2 | 2 |
| BAM | NS_RIDGE | 132 | short | 1.0057 | 1 | 4 |
| BAM | NS_RIDGE | 252 | long | 0.9642 | 2 | 3 |
| BAM | NS_RIDGE | 252 | medium | 1.2598 | 0 | 2 |
| BAM | NS_RIDGE | 252 | short | 1.0761 | 0 | 4 |
| BAM | PCA_RIDGE | 5 | long | 1.0588 | 1 | 3 |
| BAM | PCA_RIDGE | 5 | medium | 0.9270 | 2 | 2 |
| BAM | PCA_RIDGE | 5 | short | 1.1281 | 1 | 4 |
| BAM | PCA_RIDGE | 10 | long | 1.0700 | 1 | 3 |
| BAM | PCA_RIDGE | 10 | medium | 0.9480 | 1 | 2 |
| BAM | PCA_RIDGE | 10 | short | 1.0234 | 3 | 4 |
| BAM | PCA_RIDGE | 22 | long | 1.1437 | 0 | 3 |
| BAM | PCA_RIDGE | 22 | medium | 1.0054 | 1 | 2 |
| BAM | PCA_RIDGE | 22 | short | 1.0342 | 2 | 4 |
| BAM | PCA_RIDGE | 44 | long | 1.0322 | 1 | 3 |
| BAM | PCA_RIDGE | 44 | medium | 0.8448 | 2 | 2 |
| BAM | PCA_RIDGE | 44 | short | 0.9586 | 3 | 4 |
| BAM | PCA_RIDGE | 66 | long | 0.9264 | 3 | 3 |
| BAM | PCA_RIDGE | 66 | medium | 0.8846 | 2 | 2 |
| BAM | PCA_RIDGE | 66 | short | 0.9886 | 2 | 4 |
| BAM | PCA_RIDGE | 132 | long | 0.9025 | 3 | 3 |
| BAM | PCA_RIDGE | 132 | medium | 1.0055 | 1 | 2 |
| BAM | PCA_RIDGE | 132 | short | 1.0452 | 0 | 4 |
| BAM | PCA_RIDGE | 252 | long | 1.0590 | 1 | 3 |
| BAM | PCA_RIDGE | 252 | medium | 1.4070 | 0 | 2 |
| BAM | PCA_RIDGE | 252 | short | 1.1193 | 0 | 4 |
| BAM | PCA_RIDGE_VAR | 5 | long | 1.0767 | 0 | 3 |
| BAM | PCA_RIDGE_VAR | 5 | medium | 0.9674 | 1 | 2 |
| BAM | PCA_RIDGE_VAR | 5 | short | 1.2191 | 0 | 4 |
| BAM | PCA_RIDGE_VAR | 10 | long | 1.0704 | 0 | 3 |
| BAM | PCA_RIDGE_VAR | 10 | medium | 0.9599 | 1 | 2 |
| BAM | PCA_RIDGE_VAR | 10 | short | 1.1243 | 0 | 4 |
| BAM | PCA_RIDGE_VAR | 22 | long | 1.0313 | 0 | 3 |
| BAM | PCA_RIDGE_VAR | 22 | medium | 0.9159 | 2 | 2 |
| BAM | PCA_RIDGE_VAR | 22 | short | 1.0155 | 2 | 4 |
| BAM | PCA_RIDGE_VAR | 44 | long | 0.9867 | 2 | 3 |
| BAM | PCA_RIDGE_VAR | 44 | medium | 0.9107 | 2 | 2 |
| BAM | PCA_RIDGE_VAR | 44 | short | 0.9929 | 3 | 4 |
| BAM | PCA_RIDGE_VAR | 66 | long | 0.9583 | 2 | 3 |
| BAM | PCA_RIDGE_VAR | 66 | medium | 0.9214 | 2 | 2 |
| BAM | PCA_RIDGE_VAR | 66 | short | 0.9785 | 4 | 4 |
| BAM | PCA_RIDGE_VAR | 132 | long | 0.9219 | 3 | 3 |
| BAM | PCA_RIDGE_VAR | 132 | medium | 0.9617 | 2 | 2 |
| BAM | PCA_RIDGE_VAR | 132 | short | 0.9943 | 4 | 4 |
| BAM | PCA_RIDGE_VAR | 252 | long | 0.9227 | 3 | 3 |
| BAM | PCA_RIDGE_VAR | 252 | medium | 1.0213 | 0 | 2 |
| BAM | PCA_RIDGE_VAR | 252 | short | 1.0184 | 0 | 4 |
| BAM | PERSISTENCE | 5 | long | 1.0000 | 0 | 3 |
| BAM | PERSISTENCE | 5 | medium | 1.0000 | 0 | 2 |
| BAM | PERSISTENCE | 5 | short | 1.0000 | 0 | 4 |
| BAM | PERSISTENCE | 10 | long | 1.0000 | 0 | 3 |
| BAM | PERSISTENCE | 10 | medium | 1.0000 | 0 | 2 |
| BAM | PERSISTENCE | 10 | short | 1.0000 | 0 | 4 |
| BAM | PERSISTENCE | 22 | long | 1.0000 | 0 | 3 |
| BAM | PERSISTENCE | 22 | medium | 1.0000 | 0 | 2 |
| BAM | PERSISTENCE | 22 | short | 1.0000 | 0 | 4 |
| BAM | PERSISTENCE | 44 | long | 1.0000 | 0 | 3 |
| BAM | PERSISTENCE | 44 | medium | 1.0000 | 0 | 2 |
| BAM | PERSISTENCE | 44 | short | 1.0000 | 0 | 4 |
| BAM | PERSISTENCE | 66 | long | 1.0000 | 0 | 3 |
| BAM | PERSISTENCE | 66 | medium | 1.0000 | 0 | 2 |
| BAM | PERSISTENCE | 66 | short | 1.0000 | 0 | 4 |
| BAM | PERSISTENCE | 132 | long | 1.0000 | 0 | 3 |
| BAM | PERSISTENCE | 132 | medium | 1.0000 | 0 | 2 |
| BAM | PERSISTENCE | 132 | short | 1.0000 | 0 | 4 |
| BAM | PERSISTENCE | 252 | long | 1.0000 | 0 | 3 |
| BAM | PERSISTENCE | 252 | medium | 1.0000 | 0 | 2 |
| BAM | PERSISTENCE | 252 | short | 1.0000 | 0 | 4 |
| EUROPE | DNS_KALMAN_OU | 5 | long | 1.0757 | 0 | 11 |
| EUROPE | DNS_KALMAN_OU | 5 | medium | 1.0758 | 0 | 8 |
| EUROPE | DNS_KALMAN_OU | 5 | short | 1.4621 | 0 | 4 |
| EUROPE | DNS_KALMAN_OU | 10 | long | 1.0483 | 0 | 11 |
| EUROPE | DNS_KALMAN_OU | 10 | medium | 1.0560 | 0 | 8 |
| EUROPE | DNS_KALMAN_OU | 10 | short | 1.1791 | 1 | 4 |
| EUROPE | DNS_KALMAN_OU | 22 | long | 1.0526 | 0 | 11 |
| EUROPE | DNS_KALMAN_OU | 22 | medium | 1.0452 | 0 | 8 |
| EUROPE | DNS_KALMAN_OU | 22 | short | 1.1195 | 1 | 4 |
| EUROPE | DNS_KALMAN_OU | 44 | long | 1.0867 | 0 | 11 |
| EUROPE | DNS_KALMAN_OU | 44 | medium | 1.0667 | 0 | 8 |
| EUROPE | DNS_KALMAN_OU | 44 | short | 1.1270 | 1 | 4 |
| EUROPE | DNS_KALMAN_OU | 66 | long | 1.1150 | 0 | 11 |
| EUROPE | DNS_KALMAN_OU | 66 | medium | 1.1045 | 0 | 8 |
| EUROPE | DNS_KALMAN_OU | 66 | short | 1.1327 | 1 | 4 |
| EUROPE | DNS_KALMAN_OU | 132 | long | 1.1831 | 0 | 11 |
| EUROPE | DNS_KALMAN_OU | 132 | medium | 1.2065 | 0 | 8 |
| EUROPE | DNS_KALMAN_OU | 132 | short | 1.1343 | 0 | 4 |
| EUROPE | DNS_KALMAN_OU | 252 | long | 1.2304 | 0 | 11 |
| EUROPE | DNS_KALMAN_OU | 252 | medium | 1.2492 | 0 | 8 |
| EUROPE | DNS_KALMAN_OU | 252 | short | 1.1266 | 0 | 4 |
| EUROPE | NS_ELASTICNET | 5 | long | 1.0861 | 0 | 11 |
| EUROPE | NS_ELASTICNET | 5 | medium | 1.0423 | 0 | 8 |
| EUROPE | NS_ELASTICNET | 5 | short | 1.4126 | 0 | 4 |
| EUROPE | NS_ELASTICNET | 10 | long | 1.0459 | 0 | 11 |
| EUROPE | NS_ELASTICNET | 10 | medium | 1.0322 | 0 | 8 |
| EUROPE | NS_ELASTICNET | 10 | short | 1.1208 | 1 | 4 |
| EUROPE | NS_ELASTICNET | 22 | long | 1.0342 | 0 | 11 |
| EUROPE | NS_ELASTICNET | 22 | medium | 1.0165 | 0 | 8 |
| EUROPE | NS_ELASTICNET | 22 | short | 1.0445 | 1 | 4 |
| EUROPE | NS_ELASTICNET | 44 | long | 1.0488 | 0 | 11 |
| EUROPE | NS_ELASTICNET | 44 | medium | 1.0208 | 0 | 8 |
| EUROPE | NS_ELASTICNET | 44 | short | 1.0259 | 1 | 4 |
| EUROPE | NS_ELASTICNET | 66 | long | 1.0632 | 0 | 11 |
| EUROPE | NS_ELASTICNET | 66 | medium | 1.0363 | 0 | 8 |
| EUROPE | NS_ELASTICNET | 66 | short | 1.0379 | 1 | 4 |
| EUROPE | NS_ELASTICNET | 132 | long | 1.0932 | 0 | 11 |
| EUROPE | NS_ELASTICNET | 132 | medium | 1.0560 | 0 | 8 |
| EUROPE | NS_ELASTICNET | 132 | short | 1.0399 | 1 | 4 |
| EUROPE | NS_ELASTICNET | 252 | long | 1.1136 | 0 | 11 |
| EUROPE | NS_ELASTICNET | 252 | medium | 1.0912 | 0 | 8 |
| EUROPE | NS_ELASTICNET | 252 | short | 1.0546 | 0 | 4 |
| EUROPE | NS_RIDGE | 5 | long | 1.0544 | 2 | 11 |
| EUROPE | NS_RIDGE | 5 | medium | 1.0507 | 1 | 8 |
| EUROPE | NS_RIDGE | 5 | short | 1.4146 | 0 | 4 |
| EUROPE | NS_RIDGE | 10 | long | 1.0391 | 0 | 11 |
| EUROPE | NS_RIDGE | 10 | medium | 1.0868 | 0 | 8 |
| EUROPE | NS_RIDGE | 10 | short | 1.1192 | 1 | 4 |
| EUROPE | NS_RIDGE | 22 | long | 0.9928 | 9 | 11 |
| EUROPE | NS_RIDGE | 22 | medium | 1.0532 | 0 | 8 |
| EUROPE | NS_RIDGE | 22 | short | 1.0164 | 1 | 4 |
| EUROPE | NS_RIDGE | 44 | long | 1.0104 | 4 | 11 |
| EUROPE | NS_RIDGE | 44 | medium | 1.0653 | 0 | 8 |
| EUROPE | NS_RIDGE | 44 | short | 1.0133 | 1 | 4 |
| EUROPE | NS_RIDGE | 66 | long | 0.9692 | 9 | 11 |
| EUROPE | NS_RIDGE | 66 | medium | 1.0579 | 0 | 8 |
| EUROPE | NS_RIDGE | 66 | short | 1.0942 | 1 | 4 |
| EUROPE | NS_RIDGE | 132 | long | 1.0463 | 2 | 11 |
| EUROPE | NS_RIDGE | 132 | medium | 1.2221 | 0 | 8 |
| EUROPE | NS_RIDGE | 132 | short | 1.2064 | 0 | 4 |
| EUROPE | NS_RIDGE | 252 | long | 1.2266 | 0 | 11 |
| EUROPE | NS_RIDGE | 252 | medium | 1.3152 | 0 | 8 |
| EUROPE | NS_RIDGE | 252 | short | 1.1648 | 0 | 4 |
| EUROPE | PCA_RIDGE | 5 | long | 1.1014 | 3 | 11 |
| EUROPE | PCA_RIDGE | 5 | medium | 1.0661 | 0 | 8 |
| EUROPE | PCA_RIDGE | 5 | short | 1.4284 | 0 | 4 |
| EUROPE | PCA_RIDGE | 10 | long | 1.0596 | 5 | 11 |
| EUROPE | PCA_RIDGE | 10 | medium | 1.0830 | 0 | 8 |
| EUROPE | PCA_RIDGE | 10 | short | 1.1209 | 0 | 4 |
| EUROPE | PCA_RIDGE | 22 | long | 1.0023 | 10 | 11 |
| EUROPE | PCA_RIDGE | 22 | medium | 1.0506 | 0 | 8 |
| EUROPE | PCA_RIDGE | 22 | short | 1.0488 | 1 | 4 |
| EUROPE | PCA_RIDGE | 44 | long | 1.0088 | 8 | 11 |
| EUROPE | PCA_RIDGE | 44 | medium | 1.0804 | 0 | 8 |
| EUROPE | PCA_RIDGE | 44 | short | 1.0427 | 1 | 4 |
| EUROPE | PCA_RIDGE | 66 | long | 0.9867 | 10 | 11 |
| EUROPE | PCA_RIDGE | 66 | medium | 1.0625 | 1 | 8 |
| EUROPE | PCA_RIDGE | 66 | short | 1.1224 | 1 | 4 |
| EUROPE | PCA_RIDGE | 132 | long | 1.0296 | 0 | 11 |
| EUROPE | PCA_RIDGE | 132 | medium | 1.1630 | 0 | 8 |
| EUROPE | PCA_RIDGE | 132 | short | 1.1859 | 0 | 4 |
| EUROPE | PCA_RIDGE | 252 | long | 1.1582 | 0 | 11 |
| EUROPE | PCA_RIDGE | 252 | medium | 1.2019 | 0 | 8 |
| EUROPE | PCA_RIDGE | 252 | short | 1.0408 | 1 | 4 |
| EUROPE | PCA_RIDGE_VAR | 5 | long | 1.1132 | 2 | 11 |
| EUROPE | PCA_RIDGE_VAR | 5 | medium | 1.0788 | 0 | 8 |
| EUROPE | PCA_RIDGE_VAR | 5 | short | 1.4019 | 0 | 4 |
| EUROPE | PCA_RIDGE_VAR | 10 | long | 1.0442 | 6 | 11 |
| EUROPE | PCA_RIDGE_VAR | 10 | medium | 1.0502 | 0 | 8 |
| EUROPE | PCA_RIDGE_VAR | 10 | short | 1.0862 | 1 | 4 |
| EUROPE | PCA_RIDGE_VAR | 22 | long | 0.9918 | 8 | 11 |
| EUROPE | PCA_RIDGE_VAR | 22 | medium | 1.0140 | 0 | 8 |
| EUROPE | PCA_RIDGE_VAR | 22 | short | 0.9731 | 2 | 4 |
| EUROPE | PCA_RIDGE_VAR | 44 | long | 0.9497 | 10 | 11 |
| EUROPE | PCA_RIDGE_VAR | 44 | medium | 0.9969 | 6 | 8 |
| EUROPE | PCA_RIDGE_VAR | 44 | short | 0.9306 | 2 | 4 |
| EUROPE | PCA_RIDGE_VAR | 66 | long | 0.9192 | 11 | 11 |
| EUROPE | PCA_RIDGE_VAR | 66 | medium | 0.9907 | 5 | 8 |
| EUROPE | PCA_RIDGE_VAR | 66 | short | 0.9109 | 2 | 4 |
| EUROPE | PCA_RIDGE_VAR | 132 | long | 0.8413 | 11 | 11 |
| EUROPE | PCA_RIDGE_VAR | 132 | medium | 0.9698 | 5 | 8 |
| EUROPE | PCA_RIDGE_VAR | 132 | short | 0.9098 | 3 | 4 |
| EUROPE | PCA_RIDGE_VAR | 252 | long | 0.8461 | 11 | 11 |
| EUROPE | PCA_RIDGE_VAR | 252 | medium | 0.9979 | 4 | 8 |
| EUROPE | PCA_RIDGE_VAR | 252 | short | 0.9351 | 3 | 4 |
| EUROPE | PERSISTENCE | 5 | long | 1.0000 | 0 | 11 |
| EUROPE | PERSISTENCE | 5 | medium | 1.0000 | 0 | 8 |
| EUROPE | PERSISTENCE | 5 | short | 1.0000 | 0 | 4 |
| EUROPE | PERSISTENCE | 10 | long | 1.0000 | 0 | 11 |
| EUROPE | PERSISTENCE | 10 | medium | 1.0000 | 0 | 8 |
| EUROPE | PERSISTENCE | 10 | short | 1.0000 | 0 | 4 |
| EUROPE | PERSISTENCE | 22 | long | 1.0000 | 0 | 11 |
| EUROPE | PERSISTENCE | 22 | medium | 1.0000 | 0 | 8 |
| EUROPE | PERSISTENCE | 22 | short | 1.0000 | 0 | 4 |
| EUROPE | PERSISTENCE | 44 | long | 1.0000 | 0 | 11 |
| EUROPE | PERSISTENCE | 44 | medium | 1.0000 | 0 | 8 |
| EUROPE | PERSISTENCE | 44 | short | 1.0000 | 0 | 4 |
| EUROPE | PERSISTENCE | 66 | long | 1.0000 | 0 | 11 |
| EUROPE | PERSISTENCE | 66 | medium | 1.0000 | 0 | 8 |
| EUROPE | PERSISTENCE | 66 | short | 1.0000 | 0 | 4 |
| EUROPE | PERSISTENCE | 132 | long | 1.0000 | 0 | 11 |
| EUROPE | PERSISTENCE | 132 | medium | 1.0000 | 0 | 8 |
| EUROPE | PERSISTENCE | 132 | short | 1.0000 | 0 | 4 |
| EUROPE | PERSISTENCE | 252 | long | 1.0000 | 0 | 11 |
| EUROPE | PERSISTENCE | 252 | medium | 1.0000 | 0 | 8 |
| EUROPE | PERSISTENCE | 252 | short | 1.0000 | 0 | 4 |
| US_CORE | DNS_KALMAN_OU | 5 | long | 1.0178 | 1 | 2 |
| US_CORE | DNS_KALMAN_OU | 5 | medium | 1.0016 | 1 | 3 |
| US_CORE | DNS_KALMAN_OU | 5 | short | 1.3071 | 1 | 3 |
| US_CORE | DNS_KALMAN_OU | 10 | long | 1.0042 | 1 | 2 |
| US_CORE | DNS_KALMAN_OU | 10 | medium | 0.9794 | 2 | 3 |
| US_CORE | DNS_KALMAN_OU | 10 | short | 1.1384 | 1 | 3 |
| US_CORE | DNS_KALMAN_OU | 22 | long | 0.9996 | 1 | 2 |
| US_CORE | DNS_KALMAN_OU | 22 | medium | 0.9873 | 2 | 3 |
| US_CORE | DNS_KALMAN_OU | 22 | short | 1.2123 | 0 | 3 |
| US_CORE | DNS_KALMAN_OU | 44 | long | 1.0009 | 1 | 2 |
| US_CORE | DNS_KALMAN_OU | 44 | medium | 0.9943 | 2 | 3 |
| US_CORE | DNS_KALMAN_OU | 44 | short | 1.2139 | 0 | 3 |
| US_CORE | DNS_KALMAN_OU | 66 | long | 0.9908 | 2 | 2 |
| US_CORE | DNS_KALMAN_OU | 66 | medium | 0.9931 | 2 | 3 |
| US_CORE | DNS_KALMAN_OU | 66 | short | 1.2090 | 0 | 3 |
| US_CORE | DNS_KALMAN_OU | 132 | long | 1.0217 | 0 | 2 |
| US_CORE | DNS_KALMAN_OU | 132 | medium | 1.0236 | 1 | 3 |
| US_CORE | DNS_KALMAN_OU | 132 | short | 1.1913 | 0 | 3 |
| US_CORE | DNS_KALMAN_OU | 252 | long | 1.0160 | 0 | 2 |
| US_CORE | DNS_KALMAN_OU | 252 | medium | 1.0231 | 1 | 3 |
| US_CORE | DNS_KALMAN_OU | 252 | short | 1.1764 | 0 | 3 |
| US_CORE | NS_ELASTICNET | 5 | long | 1.0523 | 0 | 2 |
| US_CORE | NS_ELASTICNET | 5 | medium | 1.0147 | 1 | 3 |
| US_CORE | NS_ELASTICNET | 5 | short | 1.1962 | 1 | 3 |
| US_CORE | NS_ELASTICNET | 10 | long | 1.0322 | 0 | 2 |
| US_CORE | NS_ELASTICNET | 10 | medium | 0.9963 | 1 | 3 |
| US_CORE | NS_ELASTICNET | 10 | short | 1.0479 | 1 | 3 |
| US_CORE | NS_ELASTICNET | 22 | long | 1.0335 | 0 | 2 |
| US_CORE | NS_ELASTICNET | 22 | medium | 0.9967 | 1 | 3 |
| US_CORE | NS_ELASTICNET | 22 | short | 1.0369 | 1 | 3 |
| US_CORE | NS_ELASTICNET | 44 | long | 1.0430 | 0 | 2 |
| US_CORE | NS_ELASTICNET | 44 | medium | 0.9985 | 2 | 3 |
| US_CORE | NS_ELASTICNET | 44 | short | 1.0396 | 1 | 3 |
| US_CORE | NS_ELASTICNET | 66 | long | 1.0455 | 0 | 2 |
| US_CORE | NS_ELASTICNET | 66 | medium | 1.0074 | 0 | 3 |
| US_CORE | NS_ELASTICNET | 66 | short | 1.0404 | 1 | 3 |
| US_CORE | NS_ELASTICNET | 132 | long | 1.1033 | 0 | 2 |
| US_CORE | NS_ELASTICNET | 132 | medium | 1.0351 | 0 | 3 |
| US_CORE | NS_ELASTICNET | 132 | short | 1.0452 | 0 | 3 |
| US_CORE | NS_ELASTICNET | 252 | long | 1.1566 | 0 | 2 |
| US_CORE | NS_ELASTICNET | 252 | medium | 1.0745 | 0 | 3 |
| US_CORE | NS_ELASTICNET | 252 | short | 1.0647 | 0 | 3 |
| US_CORE | NS_RIDGE | 5 | long | 1.0372 | 1 | 2 |
| US_CORE | NS_RIDGE | 5 | medium | 1.0065 | 1 | 3 |
| US_CORE | NS_RIDGE | 5 | short | 1.1575 | 1 | 3 |
| US_CORE | NS_RIDGE | 10 | long | 1.0233 | 1 | 2 |
| US_CORE | NS_RIDGE | 10 | medium | 0.9911 | 2 | 3 |
| US_CORE | NS_RIDGE | 10 | short | 0.9577 | 1 | 3 |
| US_CORE | NS_RIDGE | 22 | long | 1.0341 | 0 | 2 |
| US_CORE | NS_RIDGE | 22 | medium | 1.0024 | 1 | 3 |
| US_CORE | NS_RIDGE | 22 | short | 0.9450 | 1 | 3 |
| US_CORE | NS_RIDGE | 44 | long | 1.0196 | 0 | 2 |
| US_CORE | NS_RIDGE | 44 | medium | 1.0086 | 1 | 3 |
| US_CORE | NS_RIDGE | 44 | short | 0.9593 | 1 | 3 |
| US_CORE | NS_RIDGE | 66 | long | 1.0449 | 0 | 2 |
| US_CORE | NS_RIDGE | 66 | medium | 1.0372 | 0 | 3 |
| US_CORE | NS_RIDGE | 66 | short | 0.9505 | 2 | 3 |
| US_CORE | NS_RIDGE | 132 | long | 1.0718 | 0 | 2 |
| US_CORE | NS_RIDGE | 132 | medium | 1.0710 | 0 | 3 |
| US_CORE | NS_RIDGE | 132 | short | 0.9705 | 2 | 3 |
| US_CORE | NS_RIDGE | 252 | long | 1.0056 | 1 | 2 |
| US_CORE | NS_RIDGE | 252 | medium | 1.1052 | 0 | 3 |
| US_CORE | NS_RIDGE | 252 | short | 1.0791 | 0 | 3 |
| US_CORE | PCA_RIDGE | 5 | long | 1.0102 | 1 | 2 |
| US_CORE | PCA_RIDGE | 5 | medium | 1.0033 | 1 | 3 |
| US_CORE | PCA_RIDGE | 5 | short | 1.1692 | 0 | 3 |
| US_CORE | PCA_RIDGE | 10 | long | 1.0091 | 0 | 2 |
| US_CORE | PCA_RIDGE | 10 | medium | 1.0124 | 1 | 3 |
| US_CORE | PCA_RIDGE | 10 | short | 0.9640 | 1 | 3 |
| US_CORE | PCA_RIDGE | 22 | long | 0.9930 | 2 | 2 |
| US_CORE | PCA_RIDGE | 22 | medium | 1.0014 | 1 | 3 |
| US_CORE | PCA_RIDGE | 22 | short | 0.9791 | 1 | 3 |
| US_CORE | PCA_RIDGE | 44 | long | 0.9877 | 2 | 2 |
| US_CORE | PCA_RIDGE | 44 | medium | 1.0090 | 1 | 3 |
| US_CORE | PCA_RIDGE | 44 | short | 0.9688 | 2 | 3 |
| US_CORE | PCA_RIDGE | 66 | long | 1.0149 | 0 | 2 |
| US_CORE | PCA_RIDGE | 66 | medium | 1.0193 | 0 | 3 |
| US_CORE | PCA_RIDGE | 66 | short | 0.9376 | 3 | 3 |
| US_CORE | PCA_RIDGE | 132 | long | 1.0614 | 0 | 2 |
| US_CORE | PCA_RIDGE | 132 | medium | 1.0653 | 0 | 3 |
| US_CORE | PCA_RIDGE | 132 | short | 0.9549 | 3 | 3 |
| US_CORE | PCA_RIDGE | 252 | long | 0.9861 | 1 | 2 |
| US_CORE | PCA_RIDGE | 252 | medium | 1.0709 | 0 | 3 |
| US_CORE | PCA_RIDGE | 252 | short | 1.0109 | 1 | 3 |
| US_CORE | PCA_RIDGE_VAR | 5 | long | 0.9950 | 1 | 2 |
| US_CORE | PCA_RIDGE_VAR | 5 | medium | 0.9945 | 1 | 3 |
| US_CORE | PCA_RIDGE_VAR | 5 | short | 1.1533 | 0 | 3 |
| US_CORE | PCA_RIDGE_VAR | 10 | long | 0.9840 | 2 | 2 |
| US_CORE | PCA_RIDGE_VAR | 10 | medium | 0.9914 | 1 | 3 |
| US_CORE | PCA_RIDGE_VAR | 10 | short | 0.9283 | 2 | 3 |
| US_CORE | PCA_RIDGE_VAR | 22 | long | 0.9837 | 2 | 2 |
| US_CORE | PCA_RIDGE_VAR | 22 | medium | 0.9892 | 1 | 3 |
| US_CORE | PCA_RIDGE_VAR | 22 | short | 0.9169 | 3 | 3 |
| US_CORE | PCA_RIDGE_VAR | 44 | long | 0.9755 | 2 | 2 |
| US_CORE | PCA_RIDGE_VAR | 44 | medium | 0.9777 | 2 | 3 |
| US_CORE | PCA_RIDGE_VAR | 44 | short | 0.8534 | 3 | 3 |
| US_CORE | PCA_RIDGE_VAR | 66 | long | 0.9706 | 2 | 2 |
| US_CORE | PCA_RIDGE_VAR | 66 | medium | 0.9722 | 3 | 3 |
| US_CORE | PCA_RIDGE_VAR | 66 | short | 0.8159 | 3 | 3 |
| US_CORE | PCA_RIDGE_VAR | 132 | long | 0.9069 | 2 | 2 |
| US_CORE | PCA_RIDGE_VAR | 132 | medium | 0.9144 | 3 | 3 |
| US_CORE | PCA_RIDGE_VAR | 132 | short | 0.7629 | 3 | 3 |
| US_CORE | PCA_RIDGE_VAR | 252 | long | 0.8386 | 2 | 2 |
| US_CORE | PCA_RIDGE_VAR | 252 | medium | 0.8810 | 3 | 3 |
| US_CORE | PCA_RIDGE_VAR | 252 | short | 0.7327 | 3 | 3 |
| US_CORE | PERSISTENCE | 5 | long | 1.0000 | 0 | 2 |
| US_CORE | PERSISTENCE | 5 | medium | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 5 | short | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 10 | long | 1.0000 | 0 | 2 |
| US_CORE | PERSISTENCE | 10 | medium | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 10 | short | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 22 | long | 1.0000 | 0 | 2 |
| US_CORE | PERSISTENCE | 22 | medium | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 22 | short | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 44 | long | 1.0000 | 0 | 2 |
| US_CORE | PERSISTENCE | 44 | medium | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 44 | short | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 66 | long | 1.0000 | 0 | 2 |
| US_CORE | PERSISTENCE | 66 | medium | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 66 | short | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 132 | long | 1.0000 | 0 | 2 |
| US_CORE | PERSISTENCE | 132 | medium | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 132 | short | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 252 | long | 1.0000 | 0 | 2 |
| US_CORE | PERSISTENCE | 252 | medium | 1.0000 | 0 | 3 |
| US_CORE | PERSISTENCE | 252 | short | 1.0000 | 0 | 3 |

The accompanying maturity×horizon heatmaps retain every maturity. Segment summaries use stable market-aware short/medium/long definitions and are treated as descriptive; isolated favorable cells are not generalized.

## 14. Statistical significance and power

Favorable aggregate DM rejections at 5% are:

None.

A positive loss difference favors persistence. Non-rejection at J+132/J+252 is not evidence of equality because effective information is extremely limited. There are 210 primary aggregate comparisons; all raw results are reported, and maturity-level results include FDR-adjusted p-values.

## 15. Full-history versus common-period robustness

| market | horizon | model | rmse_ratio_vs_persistence |
| --- | --- | --- | --- |
| BAM_COMMON | 5 | PERSISTENCE | 1.0000 |
| BAM_COMMON | 10 | PERSISTENCE | 1.0000 |
| BAM_COMMON | 22 | PCA_RIDGE_VAR | 0.9964 |
| BAM_COMMON | 44 | DNS_KALMAN_OU | 0.9385 |
| BAM_COMMON | 66 | DNS_KALMAN_OU | 0.8662 |
| BAM_COMMON | 132 | DNS_KALMAN_OU | 0.7348 |
| BAM_COMMON | 252 | DNS_KALMAN_OU | 0.5972 |
| EUROPE_COMMON | 5 | PERSISTENCE | 1.0000 |
| EUROPE_COMMON | 10 | PERSISTENCE | 1.0000 |
| EUROPE_COMMON | 22 | PERSISTENCE | 1.0000 |
| EUROPE_COMMON | 44 | PCA_RIDGE_VAR | 0.9678 |
| EUROPE_COMMON | 66 | PCA_RIDGE_VAR | 0.9447 |
| EUROPE_COMMON | 132 | PCA_RIDGE_VAR | 0.9039 |
| EUROPE_COMMON | 252 | PCA_RIDGE_VAR | 0.9250 |
| US_COMMON | 5 | PERSISTENCE | 1.0000 |
| US_COMMON | 10 | NS_RIDGE | 0.9798 |
| US_COMMON | 22 | PCA_RIDGE_VAR | 0.9672 |
| US_COMMON | 44 | PCA_RIDGE_VAR | 0.9276 |
| US_COMMON | 66 | PCA_RIDGE_VAR | 0.8995 |
| US_COMMON | 132 | PCA_RIDGE_VAR | 0.7892 |
| US_COMMON | 252 | PCA_RIDGE_VAR | 0.7599 |

Full history uses each market's longest reliable input sample. The common experiment uses 2004-09-06 through 2026-06-18. BAM is already bounded by that interval; Europe is causally subset from the same-history run because future rows cannot affect an earlier expanding-window origin; U.S. is fully re-estimated from the common start because its full run includes pre-2004 training information.

## 16. Stability through time and historical periods

Cumulative J+252 loss differentials are reported for each market's numerically best long-horizon candidate. Given only roughly 3–4 overlap-adjusted annual-horizon units, crisis-period slicing would produce scientifically misleading cells; predefined subperiod regressions are therefore not elevated to formal evidence. The plots reveal whether apparent gains are concentrated without fitting a regime model.

## 17. Scientific questions

1. **Does predictability increase with horizon?** The ratios show whether relative performance slopes downward; the decision above requires cross-market, neighboring-horizon consistency.
2. **When does persistence stop dominating?** Any numerical crossing below one is listed in the best-model table, but descriptive-only annual results are not called a threshold.
3. **Is the pattern cross-market?** The cross-market table answers this directly.
4. **Does DNS align with OU half-lives?** The DNS table and half-life diagnostic show whether competitiveness improves near estimated speeds without manipulating kappa.
5. **Does the U.S. PCA-Ridge-VAR signal strengthen?** The PCA table compares all seven horizons in both full and common histories.
6. **Are gains economically meaningful?** Improvement percentages, maturity breadth and cumulative losses are reported together.
7. **Are gains statistically defensible?** DM inference is overlap-aware; J+132/J+252 remain explicitly low-powered.

## 18. Limitations

ECB zero-coupon spot yields and U.S. Treasury par yields are not identical instruments; normalized within-market ratios are therefore primary. Market calendars differ. Monthly origins limit nominal sample size, while long forecast windows sharply reduce effective information. The fixed model family deliberately excludes macro, cross-market, nonlinear and state-dependent predictors. J+252 conclusions are descriptive, not definitive.

## 19. Reproducibility

Machine-readable outputs include feasibility, all maturity/model metrics, primary and secondary DM tests, FDR diagnostics, factor forecasts, OU half-lives, curve segments, best-model and cross-market tables, overlap audits, full/common forecasts, and at least twelve figures under `outputs_phase5/`. Tests cover arbitrary horizons, long-horizon target realization, date-derived monthly/weekly overlap, deterministic forecasts, common periods and NS/PCA reconstruction.

## 20. Final decisions

Phase 5C is justified as a descriptive, origin-observable conditional evaluation because performance changes sharply with horizon, the winning dynamics differ by market, maturity gains are broad but uneven, and cumulative annual-horizon losses vary through time. This does not authorize regime-switching estimation or new forecasting algorithms. International transmission remains deferred because Phase 5 establishes no common winning factor law and contains no direct evidence that foreign factors add incremental BAM information.

**LONG-HORIZON PREDICTABILITY SUPPORTED**

**PHASE 5C CONDITIONAL PREDICTABILITY JUSTIFIED**

**PHASE 4D INTERNATIONAL TRANSMISSION REMAINS DEFERRED**
