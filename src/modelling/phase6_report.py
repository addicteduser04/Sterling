"""Render Phase 6 one-to-five-year BAM forecasting report."""

from pathlib import Path
import numpy as np
import pandas as pd


def md(frame,d=3):
    def v(x):return f"{x:.{d}f}" if isinstance(x,(float,np.floating)) else str(x)
    c=list(frame.columns);lines=["| "+" | ".join(map(str,c))+" |","| "+" | ".join('---' for _ in c)+" |"]
    lines += ["| "+" | ".join(v(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None)];return '\n'.join(lines)


def render(project:Path):
    root=project/'outputs_phase6';audit=pd.read_csv(root/'phase6_feasibility_audit.csv');duration=pd.read_csv(root/'phase6_horizon_calendar_duration.csv');main=pd.read_csv(root/'phase6_main_accuracy_table.csv');metrics=pd.read_csv(root/'phase6_absolute_metrics.csv');sens=pd.read_csv(root/'phase6_dns_time_unit_sensitivity.csv');diag=pd.read_csv(root/'phase6_origin_diagnostics.csv');dm=pd.read_csv(root/'phase6_dm_guarded.csv');latest=pd.read_csv(root/'phase6_latest_forecast_table_percent.csv')
    diag=diag.groupby('horizon')[['ns_ar1_equilibrium_distance_bp','calendar_dns_equilibrium_distance_bp']].median().reset_index()
    selected=metrics[metrics.model.isin(['PERSISTENCE','HISTORICAL_MEAN','DIRECT_AR1_MEAN_REVERSION','NS_AR1_MEAN_REVERSION','DNS_CALENDAR_TIME_V2','DNS_PUBLICATION_TIME_V2'])]
    error=selected[['model','horizon','rmse_bp','mae_bp','median_absolute_error_bp','p75_absolute_error_bp','p90_absolute_error_bp','bias_bp','direction_accuracy','within_25bp','within_50bp','within_100bp','within_150bp']]
    latest=latest[latest.model.isin(['NS_AR1_MEAN_REVERSION','DNS_CALENDAR_TIME_V2'])]
    report=f"""# Phase 6 — One- to five-year BAM yield-curve forecasting

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

{md(audit)}

J+252 is `LIMITED_POWER`; J+504/J+756 are `DESCRIPTIVE_ONLY`; J+1008/J+1260 are `SCENARIO_ONLY`. Conventional DM inference is suppressed outside J+252. Thousands of maturity rows are never treated as independent forecast episodes.

## 4. Observation steps versus calendar duration

{md(duration)}

The median durations are 376, 753, 1,129, 1,506 and 1,883 calendar days—approximately 1.03, 2.06, 3.09, 4.12 and 5.16 years. Labels 1Y–5Y are therefore approximate.

## 5. Main historical accuracy

{md(main)}

The broad 2009–2025 evaluation differs from Phase 5C0's 2022-era sample. At J+252 the coherent publication-time DNS is 64.6 bp, NS-AR1 is 64.8 bp and calendar DNS is 68.0 bp. At J+504 NS-AR1 is best among these at 82.5 bp. Direct yield AR(1) is numerically best at J+756/J+1008 (89.5/92.3 bp), while persistence wins J+1260 at 100.0 bp. Thus simple equilibrium-oriented dynamics do not deliver monotonically improving five-year accuracy.

## 6. Absolute accuracy and error distributions

{md(error)}

At one year, publication DNS places 57.0% of maturity forecasts within 50 bp and 87.0% within 100 bp; its median error is 42.6 bp. At two years the corresponding rates fall to 36.4% and 70.9%. Beyond three years RMSE is around 90–110 bp for most models, and positive biases become large. Point precision deteriorates even when forecasts converge smoothly.

## 7. Relative accuracy versus persistence

At one year NS-AR1 and publication DNS improve RMSE by only about 3.5–3.8% over persistence in the expanded historical sample. NS-AR1 improves by 14.1% at two years and 6.8% at three years, but loses at four/five years. The relative horizon plots distinguish this from absolute error, which rises sharply through two years and then stays near an economically large plateau.

## 8. Historical mean and simple mean reversion

The expanding historical mean is leakage-safe but not universally strong: RMSE is 86.6, 93.1, 97.9, 101.9 and 107.8 bp. Its earlier J+252 success was evaluation-period dependent. NS-AR1 is materially better through three years, then becomes indistinguishable from an equilibrium scenario and eventually loses to persistence.

## 9. Corrected DNS versus NS-AR1

Calendar DNS is worse than NS-AR1 at every horizon. Publication DNS is only 0.24 bp better at J+252 and is worse thereafter. The limited-power J+252 DM p-values are 0.146 for calendar DNS versus NS-AR1 and 0.829 for publication DNS versus NS-AR1. There is no evidence that corrected DNS adds value beyond simple NS mean reversion.

## 10. Time-unit sensitivity

{md(sens)}

Correcting the mixed convention changes historical curves by about 10.7 bp RMS at J+252 and 10.0 bp at J+504. The difference shrinks at longer horizons because both specifications collapse toward their long-run mean. Calendar V2 is slightly less accurate in this sample; coherence, not ex-post RMSE, is why it is preferred scientifically.

## 11. Forecast convergence

{md(diag)}

Median NS-AR1 distance from equilibrium falls from 40.8 bp at one year to 20.0, 9.8, 5.8 and 3.4 bp from two through five years. Calendar DNS converges even faster: 25.9 bp at one year, 8.2 bp at two, and only 0.5 bp at five. Consequently, the 4Y/5Y outputs are primarily long-run equilibrium estimates rather than dynamically informative forecasts.

## 12. Historical yield-curve visualizations

For every horizon, three origins are selected algorithmically using NS-AR1 curve-error quantiles: nearest Q25 (`successful`), Q50 (`typical`) and Q75 (`difficult`). Each presentation-quality figure shows the origin/persistence curve, NS-AR1 forecast, realized curve, training-only equilibrium, and a maturity-level residual panel. No example is hand-picked.

## 13. Latest prospective BAM forecast

The latest prospective curves have no realized targets and are explicitly stored separately. Human-facing percentage forecasts for NS-AR1 and calendar DNS are:

{md(latest)}

The latest NS-AR1 curve moves from current short yields near 2.23–2.37% toward an equilibrium short end near 2.72–2.97%, while the 30-year point moves from 4.10% toward 4.48%. By three years, most of the convergence has occurred. These are model scenarios, not guaranteed future rates.

## 14. Forecast uncertainty

Predictive covariance bands are not shown. Although V2 is coherent in time units, the current pipeline has not validated joint state/measurement covariance coverage over multi-year, highly overlapping horizons. Historical out-of-sample quantiles are reported instead. At five years only about 2.2 independent episodes exist, so plotting narrow parametric bands would imply false precision.

## 15. Guarded inference

{md(dm)}

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
"""
    path=project/'docs/PHASE6_ONE_TO_FIVE_YEAR_YIELD_FORECASTING.md';path.write_text(report,encoding='utf-8');return path


if __name__=='__main__':print(render(Path('.').resolve()))
