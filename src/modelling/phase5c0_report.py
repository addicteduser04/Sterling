"""Render Phase 5C0 long-horizon validation report."""

from pathlib import Path
import numpy as np
import pandas as pd


def md(frame,digits=3):
    def v(x):return f"{x:.{digits}f}" if isinstance(x,(float,np.floating)) else str(x)
    c=list(frame.columns);lines=["| "+" | ".join(map(str,c))+" |","| "+" | ".join("---" for _ in c)+" |"]
    lines += ["| "+" | ".join(v(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None)];return "\n".join(lines)


def render(project:Path):
    root=project/'outputs_phase5c0';central=pd.read_csv(root/'phase5c0_central_table.csv');bam=pd.read_csv(root/'phase5c0_bam_core_table.csv')
    metrics=pd.read_csv(root/'phase5c0_absolute_metrics.csv');dm=pd.read_csv(root/'phase5c0_dm_comparisons.csv');cal=pd.read_csv(root/'phase5c0_calibration.csv')
    half=pd.read_csv(root/'phase5c0_bam_half_life_comparison.csv');distance=pd.read_csv(root/'phase5c0_distance_quantiles.csv');boot=pd.read_csv(root/'phase5c0_block_bootstrap.csv')
    key=[]
    for market,best in [('BAM','DNS_KALMAN_OU'),('EUROPE','PCA_RIDGE_VAR'),('US_CORE','PCA_RIDGE_VAR')]:
      p=metrics[(metrics.market.eq(market))&metrics.model.eq(best)]
      key.append(p[p.horizon.ge(44)][['market','model','horizon','rmse_bp','mae_bp','median_absolute_error_bp','p75_absolute_error_bp','p90_absolute_error_bp','within_25bp','within_50bp','within_100bp','bias_bp']])
    key=pd.concat(key)
    market_tables=[]
    for market,p in central.groupby('market'):
      q=[]
      for row in p.itertuples():
        met=metrics[(metrics.market.eq(market))&metrics.model.eq(row.best_model)&metrics.horizon.eq(row.horizon)].iloc[0]
        q.append({'Horizon':row.horizon,'Persistence RMSE bp':row.persistence_rmse_bp,'Best Model':row.best_model,'Best RMSE bp':row.best_model_rmse_bp,'Best MAE bp':met.mae_bp,'Improvement %':100*(1-row.best_vs_rw_ratio)})
      market_tables.append(f"### {market}\n\n{md(pd.DataFrame(q))}")
    favorable=dm[(dm.mean_loss_difference<0)&(dm.p_value<.05)]
    important_cal=cal[((cal.market.eq('BAM'))&cal.model.eq('DNS_KALMAN_OU'))|((cal.market.ne('BAM'))&cal.model.eq('PCA_RIDGE_VAR'))]
    report=f"""# Phase 5C0 — Long-horizon forecast validation and mean-reversion benchmarking

## 1. Motivation and decisions

Phase 5 established large relative gains at long horizons. This validation asks whether those forecasts are absolutely precise and whether sophisticated dynamics add value beyond simple training-only mean reversion. Frozen Phase 5 forecasts were not modified or rerun.

**MEDIUM-HORIZON FORECASTS ARE ABSOLUTELY INFORMATIVE; J+252 GAINS ARE MAINLY RELATIVE AND IMPRECISE**

**DNS GAINS ARE PRIMARILY EXPLAINED BY SIMPLE MEAN REVERSION**

**PHASE 5C SHOULD REMAIN DEFERRED**

BAM DNS has RMSE 26.6 bp at J+44 and 33.4 bp at J+66, with median absolute errors 13.7 and 16.8 bp. This is economically interpretable precision, not merely a ratio. At J+252 BAM DNS RMSE rises to 61.8 bp, while Europe and U.S. best-model RMSEs exceed 100 bp; those annual forecasts are directionally interesting but not precise point predictions. A simple NS-factor AR(1) is 0.6–1.0% better than BAM DNS at J+44/J+66/J+132 and 0.8% better at J+252. The expanding historical mean is better still at BAM J+252 (56.7 versus 61.8 bp). Conditional modelling should wait because the incremental sophisticated-model signal is absent, equilibrium-distance variation is weak, and annual inference remains descriptive-only.

## 2. Why persistence weakens with horizon

Persistence imposes zero change. As realized changes accumulate, its BAM RMSE grows from 13.8 bp at J+5 to 103.5 bp at J+252; Europe grows from 12.0 to 111.2 bp and the U.S. from 13.8 to 134.6 bp. The best models also deteriorate materially—BAM DNS rises from 15.1 bp at J+5 to 61.8 bp at J+252—just more slowly. A falling ratio therefore does not mean constant absolute accuracy.

## 3. Central validation table

{md(central)}

## 4. Required market absolute-accuracy tables

{chr(10).join(market_tables)}

## 5. Absolute error distributions

{md(key)}

The fixed 10/25/50/100 bp thresholds are retained for cross-market comparability. At BAM J+44, 77.6% of maturity forecasts are within 25 bp and 95.7% within 50 bp. At J+252 those proportions fall to 27.5% and 57.7%. Europe and U.S. annual 90th-percentile absolute errors are roughly 176 and 175 bp, respectively. Thus a large relative gain at one year coexists with wide economically relevant errors.

## 6. BAM DNS versus simple mean reversion

{md(bam[['horizon','PERSISTENCE','HISTORICAL_MEAN','NS_AR1_MEAN_REVERSION','DNS_KALMAN_OU','dns_vs_rw_pct','dns_vs_simple_mr_pct']])}

The DNS-versus-simple-MR percentages are negative at every reported horizon: DNS never beats NS-AR1. At J+44/J+66, both models outperform persistence by moving factors toward an origin-local long-run level. At J+252, complete historical-mean reversion performs best. The evidence supports mean reversion in BAM yields/factors, but not incremental state-space predictive value.

## 7. Europe and U.S. simple mean reversion

Europe PCA-Ridge-VAR beats the best simple benchmark at J+44/J+66/J+132 by about 1.1%, 2.4%, and 4.6%, but loses to PCA-AR1 at J+252 by 2.6%. U.S. PCA-Ridge-VAR beats the best simple benchmark by about 3.6%, 4.8%, 11.9%, and 12.3% at J+44/J+66/J+132/J+252. These are incremental numerical gains beyond generic mean reversion, especially in the U.S., but pairwise DM and moving-block intervals do not establish significance.

## 8. Factor half-lives

{md(half)}

AR(1) median half-lives are roughly 142, 232 and 186 publication observations for BAM level, slope and curvature. DNS/OU estimates 107, 180 and 144 calendar days. These are comparable in broad economic scale but not identical units. Importantly, the frozen DNS estimator fits kappa from calendar-day gaps while its forecast loop advances `delta=1` per publication step. Phase 5 forecasts are preserved, so this unit convention is disclosed rather than silently changed. A future methodological revision would need a separately versioned rerun.

## 9. Distance from equilibrium

{md(distance)}

DNS gains turn positive from J+44 and grow with horizon, but nearly all origins fall in the fixed ex-ante `LOW` standardized-distance category; only one is `MEDIUM` and none is `HIGH`. This sample cannot validate a stable distance-conditioned effect. The scatter and automatically selected factor decompositions are descriptive bridges, not a conditional model.

## 10. Magnitude and directional calibration

{md(important_cal[['market','model','horizon','mean_absolute_predicted_change_bp','mean_absolute_realized_change_bp','direction_accuracy','change_correlation','calibration_slope_realized_on_predicted']])}

BAM DNS direction accuracy increases from 67.5% at J+44 to 87.0% at J+252, but it underpredicts movement magnitude: mean absolute predicted versus realized change is 12.3 versus 17.8 bp at J+44 and 41.9 versus 84.2 bp at J+252. The annual calibration slope near 1.90 likewise indicates under-dispersed predictions. Europe and U.S. PCA forecasts show analogous underprediction of large long-horizon moves.

## 11. Statistical inference and bootstrap uncertainty

Favorable pairwise DM results at 5%:

{md(favorable[['market','model','benchmark','horizon','p_value','mean_loss_difference','n','hac_bandwidth']]) if len(favorable) else 'None.'}

DNS versus NS-AR1 has positive loss differences at every BAM horizon, so the comparison favors the simple model; at J+22 this disadvantage is significant, while long-horizon differences are not. Moving-block bootstrap intervals at J+44/J+66 all include zero:

{md(boot)}

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
"""
    path=project/'docs/PHASE5C0_LONG_HORIZON_VALIDATION.md';path.write_text(report,encoding='utf-8');return path


if __name__=='__main__':print(render(Path('.').resolve()))
