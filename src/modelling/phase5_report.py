"""Render the Phase 5A/B scientific report and evidence-driven decisions."""

from pathlib import Path

import numpy as np
import pandas as pd


def md(frame: pd.DataFrame, digits: int = 4) -> str:
    def value(x): return f"{x:.{digits}f}" if isinstance(x, (float, np.floating)) else str(x)
    cols=list(frame.columns);lines=["| "+" | ".join(map(str,cols))+" |","| "+" | ".join("---" for _ in cols)+" |"]
    lines += ["| "+" | ".join(value(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def render(project: Path) -> Path:
    root=project/"outputs_phase5";audit=pd.read_csv(root/"phase5_horizon_feasibility.csv")
    agg=pd.read_csv(root/"phase5_aggregate_metrics.csv");best=pd.read_csv(root/"phase5_best_model_by_horizon.csv")
    master=pd.read_csv(root/"phase5_master_horizon_table.csv");cross=pd.read_csv(root/"phase5_cross_market_horizon.csv")
    dm=pd.read_csv(root/"phase5_dm_primary.csv");half=pd.read_csv(root/"phase5_ou_half_lives.csv")
    factor=pd.read_csv(root/"phase5_factor_forecast_metrics.csv");segments=pd.read_csv(root/"phase5_curve_segment_results.csv")
    primary=agg[agg.market.isin(["BAM","EUROPE","US_CORE"])]
    long=primary[(primary.horizon>=44)&primary.model.ne("PERSISTENCE")]
    robust=[]
    for model,p in long.groupby("model"):
        wins=p.assign(win=p.rmse_ratio_vs_persistence<.98).groupby("horizon").win.sum()
        if (wins>=2).rolling(2).sum().ge(2).any(): robust.append(model)
    long_supported=bool(robust)
    material=best[(best.horizon>=44)&(best.rmse_improvement_pct>=2)]
    horizon_variation=(long.groupby(["market","model"]).rmse_ratio_vs_persistence.agg(lambda x:x.max()-x.min())>.05).any()
    phase5c=bool(len(material) or horizon_variation)
    long_winners=best[(best.horizon>=44)&best.rmse_ratio_vs_persistence.lt(1)]
    reconsider=any(len(set(p.market))==3 for _,p in long_winners.groupby("model"))
    decision1="LONG-HORIZON PREDICTABILITY SUPPORTED" if long_supported else "LONG-HORIZON PREDICTABILITY NOT ROBUSTLY SUPPORTED"
    decision2="PHASE 5C CONDITIONAL PREDICTABILITY JUSTIFIED" if phase5c else "PHASE 5C NOT YET JUSTIFIED"
    decision3="PHASE 4D SHOULD BE RECONSIDERED AT LONGER HORIZONS" if reconsider else "PHASE 4D INTERNATIONAL TRANSMISSION REMAINS DEFERRED"
    rw=primary[primary.model.eq("PERSISTENCE")][["market","horizon","rmse","mae"]]
    dns=primary[primary.model.eq("DNS_KALMAN_OU")].pivot(index="market",columns="horizon",values="rmse_ratio_vs_persistence").reset_index()
    ns=primary[primary.model.eq("NS_RIDGE")].pivot(index="market",columns="horizon",values="rmse_ratio_vs_persistence").reset_index()
    pca=primary[primary.model.eq("PCA_RIDGE_VAR")].pivot(index="market",columns="horizon",values="rmse_ratio_vs_persistence").reset_index()
    favorable=dm[(dm.maturity.astype(str)=="aggregate")&(dm.mean_loss_difference<0)&(dm.p_value<.05)]
    common=agg[agg.market.str.contains("COMMON")&agg.maturity.astype(str).eq("aggregate")]
    common_best=common.loc[common.groupby(["market","horizon"]).rmse.idxmin(),["market","horizon","model","rmse_ratio_vs_persistence"]]
    report=f"""# Phase 5 — Medium- and long-horizon yield-curve predictability

## 1. Motivation and Phase 4 baseline

Phase 4 found that persistence was extremely difficult to beat at J+5/J+10/J+22. Phase 5 tests, without assuming, whether direct factor changes or OU mean reversion become useful at J+44, J+66, J+132 and J+252. J+h always means the h-th subsequent published market observation on that market's own business/publication calendar—not h calendar days.

## 2. Decisions

**{decision1}**

**{decision2}**

**{decision3}**

The first decision requires economically nontrivial improvement across at least two markets and neighboring long horizons, rather than an isolated winning cell. The Phase 5C decision uses only the completed horizon, maturity and temporal-stability evidence; no Phase 5C model was implemented. No international curve was used as another market's predictor.

The central result is a genuine horizon response, but not a universal model response. At J+44/J+66, BAM DNS reaches 0.939/0.866 of persistence RMSE, Europe PCA-Ridge-VAR reaches 0.968/0.945, and U.S. PCA-Ridge-VAR reaches 0.934/0.910. At J+132/J+252 the numerical gains become larger—BAM DNS 0.735/0.597, Europe PCA-Ridge-VAR 0.904/0.925, and U.S. PCA-Ridge-VAR 0.818/0.774—but those horizons are descriptive-only. None of the favorable aggregate DM tests reaches 5%; BAM DNS approaches conventional significance at J+66/J+132/J+252 (p≈0.098/0.069/0.065), while Europe and U.S. gains remain less precisely estimated. Thus long-horizon predictability is supported by economic magnitude, neighboring-horizon consistency and maturity breadth, not by decisive conventional inference.

## 3. Ex-ante horizon feasibility and sample audit

Classification was frozen before forecasts were run. `FULLY_FEASIBLE` requires adequate raw and overlap-adjusted information; `LIMITED_POWER` permits inference with caution; `DESCRIPTIVE_ONLY` retains the forecast but rejects strong inferential claims; `NOT_FEASIBLE` would suppress modelling. No classification used forecast performance.

{md(audit[["market","horizon","total_observations","monthly_evaluation_origins","weekly_evaluation_origins","training_rows_earliest_origin","training_rows_latest_origin","monthly_maximum_overlap_lag","monthly_effective_loss_length","classification"]],2)}

J+132 and J+252 are retained for all markets but are explicitly descriptive-only. Their many overlapping raw forecasts correspond to very little independent loss-series information. Weekly counts are audit diagnostics only; monthly origins remain primary and daily origins were not introduced.

## 4. Methodology and leakage controls

The unchanged model family is Persistence, DNS-Kalman/OU, NS-Ridge, NS-ElasticNet, PCA-Ridge and PCA-Ridge-VAR. Ridge/ElasticNet estimate each Δfactor(t,h) directly. PCA is refitted at every origin. PCA-Ridge-VAR remains a recursively iterated regularized factor VAR. DNS parameters, lambda, Kalman state and OU dynamics are estimated using origin-local history, then propagated directly to h. Direct training origin s is accepted only when s+h≤t; recent unrealized outcomes are excluded. Hyperparameter validation is expanding and target-purged. Rates remain decimal.

All seven horizons use actual half-open forecast intervals `(origin,target]`. HAC bandwidth is the maximum overlapping lag measured in the monthly loss sequence, not h−1. Aggregate DM tests average maturity losses within origin before inference. Raw maturity-test p-values and Benjamini–Hochberg FDR diagnostics are both retained.

## 5. Phase 5A master horizon table — RMSE ratio versus persistence

Values below one favor the model.

{md(master)}

## 6. Persistence degradation

{md(rw)}

Persistence absolute error generally grows with h, but this alone is not model value. A factor model adds value only when its errors grow more slowly, which is captured by the ratios above.

## 7. Best model by market and horizon

{md(best[["market","horizon","model","rmse","rmse_ratio_vs_persistence","rmse_improvement_pct","p_value","hac_bandwidth","classification","evidence"]])}

## 8. Cross-market horizon comparison

{md(cross)}

The absence of a universal winner is central: an isolated gain cannot establish sovereign-curve predictability as a cross-market regularity.

Persistence stops being the numerical winner at J+44 in all three markets, but the mechanism differs: OU/DNS mean reversion drives BAM, whereas recursive PCA dynamics drive Europe and the U.S. This is a descriptive forecastability threshold, not a structural law selected in advance.

## 9. DNS and mean-reversion horizon response

{md(dns)}

Estimated OU diagnostics:

{md(half)}

The half-life comparison is diagnostic only. A forecast horizon approaching an estimated half-life does not guarantee that noisy state estimates or a fixed OU law will beat persistence.

## 10. NS-Ridge horizon response

{md(ns)}

This distinguishes factor representation from factor dynamics: improvement by NS-Ridge without DNS would point toward supervised dynamics, whereas joint failure indicates that merely changing the dynamics is insufficient.

## 11. PCA-Ridge-VAR horizon response

{md(pca)}

This directly tests whether the modest U.S. J+10/J+22 Phase 4 pattern strengthens with horizon rather than extrapolating it.

## 12. Factor predictability

{md(factor)}

Sign accuracy is secondary. Factor RMSE remains the primary factor-level diagnostic, and factor gains are not substituted for curve-yield accuracy.

## 13. Maturity and curve-segment evidence

{md(segments)}

The accompanying maturity×horizon heatmaps retain every maturity. Segment summaries use stable market-aware short/medium/long definitions and are treated as descriptive; isolated favorable cells are not generalized.

## 14. Statistical significance and power

Favorable aggregate DM rejections at 5% are:

{md(favorable[["market","model","horizon","p_value","mean_loss_difference","n","hac_bandwidth"]]) if len(favorable) else 'None.'}

A positive loss difference favors persistence. Non-rejection at J+132/J+252 is not evidence of equality because effective information is extremely limited. There are {len(dm[dm.maturity.astype(str)=='aggregate'])} primary aggregate comparisons; all raw results are reported, and maturity-level results include FDR-adjusted p-values.

## 15. Full-history versus common-period robustness

{md(common_best)}

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

**{decision1}**

**{decision2}**

**{decision3}**
"""
    path=project/"docs/PHASE5_LONG_HORIZON_PREDICTABILITY.md";path.write_text(report,encoding="utf-8");return path


if __name__=="__main__":print(render(Path(".").resolve()))
