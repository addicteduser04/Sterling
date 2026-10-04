"""Render the reproducible Phase 4 research report from saved artefacts."""

from pathlib import Path

import pandas as pd


def _md(frame: pd.DataFrame) -> str:
    def value(item):
        return f"{item:.6f}" if isinstance(item, float) else str(item)
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(value(item) for item in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def render(project: Path) -> Path:
    out = project / "outputs_phase4"
    audit = pd.read_csv(out / "phase4_curve_audit.csv")
    ns = pd.read_csv(out / "phase4_ns_summary.csv")
    dynamics = pd.read_csv(out / "phase4_factor_dynamics.csv")
    metrics = pd.read_csv(out / "phase4_master_comparison.csv")
    dm = pd.read_csv(out / "phase4_dm_tests.csv")
    primary = metrics[metrics.market.isin(["BAM", "EUROPE", "US_CORE"])]
    table = primary[["market", "model", "horizon", "rmse", "rmse_ratio_vs_persistence",
                     "mae", "bias", "n"]]
    winners = table.loc[table.groupby(["market", "horizon"]).rmse.idxmin()]
    sig = dm[(dm.maturity.astype(str) == "aggregate") & (dm.p_value < .05)]
    robust = (primary[primary.model.ne("PERSISTENCE")]
              .groupby("model").rmse_ratio_vs_persistence.max().lt(1))
    justified = bool(robust.any())
    status = "PHASE 4D JUSTIFIED" if justified else "PHASE 4D NOT YET JUSTIFIED"
    text = f"""# Phase 4 — Cross-market Nelson–Siegel forecasting

## Decision

**{status}**

The decision is based only on the pre-specified independent-market forecasts. A Phase 4D spillover model is justified only if a non-persistence specification improves RMSE in every primary market and horizon; that robustness condition is {'met' if justified else 'not met'}. No Phase 4D model was run.

## Scope and safeguards

Phase 3B remains formally deferred. Existing BAM Phase 1/2 outputs were read as the unchanged domestic benchmark. Europe and the United States were modelled independently; no foreign curve entered another market's predictors. All source rates were explicitly converted from percent per annum to decimal. Horizons J+5, J+10 and J+22 denote subsequent observations on each market's own calendar. Selection did not use macro data, neural networks, boosted trees, or forecasting outcomes to alter the input panels.

## Data audit

{_md(audit)}

Europe is the ECB AAA-rated euro-area central-government nominal zero-coupon spot curve estimated with the Svensson method. Its changing euro-area composition is a definitional caveat, not a missing-data defect. Negative observations were retained. The U.S. primary panel is the fixed complete 3M–10Y Treasury par-yield panel. The 20Y/30Y panel is secondary and begins after the documented long 30-year publication/issuance discontinuity; no maturity was interpolated. BAM remains the existing endogenous derived panel.

## Nelson–Siegel representation

{_md(ns)}

Decay was calibrated separately by market over the same fixed grid. This preserves economically comparable level/slope/curvature representations without imposing a common lambda on structurally different markets. Daily factor dynamics are:

{_md(dynamics)}

Maturity-specific fit errors, lambda-grid diagnostics, PCA loadings and common-date factor correlations are stored as machine-readable tables in `outputs_phase4/`.

## Forecast design

The model set is Persistence, DNS-Kalman, NS-Ridge, NS-ElasticNet, PCA-Ridge and PCA-Ridge-VAR. Every tuning sample is expanding-window and target-purged. Europe uses its longest reliable history from 2004; U.S. full-history training begins in 1990. A second U.S. run begins at the common 2004-09-06 boundary and ends at BAM's 2026-06-18 boundary. Evaluation origins begin in 2022, matching the retained BAM experiment. Overlap-aware HAC bandwidths are derived from actual origin/target dates.

## Primary forecast results

{_md(table)}

Best model by market and horizon:

{_md(winners)}

Aggregate DM rejections at 5%:

{_md(sig) if len(sig) else 'None.'}

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

**{status}**
"""
    path = project / "docs/PHASE4_CROSS_MARKET_NELSON_SIEGEL.md"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(render(Path(".").resolve()))
