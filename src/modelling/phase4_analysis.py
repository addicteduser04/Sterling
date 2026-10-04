"""Phase 4 descriptive diagnostics, cross-market comparisons, and report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.modelling import dns
from src.modelling.phase4_data import build_phase4_data


GRID = np.asarray((0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0))


def _factor_diagnostics(market: str, curve: pd.DataFrame, out: Path):
    maturities = dns.parse_maturities(curve.columns)
    model, factors, decay = dns.calibrate_dns(
        curve.to_numpy(), curve.index, maturities, np.ones(len(maturities)), GRID
    )
    fitted = factors @ model.loadings.T
    resid = curve.to_numpy() - fitted
    factor = pd.DataFrame(factors, index=curve.index, columns=dns.FACTOR_NAMES)
    factor.index.name = "Date"
    factor.to_csv(out / f"{market.lower()}_ns_factors.csv")
    fit = pd.DataFrame({
        "market": market, "maturity": curve.columns,
        "rmse": np.sqrt(np.mean(resid ** 2, axis=0)),
        "mae": np.mean(np.abs(resid), axis=0),
    })
    dynamics = pd.DataFrame({
        "market": market, "factor": factor.columns,
        "level_std": factor.std().to_numpy(),
        "change_std": factor.diff().std().to_numpy(),
        "lag1_autocorrelation": [factor[c].autocorr(1) for c in factor],
    })
    pca = PCA(3).fit(curve)
    load = pd.DataFrame(pca.components_.T, index=curve.columns, columns=["PC1", "PC2", "PC3"])
    load.index.name = "maturity"
    load.to_csv(out / f"{market.lower()}_pca_loadings.csv")
    decay.assign(market=market).to_csv(out / f"{market.lower()}_lambda_grid.csv", index=False)
    summary = {"market": market, "lambda": model.decay_time,
               "fit_rmse": float(np.sqrt(np.mean(resid ** 2))),
               "pc1_variance": float(pca.explained_variance_ratio_[0]),
               "pc3_cumulative": float(pca.explained_variance_ratio_.sum())}

    daily_rmse = np.sqrt(np.mean(resid ** 2, axis=1))
    dates = [daily_rmse.argmin(), np.argsort(daily_rmse)[len(daily_rmse)//2], daily_rmse.argmax()]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, label in zip(dates, ("best fit", "median fit", "worst fit")):
        ax.plot(maturities, curve.iloc[i], "o", label=f"{label}: {curve.index[i].date()}")
        ax.plot(maturities, fitted[i], "-")
    ax.set(title=f"{market}: representative Nelson–Siegel fits", xlabel="Maturity (years)", ylabel="Yield (decimal)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / f"fig_{market.lower()}_representative_fits.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5)); factor.plot(ax=ax)
    ax.set(title=f"{market}: Nelson–Siegel factors", ylabel="Decimal"); fig.tight_layout()
    fig.savefig(out / f"fig_{market.lower()}_factor_history.png", dpi=160); plt.close(fig)
    return factor, fit, dynamics, summary


def _forecast_tables(project: Path, out: Path):
    locations = {"BAM": project / "outputs_phase1", "EUROPE": out / "europe_full",
                 "US_CORE": out / "us_core_full", "US_COMMON": out / "us_core_common"}
    metrics, dm = [], []
    for market, path in locations.items():
        m = pd.read_csv(path / "phase1_metrics.csv"); m.insert(0, "market", market); metrics.append(m)
        d = pd.read_csv(path / "phase1_dm_vs_persistence.csv"); d.insert(0, "market", market); dm.append(d)
    metrics = pd.concat(metrics, ignore_index=True); dm = pd.concat(dm, ignore_index=True)
    metrics.to_csv(out / "phase4_forecast_metrics.csv", index=False)
    dm.to_csv(out / "phase4_dm_tests.csv", index=False)
    aggregate = metrics[(metrics.maturity.astype(str).eq("aggregate")) & metrics.market.ne("BAM")].copy()
    # BAM's retained Phase 2 strict-common table is the frozen domestic result.
    # Reconstruct only its six pre-specified Phase 4 benchmark rows; do not
    # silently replace it with the broader all-available Phase 1 aggregation.
    strict = pd.read_csv(project / "outputs_phase2/phase2_master_strict.csv").set_index("model")
    bam_rows = []
    for model in ("PERSISTENCE", "DNS_KALMAN_OU", "NS_RIDGE", "NS_ELASTICNET",
                  "PCA_RIDGE", "PCA_RIDGE_VAR"):
        for horizon in (5, 10, 22):
            bam_rows.append({"market": "BAM", "model": model, "representation": "retained_phase2",
                             "variant": "strict_common", "horizon": horizon, "maturity": "aggregate",
                             "n": np.nan, "mae": np.nan, "rmse": strict.loc[model, f"J+{horizon} RMSE"],
                             "bias": np.nan, "rmse_ratio_vs_persistence": strict.loc[model, f"J+{horizon} vs RW"],
                             "rmse_improvement_pct": 100 * (1 - strict.loc[model, f"J+{horizon} vs RW"]),
                             "mae_ratio_vs_persistence": np.nan, "mae_improvement_pct": np.nan,
                             "rate_unit": "decimal"})
    aggregate = pd.concat([aggregate, pd.DataFrame(bam_rows)], ignore_index=True)
    aggregate.to_csv(out / "phase4_master_comparison.csv", index=False)
    return aggregate, dm


def _forecast_figures(project: Path, out: Path, aggregate: pd.DataFrame):
    main = aggregate[aggregate.market.isin(["BAM", "EUROPE", "US_CORE"])]
    for h in (5, 10, 22):
        x = main[main.horizon.eq(h)].pivot(index="model", columns="market", values="rmse_ratio_vs_persistence")
        ax = x.plot.bar(figsize=(9, 4.5)); ax.axhline(1, color="black", lw=1)
        ax.set(title=f"Cross-market RMSE ratio at J+{h}", ylabel="RMSE / persistence RMSE")
        ax.figure.tight_layout(); ax.figure.savefig(out / f"fig_rmse_ratio_j{h}.png", dpi=160); plt.close(ax.figure)
    for market, folder in (("BAM", project / "outputs_phase1"), ("EUROPE", out / "europe_full"), ("US_CORE", out / "us_core_full")):
        f = pd.read_csv(folder / "phase1_forecasts_long.csv", parse_dates=["forecast_origin"])
        f = f[f.horizon.eq(22)]
        losses = f.groupby(["forecast_origin", "model"]).error.apply(lambda x: np.mean(x*x)).unstack()
        diff = losses.drop(columns="PERSISTENCE").sub(losses.PERSISTENCE, axis=0).cumsum()
        ax = diff.plot(figsize=(9, 4.5)); ax.axhline(0, color="black", lw=1)
        ax.set(title=f"{market}: cumulative squared-loss differential, J+22", ylabel="Model minus persistence")
        ax.figure.tight_layout(); ax.figure.savefig(out / f"fig_{market.lower()}_cumulative_loss_j22.png", dpi=160); plt.close(ax.figure)


def run(project: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    curves = build_phase4_data(project, out)
    factors, fits, dynamics, summaries = {}, [], [], []
    for market in ("BAM", "EUROPE", "US_CORE"):
        factors[market], fit, dynamic, summary = _factor_diagnostics(market, curves[market], out)
        fits.append(fit); dynamics.append(dynamic); summaries.append(summary)
    pd.concat(fits).to_csv(out / "phase4_ns_fit_by_maturity.csv", index=False)
    pd.concat(dynamics).to_csv(out / "phase4_factor_dynamics.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "phase4_ns_summary.csv", index=False)
    aligned = pd.concat({k: v for k, v in factors.items()}, axis=1).dropna()
    aligned.corr().to_csv(out / "phase4_factor_correlations.csv")
    fig, ax = plt.subplots(figsize=(7, 6)); im=ax.imshow(aligned.corr(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(9), aligned.columns.map(lambda x: f"{x[0]}-{x[1]}").tolist(), rotation=90)
    ax.set_yticks(range(9), aligned.columns.map(lambda x: f"{x[0]}-{x[1]}").tolist())
    ax.set_title("Common-date NS factor correlations"); fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out / "fig_factor_correlation_heatmap.png", dpi=160); plt.close(fig)
    aggregate, dm = _forecast_tables(project, out)
    _forecast_figures(project, out, aggregate)
    return curves, aggregate, dm, summaries


if __name__ == "__main__":
    run(Path(".").resolve(), Path("outputs_phase4").resolve())
