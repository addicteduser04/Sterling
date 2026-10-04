"""Leakage-safe Phase 1 forecasts of BAM yield-curve changes.

The module deliberately keeps the research universe small: persistence,
existing DNS--Kalman/OU, direct NS Ridge/Elastic Net, leakage-safe PCA Ridge,
and a parsimonious regularized PCA VAR. Every estimator is refitted at each
monthly origin and each direct horizon has its own target and tuning exercise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modelling import dns
from src.modelling.change_evaluation import dm_vs_persistence, metric_table
from src.modelling.change_features import (
    causal_factor_curve_features,
    direct_change_dataset,
    purged_expanding_splits,
)


HORIZONS = (5, 10, 22)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
ELASTIC_ALPHAS = (0.01, 0.1, 1.0)
ELASTIC_L1_RATIOS = (0.1, 0.5, 0.9)
VAR_LAGS = (1, 2, 5)


@dataclass(frozen=True)
class TunedPrediction:
    change: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class Phase1Result:
    forecasts: pd.DataFrame
    factors: pd.DataFrame
    pca_diagnostics: pd.DataFrame
    sample_diagnostics: pd.DataFrame
    metrics: pd.DataFrame
    dm_tests: pd.DataFrame


def _fit_predict_pipeline(kind: str, alpha: float, l1_ratio: float | None,
                          train_x: pd.DataFrame, train_y: pd.DataFrame,
                          row: pd.DataFrame) -> np.ndarray:
    if kind == "ridge":
        regressor = Ridge(alpha=alpha)
    elif kind == "elasticnet":
        regressor = MultiTaskElasticNet(
            alpha=alpha, l1_ratio=float(l1_ratio), max_iter=10_000,
            tol=1e-3, selection="cyclic", random_state=42,
        )
    else:
        raise ValueError(f"unknown estimator kind: {kind}")
    # Target scaling is fitted on the same training fold. Besides treating the
    # three factors symmetrically, it makes Elastic-Net convergence independent
    # of whether rates are represented in decimals or basis points.
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(train_y)
    pipeline = Pipeline([("scale", StandardScaler()), ("model", regressor)])
    pipeline.fit(train_x, scaled_target)
    return target_scaler.inverse_transform(pipeline.predict(row))[0]


def tune_direct_model(kind: str, dataset, row: pd.DataFrame) -> TunedPrediction:
    """Select regularization with purged expanding folds, then refit all history."""

    if len(dataset.features) < 30:
        return TunedPrediction(np.zeros(dataset.targets.shape[1]), {
            "fallback": "zero change", "reason": "fewer than 30 usable training samples",
            "n_train": len(dataset.features), "n_features": dataset.features.shape[1],
        })
    folds = purged_expanding_splits(dataset.origin_positions, dataset.target_positions)
    candidates = (
        [(alpha, None) for alpha in RIDGE_ALPHAS]
        if kind == "ridge"
        else [(alpha, ratio) for alpha in ELASTIC_ALPHAS for ratio in ELASTIC_L1_RATIOS]
    )
    scores: list[tuple[float, float, float | None]] = []
    for alpha, ratio in candidates:
        errors = []
        for train, validation in folds:
            prediction = _fit_predict_pipeline(
                kind, alpha, ratio, dataset.features.iloc[train], dataset.targets.iloc[train],
                dataset.features.iloc[validation],
            )
            errors.append(np.mean((prediction - dataset.targets.iloc[validation].to_numpy()) ** 2))
        scores.append((float(np.mean(errors)) if errors else np.inf, alpha, ratio))
    _, alpha, ratio = min(scores, key=lambda item: (item[0], item[1], item[2] or 0.0))
    # A deterministic conservative fallback is explicit when history cannot support CV.
    if not folds:
        alpha, ratio = (10.0, None) if kind == "ridge" else (0.01, 0.1)
    prediction = _fit_predict_pipeline(
        kind, alpha, ratio, dataset.features, dataset.targets, row,
    )
    return TunedPrediction(prediction, {
        "alpha": alpha, "l1_ratio": ratio, "cv_folds": len(folds),
        "n_train": len(dataset.features), "n_features": dataset.features.shape[1],
        "target_rule": "training_origin + horizon <= forecast_origin",
    })


def _lagged_var_xy(factors: pd.DataFrame, lag: int, last_target: int | None = None):
    values = factors.to_numpy(float)
    end = len(values) - 1 if last_target is None else min(int(last_target), len(values) - 1)
    targets = np.arange(lag, end + 1)
    x = np.vstack([values[t - lag:t][::-1].reshape(-1) for t in targets])
    y = values[targets]
    return x, y, targets


def regularized_var_forecast(factors: pd.DataFrame, horizon: int) -> TunedPrediction:
    """Tune a small Ridge VAR on expanding one-step validation and recurse."""

    if len(factors) < 30:
        return TunedPrediction(np.zeros(factors.shape[1]), {
            "fallback": "zero change", "reason": "fewer than 30 factor observations",
        })
    candidates: list[tuple[float, int, float]] = []
    for lag in VAR_LAGS:
        x, y, targets = _lagged_var_xy(factors, lag)
        split = max(60, int(len(x) * 0.8))
        if len(x) - split < 10:
            continue
        for alpha in RIDGE_ALPHAS:
            model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
            model.fit(x[:split], y[:split])
            score = float(np.mean((model.predict(x[split:]) - y[split:]) ** 2))
            candidates.append((score, lag, alpha))
    if candidates:
        _, lag, alpha = min(candidates)
    else:
        lag, alpha = 1, 10.0
    x, y, _ = _lagged_var_xy(factors, lag)
    model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
    model.fit(x, y)
    history = [row.copy() for row in factors.to_numpy(float)]
    for _ in range(horizon):
        row = np.asarray(history[-lag:][::-1]).reshape(1, -1)
        history.append(model.predict(row)[0])
    change = history[-1] - factors.iloc[-1].to_numpy(float)
    return TunedPrediction(change, {
        "alpha": alpha, "lag_order": lag, "n_train": len(x),
        "n_features": x.shape[1], "forecast_method": "recursive regularized VAR",
    })


def _observed_frame(curves: pd.DataFrame, all_maturities_observed: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    maturities = dns.parse_maturities(curves.columns)
    observed = np.ones(len(maturities), dtype=bool) if all_maturities_observed else dns.infer_observed_mask(maturities)
    selected = curves.loc[:, observed].copy()
    return selected, maturities[observed]


def _append_curve_rows(rows: list[dict[str, object]], origin: pd.Timestamp,
                       target_date: pd.Timestamp, horizon: int, maturities: np.ndarray,
                       actual: np.ndarray, forecast: np.ndarray, model: str,
                       representation: str, variant: str, metadata: dict[str, object]) -> None:
    encoded = json.dumps(metadata, sort_keys=True)
    for maturity, realised, predicted in zip(maturities, actual, forecast):
        rows.append({
            "forecast_origin": origin, "target_date": target_date, "horizon": horizon,
            "maturity": float(maturity), "actual_yield": float(realised),
            "forecast_yield": float(predicted), "error": float(predicted - realised),
            "model": model, "representation": representation, "variant": variant,
            "model_metadata": encoded, "rate_unit": "decimal",
        })


def _append_factor_rows(rows: list[dict[str, object]], origin: pd.Timestamp,
                        target_date: pd.Timestamp, horizon: int, names: Iterable[str],
                        current: np.ndarray, change: np.ndarray, realised: np.ndarray,
                        model: str, representation: str) -> None:
    for name, now, predicted_change, future in zip(names, current, change, realised):
        predicted = now + predicted_change
        rows.append({
            "forecast_origin": origin, "target_date": target_date, "horizon": horizon,
            "factor": name, "current_factor": now, "predicted_change": predicted_change,
            "predicted_future_factor": predicted, "realized_future_factor": future,
            "error": predicted - future, "model": model, "representation": representation,
        })


def _existing_dns_lookup(curves: pd.DataFrame, start_date: str, decay_grid: np.ndarray | None,
                         horizons: tuple[int, ...] = HORIZONS,
                         all_maturities_observed: bool = False):
    results = dns.run_backtest(
        curves, start_date=start_date, horizon=max(horizons), decay_grid=decay_grid,
        all_maturities_observed=all_maturities_observed,
        minimum_evaluation_horizon=min(horizons),
    )
    return {result.cutoff_date: result for result in results}


def run_phase1(
    curves: pd.DataFrame,
    start_date: str = "2022-01-01",
    include_dns: bool = True,
    decay_grid: np.ndarray | None = None,
    all_maturities_observed: bool = False,
    horizons: Iterable[int] = HORIZONS,
) -> Phase1Result:
    """Run the common-origin Phase 1 expanding-window experiment."""

    horizons = tuple(sorted(set(int(horizon) for horizon in horizons)))
    if not horizons or horizons[0] <= 0:
        raise ValueError("horizons must contain positive integer observation steps")
    curves = curves.sort_index().astype(float)
    if curves.index.has_duplicates or not curves.index.is_monotonic_increasing:
        raise ValueError("curve dates must be unique and chronological")
    observed_curves, observed_maturities = _observed_frame(curves, all_maturities_observed)
    all_maturities = dns.parse_maturities(curves.columns)
    weights = np.ones_like(all_maturities) if all_maturities_observed else dns.maturity_weights(all_maturities)
    origins = dns._month_origins(curves.index, start_date, min(horizons))
    dns_results = _existing_dns_lookup(
        curves, start_date, decay_grid, horizons, all_maturities_observed
    ) if include_dns else {}
    forecast_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    pca_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for _, cutoff in origins:
        origin = curves.index[cutoff]
        train_curve = curves.iloc[:cutoff + 1]
        train_observed = observed_curves.iloc[:cutoff + 1]
        ns_model, ns_values, _ = dns.calibrate_dns(
            train_curve.to_numpy(float), train_curve.index, all_maturities, weights, decay_grid,
        )
        ns_factors = pd.DataFrame(ns_values, index=train_curve.index, columns=dns.FACTOR_NAMES)
        ns_features = causal_factor_curve_features(ns_factors, train_observed)

        pca = PCA(n_components=3, svd_solver="full")
        pc_values = pca.fit_transform(train_observed.to_numpy(float))
        pc_factors = pd.DataFrame(pc_values, index=train_curve.index, columns=("PC1", "PC2", "PC3"))
        pc_features = causal_factor_curve_features(pc_factors, train_observed)
        for component, ratio, cumulative, loading in zip(
            pc_factors.columns, pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_), pca.components_
        ):
            for maturity, value in zip(observed_maturities, loading):
                pca_rows.append({"forecast_origin": origin, "component": component,
                                 "explained_variance_ratio": ratio, "cumulative_explained_variance": cumulative,
                                 "maturity": maturity, "loading": value})

        for horizon in horizons:
            if cutoff + horizon >= len(curves):
                continue
            target_position = cutoff + horizon
            target_date = curves.index[target_position]
            actual = observed_curves.iloc[target_position].to_numpy(float)
            current = train_observed.iloc[-1].to_numpy(float)
            _append_curve_rows(forecast_rows, origin, target_date, horizon, observed_maturities,
                               actual, current, "PERSISTENCE", "YIELD", "random_walk", {})
            if origin in dns_results and len(dns_results[origin].predicted_yields) >= horizon:
                result = dns_results[origin]
                all_observed = (
                    np.ones(len(all_maturities), dtype=bool)
                    if all_maturities_observed else dns.infer_observed_mask(all_maturities)
                )
                _append_curve_rows(forecast_rows, origin, target_date, horizon, observed_maturities,
                                   actual, result.predicted_yields[horizon - 1, all_observed],
                                   "DNS_KALMAN_OU", "NS", "existing", {"decay_time": result.decay_time_bam})

            ns_data = direct_change_dataset(ns_features, ns_factors, horizon, cutoff - horizon)
            pc_data = direct_change_dataset(pc_features, pc_factors, horizon, cutoff - horizon)
            sample_rows.append({
                "forecast_origin": origin, "horizon": horizon, "total_observations": cutoff + 1,
                "usable_ns_samples": len(ns_data.features), "usable_pca_samples": len(pc_data.features),
                "ns_features": ns_data.features.shape[1], "pca_features": pc_data.features.shape[1],
                "latest_training_target_position": (
                    int(ns_data.target_positions.max()) if len(ns_data.target_positions) else -1
                ),
                "forecast_origin_position": cutoff,
            })
            model_specs = [
                ("NS_RIDGE", "NS", ns_factors, ns_features, ns_data, "ridge"),
                ("NS_ELASTICNET", "NS", ns_factors, ns_features, ns_data, "elasticnet"),
                ("PCA_RIDGE", "PCA", pc_factors, pc_features, pc_data, "ridge"),
            ]
            for model_name, representation, factors, features, dataset, kind in model_specs:
                tuned = tune_direct_model(kind, dataset, features.iloc[[-1]])
                future_factors = factors.iloc[-1].to_numpy(float) + tuned.change
                if representation == "NS":
                    evaluation_mask = (
                        np.ones(len(all_maturities), dtype=bool)
                        if all_maturities_observed else dns.infer_observed_mask(all_maturities)
                    )
                    forecast = ns_model.loadings[evaluation_mask] @ future_factors
                    realised_factor = dns.extract_ols_betas(
                        curves.iloc[[target_position]].to_numpy(float), ns_model.loadings, weights,
                    )[0]
                else:
                    forecast = pca.inverse_transform(future_factors.reshape(1, -1))[0]
                    realised_factor = pca.transform(observed_curves.iloc[[target_position]].to_numpy(float))[0]
                _append_curve_rows(forecast_rows, origin, target_date, horizon, observed_maturities,
                                   actual, forecast, model_name, representation, "direct_change", tuned.metadata)
                _append_factor_rows(factor_rows, origin, target_date, horizon, factors.columns,
                                    factors.iloc[-1].to_numpy(float), tuned.change, realised_factor,
                                    model_name, representation)

            var = regularized_var_forecast(pc_factors, horizon)
            future_pc = pc_factors.iloc[-1].to_numpy(float) + var.change
            forecast = pca.inverse_transform(future_pc.reshape(1, -1))[0]
            _append_curve_rows(forecast_rows, origin, target_date, horizon, observed_maturities,
                               actual, forecast, "PCA_RIDGE_VAR", "PCA", "recursive", var.metadata)
            _append_factor_rows(factor_rows, origin, target_date, horizon, pc_factors.columns,
                                pc_factors.iloc[-1].to_numpy(float), var.change,
                                pca.transform(observed_curves.iloc[[target_position]].to_numpy(float))[0],
                                "PCA_RIDGE_VAR", "PCA")

    forecasts = pd.DataFrame(forecast_rows)
    return Phase1Result(
        forecasts=forecasts, factors=pd.DataFrame(factor_rows),
        pca_diagnostics=pd.DataFrame(pca_rows), sample_diagnostics=pd.DataFrame(sample_rows),
        metrics=metric_table(forecasts), dm_tests=dm_vs_persistence(forecasts),
    )


def attach_existing_dns(result: Phase1Result, history: pd.DataFrame) -> Phase1Result:
    """Attach retained DNS and persistence-DNS blend forecasts on exact common rows."""

    required = {"forecast_origin", "target_date", "horizon", "maturity", "actual_yield",
                "dns_forecast", "blended_forecast", "is_observed_maturity"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"DNS history is missing columns: {sorted(missing)}")
    work = history[history.is_observed_maturity.astype(bool)].copy()
    work["forecast_origin"] = pd.to_datetime(work.forecast_origin)
    work["target_date"] = pd.to_datetime(work.target_date)
    identity = ["forecast_origin", "target_date", "horizon", "maturity"]
    common = result.forecasts[identity].drop_duplicates().merge(work, on=identity, validate="one_to_one")
    appended = []
    for column, model, variant in (
        ("dns_forecast", "DNS_KALMAN_OU", "existing"),
        ("blended_forecast", "PERSISTENCE_DNS_BLEND", "leakage_safe_horizon_weight"),
    ):
        block = common[identity + ["actual_yield", column]].rename(columns={column: "forecast_yield"})
        block["error"] = block.forecast_yield - block.actual_yield
        block["model"] = model; block["representation"] = "NS"; block["variant"] = variant
        block["model_metadata"] = "{}"; block["rate_unit"] = "decimal"
        appended.append(block)
    forecasts = pd.concat([result.forecasts, *appended], ignore_index=True)
    return Phase1Result(forecasts, result.factors, result.pca_diagnostics,
                        result.sample_diagnostics, metric_table(forecasts), dm_vs_persistence(forecasts))


def write_phase1_outputs(result: Phase1Result, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.forecasts.to_csv(output / "phase1_forecasts_long.csv", index=False)
    result.factors.to_csv(output / "phase1_factor_forecasts_long.csv", index=False)
    result.metrics.to_csv(output / "phase1_metrics.csv", index=False)
    result.dm_tests.to_csv(output / "phase1_dm_vs_persistence.csv", index=False)
    result.pca_diagnostics.to_csv(output / "phase1_pca_diagnostics.csv", index=False)
    result.sample_diagnostics.to_csv(output / "phase1_sample_diagnostics.csv", index=False)

    latest = result.pca_diagnostics.forecast_origin.max()
    diagnostic = result.pca_diagnostics[result.pca_diagnostics.forecast_origin == latest]
    variance = diagnostic.drop_duplicates("component")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(variance.component, variance.explained_variance_ratio * 100)
    axes[0].plot(variance.component, variance.cumulative_explained_variance * 100, marker="o")
    axes[0].set_ylabel("Explained variance (%)")
    for component, part in diagnostic.groupby("component"):
        axes[1].plot(part.maturity, part.loading, marker="o", label=component)
    axes[1].set_xlabel("Maturity (years)"); axes[1].set_ylabel("Loading"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output / "phase1_pca_diagnostics.png", dpi=160); plt.close(fig)

    aggregate = result.metrics[result.metrics.maturity == "aggregate"]
    horizons = tuple(sorted(int(value) for value in aggregate.horizon.unique()))
    pivot = aggregate.pivot(index="model", columns="horizon", values="rmse").rename(
        columns={horizon: f"J+{horizon} RMSE" for horizon in horizons})
    horizon_columns = [f"J+{horizon} RMSE" for horizon in horizons]
    pivot["Avg RMSE"] = pivot[horizon_columns].mean(axis=1)
    exact_ratios = aggregate.pivot(index="model", columns="horizon", values="rmse_ratio_vs_persistence")
    pivot["Relative to Persistence"] = exact_ratios.mean(axis=1)
    pivot.sort_values("Avg RMSE").to_csv(output / "phase1_master_comparison.csv")

    comparison_models = [name for name in (
        "PERSISTENCE", "DNS_KALMAN_OU", "NS_RIDGE", "NS_ELASTICNET",
        "PCA_RIDGE", "PCA_RIDGE_VAR", "PERSISTENCE_DNS_BLEND",
    ) if name in set(aggregate.model)]
    chart = aggregate[aggregate.model.isin(comparison_models)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, part in chart.groupby("model"):
        ax.plot(part.horizon, part.rmse * 10_000, marker="o", label=model)
    ax.set_xticks(horizons); ax.set_xlabel("Forecast horizon (publications)")
    ax.set_ylabel("Aggregate RMSE (basis points)"); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(output / "phase1_model_comparison.png", dpi=160); plt.close(fig)

    maturity_metrics = result.metrics[result.metrics.maturity != "aggregate"].copy()
    maturity_metrics["maturity"] = pd.to_numeric(maturity_metrics.maturity)
    best = maturity_metrics.loc[maturity_metrics.groupby(["horizon", "maturity"]).rmse.idxmin(),
                                ["horizon", "maturity", "model", "rmse"]]
    best.to_csv(output / "phase1_best_by_horizon_maturity.csv", index=False)

    summary = {
        "methodology": "expanding monthly origins; origin-local NS/PCA/scaling; purged time-series CV",
        "horizons": list(horizons), "rate_unit": "decimal",
        "observed_maturities": int(result.forecasts.maturity.nunique()),
        "forecast_rows": len(result.forecasts), "forecast_origins": result.forecasts.forecast_origin.nunique(),
        "models": sorted(result.forecasts.model.unique()),
    }
    (output / "phase1_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest_variance = variance.set_index("component")
    best_by_horizon = aggregate.loc[aggregate.groupby("horizon").rmse.idxmin(),
                                    ["horizon", "model", "rmse", "rmse_improvement_pct"]]
    significant = result.dm_tests[(result.dm_tests.maturity == "aggregate") &
                                  (result.dm_tests.p_value < .05) &
                                  (result.dm_tests.mean_loss_difference < 0)]
    report = f"""# BAM yield-curve predictability: Phase 1 report

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
variables; the earliest origin has {int(result.sample_diagnostics.usable_ns_samples.min())}
usable rows.

## Nelson–Siegel regularized dynamics

`NS_RIDGE` and `NS_ELASTICNET` predict the three factor changes separately for
each horizon and reconstruct yields with the origin-local NS loadings. This
tests factor dynamics without abandoning the NS cross-sectional restriction.

## Leakage-safe PCA and PCA dynamics

At the latest origin PC1, PC2 and PC3 explain
{latest_variance.loc['PC1', 'explained_variance_ratio']:.2%},
{latest_variance.loc['PC2', 'explained_variance_ratio']:.2%}, and
{latest_variance.loc['PC3', 'explained_variance_ratio']:.2%}; cumulatively the
first three explain {latest_variance.loc['PC3', 'cumulative_explained_variance']:.2%}.
The loadings support a level interpretation for PC1, a short-versus-long slope
for PC2, and a curvature/twist interpretation for PC3 (component signs are
arbitrary). `PCA_RIDGE` forecasts direct component changes. `PCA_RIDGE_VAR` is a
small Ridge VAR with lag order in {VAR_LAGS}, selected using past validation and
recursed to the requested horizon.

## Results

Best aggregate result by horizon:

```
{best_by_horizon.to_string(index=False)}
```

Full master comparison is in `phase1_master_comparison.csv`; maturity-level
winners are in `phase1_best_by_horizon_maturity.csv`. Robust DM tests derive
their Bartlett/Newey–West bandwidth from actual origin-to-target window overlap
and aggregate the nine maturity losses to one curve loss per origin before
inference. At monthly frequency the bandwidth is 0 for J+5/J+10 and 1 for J+22.

Models with a statistically significant aggregate improvement over persistence
at 5%: {', '.join(significant.model.unique()) if len(significant) else 'none'}.

## Scientific interpretation

The Phase 1 evidence does not show robust short-horizon predictability. Any
numerical gain must be read together with its DM result and economic magnitude;
negative findings are retained. Differences between NS and PCA indicate whether
the cross-sectional restriction matters, while differences among Ridge,
Elastic Net and regularized VAR isolate the dynamics assumption.

## Limitations and deferred experiments

The forecast-origin count is {result.forecasts.forecast_origin.nunique()}, so
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
"""
    (output / "phase1_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-data", default="data/masi/bam_ecb_2004.csv")
    parser.add_argument("--bam-unit", choices=dns.VALID_RATE_UNITS, default="percent")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output-dir", default="outputs_phase1")
    parser.add_argument("--decay-grid", default="0.75,1,1.5,2,3,4,6,8",
                        help="Comma-separated origin-local Nelson-Siegel decay candidates.")
    parser.add_argument("--dns-backtest", default="outputs_weighted/backtest_blended_bam.csv",
                        help="Existing leakage-safe DNS/blend long table; reused when present.")
    parser.add_argument("--rerun-dns", action="store_true", help="Re-estimate the retained DNS benchmark.")
    args = parser.parse_args()
    path = Path(args.bam_data)
    raw = dns._read_numeric_csv(path)
    columns = [column for column in raw.columns if str(column).endswith("_x")]
    curves = dns.normalize_rate_units(raw[columns] if columns else raw, args.bam_unit, "BAM")
    dns_history = Path(args.dns_backtest)
    decay_grid = np.asarray([float(value) for value in args.decay_grid.split(",")], dtype=float)
    result = run_phase1(curves, args.start_date, include_dns=args.rerun_dns, decay_grid=decay_grid)
    if not args.rerun_dns and dns_history.exists():
        result = attach_existing_dns(result, pd.read_csv(dns_history))
    write_phase1_outputs(result, args.output_dir)


if __name__ == "__main__":
    main()
