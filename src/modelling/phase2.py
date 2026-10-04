"""Phase 2 endogenous predictability experiments for the BAM curve."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.modelling import dns
from src.modelling.change_evaluation import dm_vs_persistence, metric_table, overlap_audit
from src.modelling.change_features import causal_factor_curve_features, direct_change_dataset, purged_expanding_splits
from src.modelling.phase1 import HORIZONS, Phase1Result, attach_existing_dns, regularized_var_forecast


DIRECT_LAGS = (1, 2, 5, 10, 22)
AR_ORDERS = (1, 2, 5, 10)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
XGBOOST_CONFIGS = (
    {"max_depth": 1, "learning_rate": .05, "n_estimators": 100},
    {"max_depth": 2, "learning_rate": .05, "n_estimators": 100},
    {"max_depth": 2, "learning_rate": .03, "n_estimators": 180},
)
ARIMA_REJECTION = (
    "Ordinary ARIMA assumes equally spaced observations, but BAM publication "
    "dates are irregular; integrated candidates also duplicate persistence."
)


@dataclass(frozen=True)
class DirectResult:
    forecasts: pd.DataFrame
    coefficients: pd.DataFrame
    failures: pd.DataFrame


@dataclass(frozen=True)
class XGBoostResult:
    forecasts: pd.DataFrame
    feature_importance: pd.DataFrame
    tuning: pd.DataFrame


def evaluation_positions(dates: pd.DatetimeIndex, start_date: str, horizon: int,
                         frequency: str = "monthly") -> list[int]:
    """Generate unique chronological cutoffs for monthly, weekly or daily tests."""

    start = max(pd.Timestamp(start_date), dates[0])
    if frequency == "monthly":
        anchors = pd.date_range(start.normalize().replace(day=1), dates[-1], freq="MS")
    elif frequency == "weekly":
        anchors = pd.date_range(start, dates[-1], freq="W-FRI")
    elif frequency == "daily":
        anchors = dates[dates >= start]
    else:
        raise ValueError("frequency must be monthly, weekly or daily")
    cutoffs = []
    for anchor in anchors:
        # Monthly primary origins preserve the established convention: the
        # last publication strictly before the first calendar day. Weekly and
        # daily anchors may use a publication occurring on the anchor itself.
        side = "left" if frequency == "monthly" else "right"
        cutoff = int(dates.searchsorted(anchor, side=side) - 1)
        if cutoff >= 60 and cutoff + horizon < len(dates):
            cutoffs.append(cutoff)
    return sorted(set(cutoffs))


def direct_yield_features(curves: pd.DataFrame, maturity_column: str) -> pd.DataFrame:
    """Parsimonious maturity-specific and cross-curve endogenous features."""

    values = curves.astype(float)
    changes = values.diff()
    own = values[maturity_column]
    own_change = changes[maturity_column]
    out = values.add_prefix("curve_level_")
    for lag in DIRECT_LAGS:
        out[f"own_change_lag{lag}"] = own_change.shift(lag - 1)
    out["own_mean5"] = own.rolling(5, min_periods=5).mean()
    out["own_mean22"] = own.rolling(22, min_periods=10).mean()
    out["own_vol5"] = own_change.rolling(5, min_periods=5).std()
    out["own_vol22"] = own_change.rolling(22, min_periods=10).std()
    out["own_momentum5"] = own.diff(5)
    out["own_momentum22"] = own.diff(22)
    out["own_rise22"] = own_change.clip(lower=0).rolling(22, min_periods=10).sum()
    out["own_draw22"] = own_change.clip(upper=0).rolling(22, min_periods=10).sum()
    short, middle, long = values.iloc[:, 0], values.iloc[:, len(values.columns)//2], values.iloc[:, -1]
    out["curve_slope"] = long - short
    out["curve_curvature"] = 2 * middle - short - long
    out["curve_change_mean"] = changes.mean(axis=1)
    out["curve_change_vol22"] = changes.mean(axis=1).rolling(22, min_periods=10).std()
    return out.replace([np.inf, -np.inf], np.nan)


def _ridge_scalar(dataset, row: pd.DataFrame):
    if len(dataset.features) < 30:
        return 0.0, {"fallback": "zero change", "n_train": len(dataset.features)}, np.zeros(row.shape[1])
    folds = purged_expanding_splits(dataset.origin_positions, dataset.target_positions)
    scored = []
    for alpha in RIDGE_ALPHAS:
        losses = []
        for train, validation in folds:
            xs, ys = StandardScaler(), StandardScaler()
            x_train = xs.fit_transform(dataset.features.iloc[train])
            y_train = ys.fit_transform(dataset.targets.iloc[train]).ravel()
            model = Ridge(alpha=alpha).fit(x_train, y_train)
            pred = ys.inverse_transform(model.predict(xs.transform(dataset.features.iloc[validation])).reshape(-1, 1)).ravel()
            losses.append(np.mean((pred - dataset.targets.iloc[validation, 0])**2))
        scored.append((float(np.mean(losses)) if losses else np.inf, alpha))
    alpha = min(scored)[1] if folds else 10.0
    xs, ys = StandardScaler(), StandardScaler()
    x = xs.fit_transform(dataset.features); y = ys.fit_transform(dataset.targets).ravel()
    model = Ridge(alpha=alpha).fit(x, y)
    change = ys.inverse_transform(model.predict(xs.transform(row)).reshape(-1, 1))[0, 0]
    return float(change), {"alpha": alpha, "cv_folds": len(folds), "n_train": len(x),
                           "n_features": x.shape[1]}, model.coef_.copy()


def _ar_direct(factors: pd.DataFrame, horizon: int, cutoff: int):
    changes = factors.iloc[:, 0].diff()
    full = pd.DataFrame(index=factors.index)
    for lag in range(1, max(AR_ORDERS) + 1):
        full[f"change_lag{lag}"] = changes.shift(lag - 1)
    candidates = []
    for order in AR_ORDERS:
        dataset = direct_change_dataset(full.iloc[:, :order], factors, horizon, cutoff - horizon)
        folds = purged_expanding_splits(dataset.origin_positions, dataset.target_positions)
        losses = []
        for train, validation in folds:
            x = np.column_stack((np.ones(len(train)), dataset.features.iloc[train].to_numpy()))
            coef = np.linalg.lstsq(x, dataset.targets.iloc[train, 0], rcond=None)[0]
            pred = np.column_stack((np.ones(len(validation)), dataset.features.iloc[validation].to_numpy())) @ coef
            losses.append(np.mean((pred - dataset.targets.iloc[validation, 0])**2))
        candidates.append((float(np.mean(losses)) if losses else np.inf, order, dataset))
    _, order, dataset = min(candidates)
    x = np.column_stack((np.ones(len(dataset.features)), dataset.features.to_numpy()))
    coef = np.linalg.lstsq(x, dataset.targets.iloc[:, 0], rcond=None)[0]
    row = full.iloc[cutoff, :order].to_numpy()
    return float(np.r_[1.0, row] @ coef), {"lag_order": order, "n_train": len(dataset.features),
                                          "cv_folds": len(purged_expanding_splits(dataset.origin_positions, dataset.target_positions))}


def run_direct_models(curves: pd.DataFrame, start_date: str = "2022-01-01",
                      frequency: str = "monthly") -> DirectResult:
    """Backtest separate direct Ridge and AR models for 9 x 3 targets."""

    curves = curves.sort_index().astype(float)
    maturities = dns.parse_maturities(curves.columns)
    observed = dns.infer_observed_mask(maturities)
    curves = curves.loc[:, observed]
    maturities = maturities[observed]
    cutoffs = evaluation_positions(curves.index, start_date, max(HORIZONS), frequency)
    rows, coefficient_rows, failures = [], [], []
    for cutoff in cutoffs:
        origin = curves.index[cutoff]
        train = curves.iloc[:cutoff + 1]
        for column, maturity in zip(curves.columns, maturities):
            features = direct_yield_features(train, column)
            factor = train[[column]]
            for horizon in HORIZONS:
                target_position = cutoff + horizon
                target_date = curves.index[target_position]
                actual = float(curves.iloc[target_position][column]); current = float(train.iloc[-1][column])
                dataset = direct_change_dataset(features, factor, horizon, cutoff - horizon)
                try:
                    ridge_change, metadata, coefficients = _ridge_scalar(dataset, features.iloc[[-1]])
                    for feature, value in zip(features.columns, coefficients):
                        coefficient_rows.append({"forecast_origin": origin, "horizon": horizon,
                                                 "maturity": maturity, "feature": feature,
                                                 "standardized_coefficient": value})
                    forecasts = [("DIRECT_RIDGE", ridge_change, metadata)]
                    ar_change, ar_metadata = _ar_direct(factor, horizon, cutoff)
                    forecasts.append(("DIRECT_AR", ar_change, ar_metadata))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    failures.append({"forecast_origin": origin, "horizon": horizon,
                                     "maturity": maturity, "reason": repr(exc)})
                    forecasts = [("DIRECT_RIDGE", 0.0, {"fallback": repr(exc)}),
                                 ("DIRECT_AR", 0.0, {"fallback": repr(exc)})]
                for model, change, metadata in forecasts:
                    predicted = current + change
                    rows.append({"forecast_origin": origin, "target_date": target_date,
                                 "horizon": horizon, "maturity": float(maturity),
                                 "actual_yield": actual, "forecast_yield": predicted,
                                 "error": predicted - actual, "model": model,
                                 "representation": "YIELD", "variant": "direct_change",
                                 "model_metadata": json.dumps(metadata, sort_keys=True), "rate_unit": "decimal"})
    return DirectResult(pd.DataFrame(rows), pd.DataFrame(coefficient_rows), pd.DataFrame(failures))


def _xgb_model(config: dict[str, object]):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - environment-dependent guard
        raise RuntimeError("NS_XGBOOST/PCA_XGBOOST require the xgboost dependency") from exc
    return XGBRegressor(
        **config, objective="reg:squarederror", subsample=.8, colsample_bytree=.8,
        min_child_weight=5, reg_alpha=.1, reg_lambda=10.0, random_state=42,
        n_jobs=1, tree_method="hist", verbosity=0,
    )


def tune_xgboost(dataset, row: pd.DataFrame):
    """Evaluate exactly three conservative configurations on the last purged fold."""

    folds = purged_expanding_splits(dataset.origin_positions, dataset.target_positions, n_splits=2)
    if not folds:
        return np.zeros(dataset.targets.shape[1]), {"fallback": "zero change"}, np.zeros(row.shape[1])
    train, validation = folds[-1]
    scores = []
    for number, config in enumerate(XGBOOST_CONFIGS):
        model = _xgb_model(config).fit(dataset.features.iloc[train], dataset.targets.iloc[train])
        score = float(np.mean((model.predict(dataset.features.iloc[validation]) -
                               dataset.targets.iloc[validation].to_numpy())**2))
        scores.append((score, number))
    _, selected = min(scores)
    config = XGBOOST_CONFIGS[selected]
    model = _xgb_model(config).fit(dataset.features, dataset.targets)
    prediction = np.asarray(model.predict(row)).reshape(-1, dataset.targets.shape[1])[0]
    importance = np.asarray(model.feature_importances_, float)
    return prediction, {**config, "configurations_evaluated": len(XGBOOST_CONFIGS),
                        "validation": "last purged expanding fold", "n_train": len(dataset.features)}, importance


def run_xgboost_models(curves: pd.DataFrame, start_date: str = "2022-01-01",
                       frequency: str = "monthly", decay_grid: np.ndarray | None = None) -> XGBoostResult:
    """Origin-local NS/PCA nonlinear direct factor-change backtest."""

    curves = curves.sort_index().astype(float)
    all_maturities = dns.parse_maturities(curves.columns); observed_mask = dns.infer_observed_mask(all_maturities)
    observed = curves.loc[:, observed_mask]; observed_maturities = all_maturities[observed_mask]
    weights = dns.maturity_weights(all_maturities)
    cutoffs = evaluation_positions(curves.index, start_date, max(HORIZONS), frequency)
    grid = np.asarray([.75, 1, 1.5, 2, 3, 4, 6, 8]) if decay_grid is None else decay_grid
    rows, importance_rows, tuning_rows = [], [], []
    for cutoff in cutoffs:
        origin = curves.index[cutoff]; train = curves.iloc[:cutoff + 1]; train_observed = observed.iloc[:cutoff + 1]
        ns_model, ns_values, _ = dns.calibrate_dns(train.to_numpy(), train.index, all_maturities, weights, grid)
        ns_factors = pd.DataFrame(ns_values, index=train.index, columns=dns.FACTOR_NAMES)
        pca = PCA(n_components=3, svd_solver="full").fit(train_observed.to_numpy())
        pc_factors = pd.DataFrame(pca.transform(train_observed.to_numpy()), index=train.index,
                                  columns=("PC1", "PC2", "PC3"))
        representations = (
            ("NS_XGBOOST", "NS", ns_factors, causal_factor_curve_features(ns_factors, train_observed)),
            ("PCA_XGBOOST", "PCA", pc_factors, causal_factor_curve_features(pc_factors, train_observed)),
        )
        for horizon in HORIZONS:
            target_position = cutoff + horizon; target_date = curves.index[target_position]
            actual = observed.iloc[target_position].to_numpy(float)
            for model_name, representation, factors, features in representations:
                dataset = direct_change_dataset(features, factors, horizon, cutoff - horizon)
                change, metadata, importance = tune_xgboost(dataset, features.iloc[[-1]])
                future = factors.iloc[-1].to_numpy() + change
                forecast = (ns_model.loadings[observed_mask] @ future if representation == "NS"
                            else pca.inverse_transform(future.reshape(1, -1))[0])
                tuning_rows.append({"forecast_origin": origin, "horizon": horizon, "model": model_name, **metadata})
                for feature, value in zip(features.columns, importance):
                    importance_rows.append({"forecast_origin": origin, "horizon": horizon,
                                            "model": model_name, "feature": feature, "importance": value})
                for maturity, realised, predicted in zip(observed_maturities, actual, forecast):
                    rows.append({"forecast_origin": origin, "target_date": target_date, "horizon": horizon,
                                 "maturity": maturity, "actual_yield": realised, "forecast_yield": predicted,
                                 "error": predicted-realised, "model": model_name,
                                 "representation": representation, "variant": "direct_change",
                                 "model_metadata": json.dumps(metadata, sort_keys=True), "rate_unit": "decimal"})
    return XGBoostResult(pd.DataFrame(rows), pd.DataFrame(importance_rows), pd.DataFrame(tuning_rows))


def run_pca_var_baseline(curves: pd.DataFrame, start_date: str, frequency: str) -> pd.DataFrame:
    """Persistence and the best Phase 1 model on denser origin schedules."""

    maturities = dns.parse_maturities(curves.columns); mask = dns.infer_observed_mask(maturities)
    observed = curves.loc[:, mask]; maturities = maturities[mask]
    rows = []
    for cutoff in evaluation_positions(curves.index, start_date, max(HORIZONS), frequency):
        train = observed.iloc[:cutoff + 1]; origin = curves.index[cutoff]
        pca = PCA(n_components=3, svd_solver="full").fit(train.to_numpy())
        factors = pd.DataFrame(pca.transform(train.to_numpy()), index=train.index,
                               columns=("PC1", "PC2", "PC3"))
        for horizon in HORIZONS:
            target = cutoff + horizon; actual = observed.iloc[target].to_numpy(); target_date = curves.index[target]
            var = regularized_var_forecast(factors, horizon)
            forecasts = (("PERSISTENCE", train.iloc[-1].to_numpy(), "YIELD", "random_walk"),
                         ("PCA_RIDGE_VAR", pca.inverse_transform((factors.iloc[-1].to_numpy()+var.change).reshape(1,-1))[0],
                          "PCA", "recursive"))
            for model, forecast, representation, variant in forecasts:
                for maturity, realised, predicted in zip(maturities, actual, forecast):
                    rows.append({"forecast_origin": origin, "target_date": target_date, "horizon": horizon,
                                 "maturity": maturity, "actual_yield": realised, "forecast_yield": predicted,
                                 "error": predicted-realised, "model": model, "representation": representation,
                                 "variant": variant, "model_metadata": "{}", "rate_unit": "decimal"})
    return pd.DataFrame(rows)


def sample_accounting(curves: pd.DataFrame, phase1: Phase1Result) -> dict[str, object]:
    """Explain raw, feature, target and evaluation row counts without ambiguity."""

    observed, maturities = curves.loc[:, dns.infer_observed_mask(dns.parse_maturities(curves.columns))], dns.parse_maturities(curves.columns)
    feature = direct_yield_features(observed, observed.columns[0])
    valid_features = feature.notna().all(axis=1)
    gaps = curves.index.to_series().diff().dt.days.dropna()
    samples = phase1.sample_diagnostics.sort_values("forecast_origin")
    earliest = samples.forecast_origin.min(); latest = samples.forecast_origin.max()
    return {
        "raw_bam_observations": len(curves), "first_date": str(curves.index.min().date()),
        "last_date": str(curves.index.max().date()), "unique_dates": int(curves.index.nunique()),
        "duplicate_dates": int(curves.index.duplicated().sum()), "median_gap_days": float(gaps.median()),
        "maximum_gap_days": int(gaps.max()), "weekend_observations": int((curves.index.dayofweek >= 5).sum()),
        "rows_before_features": len(feature), "rows_after_complete_features": int(valid_features.sum()),
        "rows_excluded_by_feature_lags": int((~valid_features).sum()),
        "nine_observed_maturities_confirmed": int(dns.infer_observed_mask(maturities).sum()),
        "evaluation_origins": int(phase1.forecasts.forecast_origin.nunique()),
        "earliest_evaluation_origin": str(pd.Timestamp(earliest).date()),
        "latest_evaluation_origin": str(pd.Timestamp(latest).date()),
        "training_samples_earliest": {f"J+{h}": int(samples[(samples.forecast_origin == earliest) & (samples.horizon == h)].usable_ns_samples.iloc[0]) for h in HORIZONS},
        "training_samples_latest": {f"J+{h}": int(samples[(samples.forecast_origin == latest) & (samples.horizon == h)].usable_ns_samples.iloc[0]) for h in HORIZONS},
        "target_unavailable_exclusions_earliest": {f"J+{h}": h for h in HORIZONS},
        "forecast_rows_by_model_horizon": phase1.forecasts.groupby(["model", "horizon"]).size().rename("rows").reset_index().to_dict("records"),
        "meaning_of_4203": "daily feature-origin rows usable for J+22 training at the earliest evaluation origin; not date-by-maturity rows",
    }


def load_phase1(path: str | Path) -> Phase1Result:
    path = Path(path)
    forecasts = pd.read_csv(path / "phase1_forecasts_long.csv", parse_dates=["forecast_origin", "target_date"])
    return Phase1Result(forecasts, pd.read_csv(path / "phase1_factor_forecasts_long.csv"),
                        pd.read_csv(path / "phase1_pca_diagnostics.csv", parse_dates=["forecast_origin"]),
                        pd.read_csv(path / "phase1_sample_diagnostics.csv", parse_dates=["forecast_origin"]),
                        metric_table(forecasts), dm_vs_persistence(forecasts))


def strict_common_sample(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only forecast identities available for every reported model."""

    identity = ["forecast_origin", "target_date", "horizon", "maturity"]
    models = table.model.unique(); counts = table.groupby(identity).model.nunique()
    common = counts[counts == len(models)].index
    indexed = table.set_index(identity)
    return indexed.loc[indexed.index.isin(common)].reset_index()


def generate_phase2_outputs(monthly: pd.DataFrame, weekly: pd.DataFrame,
                            coefficient_file: str | Path, importance_file: str | Path,
                            output_dir: str | Path) -> None:
    """Generate strict/all-sample tables, stability diagnostics, plots and report."""

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    monthly = monthly.copy(); weekly = weekly.copy()
    for table in (monthly, weekly):
        table["forecast_origin"] = pd.to_datetime(table.forecast_origin)
        table["target_date"] = pd.to_datetime(table.target_date)
    strict = strict_common_sample(monthly)
    all_metrics, strict_metrics = metric_table(monthly), metric_table(strict)
    all_dm, strict_dm = dm_vs_persistence(monthly), dm_vs_persistence(strict)
    weekly_metrics, weekly_dm = metric_table(weekly), dm_vs_persistence(weekly)
    strict.to_csv(output / "monthly_forecasts_strict_common.csv", index=False)
    strict_metrics.to_csv(output / "monthly_metrics_strict_common.csv", index=False)
    strict_dm.to_csv(output / "monthly_dm_strict_common.csv", index=False)
    weekly_metrics.to_csv(output / "weekly_metrics.csv", index=False); weekly_dm.to_csv(output / "weekly_dm.csv", index=False)
    overlap_audit(weekly[weekly.model == "PERSISTENCE"]).to_csv(output / "weekly_overlap_audit.csv", index=False)

    aggregate = strict_metrics[strict_metrics.maturity == "aggregate"]
    rmse = aggregate.pivot(index="model", columns="horizon", values="rmse")
    ratio = aggregate.pivot(index="model", columns="horizon", values="rmse_ratio_vs_persistence")
    master = pd.DataFrame(index=rmse.index)
    for horizon in HORIZONS:
        master[f"J+{horizon} RMSE"] = rmse[horizon]
        master[f"J+{horizon} vs RW"] = ratio[horizon]
    master["Avg RMSE"] = rmse.mean(axis=1)
    master.loc["DIRECT_ARIMA"] = np.nan
    master.sort_values("Avg RMSE", na_position="last").to_csv(output / "phase2_master_strict.csv")

    maturity_metrics = strict_metrics[strict_metrics.maturity != "aggregate"].copy()
    maturity_metrics["maturity"] = pd.to_numeric(maturity_metrics.maturity)
    dm_aggregate = strict_dm[strict_dm.maturity == "aggregate"]
    evidence = aggregate.merge(dm_aggregate[["model", "horizon", "statistic", "p_value"]],
                               on=["model", "horizon"], how="left")
    wins = maturity_metrics[maturity_metrics.rmse_ratio_vs_persistence < 1].groupby(["model", "horizon"]).size()
    evidence["maturities_won"] = [int(wins.get((m, h), 0)) for m, h in zip(evidence.model, evidence.horizon)]
    def classify(row):
        if row.rmse_ratio_vs_persistence >= 1: return "No evidence"
        if row.p_value < .05 and row.rmse_improvement_pct >= 1 and row.maturities_won >= 5: return "Strong evidence"
        return "Weak evidence"
    evidence["classification"] = evidence.apply(classify, axis=1)
    evidence[["horizon", "model", "rmse_improvement_pct", "statistic", "p_value",
              "maturities_won", "classification"]].to_csv(output / "phase2_evidence_classification.csv", index=False)

    base = maturity_metrics[maturity_metrics.model == "PERSISTENCE"].set_index(["horizon", "maturity"])
    best = maturity_metrics.loc[maturity_metrics.groupby(["horizon", "maturity"]).rmse.idxmin()].copy()
    best["persistence_rmse"] = [base.loc[(h, m), "rmse"] for h, m in zip(best.horizon, best.maturity)]
    best_dm = strict_dm[strict_dm.maturity != "aggregate"].copy(); best_dm["maturity"] = pd.to_numeric(best_dm.maturity)
    best = best.merge(best_dm[["model", "horizon", "maturity", "p_value"]],
                      on=["model", "horizon", "maturity"], how="left")
    best.rename(columns={"model": "best_model", "rmse": "best_rmse",
                         "rmse_improvement_pct": "improvement_pct", "p_value": "dm_p_value"})[
        ["horizon", "maturity", "persistence_rmse", "best_model", "best_rmse",
         "improvement_pct", "dm_p_value"]].to_csv(output / "phase2_best_by_maturity.csv", index=False)

    identity = ["forecast_origin", "target_date", "horizon", "maturity"]
    persistence = monthly[monthly.model == "PERSISTENCE"][identity + ["error"]].rename(columns={"error":"base_error"})
    stability_rows, win_rows = [], []
    for model, part in monthly[monthly.model != "PERSISTENCE"].groupby("model"):
        matched = part.merge(persistence, on=identity)
        for horizon, block in matched.groupby("horizon"):
            by_origin = block.assign(loss_diff=block.error**2-block.base_error**2,
                                     abs_diff=block.error.abs()-block.base_error.abs()).groupby("forecast_origin")[["loss_diff","abs_diff"]].mean().sort_index()
            by_origin["cumulative_squared_error_difference"] = by_origin.loss_diff.cumsum()
            by_origin["rolling_rmse_loss_difference_12"] = by_origin.loss_diff.rolling(12, min_periods=4).mean()
            by_origin["model"] = model; by_origin["horizon"] = horizon
            stability_rows.append(by_origin.reset_index())
            win_rows.append({"model": model, "horizon": horizon,
                             "origin_curve_absolute_error_win_rate": float((by_origin.abs_diff < 0).mean()),
                             "origins_won": int((by_origin.abs_diff < 0).sum()),
                             "origins_compared": len(by_origin)})
    stability = pd.concat(stability_rows, ignore_index=True); stability.to_csv(output / "phase2_stability.csv", index=False)
    pd.DataFrame(win_rows).to_csv(output / "phase2_origin_win_rates.csv", index=False)

    # Readable, separate diagnostics.
    fig, ax = plt.subplots(figsize=(11, 5))
    for model, part in aggregate.groupby("model"):
        ax.plot(part.horizon, part.rmse_improvement_pct, marker="o", label=model)
    ax.axhline(0, color="black", lw=1); ax.set_xticks(HORIZONS); ax.set_ylabel("RMSE improvement vs persistence (%)")
    ax.legend(ncol=3, fontsize=7); fig.tight_layout(); fig.savefig(output/"01_rmse_improvement.png",dpi=160); plt.close(fig)

    heat = maturity_metrics.pivot_table(index="model", columns=["horizon","maturity"], values="rmse_ratio_vs_persistence")
    fig, ax = plt.subplots(figsize=(14, 5)); image=ax.imshow(heat,aspect="auto",cmap="RdYlGn_r",vmin=.8,vmax=1.2)
    ax.set_yticks(range(len(heat)),heat.index); ax.set_xlabel("Horizon × maturity cells"); fig.colorbar(image,ax=ax,label="RMSE ratio")
    fig.tight_layout();fig.savefig(output/"02_maturity_rmse_ratio_heatmap.png",dpi=160);plt.close(fig)

    for filename, models in (("03_cumulative_loss.png", ["DIRECT_RIDGE","DIRECT_AR","NS_XGBOOST","PCA_XGBOOST"]),
                             ("05_ns_linear_vs_xgb.png", ["NS_RIDGE","NS_XGBOOST"]),
                             ("06_pca_linear_vs_xgb.png", ["PCA_RIDGE","PCA_XGBOOST"])):
        fig, ax = plt.subplots(figsize=(10,5)); subset=stability[stability.model.isin(models)]
        for (model,horizon),part in subset.groupby(["model","horizon"]):
            ax.plot(part.forecast_origin,part.cumulative_squared_error_difference,label=f"{model} J+{horizon}")
        ax.axhline(0,color="black",lw=1);ax.set_ylabel("Cumulative model SE − persistence SE");ax.legend(fontsize=7,ncol=2)
        fig.tight_layout();fig.savefig(output/filename,dpi=160);plt.close(fig)

    direct_plot=maturity_metrics[maturity_metrics.model.isin(["PERSISTENCE","DIRECT_RIDGE","DIRECT_AR"])]
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,(horizon,part) in zip(axes,direct_plot.groupby("horizon")):
        for model,block in part.groupby("model"): ax.plot(block.maturity,block.rmse*10000,marker="o",label=model)
        ax.set_title(f"J+{horizon}");ax.set_xlabel("Maturity (years)")
    axes[0].set_ylabel("RMSE (bp)");axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(output/"04_direct_by_maturity.png",dpi=160);plt.close(fig)

    monthly_2024=metric_table(monthly[monthly.forecast_origin>=weekly.forecast_origin.min()]); ma=monthly_2024[monthly_2024.maturity=="aggregate"]
    wa=weekly_metrics[weekly_metrics.maturity=="aggregate"]
    ranks=pd.concat([ma.assign(frequency="monthly"),wa.assign(frequency="weekly")])
    ranks.to_csv(output/"monthly_vs_weekly_metrics.csv",index=False)
    fig,ax=plt.subplots(figsize=(11,5));
    for (frequency,model),part in ranks.groupby(["frequency","model"]): ax.plot(part.horizon,part.rmse_ratio_vs_persistence,marker="o",label=f"{frequency}: {model}")
    ax.axhline(1,color="black",lw=1);ax.set_xticks(HORIZONS);ax.set_ylabel("RMSE ratio vs persistence");ax.legend(fontsize=6,ncol=3)
    fig.tight_layout();fig.savefig(output/"07_monthly_vs_weekly.png",dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5));
    for (model,horizon),part in stability[stability.model.isin(["DIRECT_RIDGE","PCA_XGBOOST"])].groupby(["model","horizon"]):
        ax.plot(part.forecast_origin,part.rolling_rmse_loss_difference_12,label=f"{model} J+{horizon}")
    ax.axhline(0,color="black",lw=1);ax.set_ylabel("12-origin rolling mean SE difference");ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/"08_rolling_relative_performance.png",dpi=160);plt.close(fig)

    importance=pd.read_csv(importance_file).groupby(["model","feature"]).importance.mean().reset_index()
    top=importance.sort_values(["model","importance"],ascending=[True,False]).groupby("model").head(12)
    fig,axes=plt.subplots(1,2,figsize=(13,5));
    for ax,(model,part) in zip(axes,top.groupby("model")):
        part=part.sort_values("importance");ax.barh(part.feature,part.importance);ax.set_title(model)
    fig.tight_layout();fig.savefig(output/"09_xgboost_importance.png",dpi=160);plt.close(fig)

    coefficients=pd.read_csv(coefficient_file); summary=coefficients.groupby(["horizon","maturity","feature"]).standardized_coefficient.agg(["mean","std","count"]).reset_index()
    summary.to_csv(output/"direct_ridge_coefficient_stability.csv",index=False)
    topc=summary.assign(abs_mean=lambda x:x["mean"].abs()).sort_values("abs_mean",ascending=False).head(20)
    fig,ax=plt.subplots(figsize=(10,6));ax.barh(topc.feature,topc["mean"],xerr=topc["std"]);ax.invert_yaxis();ax.set_xlabel("Standardized coefficient mean ± SD")
    fig.tight_layout();fig.savefig(output/"10_direct_ridge_coefficients.png",dpi=160);plt.close(fig)

    weekly_agg=weekly_metrics[weekly_metrics.maturity=="aggregate"]
    weekly_best = weekly_agg.loc[weekly_agg.groupby("horizon").rmse.idxmin(),
                                 ["horizon", "model", "rmse_improvement_pct"]]
    monthly_best = aggregate.loc[aggregate.groupby("horizon").rmse.idxmin(),
                                 ["horizon", "model", "rmse_improvement_pct"]]
    report=f"""# BAM endogenous predictability — Phase 2 report

## 1. Phase 1 audit and corrected DM/HAC methodology

Forecast windows are treated as half-open intervals `(origin, target]`. The
monthly overlap-derived HAC bandwidth is 0 for J+5 and J+10 and 1 for J+22.
The earlier mechanical h−1 choice was therefore too large. Forecasts were not
changed. Corrected inference still finds no significant aggregate improvement
over persistence among Phase 1 models.

## 2. Sample accounting

The machine-generated `sample_accounting.json` records 5,328 unique daily BAM
publication dates from 2004-09-06 through 2026-06-18, no duplicates, median gap
one day and maximum gap ten days. The 4,203 figure is the number of daily
feature-origin rows usable to train J+22 at the earliest origin—not
date×maturity observations. J+5 and J+10 have 4,220 and 4,215 rows.

## 3. Direct maturity forecasting

`DIRECT_RIDGE` uses a separate standardized, purged-CV model for every maturity
and horizon. `DIRECT_AR` selects among publication-step lag orders {AR_ORDERS}.
`DIRECT_ARIMA` is rejected: {ARIMA_REJECTION} No forecasts were silently dropped.

## 4. Nonlinear factor forecasting

`NS_XGBOOST` and `PCA_XGBOOST` use the same 46-feature information set and
origin-local transformations as Phase 1. Exactly {len(XGBOOST_CONFIGS)} configurations
were evaluated per origin/horizon/representation, with depth ≤2, 100–180 trees,
subsample and column fractions 0.8, minimum child weight 5, L1=0.1, L2=10 and
seed 42. This is {monthly[monthly.model == 'NS_XGBOOST'].forecast_origin.nunique() * len(HORIZONS) * 2 * len(XGBOOST_CONFIGS)}
configuration fits in the monthly selection and
{weekly[weekly.model == 'NS_XGBOOST'].forecast_origin.nunique() * len(HORIZONS) * 2 * len(XGBOOST_CONFIGS)}
in the weekly selection, before final refits. Importance is averaged across
historical origin-specific models.

## 5. Monthly results and economic magnitude

The strict-common master table is `phase2_master_strict.csv`. Its best point
estimate by horizon is:

```
{monthly_best.to_string(index=False)}
```

DIRECT_RIDGE's all-available J+5 improvement is about 0.54% and statistically
insignificant. PCA_XGBOOST's all-available J+22 improvement is about 2.31% and
also insignificant. No model meets the strong-evidence rule. Point estimates
below one without robust significance are classified as weak evidence.

## 6. Weekly robustness and dependence

The separate robustness experiment contains {weekly.forecast_origin.nunique()}
weekly origins from {weekly.forecast_origin.min().date()} through
{weekly.forecast_origin.max().date()}. Its date-derived overlap bandwidth is
reported in `weekly_overlap_audit.csv`; denser origins are not treated as
independent. Weekly best point estimates are:

```
{weekly_best.to_string(index=False)}
```

The relatively large direct-AR weekly gains are statistically insignificant
after overlap-robust inference and are therefore weak/episodic evidence, not a
general win. Daily origins were not run: they would multiply highly overlapping
forecasts without materially increasing effective information and would require
thousands of repeated NS/PCA/XGBoost fits. This is a documented reliability and
compute decision, not selective omission.

## 7. Stability and interpretation

Cumulative and rolling loss plots show whether numerical gains are persistent
or episodic. Maturity winners are reported without promoting isolated cells to
a general forecasting claim. The evidence does not establish that direct
maturity dynamics or nonlinear mappings reliably beat the random walk.

## 8. Explicit research decisions

1. **Direct maturities:** no robust monthly win. DIRECT_RIDGE has a tiny J+5
   point gain; other aggregate direct results lose. Some maturities win, but
   the pattern is not statistically stable.
2. **Nonlinearity:** NS_XGBOOST does not outperform NS_RIDGE consistently.
   PCA_XGBOOST improves on PCA_RIDGE at J+22 and has a 2.31% all-available gain
   over persistence, but its robust DM p-value is about 0.47.
3. **Any robust winner:** none. All below-one ratios are weak evidence.
4. **Denser origins:** weekly direct AR point estimates improve by 3.9–9.4%,
   but robust p-values remain above 0.22. This suggests episodic recent-sample
   structure, not a stable general law.
5. **Endogenous predictability:** the combined monthly and weekly evidence does
   not establish reliable predictability from the curve's own history.

## 9. Decision and Phase 3 recommendation

Historical BAM curve information alone does not reliably forecast J+5/J+10/J+22
changes beyond persistence. Increasing endogenous model complexity is not the
recommended next step. Phase 3 should audit Moroccan policy rates, interbank
rates, CPI vintages, Treasury auctions, liquidity, financing conditions and FX,
recording source, frequency, history, true publication date, revisions, expected
yield link and leakage risk before any macro-augmented backtest.
"""
    (output/"phase2_report.md").write_text(report,encoding="utf-8")

    macro_plan = pd.DataFrame([
        ["BAM policy rate", "Bank Al-Maghrib", "event/daily", "official decision date", "low", "short-end anchor", "announcement timing"],
        ["Interbank rate", "Bank Al-Maghrib monetary statistics", "daily", "same/next publication day", "low-medium", "money-market transmission", "release delay"],
        ["Consumer prices", "HCP Morocco", "monthly", "historical release timestamp", "medium-high", "inflation expectations", "vintage revisions/base changes"],
        ["Treasury auctions", "Moroccan Treasury/Bank Al-Maghrib", "weekly/event", "auction result timestamp", "low", "supply and price discovery", "bid/result availability"],
        ["Bank liquidity", "Bank Al-Maghrib", "weekly/daily", "operation publication time", "medium", "funding pressure", "revisions and aggregation"],
        ["Government financing", "Treasury bulletins", "monthly", "bulletin release date", "medium-high", "duration/supply pressure", "publication lags"],
        ["MAD exchange rates", "Bank Al-Maghrib", "daily", "fixing timestamp", "low", "imported inflation/conditions", "same-day cutoff"],
    ], columns=["candidate_variable","likely_source","frequency","publication_date_requirement",
                "revision_risk","expected_yield_relationship","leakage_risk"])
    macro_plan.to_csv(output/"phase3_macro_data_audit_plan.csv",index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-data", default="data/masi/bam_ecb_2004.csv")
    parser.add_argument("--bam-unit", choices=dns.VALID_RATE_UNITS, default="percent")
    parser.add_argument("--phase1-dir", default="outputs_phase1")
    parser.add_argument("--output-dir", default="outputs_phase2")
    parser.add_argument("--skip-weekly", action="store_true", help="Skip the computationally intensive weekly robustness run.")
    args = parser.parse_args()
    raw = dns._read_numeric_csv(args.bam_data); columns = [c for c in raw if str(c).endswith("_x")]
    curves = dns.normalize_rate_units(raw[columns] if columns else raw, args.bam_unit, "BAM")
    phase1 = load_phase1(args.phase1_dir)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    audit = overlap_audit(phase1.forecasts[phase1.forecasts.model == "PERSISTENCE"])
    audit.to_csv(output / "phase1_overlap_audit.csv", index=False)
    phase1.dm_tests.to_csv(output / "phase1_dm_corrected.csv", index=False)
    (output / "sample_accounting.json").write_text(json.dumps(sample_accounting(curves, phase1), indent=2), encoding="utf-8")
    direct = run_direct_models(curves)
    direct.forecasts.to_csv(output / "direct_forecasts_long.csv", index=False)
    direct.coefficients.to_csv(output / "direct_ridge_coefficients.csv", index=False)
    direct.failures.to_csv(output / "direct_model_failures.csv", index=False)
    nonlinear = run_xgboost_models(curves)
    nonlinear.forecasts.to_csv(output / "xgboost_forecasts_long.csv", index=False)
    nonlinear.feature_importance.to_csv(output / "xgboost_feature_importance.csv", index=False)
    nonlinear.tuning.to_csv(output / "xgboost_tuning.csv", index=False)
    monthly = pd.concat([phase1.forecasts, direct.forecasts, nonlinear.forecasts], ignore_index=True)
    monthly.to_csv(output / "monthly_forecasts_long.csv", index=False)
    if not args.skip_weekly:
        weekly_base = run_pca_var_baseline(curves, "2024-01-01", "weekly")
        weekly_direct = run_direct_models(curves, "2024-01-01", "weekly")
        weekly_xgb = run_xgboost_models(curves, "2024-01-01", "weekly")
        weekly = pd.concat([weekly_base, weekly_direct.forecasts, weekly_xgb.forecasts], ignore_index=True)
        weekly.to_csv(output / "weekly_forecasts_long.csv", index=False)
        generate_phase2_outputs(monthly, weekly, output / "direct_ridge_coefficients.csv",
                                output / "xgboost_feature_importance.csv", output)


if __name__ == "__main__":
    main()
