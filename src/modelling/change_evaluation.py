"""Evaluation tools for common long-form yield forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


REQUIRED_FORECAST_COLUMNS = {
    "forecast_origin", "target_date", "horizon", "maturity", "actual_yield",
    "forecast_yield", "error", "model", "representation", "variant",
}


def validate_forecast_table(table: pd.DataFrame) -> None:
    missing = REQUIRED_FORECAST_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"forecast table is missing columns: {sorted(missing)}")
    if not table.empty:
        horizons = pd.to_numeric(table["horizon"], errors="coerce")
        if horizons.isna().any() or (horizons <= 0).any() or (horizons % 1 != 0).any():
            raise ValueError("forecast horizons must be positive integer observation steps")


def metric_table(table: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE and bias by model/horizon/maturity plus curve aggregate."""

    validate_forecast_table(table)
    rows: list[dict[str, object]] = []
    keys = ["model", "representation", "variant", "horizon"]
    groups = [(key + (float(maturity),), part) for key, block in table.groupby(keys)
              for maturity, part in block.groupby("maturity")]
    groups += [(key + ("aggregate",), block) for key, block in table.groupby(keys)]
    identity = ["forecast_origin", "target_date", "horizon", "maturity"]
    persistence = table[table.model == "PERSISTENCE"][identity + ["error"]].rename(
        columns={"error": "persistence_error"})
    for labels, part in groups:
        model, representation, variant, horizon, maturity = labels
        error = part.error.to_numpy(float)
        rmse = float(np.sqrt(np.mean(error**2)))
        matched = part.merge(persistence, on=identity, validate="one_to_one")
        base = float(np.sqrt(np.mean(matched.persistence_error.to_numpy(float) ** 2)))
        base_mae = float(np.mean(np.abs(matched.persistence_error.to_numpy(float))))
        mae = float(np.mean(np.abs(error)))
        rows.append({
            "model": model, "representation": representation, "variant": variant,
            "horizon": int(horizon), "maturity": maturity, "n": len(error),
            "mae": mae, "rmse": rmse,
            "bias": float(np.mean(error)), "rmse_ratio_vs_persistence": rmse / base if base else np.nan,
            "rmse_improvement_pct": 100.0 * (base - rmse) / base if base else np.nan,
            "mae_ratio_vs_persistence": mae / base_mae if base_mae else np.nan,
            "mae_improvement_pct": 100.0 * (base_mae - mae) / base_mae if base_mae else np.nan,
            "rate_unit": "decimal",
        })
    return pd.DataFrame(rows)


def overlap_bandwidth(origins: pd.Series | pd.DatetimeIndex,
                      targets: pd.Series | pd.DatetimeIndex) -> int:
    """Maximum evaluation-origin lag whose forecast windows actually overlap.

    Windows are treated as half-open ``(origin, target]`` intervals. Thus a
    later origin exactly equal to the earlier target does not overlap it.
    """

    frame = pd.DataFrame({"origin": pd.to_datetime(origins), "target": pd.to_datetime(targets)})
    frame = frame.drop_duplicates().sort_values("origin").reset_index(drop=True)
    if (frame.target <= frame.origin).any() or frame.origin.duplicated().any():
        raise ValueError("forecast windows require unique origins strictly before targets")
    bandwidth = 0
    for i in range(len(frame)):
        later = frame.index[(frame.index > i) & (frame.origin < frame.target.iloc[i])]
        if len(later):
            bandwidth = max(bandwidth, int(later.max() - i))
    return bandwidth


def overlap_audit(table: pd.DataFrame) -> pd.DataFrame:
    """Machine-readable overlap structure for each evaluation horizon."""

    rows = []
    windows = table[["forecast_origin", "target_date", "horizon"]].drop_duplicates()
    for horizon, part in windows.groupby("horizon"):
        part = part.sort_values("forecast_origin")
        gaps = part.forecast_origin.diff().dt.days.dropna()
        overlaps = []
        values = part.reset_index(drop=True)
        for i, row in values.iterrows():
            overlaps.append(int(((values.index > i) & (values.forecast_origin < row.target_date)).sum()))
        rows.append({"horizon": int(horizon), "n_origins": len(part),
                     "median_origin_gap_days": float(gaps.median()),
                     "min_origin_gap_days": int(gaps.min()), "max_origin_gap_days": int(gaps.max()),
                     "overlap_bandwidth": overlap_bandwidth(part.forecast_origin, part.target_date),
                     "origins_overlapping_next": int(np.sum(np.asarray(overlaps) > 0)),
                     "mean_subsequent_overlaps": float(np.mean(overlaps))})
    return pd.DataFrame(rows)


def dm_hac(error_model: np.ndarray, error_benchmark: np.ndarray, horizon: int,
           bandwidth: int | None = None) -> dict[str, float | int]:
    """Two-sided DM test with Bartlett HAC at a supplied dependence bandwidth.

    Inputs are one loss-bearing time series each. The loss differential is
    squared model error minus squared benchmark error, so a negative statistic
    favours the model.
    """

    model = np.asarray(error_model, float)
    benchmark = np.asarray(error_benchmark, float)
    valid = np.isfinite(model) & np.isfinite(benchmark)
    differential = model[valid] ** 2 - benchmark[valid] ** 2
    n = len(differential)
    lag_count = horizon - 1 if bandwidth is None else int(bandwidth)
    if lag_count < 0:
        raise ValueError("HAC bandwidth cannot be negative")
    if n < max(10, lag_count + 3):
        return {"statistic": np.nan, "p_value": np.nan, "mean_loss_difference": np.nan, "n": n}
    centered = differential - differential.mean()
    variance = float(centered @ centered / n)
    max_lag = min(lag_count, n - 2)
    for lag in range(1, max_lag + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        variance += 2.0 * (1.0 - lag / (max_lag + 1)) * covariance
    statistic = float(differential.mean() / np.sqrt(max(variance / n, 1e-30)))
    return {
        "statistic": statistic,
        "p_value": float(2.0 * student_t.sf(abs(statistic), df=n - 1)),
        "mean_loss_difference": float(differential.mean()), "n": n,
        "hac_bandwidth": max_lag,
    }


def dm_vs_persistence(table: pd.DataFrame) -> pd.DataFrame:
    """DM comparisons using bandwidth derived from actual forecast-window overlap."""

    validate_forecast_table(table)
    identity = ["forecast_origin", "target_date", "horizon", "maturity"]
    base = table[table.model == "PERSISTENCE"][identity + ["error"]].rename(columns={"error": "base_error"})
    rows: list[dict[str, object]] = []
    for model, model_rows in table[table.model != "PERSISTENCE"].groupby("model"):
        matched = model_rows.merge(base, on=identity, validate="one_to_one")
        for (horizon, maturity), part in matched.groupby(["horizon", "maturity"]):
            bandwidth = overlap_bandwidth(part.forecast_origin, part.target_date)
            rows.append({"model": model, "horizon": int(horizon), "maturity": maturity,
                         **dm_hac(part.error, part.base_error, int(horizon), bandwidth)})
        for horizon, part in matched.groupby("horizon"):
            # One observation per origin avoids treating nine tenors as independent.
            losses = part.assign(model_loss=part.error**2, base_loss=part.base_error**2).groupby("forecast_origin")[["model_loss", "base_loss"]].mean()
            signed_model = np.sqrt(losses.model_loss)
            signed_base = np.sqrt(losses.base_loss)
            windows = part[["forecast_origin", "target_date"]].drop_duplicates()
            bandwidth = overlap_bandwidth(windows.forecast_origin, windows.target_date)
            rows.append({"model": model, "horizon": int(horizon), "maturity": "aggregate",
                         **dm_hac(signed_model, signed_base, int(horizon), bandwidth)})
    return pd.DataFrame(rows)
