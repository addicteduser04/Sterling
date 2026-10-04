"""Causal, compact features for direct yield-curve change forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_LAGS = (1, 2, 5, 10, 22)


@dataclass(frozen=True)
class DirectDataset:
    """A direct-horizon supervised sample with explicit positional metadata."""

    features: pd.DataFrame
    targets: pd.DataFrame
    origin_positions: np.ndarray
    target_positions: np.ndarray


def causal_factor_curve_features(
    factors: pd.DataFrame,
    yields: pd.DataFrame,
    lags: Sequence[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Build interpretable features using information at or before each row only.

    The yield frame is expected to contain the nine observed maturities ordered
    from short to long. Rolling windows include the current observation, which
    is available at the forecast origin; no centred/backward shift is used.
    """

    if not factors.index.equals(yields.index):
        raise ValueError("factors and yields must have identical indexes")
    if factors.shape[1] != 3 or yields.shape[1] < 3:
        raise ValueError("three factors and at least three maturities are required")
    if not factors.index.is_monotonic_increasing or factors.index.has_duplicates:
        raise ValueError("the index must be unique and chronological")

    out = factors.astype(float).add_prefix("factor_")
    factor_changes = factors.diff()
    yield_changes = yields.diff()
    for lag in lags:
        if int(lag) < 1:
            raise ValueError("lags must be positive integers")
        shifted_factor_change = factor_changes.shift(int(lag) - 1)
        shifted_yield_change = yield_changes.shift(int(lag) - 1)
        out = out.join(shifted_factor_change.add_suffix(f"_change_lag{lag}"))
        # Keep cross-sectional size disciplined: short, middle and long points.
        selected = shifted_yield_change.iloc[:, [0, len(yields.columns) // 2, -1]]
        out = out.join(selected.add_suffix(f"_change_lag{lag}"))

    short, middle, long = yields.iloc[:, 0], yields.iloc[:, len(yields.columns) // 2], yields.iloc[:, -1]
    out["yield_short"] = short
    out["yield_middle"] = middle
    out["yield_long"] = long
    out["curve_slope_long_short"] = long - short
    out["curve_curvature"] = 2.0 * middle - short - long
    out["curve_momentum_5"] = yields.mean(axis=1).diff(5)
    out["curve_volatility_22"] = yield_changes.mean(axis=1).rolling(22, min_periods=10).std()
    for column in factors.columns:
        out[f"{column}_mean22"] = factors[column].rolling(22, min_periods=10).mean()
        out[f"{column}_vol22"] = factor_changes[column].rolling(22, min_periods=10).std()
    return out.replace([np.inf, -np.inf], np.nan)


def direct_change_dataset(
    features: pd.DataFrame,
    factors: pd.DataFrame,
    horizon: int,
    last_usable_origin: int | None = None,
) -> DirectDataset:
    """Align ``X(s)`` with ``factor(s+h)-factor(s)`` without future leakage."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not features.index.equals(factors.index):
        raise ValueError("features and factors must have identical indexes")
    n = len(features)
    limit = n - horizon - 1 if last_usable_origin is None else int(last_usable_origin)
    limit = min(limit, n - horizon - 1)
    positions = np.arange(max(limit + 1, 0), dtype=int)
    if len(positions):
        valid = features.iloc[positions].notna().all(axis=1).to_numpy()
        valid &= factors.iloc[positions].notna().all(axis=1).to_numpy()
        valid &= factors.iloc[positions + horizon].notna().all(axis=1).to_numpy()
        positions = positions[valid]
    targets = factors.iloc[positions + horizon].to_numpy() - factors.iloc[positions].to_numpy()
    return DirectDataset(
        features.iloc[positions].copy(),
        pd.DataFrame(targets, index=features.index[positions], columns=factors.columns),
        positions,
        positions + horizon,
    )


def purged_expanding_splits(
    origin_positions: np.ndarray,
    target_positions: np.ndarray,
    n_splits: int = 3,
    min_train: int = 60,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding folds whose training targets predate validation origins."""

    origins = np.asarray(origin_positions, int)
    targets = np.asarray(target_positions, int)
    if len(origins) != len(targets):
        raise ValueError("origin and target position arrays must have equal length")
    if len(origins) < min_train + n_splits:
        return []
    validation_starts = np.linspace(min_train, len(origins) - 1, n_splits + 1, dtype=int)[:-1]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold, start in enumerate(validation_starts):
        end = validation_starts[fold + 1] if fold + 1 < len(validation_starts) else len(origins)
        validation = np.arange(start, end)
        training = np.flatnonzero(targets <= origins[start])
        training = training[training < start]
        if len(training) >= min_train and len(validation):
            folds.append((training, validation))
    return folds
