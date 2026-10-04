"""Real-time macro observation validation and leakage-safe as-of alignment.

Phase 3 deliberately separates reference dates from availability dates.  This
module contains no source-specific assumptions: parsers must first emit the
standard observation schema, after which every merge is governed by
``available_from``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


MACRO_SCHEMA = (
    "series_id", "reference_date", "publication_date", "available_from",
    "value", "vintage", "source", "frequency", "unit", "is_revised",
    "timing_quality", "vintage_quality", "metadata",
)

TIMING_QUALITIES = {
    "EXACT_TIMESTAMP", "VERIFIED_DATE", "CONSERVATIVE_NEXT_ORIGIN",
    "REFERENCE_DATE_ONLY", "UNVERIFIED",
}
VINTAGE_QUALITIES = {
    "REAL_TIME_VINTAGE", "FIRST_RELEASE", "UNREVISED_ADMIN",
    "LATEST_REVISED", "UNKNOWN",
}


@dataclass(frozen=True)
class MacroFeatureBlock:
    """A frozen, ex-ante collection of standardized macro series IDs."""

    name: str
    series_ids: tuple[str, ...]


def validate_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize macro observations without imputing any value."""

    missing = set(MACRO_SCHEMA) - set(observations.columns)
    if missing:
        raise ValueError(f"macro observations missing schema fields: {sorted(missing)}")
    out = observations.loc[:, MACRO_SCHEMA].copy()
    for column in ("reference_date", "publication_date", "available_from"):
        out[column] = pd.to_datetime(out[column], errors="raise")
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    if not np.isfinite(out["value"]).all():
        raise ValueError("macro values must be finite")
    if (out["available_from"] < out["publication_date"]).any():
        raise ValueError("available_from cannot precede publication_date")
    if invalid := set(out["timing_quality"]) - TIMING_QUALITIES:
        raise ValueError(f"invalid timing quality: {sorted(invalid)}")
    if invalid := set(out["vintage_quality"]) - VINTAGE_QUALITIES:
        raise ValueError(f"invalid vintage quality: {sorted(invalid)}")
    if out.duplicated(["series_id", "available_from", "vintage"]).any():
        raise ValueError("duplicate series/availability/vintage observations")
    return out.sort_values(["series_id", "available_from", "vintage"], kind="stable")


def asof_macro_features(
    origins: Sequence[pd.Timestamp],
    observations: pd.DataFrame,
    series_ids: Sequence[str] | None = None,
    max_stale_days: Mapping[str, int | None] | None = None,
) -> pd.DataFrame:
    """Return the last actually available value at each forecast origin.

    No backward fill or interpolation is performed.  When multiple vintages of
    a reference period are available, the most recently published vintage known
    at the origin wins naturally.  ``max_stale_days`` can invalidate old state
    values; omitted/None limits mean that carry-forward is economically valid.
    """

    data = validate_observations(observations)
    origin_index = pd.DatetimeIndex(pd.to_datetime(origins), name="forecast_origin")
    if not origin_index.is_monotonic_increasing or origin_index.has_duplicates:
        raise ValueError("forecast origins must be unique and chronological")
    wanted = list(series_ids) if series_ids is not None else sorted(data.series_id.unique())
    unknown = set(wanted) - set(data.series_id.unique())
    if unknown:
        raise ValueError(f"unknown macro series: {sorted(unknown)}")
    result = pd.DataFrame(index=origin_index, columns=wanted, dtype=float)
    stale_limits = dict(max_stale_days or {})
    left = pd.DataFrame({"forecast_origin": origin_index})
    for series_id in wanted:
        right = data.loc[data.series_id.eq(series_id), ["available_from", "value"]]
        right = right.sort_values("available_from")
        merged = pd.merge_asof(
            left, right, left_on="forecast_origin", right_on="available_from",
            direction="backward", allow_exact_matches=True,
        )
        limit = stale_limits.get(series_id)
        if limit is not None:
            age = (merged.forecast_origin - merged.available_from).dt.total_seconds() / 86400
            merged.loc[age > int(limit), "value"] = np.nan
        result[series_id] = merged.value.to_numpy()
    return result


def matched_complete_origins(curve: pd.DataFrame, macro: pd.DataFrame) -> pd.DatetimeIndex:
    """Origins usable by both matched models on an identical complete sample."""

    common = curve.index.intersection(macro.index)
    usable = curve.loc[common].notna().all(axis=1) & macro.loc[common].notna().all(axis=1)
    return pd.DatetimeIndex(common[usable])
