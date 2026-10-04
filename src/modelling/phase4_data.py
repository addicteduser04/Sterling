"""Deterministic market-specific curve ingestion and Phase 4A audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling.dns import normalize_rate_units, parse_maturities


EUROPE_TENORS = (
    "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y",
    "10Y", "11Y", "12Y", "13Y", "14Y", "15Y", "16Y", "17Y", "18Y",
    "19Y", "20Y", "30Y",
)
US_CORE_TENORS = ("3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y")
US_EXTENDED_TENORS = US_CORE_TENORS + ("20Y", "30Y")
US_COLUMN_MAP = {
    "3 Mo": "3M", "6 Mo": "6M", "1 Yr": "1Y", "2 Yr": "2Y", "3 Yr": "3Y",
    "5 Yr": "5Y", "7 Yr": "7Y", "10 Yr": "10Y", "20 Yr": "20Y", "30 Yr": "30Y",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_europe_curve(root: Path) -> pd.DataFrame:
    """Load 23 ECB AAA Svensson spot-rate exports and convert percent to decimal."""

    series = []
    for tenor in EUROPE_TENORS:
        path = root / f"ECB Data Portal_{tenor}.csv"
        raw = pd.read_csv(path)
        expected_key = f"YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor}"
        if expected_key not in raw.columns[-1]:
            raise ValueError(f"unexpected ECB series definition in {path.name}")
        frame = pd.DataFrame({
            "Date": pd.to_datetime(raw["DATE"], errors="raise"),
            tenor: pd.to_numeric(raw.iloc[:, -1], errors="raise"),
        }).set_index("Date")
        series.append(frame)
    curve = pd.concat(series, axis=1, join="outer").sort_index()
    if curve.index.has_duplicates or curve.isna().any().any():
        raise ValueError("ECB fixed-maturity panel is not complete and unique")
    parse_maturities(curve.columns)
    return normalize_rate_units(curve, "percent", "ECB AAA spot curve")


def load_us_curve(raw_dir: Path, panel: str = "core") -> pd.DataFrame:
    """Load official annual Treasury par-yield files into a fixed observed panel."""

    frames = [pd.read_csv(path) for path in sorted(raw_dir.glob("daily_treasury_yield_curve_*.csv"))]
    if not frames:
        raise FileNotFoundError("no official Treasury annual files")
    raw = pd.concat(frames, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y", errors="raise")
    raw = raw.sort_values("Date", kind="stable").drop_duplicates("Date", keep="last").set_index("Date")
    renamed = raw.rename(columns=US_COLUMN_MAP)
    tenors = US_CORE_TENORS if panel == "core" else US_EXTENDED_TENORS if panel == "extended" else None
    if tenors is None:
        raise ValueError("US panel must be 'core' or 'extended'")
    curve = renamed.loc[:, tenors].apply(pd.to_numeric, errors="coerce")
    # No maturity is imputed. A fixed panel means retaining only dates on which
    # every selected official observed maturity was published.
    curve = curve.dropna(how="any")
    if panel == "extended":
        # Treasury suspended 30-year issuance in 2002 and resumed it in 2006.
        # Keep the secondary panel contiguous after the largest publication gap.
        gaps = curve.index.to_series().diff().dt.days
        if gaps.max() > 30:
            curve = curve.loc[gaps.idxmax():]
    if curve.index.has_duplicates or not curve.index.is_monotonic_increasing:
        raise ValueError("Treasury dates must be unique and chronological")
    parse_maturities(curve.columns)
    return normalize_rate_units(curve, "percent", f"US Treasury {panel} par curve")


def load_bam_phase4_curve(path: Path) -> pd.DataFrame:
    """Load the existing derived BAM panel without changing historical results."""

    raw = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    columns = [column for column in raw.columns if column.endswith("_x")]
    curve = raw[columns].rename(columns=lambda value: value[:-2])
    return normalize_rate_units(curve, "percent", "existing BAM derived curve")


def curve_audit(market: str, curve: pd.DataFrame, source_unit: str, definition: str) -> dict:
    gaps = curve.index.to_series().diff().dt.days.dropna()
    values = curve.to_numpy(float)
    return {
        "market": market, "definition": definition, "source_unit": source_unit,
        "internal_unit": "decimal", "first_observation": curve.index.min().date().isoformat(),
        "last_observation": curve.index.max().date().isoformat(), "observations": len(curve),
        "maturity_count": curve.shape[1], "maturities": "|".join(curve.columns),
        "median_gap_days": float(gaps.median()), "maximum_gap_days": int(gaps.max()),
        "gaps_over_3_days": int((gaps > 3).sum()), "duplicate_dates": int(curve.index.duplicated().sum()),
        "missing_cells": int(curve.isna().sum().sum()), "negative_cells": int((values < 0).sum()),
        "minimum_decimal_yield": float(np.nanmin(values)), "maximum_decimal_yield": float(np.nanmax(values)),
    }


def build_phase4_data(project: Path, output: Path) -> dict[str, pd.DataFrame]:
    europe = load_europe_curve(project / "data/taux_europe")
    us_core = load_us_curve(project / "data/taux_us/raw", "core")
    us_extended = load_us_curve(project / "data/taux_us/raw", "extended")
    bam = load_bam_phase4_curve(project / "data/combined/bam_ecb_2004.csv")
    (project / "data/taux_europe/processed").mkdir(parents=True, exist_ok=True)
    (project / "data/taux_us/processed").mkdir(parents=True, exist_ok=True)
    europe.to_csv(project / "data/taux_europe/processed/ecb_aaa_spot_curve_decimal.csv")
    us_core.to_csv(project / "data/taux_us/processed/us_treasury_core_decimal.csv")
    us_extended.to_csv(project / "data/taux_us/processed/us_treasury_extended_decimal.csv")
    output.mkdir(parents=True, exist_ok=True)
    audits = pd.DataFrame([
        curve_audit("BAM", bam, "percent in existing derived panel", "BAM endogenous interpolated sovereign curve"),
        curve_audit("EUROPE", europe, "percent per annum", "ECB AAA euro-area central-government Svensson zero-coupon spot curve"),
        curve_audit("US_CORE", us_core, "percent per annum", "US Treasury constant-maturity par yields; fixed 3M-10Y core"),
        curve_audit("US_EXTENDED", us_extended, "percent per annum", "US Treasury constant-maturity par yields; fixed core plus 20Y/30Y"),
    ])
    audits.to_csv(output / "phase4_curve_audit.csv", index=False)
    manifest = []
    for path in sorted((project / "data/taux_us/raw").glob("*.csv")):
        year = path.stem.rsplit("_", 1)[-1]
        manifest.append({
            "path": str(path.relative_to(project)), "year": int(year), "sha256": sha256(path),
            "size_bytes": path.stat().st_size, "retrieval_date": "2026-08-11",
            "source_url": f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv",
            "source": "U.S. Department of the Treasury", "source_unit": "percent per annum",
        })
    pd.DataFrame(manifest).to_csv(project / "data/metadata/us_treasury_source_manifest.csv", index=False)
    pd.DataFrame([{
        "market": "EUROPE", "provider": "European Central Bank",
        "dataset": "YC.B.U2.EUR.4F.G_N_A.SV_C_YM",
        "definition": "AAA euro-area central-government nominal zero-coupon spot curve; Svensson method",
        "frequency": "daily businessweek; end-period", "source_unit": "percent per annum",
        "retrieval_date": "2026-08-11",
        "source_url": "https://data.ecb.europa.eu/data/datasets/YC",
        "methodology_url": "https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html",
    }]).to_csv(project / "data/metadata/ecb_yield_curve_source_manifest.csv", index=False)
    metadata = {
        "panels": {"US_CORE": list(US_CORE_TENORS), "US_EXTENDED": list(US_EXTENDED_TENORS),
                   "EUROPE": list(EUROPE_TENORS)},
        "transform": "explicit percent-to-decimal division by 100; no date or maturity imputation",
        "horizon_semantics": "J+h is h subsequent market observations, not h calendar days",
    }
    (output / "phase4_data_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"BAM": bam, "EUROPE": europe, "US_CORE": us_core, "US_EXTENDED": us_extended}
