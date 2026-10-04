"""Reproducible Phase 3A audit and accepted-source processing CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

import pandas as pd

from .macro_data import MACRO_SCHEMA, asof_macro_features, validate_observations


BLOCKS: Final[dict[str, tuple[str, ...]]] = {
    "CURVE": ("bam_yield_curve",),
    "MACRO_DOMESTIC_V1": (
        "bam_policy_rate", "hcp_cpi_headline", "bam_interbank_weighted_average",
    ),
    "MOROCCO_MARKET_V1": (),
    "MACRO_EXTERNAL_V1": ("ecb_aaa_zero_coupon_curve",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_policy_rate(path: Path) -> pd.DataFrame:
    """Parse official BAM decision-date states; percentages become decimals."""

    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["Date"], dayfirst=True, errors="raise")
    values = pd.to_numeric(raw["Taux directeur"].str.rstrip("%").str.replace(",", ".")) / 100
    rows = pd.DataFrame({
        "series_id": "bam_policy_rate", "reference_date": dates,
        "publication_date": dates, "available_from": dates, "value": values,
        "vintage": "unrevised_event_history", "source": "Bank Al-Maghrib",
        "frequency": "event", "unit": "decimal_rate", "is_revised": False,
        "timing_quality": "VERIFIED_DATE", "vintage_quality": "UNREVISED_ADMIN",
        "metadata": "Decision date is used as date-level availability; no intraday BAM origins.",
    })
    return validate_observations(rows)


def parse_cpi(path: Path, release_calendar_path: Path) -> pd.DataFrame:
    """Join latest HCP CPI levels to independently verified first-release dates.

    The value file's ``Mois`` field is a reference period, never an availability
    date.  An incomplete calendar is rejected so a partial reconstruction cannot
    accidentally become a model input.
    """

    raw = pd.read_csv(path)
    if set(raw.columns) != {"Mois", "IPC"}:
        raise ValueError("unexpected HCP CPI value schema")
    reference = pd.to_datetime(
        raw["Mois"].str.extract(r"^(\d{4})M(\d{1,2})$").agg("-".join, axis=1),
        format="%Y-%m", errors="raise",
    ) + pd.offsets.MonthEnd(0)
    values = pd.DataFrame({"reference_date": reference, "value": raw["IPC"]})

    calendar = pd.read_csv(release_calendar_path)
    required = {"reference_period", "publication_date", "source_url", "status"}
    if missing := required - set(calendar.columns):
        raise ValueError(f"CPI release calendar missing fields: {sorted(missing)}")
    calendar["reference_date"] = (
        pd.to_datetime(calendar["reference_period"], format="%Y-%m", errors="raise")
        + pd.offsets.MonthEnd(0)
    )
    calendar["publication_date"] = pd.to_datetime(calendar["publication_date"], errors="raise")
    verified = calendar.loc[calendar["status"].eq("verified_first_release")].copy()
    if verified.duplicated("reference_date").any():
        raise ValueError("duplicate verified CPI release dates")
    joined = values.merge(verified, on="reference_date", how="left", validate="one_to_one")
    missing_periods = joined.loc[joined.publication_date.isna(), "reference_date"]
    if not missing_periods.empty:
        periods = missing_periods.dt.to_period("M").astype(str).tolist()
        raise ValueError(f"CPI first-release calendar incomplete: {periods}")
    if (joined.publication_date <= joined.reference_date).any():
        raise ValueError("CPI publication must follow its reference month")

    rows = pd.DataFrame({
        "series_id": "hcp_cpi_headline",
        "reference_date": joined.reference_date,
        "publication_date": joined.publication_date,
        "available_from": joined.publication_date,
        "value": joined.value,
        "vintage": "first_release_date_with_latest_level",
        "source": "Haut-Commissariat au Plan",
        "frequency": "monthly",
        "unit": "index_2017_100",
        "is_revised": True,
        "timing_quality": "VERIFIED_DATE",
        "vintage_quality": "LATEST_REVISED",
        "metadata": joined.source_url.map(
            lambda url: f"First-release date verified at {url}; value is from latest local extract."
        ),
    })
    return validate_observations(rows)


def parse_interbank_exports(paths: list[Path]) -> pd.DataFrame:
    """Parse non-contiguous official BAM exports without concealing their gap."""

    frames = []
    for path in paths:
        raw = pd.read_csv(path, sep=";", skiprows=2, na_values=["-"])
        raw["source_file"] = path.name
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    raw["reference_date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="raise")
    raw["value"] = pd.to_numeric(
        raw["Taux Moyen Pondéré"].str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False).str.strip(), errors="raise"
    ) / 100
    raw["volume_mad_millions"] = pd.to_numeric(raw["Volume JJ"], errors="coerce")
    raw["outstanding_mad_millions"] = pd.to_numeric(raw["Encours"], errors="coerce")
    raw = raw.sort_values("reference_date", kind="stable").drop_duplicates("reference_date", keep="last")
    available = raw.reference_date + pd.Timedelta(days=1)
    rows = pd.DataFrame({
        "series_id": "bam_interbank_weighted_average",
        "reference_date": raw.reference_date,
        "publication_date": available,
        "available_from": available,
        "value": raw.value,
        "vintage": "current_official_historical_export",
        "source": "Bank Al-Maghrib",
        "frequency": "daily",
        "unit": "decimal_rate",
        "is_revised": True,
        "timing_quality": "CONSERVATIVE_NEXT_ORIGIN",
        "vintage_quality": "UNKNOWN",
        "metadata": raw.apply(
            lambda row: (
                f"{row.source_file}; volume_mad_millions={row.volume_mad_millions}; "
                f"outstanding_mad_millions={row.outstanding_mad_millions}; export has a middle gap"
            ), axis=1,
        ),
    })
    return validate_observations(rows)


def family_audit(project: Path) -> pd.DataFrame:
    """Describe the economic role and safe status of every known local family."""

    rows = [
        ("TAUX", "CURVE", "data/TAUX/raw/Data.xlsm", "data/TAUX/processed/curves_interp(20250903).csv", "target/endogenous BAM yield curve", "archived processed file currently absent; raw preserved"),
        ("IPC", "MACRO_DOMESTIC_V1", "data/IPC/*.csv; data/IPC/*.xlsx", "data/processed/macro/hcp_cpi_observations.csv", "Moroccan CPI/inflation", "113/113 release dates verified; levels remain latest-revised rather than first-release vintages"),
        ("taux_directeur", "MACRO_DOMESTIC_V1", "data/taux_directeur/taux_directeur.csv", "data/processed/macro/bam_policy_rate_observations.csv", "BAM policy decision state", "accepted at date-level; percent converted once to decimal"),
        ("taux_europe", "MACRO_EXTERNAL_V1", "data/taux_europe/ECB Data Portal_*.csv", "data/taux_europe/ECB Data Portal 2004.csv", "ECB AAA euro-area zero-coupon curve", "audited external block; not domestic V1"),
        ("interbank", "MACRO_DOMESTIC_V1", "data/interbank/raw/*.csv", "data/interbank/processed/bam_interbank_observations.csv", "BAM weighted-average interbank rate", "oldest/newest 1000 preserved; 2003-10-25 to 2023-11-09 gap blocks long-sample use"),
        ("bam_weekly", "MACRO_DOMESTIC_V1 candidates", "data/bam_weekly/raw/*.pdf", "data/bam_weekly/processed/*.csv", "liquidity/refinancing and Treasury auctions", "values recovered but REFERENCE_DATE_ONLY; ineligible until release timing is verified"),
        ("processed", "mixed/legacy", "not applicable", "data/processed and archived data/preprocessed", "legacy aligned outputs", "do not reuse without availability-date audit"),
        ("metadata", "provenance", "data/metadata", "data/metadata", "catalogue and checksums", "extend existing metadata; do not duplicate"),
    ]
    columns = ["family", "information_block", "raw_inputs", "processed_outputs", "economic_role", "audit_status"]
    out = pd.DataFrame(rows, columns=columns)
    out["raw_present"] = out.raw_inputs.map(
        lambda value: "n/a" if value in {"none in working tree", "not applicable"} else "see inventory"
    )
    return out


def cpi_release_coverage(value_path: Path, calendar_path: Path) -> pd.DataFrame:
    """Report calendar coverage without making incomplete CPI model-ready."""

    values = pd.read_csv(value_path)
    values["reference_period"] = values["Mois"].str.replace("M", "-", regex=False).map(
        lambda value: "-".join([value.split("-")[0], value.split("-")[1].zfill(2)])
    )
    calendar = pd.read_csv(calendar_path)
    columns = ["reference_period", "publication_date", "source_url", "status"]
    out = values[["reference_period", "IPC"]].merge(
        calendar[columns], on="reference_period", how="left", validate="one_to_one"
    )
    out["release_date_verified"] = out.status.eq("verified_first_release")
    out["value_vintage"] = "latest_extract_not_first_release_verified"
    return out.sort_values("reference_period", kind="stable").reset_index(drop=True)


def build_audit(project: Path, output: Path) -> None:
    catalogue_path = project / "data/metadata/macro_catalogue.csv"
    catalogue = pd.read_csv(catalogue_path)
    output.mkdir(parents=True, exist_ok=True)
    catalogue.to_csv(output / "coverage_table.csv", index=False)
    leakage = catalogue[["variable", "reference_date_rule", "publication_date_rule",
                         "earliest_usable_origin", "leakage_risk"]]
    leakage.to_csv(output / "leakage_table.csv", index=False)
    mapping = catalogue[["variable", "expected_yield_effect", "curve_segment", "expected_horizon"]]
    mapping.to_csv(output / "economic_mapping.csv", index=False)
    family_audit(project).to_csv(output / "family_audit.csv", index=False)
    (output / "information_blocks.json").write_text(
        json.dumps({name: list(series) for name, series in BLOCKS.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    cpi_values = project / "data/IPC/Indice des prix à la consommation (Mensuel) (Base 100 2017)_2026-06-25.csv"
    cpi_calendar = project / "data/metadata/hcp_cpi_release_calendar.csv"
    cpi_release_coverage(cpi_values, cpi_calendar).to_csv(
        output / "cpi_release_coverage.csv", index=False
    )

    raw_policy = project / "data/taux_directeur/taux_directeur.csv"
    processed_dir = project / "data/processed/macro"
    processed_dir.mkdir(parents=True, exist_ok=True)
    policy = parse_policy_rate(raw_policy)
    policy.to_csv(processed_dir / "bam_policy_rate_observations.csv", index=False)
    cpi = parse_cpi(cpi_values, cpi_calendar)
    cpi.to_csv(processed_dir / "hcp_cpi_observations.csv", index=False)
    interbank = parse_interbank_exports(sorted((project / "data/interbank/raw").glob("*.csv")))
    interbank.to_csv(project / "data/interbank/processed/bam_interbank_observations.csv", index=False)

    accepted = pd.concat([policy, cpi, interbank], ignore_index=True)
    monthly_origins = pd.DatetimeIndex(pd.read_csv(
        project / "outputs_phase2/monthly_forecasts_strict_common.csv", usecols=["forecast_origin"]
    ).forecast_origin.drop_duplicates().pipe(pd.to_datetime).sort_values())
    weekly_origins = pd.DatetimeIndex(pd.read_csv(
        project / "outputs_phase2/weekly_forecasts_long.csv", usecols=["forecast_origin"]
    ).forecast_origin.drop_duplicates().pipe(pd.to_datetime).sort_values())
    stale = {"bam_policy_rate": None, "hcp_cpi_headline": 62, "bam_interbank_weighted_average": 7}
    monthly_features = asof_macro_features(monthly_origins, accepted, list(BLOCKS["MACRO_DOMESTIC_V1"]), stale)
    weekly_features = asof_macro_features(weekly_origins, accepted, list(BLOCKS["MACRO_DOMESTIC_V1"]), stale)
    monthly_complete = monthly_features.notna().all(axis=1)
    weekly_complete = weekly_features.notna().all(axis=1)
    pd.DataFrame([
        {"frequency": "monthly", "total_origins": len(monthly_origins), "complete_origins": int(monthly_complete.sum()),
         "sample_reduction": int((~monthly_complete).sum()), "coverage_start": monthly_features.index[monthly_complete].min()},
        {"frequency": "weekly", "total_origins": len(weekly_origins), "complete_origins": int(weekly_complete.sum()),
         "sample_reduction": int((~weekly_complete).sum()), "coverage_start": weekly_features.index[weekly_complete].min()},
    ]).to_csv(output / "domestic_candidate_origin_coverage.csv", index=False)
    pd.concat([
        pd.DataFrame({"frequency": "monthly", "series_id": monthly_features.columns,
                      "missing_count": monthly_features.isna().sum().to_numpy(),
                      "missing_pct": monthly_features.isna().mean().mul(100).to_numpy()}),
        pd.DataFrame({"frequency": "weekly", "series_id": weekly_features.columns,
                      "missing_count": weekly_features.isna().sum().to_numpy(),
                      "missing_pct": weekly_features.isna().mean().mul(100).to_numpy()}),
    ], ignore_index=True).to_csv(output / "domestic_candidate_missingness.csv", index=False)
    pd.DataFrame([
        {"series_id": "bam_policy_rate", "max_stale_days": None, "rule": "administrative state carried until next decision"},
        {"series_id": "hcp_cpi_headline", "max_stale_days": 62, "rule": "monthly release; invalidate after 62 days"},
        {"series_id": "bam_interbank_weighted_average", "max_stale_days": 7, "rule": "daily market rate; gap is never bridged beyond seven days"},
    ]).to_csv(output / "domestic_candidate_staleness.csv", index=False)

    pd.DataFrame([
        {"series_id": "bam_policy_rate", "accepted_timing": True, "timing_quality": "VERIFIED_DATE", "vintage_quality": "UNREVISED_ADMIN", "coverage": "2006-12-19 to 2026-03-19", "unresolved_issue": "intraday decision time absent"},
        {"series_id": "hcp_cpi_headline", "accepted_timing": True, "timing_quality": "VERIFIED_DATE", "vintage_quality": "LATEST_REVISED", "coverage": "2017-01 to 2026-05; 113/113 release dates", "unresolved_issue": "first-release values/vintages not preserved"},
        {"series_id": "bam_interbank_weighted_average", "accepted_timing": True, "timing_quality": "CONSERVATIVE_NEXT_ORIGIN", "vintage_quality": "UNKNOWN", "coverage": "2001-01-01 to 2003-10-24 and 2023-11-10 to 2026-08-09", "unresolved_issue": "7,322-day middle gap; publication timestamps unavailable"},
        {"series_id": "bam_liquidity_recovered", "accepted_timing": False, "timing_quality": "REFERENCE_DATE_ONLY", "vintage_quality": "UNREVISED_ADMIN", "coverage": "2019-10-30 to 2026-08-05", "unresolved_issue": "weekly PDF public-release dates/timestamps not established"},
        {"series_id": "treasury_auctions_recovered", "accepted_timing": False, "timing_quality": "REFERENCE_DATE_ONLY", "vintage_quality": "UNREVISED_ADMIN", "coverage": "2019-10-29 to 2026-08-04", "unresolved_issue": "result-publication dates absent; demanded amounts and settlement dates unavailable"},
    ]).to_csv(output / "phase3a3_recovery_status.csv", index=False)
    missingness = pd.DataFrame([{
        "variable": row.variable,
        "obs_count": len(policy) if row.accepted == "Yes" else 0,
        "missing_pct": 0.0 if row.accepted == "Yes" else None,
        "max_stale_age_days": "unbounded state carry" if row.series_id == "bam_policy_rate" else "not available",
        "treatment": row.treatment,
    } for row in catalogue.itertuples(index=False)])
    missingness.to_csv(output / "missingness_table.csv", index=False)
    sample_accounting = {
        "bam_full_start": "2004-09-06", "bam_full_end": "2026-06-18",
        "accepted_macro_series": ["bam_policy_rate", "hcp_cpi_headline", "bam_interbank_weighted_average"],
        "accepted_macro_features": 3,
        "domestic_v1_frozen": False,
        "accepted_macro_start": str(policy.reference_date.min().date()),
        "accepted_macro_end": str(policy.reference_date.max().date()),
        "policy_observations": len(policy),
        "monthly_origins": len(monthly_origins), "weekly_origins": len(weekly_origins),
        "monthly_complete_origins": int(monthly_complete.sum()),
        "weekly_complete_origins": int(weekly_complete.sum()),
        "earliest_origin_training_observations": None, "effective_common_sample": None,
        "reason_counts_unavailable": (
            "Forecasting is gated: MACRO_DOMESTIC_V1 is not frozen. The interbank history has a "
            "7,322-day gap, while recovered liquidity and auction records lack verified release timing."
        ),
    }
    (output / "sample_accounting.json").write_text(
        json.dumps(sample_accounting, indent=2) + "\n", encoding="utf-8"
    )
    manifest_files = [
        {"path": str(raw_policy.relative_to(project)), "sha256": sha256(raw_policy),
         "parser": "parse_policy_rate", "rows": len(policy)},
        {"path": str(cpi_values.relative_to(project)), "sha256": sha256(cpi_values),
         "parser": "parse_cpi", "rows": len(cpi)},
        {"path": str(cpi_calendar.relative_to(project)), "sha256": sha256(cpi_calendar),
         "parser": "parse_cpi/release calendar", "rows": len(cpi)},
    ]
    for interbank_path in sorted((project / "data/interbank/raw").glob("*.csv")):
        manifest_files.append({
            "path": str(interbank_path.relative_to(project)), "sha256": sha256(interbank_path),
            "parser": "parse_interbank_exports", "rows": 1000,
        })
    manifest = {
        "retrieval_recorded_at": "2026-08-10",
        "files": manifest_files,
        "schema": list(MACRO_SCHEMA),
    }
    metadata_dir = project / "data/metadata"
    (metadata_dir / "macro_source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    weekly_sources = pd.read_csv(project / "data/bam_weekly/raw/source_urls.csv")
    weekly_sources["sha256"] = weekly_sources.local_file.map(
        lambda name: sha256(project / "data/bam_weekly/raw" / name)
    )
    weekly_sources["size_bytes"] = weekly_sources.local_file.map(
        lambda name: (project / "data/bam_weekly/raw" / name).stat().st_size
    )
    weekly_sources.to_csv(metadata_dir / "bam_weekly_source_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_phase3a"))
    args = parser.parse_args()
    build_audit(args.project_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
