"""Phase 5 long-horizon experiment and ex-ante feasibility audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling import dns
from src.modelling.change_evaluation import overlap_bandwidth
from src.modelling.phase1 import run_phase1, write_phase1_outputs
from src.modelling.phase4 import DECAY_GRID, load_market
from src.modelling.phase4_data import load_bam_phase4_curve


HORIZONS = (5, 10, 22, 44, 66, 132, 252)
START_DATE = "2022-01-01"
COMMON_START = "2004-09-06"
COMMON_END = "2026-06-18"


def market_curve(project: Path, market: str, common_period: bool = False) -> pd.DataFrame:
    if market == "BAM":
        curve = load_bam_phase4_curve(project / "data/combined/bam_ecb_2004.csv")
    else:
        curve = load_market(project, market, common_period=False)
    return curve.loc[COMMON_START:COMMON_END] if common_period else curve


def evaluation_windows(index: pd.DatetimeIndex, horizon: int, frequency: str) -> pd.DataFrame:
    first = max(pd.Timestamp(START_DATE), index[0])
    anchors = (pd.date_range(first.normalize().replace(day=1), index[-1], freq="MS")
               if frequency == "monthly" else pd.date_range(first, index[-1], freq="7D"))
    rows = []
    for anchor in anchors:
        cutoff = int(index.searchsorted(anchor, side="left") - 1)
        if cutoff >= 60 and cutoff + horizon < len(index):
            rows.append({"origin": index[cutoff], "target": index[cutoff + horizon], "cutoff": cutoff})
    return pd.DataFrame(rows)


def _classification(n: int, bandwidth: int, earliest_train: int) -> tuple[str, str]:
    effective = n / (bandwidth + 1) if n else 0
    if n == 0 or earliest_train < 30:
        return "NOT_FEASIBLE", "no realized evaluation sample or insufficient direct-training history"
    if effective >= 20 and n >= 36:
        return "FULLY_FEASIBLE", "adequate raw and overlap-adjusted evaluation information"
    if effective >= 8 and n >= 24:
        return "LIMITED_POWER", "usable, but overlap materially limits independent information"
    if effective >= 3 and n >= 12:
        return "DESCRIPTIVE_ONLY", "forecastable, but overlap-adjusted inference is very low-powered"
    return "NOT_FEASIBLE", "effective loss-series information is scientifically insufficient"


def feasibility_audit(project: Path) -> pd.DataFrame:
    rows = []
    for market in ("BAM", "EUROPE", "US_CORE"):
        curve = market_curve(project, market)
        for horizon in HORIZONS:
            monthly = evaluation_windows(curve.index, horizon, "monthly")
            weekly = evaluation_windows(curve.index, horizon, "weekly")
            bandwidth = overlap_bandwidth(monthly.origin, monthly.target) if len(monthly) else 0
            weekly_bandwidth = overlap_bandwidth(weekly.origin, weekly.target) if len(weekly) else 0
            earliest = int(monthly.cutoff.iloc[0] - horizon + 1) if len(monthly) else 0
            latest = int(monthly.cutoff.iloc[-1] - horizon + 1) if len(monthly) else 0
            status, reason = _classification(len(monthly), bandwidth, earliest)
            rows.append({
                "market": market, "horizon": horizon, "total_observations": len(curve),
                "earliest_possible_training_target": curve.index[horizon].date().isoformat(),
                "latest_possible_forecast_origin": curve.index[-horizon-1].date().isoformat(),
                "monthly_evaluation_origins": len(monthly), "weekly_evaluation_origins": len(weekly),
                "training_rows_earliest_origin": earliest, "training_rows_latest_origin": latest,
                "realized_out_of_sample_forecasts": len(monthly),
                "monthly_maximum_overlap_lag": bandwidth, "weekly_maximum_overlap_lag": weekly_bandwidth,
                "monthly_effective_loss_length": round(len(monthly) / (bandwidth + 1), 2),
                "weekly_effective_loss_length": round(len(weekly) / (weekly_bandwidth + 1), 2),
                "overlap_structure": "actual half-open intervals (origin, target]",
                "hac_rule": "date-derived maximum overlap in loss-sequence units",
                "classification": status, "justification": reason,
            })
    return pd.DataFrame(rows)


def run_market(project: Path, market: str, output: Path, common_period: bool = False) -> None:
    curve = market_curve(project, market, common_period)
    result = run_phase1(curve, start_date=START_DATE, include_dns=True, decay_grid=DECAY_GRID,
                        all_maturities_observed=market != "BAM", horizons=HORIZONS)
    write_phase1_outputs(result, output)
    from src.modelling.change_evaluation import overlap_audit
    audit = overlap_audit(result.forecasts)
    audit.insert(0, "market", market); audit.insert(1, "origin_frequency", "monthly")
    audit["hac_rule"] = "date-derived maximum overlap in loss-sequence units"
    audit.to_csv(output / "phase5_overlap_audit.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("BAM", "EUROPE", "US_CORE"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--common-period", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(); project = Path(".").resolve()
    if args.audit:
        destination = project / "outputs_phase5"; destination.mkdir(exist_ok=True)
        feasibility_audit(project).to_csv(destination / "phase5_horizon_feasibility.csv", index=False)
    else:
        if not args.market or not args.output:
            parser.error("--market and --output are required unless --audit is used")
        run_market(project, args.market, args.output.resolve(), args.common_period)


if __name__ == "__main__":
    main()
