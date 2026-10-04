"""Market-agnostic Phase 4 independent sovereign-curve experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling.phase1 import run_phase1, write_phase1_outputs
from src.modelling.phase4_data import load_europe_curve, load_us_curve


DECAY_GRID = np.asarray((0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0))


def load_market(project: Path, market: str, common_period: bool = False) -> pd.DataFrame:
    if market == "EUROPE":
        curve = load_europe_curve(project / "data/taux_europe")
    elif market == "US_CORE":
        curve = load_us_curve(project / "data/taux_us/raw", "core")
    elif market == "US_EXTENDED":
        curve = load_us_curve(project / "data/taux_us/raw", "extended")
    else:
        raise ValueError(f"unsupported independent Phase 4 market: {market}")
    if common_period:
        curve = curve.loc["2004-09-06":"2026-06-18"]
    return curve


def run_market_backtest(
    market: str,
    curve_data: pd.DataFrame,
    output_dir: Path,
    start_date: str = "2022-01-01",
    horizons: tuple[int, ...] = (5, 10, 22),
) -> None:
    """Reuse Phase 1 with a fixed market grid and no cross-market predictors."""

    result = run_phase1(
        curve_data, start_date=start_date, include_dns=True, decay_grid=DECAY_GRID,
        all_maturities_observed=True, horizons=horizons,
    )
    write_phase1_outputs(result, output_dir)
    metadata = {
        "market": market, "start_date": start_date,
        "curve_start": curve_data.index.min().date().isoformat(),
        "curve_end": curve_data.index.max().date().isoformat(),
        "curve_observations": len(curve_data), "maturities": list(curve_data.columns),
        "internal_unit": "decimal", "seed": 42,
        "horizons": list(horizons),
        "horizon_semantics": "subsequent market observations",
        "cross_market_predictors": False,
    }
    (output_dir / "phase4_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("EUROPE", "US_CORE", "US_EXTENDED"), required=True)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--common-period", action="store_true")
    parser.add_argument("--horizons", default="5,10,22")
    args = parser.parse_args()
    project = args.project.resolve()
    curve = load_market(project, args.market, args.common_period)
    horizons = tuple(int(value) for value in args.horizons.split(","))
    run_market_backtest(args.market, curve, args.output.resolve(), args.start_date, horizons)


if __name__ == "__main__":
    main()
