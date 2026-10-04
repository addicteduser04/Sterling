from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling import dns
from src.modelling.phase4 import load_market
from src.modelling.phase4_data import load_europe_curve, load_us_curve
from src.modelling.phase1 import run_phase1


PROJECT = Path(__file__).resolve().parents[1]


def test_europe_fixed_panel_and_explicit_decimal_conversion():
    curve = load_europe_curve(PROJECT / "data/taux_europe")
    raw = pd.read_csv(PROJECT / "data/taux_europe/ECB Data Portal_3M.csv")
    assert curve.shape[1] == 23
    assert curve.columns[0] == "3M" and curve.columns[-1] == "30Y"
    assert np.isclose(curve.iloc[0, 0], float(raw.iloc[0, -1]) / 100)
    assert (curve.to_numpy() < 0).any()


def test_us_panels_are_observed_complete_and_decimal():
    core = load_us_curve(PROJECT / "data/taux_us/raw", "core")
    extended = load_us_curve(PROJECT / "data/taux_us/raw", "extended")
    assert list(core.columns) == ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
    assert not core.isna().any().any() and not extended.isna().any().any()
    assert core.to_numpy().max() < .10
    assert extended.index.min() == pd.Timestamp("2006-02-09")
    assert extended.index.to_series().diff().dt.days.max() < 30


def test_foreign_maturities_are_all_observed_and_parse_exactly():
    curve = load_market(PROJECT, "US_CORE")
    np.testing.assert_allclose(dns.parse_maturities(curve.columns), [.25, .5, 1, 2, 3, 5, 7, 10])
    mask = dns.infer_observed_mask(dns.parse_maturities(curve.columns))
    # The BAM-specific default mask excludes some tenors, making the Phase 4
    # all-observed opt-in a necessary and test-covered modelling rule.
    assert not mask.all()


def test_negative_yields_fit_without_clipping():
    curve = load_market(PROJECT, "EUROPE")
    row = curve.loc[curve.min(axis=1).idxmin()]
    loadings = dns.nelson_siegel_loadings(dns.parse_maturities(row.index), 1.5)
    beta = np.linalg.lstsq(loadings, row.to_numpy(), rcond=None)[0]
    fitted = loadings @ beta
    assert np.isfinite(beta).all() and np.isfinite(fitted).all()
    assert row.min() < 0


def test_common_period_alignment_and_market_isolation():
    full = load_market(PROJECT, "US_CORE")
    common = load_market(PROJECT, "US_CORE", common_period=True)
    assert common.index.min() >= pd.Timestamp("2004-09-06")
    assert common.index.max() <= pd.Timestamp("2026-06-18")
    pd.testing.assert_frame_equal(common, full.loc[common.index])
    assert not any(column.endswith("_x") or column.endswith("_y") for column in common.columns)


def test_all_observed_forecast_path_keeps_non_bam_tenors():
    dates = pd.bdate_range("2020-01-01", periods=145)
    maturities = np.asarray([.25, .5, 1, 2, 3, 5, 7, 10])
    loadings = dns.nelson_siegel_loadings(maturities, 1.5)
    factors = np.column_stack((np.linspace(.02, .025, len(dates)),
                               np.full(len(dates), -.01), np.full(len(dates), .003)))
    curve = pd.DataFrame(factors @ loadings.T, index=dates,
                         columns=["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"])
    result = run_phase1(curve, start_date=str(dates[-30].date()), include_dns=False,
                        decay_grid=np.asarray([1.5]), all_maturities_observed=True)
    assert result.forecasts.groupby(["model", "forecast_origin", "horizon"]).size().eq(8).all()
