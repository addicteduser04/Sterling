import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.decomposition import PCA

from src.modelling import dns
from src.modelling.change_evaluation import overlap_bandwidth, validate_forecast_table
from src.modelling.change_features import causal_factor_curve_features, direct_change_dataset
from src.modelling.phase1 import run_phase1
from src.modelling.phase5 import HORIZONS, evaluation_windows, feasibility_audit, market_curve


def synthetic(n=430):
    dates = pd.bdate_range("2018-01-01", periods=n)
    maturities = np.asarray([.25, .5, 1, 2, 3, 5, 7, 10])
    loadings = dns.nelson_siegel_loadings(maturities, 2.0)
    t = np.arange(n)
    factors = pd.DataFrame(np.c_[.025 + t * 1e-6, -.01 + .001*np.sin(t/20),
                                     .003*np.cos(t/31)], index=dates, columns=dns.FACTOR_NAMES)
    curves = pd.DataFrame(factors.to_numpy() @ loadings.T, index=dates,
                          columns=["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"])
    return factors, curves


@pytest.mark.parametrize("horizon", [44, 66, 132, 252])
def test_long_horizon_target_alignment_and_realization(horizon):
    factors, curves = synthetic()
    features = causal_factor_curve_features(factors, curves)
    cutoff = 400
    data = direct_change_dataset(features, factors, horizon, cutoff - horizon)
    assert np.all(data.target_positions == data.origin_positions + horizon)
    assert data.target_positions.max() <= cutoff
    np.testing.assert_allclose(data.targets.iloc[-1],
                               factors.iloc[cutoff] - factors.iloc[cutoff-horizon])


def test_arbitrary_horizon_forecast_schema_and_determinism():
    _, curves = synthetic(180)
    kwargs = dict(start_date=str(curves.index[-80].date()), include_dns=False,
                  decay_grid=np.asarray([2.0]), all_maturities_observed=True,
                  horizons=(44, 66))
    one, two = run_phase1(curves, **kwargs), run_phase1(curves, **kwargs)
    assert set(one.forecasts.horizon) == {44, 66}
    pd.testing.assert_frame_equal(one.forecasts, two.forecasts)
    validate_forecast_table(one.forecasts)


@pytest.mark.parametrize("horizon,monthly,weekly", [(44, 2, 8), (66, 3, 13),
                                                     (132, 6, 26), (252, 11, 50)])
def test_long_horizon_date_derived_overlap(horizon, monthly, weekly):
    dates = pd.bdate_range("2010-01-01", "2025-12-31")
    m = evaluation_windows(dates, horizon, "monthly")
    w = evaluation_windows(dates, horizon, "weekly")
    assert overlap_bandwidth(m.origin, m.target) == monthly
    assert overlap_bandwidth(w.origin, w.target) == weekly


def test_long_horizon_ns_and_pca_reconstruction():
    factors, curves = synthetic()
    maturity = dns.parse_maturities(curves.columns)
    loading = dns.nelson_siegel_loadings(maturity, 2.0)
    reconstructed = factors.iloc[-1].to_numpy() @ loading.T
    np.testing.assert_allclose(reconstructed, curves.iloc[-1])
    pca = PCA(3).fit(curves.iloc[:-1])
    score = pca.transform(curves.iloc[[-1]])
    assert pca.inverse_transform(score).shape == (1, curves.shape[1])


def test_common_period_and_market_specific_feasibility():
    project_root = Path(__file__).resolve().parents[1]
    for market in ("BAM", "EUROPE", "US_CORE"):
        curve = market_curve(project_root, market, common_period=True)
        assert curve.index.min() >= pd.Timestamp("2004-09-06")
        assert curve.index.max() <= pd.Timestamp("2026-06-18")
    audit = feasibility_audit(project_root)
    assert len(audit) == 3 * len(HORIZONS)
    assert set(audit.classification) <= {"FULLY_FEASIBLE", "LIMITED_POWER",
                                         "DESCRIPTIVE_ONLY", "NOT_FEASIBLE"}
