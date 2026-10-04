import numpy as np
import pandas as pd

from src.modelling import dns
from src.modelling.phase2 import (ARIMA_REJECTION, XGBOOST_CONFIGS, direct_yield_features,
                                  evaluation_positions, run_direct_models, run_xgboost_models,
                                  strict_common_sample)


MATURITIES = np.array([0.25, 0.5, 1, 2, 5, 10, 15, 20, 30.0])
COLUMNS = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "15Y", "20Y", "30Y"]


def curves(n=145):
    dates = pd.bdate_range("2020-01-01", periods=n)
    loadings = dns.nelson_siegel_loadings(MATURITIES, 1.5)
    t = np.arange(n)
    beta = np.column_stack([.03 + t*2e-6, -.01 + .001*np.sin(t/10), .003*np.cos(t/15)])
    return pd.DataFrame(beta @ loadings.T, index=dates, columns=COLUMNS)


def test_direct_features_are_maturity_specific_and_causal():
    frame = curves(220)
    before = direct_yield_features(frame, "3M")
    changed = frame.copy(); changed.iloc[-1] = 9
    after = direct_yield_features(changed, "3M")
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])
    assert {"own_change_lag1", "own_change_lag22", "curve_slope", "curve_curvature"} <= set(before)


def test_origin_generation_is_unique_and_targets_exist():
    dates = curves().index
    for frequency in ("monthly", "weekly", "daily"):
        positions = evaluation_positions(dates, str(dates[-35].date()), 22, frequency)
        assert positions == sorted(set(positions))
        assert all(position + 22 < len(dates) for position in positions)
    expected = [cutoff for _, cutoff in dns._month_origins(dates, str(dates[-35].date()), 22)]
    assert evaluation_positions(dates, str(dates[-35].date()), 22, "monthly") == expected


def test_direct_models_cover_nine_maturities_three_horizons_and_reconstruct():
    frame = curves()
    result = run_direct_models(frame, str(frame.index[-30].date()))
    assert set(result.forecasts.model) == {"DIRECT_RIDGE", "DIRECT_AR"}
    assert set(result.forecasts.horizon) == {5, 10, 22}
    assert result.forecasts.maturity.nunique() == 9
    assert result.forecasts.groupby(["model", "forecast_origin", "horizon"]).size().eq(9).all()
    assert result.failures.empty
    # Forecasts are reconstructed yield levels, while metadata records direct changes.
    assert np.isfinite(result.forecasts.forecast_yield).all()
    assert result.forecasts.rate_unit.eq("decimal").all()


def test_xgboost_space_is_bounded_and_forecasts_are_deterministic():
    assert len(XGBOOST_CONFIGS) == 3
    assert max(config["max_depth"] for config in XGBOOST_CONFIGS) <= 3
    assert max(config["n_estimators"] for config in XGBOOST_CONFIGS) <= 200
    frame = curves(220)
    kwargs = dict(start_date=str(frame.index[-30].date()), decay_grid=np.array([1.5]))
    one = run_xgboost_models(frame, **kwargs)
    two = run_xgboost_models(frame, **kwargs)
    assert set(one.forecasts.model) == {"NS_XGBOOST", "PCA_XGBOOST"}
    assert one.forecasts.groupby(["model", "forecast_origin", "horizon"]).size().eq(9).all()
    np.testing.assert_array_equal(one.forecasts.forecast_yield, two.forecasts.forecast_yield)
    assert one.tuning.configurations_evaluated.eq(3).all()


def test_arima_rejection_and_strict_common_alignment_are_explicit():
    assert "equally spaced" in ARIMA_REJECTION and "irregular" in ARIMA_REJECTION
    rows = []
    for model in ("PERSISTENCE", "A", "B"):
        for origin in pd.bdate_range("2024-01-01", periods=3):
            if model == "B" and origin == pd.Timestamp("2024-01-02"):
                continue
            rows.append({"forecast_origin": origin, "target_date": origin + pd.offsets.BDay(5),
                         "horizon": 5, "maturity": .25, "model": model})
    common = strict_common_sample(pd.DataFrame(rows))
    assert common.forecast_origin.nunique() == 2
    assert common.groupby("forecast_origin").model.nunique().eq(3).all()
