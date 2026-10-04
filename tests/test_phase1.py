import numpy as np
import pandas as pd

from src.modelling import dns
from src.modelling.change_evaluation import dm_hac, metric_table, overlap_bandwidth
from src.modelling.change_features import (
    causal_factor_curve_features,
    direct_change_dataset,
    purged_expanding_splits,
)
from src.modelling.phase1 import attach_existing_dns, regularized_var_forecast, run_phase1, tune_direct_model


MATURITIES = np.array([0.25, 0.5, 1, 2, 5, 10, 15, 20, 30.0])
COLUMNS = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "15Y", "20Y", "30Y"]


def synthetic_curve(n=150):
    dates = pd.bdate_range("2020-01-01", periods=n)
    loadings = dns.nelson_siegel_loadings(MATURITIES, 1.5)
    t = np.arange(n)
    factors = pd.DataFrame(np.column_stack([
        .03 + t * 2e-6, -.01 + .001 * np.sin(t / 11), .003 * np.cos(t / 17),
    ]), index=dates, columns=dns.FACTOR_NAMES)
    curves = pd.DataFrame(factors.to_numpy() @ loadings.T, index=dates, columns=COLUMNS)
    return factors, curves


def test_causal_features_do_not_change_when_future_changes():
    factors, curves = synthetic_curve()
    before = causal_factor_curve_features(factors, curves)
    changed_factors, changed_curves = factors.copy(), curves.copy()
    changed_factors.iloc[-1] = 9
    changed_curves.iloc[-1] = 9
    after = causal_factor_curve_features(changed_factors, changed_curves)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


def test_direct_horizon_alignment_and_completed_target_rule():
    factors, curves = synthetic_curve()
    features = causal_factor_curve_features(factors, curves)
    data = direct_change_dataset(features, factors, 22, last_usable_origin=100)
    assert np.all(data.target_positions == data.origin_positions + 22)
    assert data.target_positions.max() <= 122
    first = data.origin_positions[0]
    np.testing.assert_allclose(data.targets.iloc[0], factors.iloc[first + 22] - factors.iloc[first])


def test_purged_cv_never_trains_on_target_after_validation_origin():
    origins = np.arange(120)
    targets = origins + 22
    for train, validation in purged_expanding_splits(origins, targets, min_train=30):
        assert targets[train].max() <= origins[validation].min()


def test_direct_models_are_deterministic_and_have_three_outputs():
    factors, curves = synthetic_curve()
    features = causal_factor_curve_features(factors, curves)
    data = direct_change_dataset(features, factors, 5, 140)
    one = tune_direct_model("ridge", data, features.iloc[[-1]])
    two = tune_direct_model("ridge", data, features.iloc[[-1]])
    assert one.change.shape == (3,)
    np.testing.assert_array_equal(one.change, two.change)
    assert one.metadata["cv_folds"] > 0


def test_short_history_fallback_is_zero_change():
    factors, curves = synthetic_curve(28)
    features = causal_factor_curve_features(factors, curves).fillna(0)
    data = direct_change_dataset(features, factors, 5, 15)
    prediction = tune_direct_model("ridge", data, features.iloc[[-1]])
    np.testing.assert_array_equal(prediction.change, np.zeros(3))
    var = regularized_var_forecast(factors, 5)
    np.testing.assert_array_equal(var.change, np.zeros(3))


def test_metrics_and_overlapping_horizon_dm():
    rows = []
    for model, scale in (("PERSISTENCE", 1.0), ("NS_RIDGE", .5)):
        for i, origin in enumerate(pd.bdate_range("2024-01-01", periods=40)):
            rows.append({"forecast_origin": origin, "target_date": origin + pd.offsets.BDay(5),
                         "horizon": 5, "maturity": .25, "actual_yield": .03,
                         "forecast_yield": .03 + scale * (i + 1) * 1e-5,
                         "error": scale * (i + 1) * 1e-5, "model": model,
                         "representation": "YIELD" if model == "PERSISTENCE" else "NS",
                         "variant": "test"})
    metrics = metric_table(pd.DataFrame(rows))
    ridge = metrics[(metrics.model == "NS_RIDGE") & (metrics.maturity == "aggregate")].iloc[0]
    assert ridge.rmse_ratio_vs_persistence == .5
    result = dm_hac(np.arange(40) * .5, np.arange(40), horizon=5)
    assert result["statistic"] < 0 and result["n"] == 40


def test_date_derived_overlap_bandwidth_cases():
    monthly = pd.date_range("2024-01-31", periods=5, freq="ME")
    assert overlap_bandwidth(monthly, monthly + pd.offsets.BDay(5)) == 0
    # A J+22 window can overlap the next monthly origin, but not mechanically 21 lags.
    assert overlap_bandwidth(monthly, monthly + pd.offsets.BDay(22)) == 1
    weekly = pd.date_range("2024-01-05", periods=8, freq="7D")
    assert overlap_bandwidth(weekly, weekly + pd.offsets.BDay(11)) == 2
    daily = pd.bdate_range("2024-01-01", periods=30)
    assert overlap_bandwidth(daily, daily + pd.offsets.BDay(5)) == 4


def test_small_end_to_end_has_common_schema_and_leakage_safe_pca():
    _, curves = synthetic_curve(145)
    result = run_phase1(curves, start_date=str(curves.index[-30].date()),
                        include_dns=False, decay_grid=np.array([1.5]))
    assert set(result.forecasts.model) == {
        "PERSISTENCE", "NS_RIDGE", "NS_ELASTICNET", "PCA_RIDGE", "PCA_RIDGE_VAR",
    }
    assert set(result.forecasts.horizon) == {5, 10, 22}
    assert result.forecasts.groupby(["model", "forecast_origin", "horizon"]).size().eq(9).all()
    assert (result.sample_diagnostics.latest_training_target_position
            <= result.sample_diagnostics.forecast_origin_position).all()
    assert result.pca_diagnostics.groupby(["forecast_origin", "component"]).size().eq(9).all()
    assert result.forecasts.rate_unit.eq("decimal").all()


def test_existing_dns_is_attached_only_on_exact_common_rows():
    _, curves = synthetic_curve(145)
    result = run_phase1(curves, start_date=str(curves.index[-30].date()),
                        include_dns=False, decay_grid=np.array([1.5]))
    base = result.forecasts[result.forecasts.model == "PERSISTENCE"].copy()
    history = base[["forecast_origin", "target_date", "horizon", "maturity", "actual_yield"]].copy()
    history["is_observed_maturity"] = True
    history["dns_forecast"] = history.actual_yield + 1e-4
    history["blended_forecast"] = history.actual_yield + 5e-5
    attached = attach_existing_dns(result, history)
    assert {"DNS_KALMAN_OU", "PERSISTENCE_DNS_BLEND"} <= set(attached.forecasts.model)
    assert np.allclose(attached.forecasts[attached.forecasts.model == "DNS_KALMAN_OU"].error, 1e-4)
