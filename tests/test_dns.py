import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.modelling import dns


MATURITIES = np.array([0.25, 0.5, 1, 2, 5, 10, 15, 20, 30.0])
COLUMNS_X = ["3M_x", "6M_x", "1Y_x", "2Y_x", "5Y_x", "10Y_x", "15Y_x", "20Y_x", "30Y_x"]
COLUMNS_Y = [name[:-1] + "y" for name in COLUMNS_X]


def synthetic_panels(n=100):
    dates = pd.bdate_range("2020-01-01", periods=n)
    loadings = dns.nelson_siegel_loadings(MATURITIES, 1.5)
    t = np.arange(n)
    bam_beta = np.column_stack((0.035 + t * 1e-5, -0.012 + np.sin(t / 9) * 0.001, np.cos(t / 13) * 0.004))
    ecb_beta = np.column_stack((0.025 + t * 8e-6, -0.009 + np.sin(t / 11) * 0.001, np.cos(t / 15) * 0.003))
    return (
        pd.DataFrame(bam_beta @ loadings.T, index=dates, columns=COLUMNS_X),
        pd.DataFrame(ecb_beta @ loadings.T, index=dates, columns=COLUMNS_Y),
    )


def simple_model():
    loadings = dns.nelson_siegel_loadings(MATURITIES, 1.5)
    dynamics = dns.OUParameters(np.array([0.03, -0.01, 0.003]), np.full(3, 0.02), np.eye(3) * 1e-7)
    return dns.DNSModel(1.5, loadings, dynamics, np.eye(9) * 1e-6, np.array([0.03, -0.01, 0.003]), np.eye(3) * 1e-3, np.linalg.cond(loadings))


def test_declared_unit_normalization_and_negative_rates():
    raw = pd.DataFrame({"bam": [0.035], "ecb": [-0.5]})
    assert dns.normalize_rate_units(raw[["bam"]], "decimal", "BAM").iloc[0, 0] == 0.035
    assert dns.normalize_rate_units(pd.DataFrame({"ecb": [3.5]}), "percent", "ECB").iloc[0, 0] == 0.035
    assert dns.normalize_rate_units(raw[["ecb"]], "percent", "ECB").iloc[0, 0] == -0.005


def test_conversion_exactly_once():
    once = dns.normalize_rate_units(pd.DataFrame({"r": [3.5]}), "percent", "ECB")
    twice_guard = dns.normalize_rate_units(once, "decimal", "internal")
    pd.testing.assert_frame_equal(once, twice_guard)


def test_strict_validation():
    with pytest.raises(ValueError, match="non numériques"):
        dns.normalize_rate_units(pd.DataFrame({"r": ["bad"]}), "decimal", "BAM")
    with pytest.raises(ValueError, match="infinis"):
        dns.normalize_rate_units(pd.DataFrame({"r": [np.inf]}), "decimal", "ECB")
    with pytest.raises(ValueError, match="unité déclarée"):
        dns.normalize_rate_units(pd.DataFrame({"r": [3.5]}), "decimal", "BAM")
    assert np.isnan(dns.normalize_rate_units(pd.DataFrame({"r": [np.nan, -0.005]}), "decimal", "ECB").iloc[0, 0])


def test_mixed_csv_is_normalized_before_factor_extraction(tmp_path):
    bam, ecb = synthetic_panels(3)
    mixed = pd.concat((bam, ecb * 100), axis=1)
    path = tmp_path / "mixed.csv"
    mixed.to_csv(path)
    loaded_bam, loaded_ecb = dns.load_combined_data(path)
    pd.testing.assert_frame_equal(loaded_bam, bam, check_freq=False)
    pd.testing.assert_frame_equal(loaded_ecb, ecb, check_freq=False)
    weights = np.ones(9)
    b1 = dns.extract_ols_betas(loaded_bam.to_numpy(), simple_model().loadings, weights)
    b2 = dns.extract_ols_betas(loaded_ecb.to_numpy(), simple_model().loadings, weights)
    assert np.max(np.abs(b1)) < 0.1 and np.max(np.abs(b2)) < 0.1


def test_plotting_uses_percent_display_without_mutation(tmp_path):
    bam, _ = synthetic_panels(12)
    model = simple_model()
    historical = dns.extract_ols_betas(bam.to_numpy(), model.loadings, np.ones(9))
    forecast = historical[-2:].copy()
    before = forecast.copy()
    dns.plot_forecast(bam.index, historical, pd.bdate_range(bam.index[-1], periods=2), forecast, bam.iloc[-1].to_numpy(), model, MATURITIES, tmp_path, "test")
    np.testing.assert_array_equal(forecast, before)
    assert (tmp_path / "forecast_curve_1month_test.png").exists()


def test_cli_help_and_import_have_no_side_effects(tmp_path):
    env = {**os.environ, "MPLBACKEND": "Agg"}
    help_result = subprocess.run([sys.executable, "src/modelling/dns.py", "--help"], cwd=Path(__file__).parents[1], env=env, text=True, capture_output=True)
    assert help_result.returncode == 0
    assert "--bam-unit {decimal,percent}" in help_result.stdout
    imported = subprocess.run([sys.executable, "-c", "import src.modelling.dns; print('ok')"], cwd=tmp_path, env={**env, "PYTHONPATH": str(Path(__file__).parents[1])}, text=True, capture_output=True)
    assert imported.returncode == 0 and imported.stdout.strip() == "ok"
    assert list(tmp_path.iterdir()) == []


def test_bce_lag_ordering():
    model = simple_model()
    influence = np.eye(3)
    state = np.array([0.03, -0.01, 0.003])
    ecb_t = np.array([0.04, -0.02, 0.006])
    forecast = dns.forecast_states(state, model, 1, ecb_t, model, influence)[0]
    a, c, _ = dns.transition_matrices(model.dynamics, 1)
    expected = a @ state + c + influence @ (ecb_t - model.dynamics.long_run_mean)
    np.testing.assert_allclose(forecast, expected)


def test_kalman_covariance_psd_and_missing_curve_propagates():
    bam, _ = synthetic_panels(4)
    bam.iloc[2] = np.nan
    model = simple_model()
    result = dns.kalman_filter(bam.to_numpy(), bam.index, model)
    for covariance in result.covariances:
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    a, c, _ = dns.transition_matrices(model.dynamics, (bam.index[2] - bam.index[1]).days)
    np.testing.assert_allclose(result.filtered[2], a @ result.filtered[1] + c)
    assert not np.allclose(result.filtered[2], result.filtered[1])


def test_no_lookahead_and_small_end_to_end_backtests():
    bam, ecb = synthetic_panels(92)
    kwargs = dict(start_date=str(bam.index[75].date()), horizon=2, decay_grid=np.array([1.5]))
    base = dns.run_backtest(bam, **kwargs)
    enriched = dns.run_backtest(bam, ecb, **kwargs)
    assert base and enriched
    changed = bam.copy()
    first = base[0]
    changed.loc[changed.index > first.future_dates[-1]] += 0.2
    repeated = dns.run_backtest(changed, **kwargs)
    np.testing.assert_allclose(base[0].predicted_yields, repeated[0].predicted_yields)


def test_csv_tables_declare_decimal_unit():
    assert dns.backtest_table([synthetic_result()]).loc[0, "rate_unit"] == "decimal"


def synthetic_result():
    zeros = np.zeros((2, 3))
    curves = np.zeros((2, 9))
    return dns.BacktestOrigin(pd.Timestamp("2020-01-01"), pd.Timestamp("2019-12-31"), pd.bdate_range("2020-01-01", periods=2), 1.5, None, zeros, zeros, curves, curves, curves, 0, 0, 0, 0, 0, 0, 0, np.zeros(9), np.zeros(3))


def test_convex_blend_boundaries_and_intermediate():
    persistence = np.array([0.02, 0.03])
    forecast = np.array([0.04, 0.01])
    np.testing.assert_array_equal(dns.combine_persistence_and_dns(persistence, forecast, 0), persistence)
    np.testing.assert_array_equal(dns.combine_persistence_and_dns(persistence, forecast, 1), forecast)
    np.testing.assert_allclose(
        dns.combine_persistence_and_dns(persistence, forecast, 0.25),
        0.75 * persistence + 0.25 * forecast,
    )


def test_blend_weight_is_bounded_and_zero_denominator_is_safe():
    p = np.array([0.01, 0.02])
    assert dns.estimate_convex_blend_weight(p, p, np.array([0.03, 0.04])) == 0
    assert dns.estimate_convex_blend_weight(p, p + 0.01, p + 1) == 1
    assert dns.estimate_convex_blend_weight(p, p + 0.01, p - 1) == 0


def blend_history():
    rows = []
    origins = pd.bdate_range("2020-01-01", periods=8)
    for horizon in (5, 10, 22):
        for i, origin in enumerate(origins):
            target = origin + pd.offsets.BDay(2)
            for maturity in MATURITIES:
                rows.append({
                    "forecast_origin": origin,
                    "target_date": target,
                    "horizon": horizon,
                    "maturity": maturity,
                    "is_observed_maturity": True,
                    "actual_yield": 0.03 + horizon * 1e-5,
                    "persistence_forecast": 0.03,
                    "dns_forecast": 0.031 + horizon * 1e-5,
                })
    return pd.DataFrame(rows)


def test_leakage_safe_weights_wait_for_completed_origins_and_are_horizon_specific():
    history = blend_history()
    weights = dns.estimate_leakage_safe_weight_history(history, min_origins=2)
    first = weights.sort_values("forecast_origin").iloc[0]
    assert first.weight == 0 and not first.estimated and "insufficient" in first.fallback_reason
    # At the third business-day origin only the first target is known, not two.
    origin = sorted(history.forecast_origin.unique())[2]
    assert (weights[weights.forecast_origin == origin].n_completed_origins == 1).all()
    final = weights[weights.forecast_origin == weights.forecast_origin.max()]
    assert set(final.horizon) == {5, 10, 22}
    assert final.groupby("horizon").weight.nunique().eq(1).all()


def test_future_changes_cannot_change_earlier_blend_weights():
    history = blend_history()
    before = dns.estimate_leakage_safe_weight_history(history, min_origins=2)
    cutoff = sorted(history.forecast_origin.unique())[4]
    changed = history.copy()
    changed.loc[changed.target_date > cutoff, "actual_yield"] = 0.9
    after = dns.estimate_leakage_safe_weight_history(changed, min_origins=2)
    cols = ["forecast_origin", "horizon", "weight", "n_completed_origins"]
    pd.testing.assert_frame_equal(before[before.forecast_origin <= cutoff][cols].reset_index(drop=True), after[after.forecast_origin <= cutoff][cols].reset_index(drop=True))


def test_interpolated_maturities_are_excluded_by_default():
    history = blend_history()
    extra = history.iloc[:8].copy()
    extra["maturity"] = 3.0
    extra["is_observed_maturity"] = False
    extra["actual_yield"] = 0.9
    augmented = pd.concat((history, extra), ignore_index=True)
    pd.testing.assert_frame_equal(
        dns.estimate_leakage_safe_weight_history(history, 2),
        dns.estimate_leakage_safe_weight_history(augmented, 2),
    )


def test_blend_min_origins_validation_and_cli_help():
    with pytest.raises(ValueError, match="positive integer"):
        dns.estimate_leakage_safe_weight_history(blend_history(), 0)
    result = subprocess.run(
        [sys.executable, "src/modelling/dns.py", "--help"],
        cwd=Path(__file__).parents[1], text=True, capture_output=True,
    )
    assert result.returncode == 0 and "--blend-min-origins" in result.stdout


def test_synthetic_blend_output_end_to_end(tmp_path):
    results = []
    for i, cutoff in enumerate(pd.bdate_range("2020-01-01", periods=6, freq="5B")):
        dates = pd.bdate_range(cutoff + pd.offsets.BDay(1), periods=22)
        persistence = np.full((22, 9), 0.03)
        predicted = np.full((22, 9), 0.032)
        actual = np.full((22, 9), 0.031)
        results.append(dns.BacktestOrigin(
            cutoff, cutoff, dates, 1.5, None, np.zeros((22, 3)), np.zeros((22, 3)),
            predicted, actual, persistence, 0, 0, 0, 0, 0, 0, 0,
            np.zeros(9), np.zeros(3),
        ))
    future = pd.bdate_range("2020-03-01", periods=22)
    final = {"forecast_yields": pd.DataFrame(np.full((22, 9), 0.032), index=future, columns=COLUMNS_X)}
    last = pd.Series(np.full(9, 0.03), index=COLUMNS_X, name=pd.Timestamp("2020-02-28"))
    table, metadata = dns.generate_blend_outputs(results, final, last, MATURITIES, "bam", tmp_path, 2)
    assert set(table.horizon) == {5, 10, 22}
    assert table.rate_unit.eq("decimal").all() and table.model_variant.eq("bam").all()
    assert metadata["internal_yield_unit"] == "decimal"
    for filename in (
        "backtest_blended_bam.csv", "forecast_yields_blended_bam.csv",
        "forecast_blend_metadata_bam.json", "comparison_blended_bam.png",
    ):
        assert (tmp_path / filename).exists()
