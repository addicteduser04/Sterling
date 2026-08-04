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
