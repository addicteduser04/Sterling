from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "masi" / "bam_ecb_2004.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_maturity(column_name: str) -> float:
    """Convert a maturity label such as '3M_y' or '10Y_y' into years."""
    match = re.match(r"^(\d+)([MY])(?:_y)?$", column_name)
    if not match:
        raise ValueError(f"Unsupported maturity column: {column_name}")

    value = float(match.group(1))
    unit = match.group(2)
    return value / 12.0 if unit == "M" else value


def nelson_siegel_loadings(maturities, lambda_val: float = 0.0609) -> np.ndarray:
    """Create the Nelson-Siegel loading matrix for a set of maturities."""
    maturities = np.asarray(maturities, dtype=float)
    term1 = np.where(
        maturities == 0,
        1.0,
        (1.0 - np.exp(-lambda_val * maturities)) / (lambda_val * maturities),
    )
    term2 = term1 - np.exp(-lambda_val * maturities)
    return np.column_stack([np.ones_like(maturities), term1, term2])


def fit_dynamic_nelson_siegel_kalman(
    yield_curve: pd.DataFrame,
    maturities=None,
    lambda_val: float = 0.0609,
    process_noise: float = 1e-3,
    observation_noise: float = 1e-2,
    forecast_horizon: int = 0,
):
    """Estimate dynamic Nelson-Siegel factors with a simple Kalman filter."""
    if maturities is None:
        maturities = [parse_maturity(column) for column in yield_curve.columns]

    design_matrix = nelson_siegel_loadings(maturities, lambda_val=lambda_val)

    n_steps = len(yield_curve)
    n_factors = design_matrix.shape[1]
    state_mean = np.zeros(n_factors)
    state_cov = np.eye(n_factors)

    filtered_states = np.zeros((n_steps, n_factors))
    filtered_covariances = np.zeros((n_steps, n_factors, n_factors))

    for idx, row in enumerate(yield_curve.itertuples(index=False, name=None)):
        obs = np.asarray(row, dtype=float)
        valid_mask = ~np.isnan(obs)
        if not np.any(valid_mask):
            continue

        design = design_matrix[valid_mask]
        obs_valid = obs[valid_mask]
        obs_cov = np.eye(int(valid_mask.sum())) * observation_noise

        pred_state = state_mean
        pred_cov = state_cov + np.eye(n_factors) * process_noise

        innovation = obs_valid - design @ pred_state
        innovation_cov = design @ pred_cov @ design.T + obs_cov
        kalman_gain = pred_cov @ design.T @ np.linalg.pinv(innovation_cov)

        state_mean = pred_state + kalman_gain @ innovation
        state_cov = (np.eye(n_factors) - kalman_gain @ design) @ pred_cov

        filtered_states[idx] = state_mean
        filtered_covariances[idx] = state_cov

    beta_df = pd.DataFrame(filtered_states, index=yield_curve.index, columns=["beta0", "beta1", "beta2"])
    fitted_yields = beta_df.to_numpy() @ design_matrix.T
    fitted_yields_df = pd.DataFrame(fitted_yields, index=yield_curve.index, columns=yield_curve.columns)

    forecast_df = None
    if forecast_horizon > 0:
        forecast_states = np.zeros((forecast_horizon, n_factors))
        forecast_cov = state_cov
        state_forecast = state_mean
        for step in range(forecast_horizon):
            state_forecast = state_forecast
            forecast_cov = forecast_cov + np.eye(n_factors) * process_noise
            forecast_states[step] = state_forecast

        forecast_df = pd.DataFrame(forecast_states, columns=["beta0", "beta1", "beta2"])
        forecast_yields = forecast_df.to_numpy() @ design_matrix.T
        forecast_yields_df = pd.DataFrame(forecast_yields, columns=yield_curve.columns)
        return {
            "betas": beta_df,
            "fitted_yields": fitted_yields_df,
            "forecast_betas": forecast_df,
            "forecast_yields": forecast_yields_df,
        }

    return {"betas": beta_df, "fitted_yields": fitted_yields_df}


def load_yield_curve_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the ECB daily yield curve data and keep the yield columns only."""
    df = pd.read_csv(path, index_col=0)
    if "Date" in df.columns:
        df = df.set_index("Date")

    yield_columns = [column for column in df.columns if column.endswith("_y")]
    if not yield_columns:
        raise ValueError("No yield columns ending with '_y' were found in the input data.")

    yield_curve = df[yield_columns].copy()
    yield_curve.index = pd.to_datetime(yield_curve.index)
    yield_curve = yield_curve.sort_index()
    yield_curve = yield_curve.apply(pd.to_numeric, errors="coerce")
    yield_curve = yield_curve.dropna(how="all")
    return yield_curve


def main() -> None:
    yield_curve = load_yield_curve_data(DATA_PATH)
    maturities = [parse_maturity(column) for column in yield_curve.columns]

    results = fit_dynamic_nelson_siegel_kalman(
        yield_curve=yield_curve,
        maturities=maturities,
        lambda_val=0.0609,
        forecast_horizon=6,
    )

    results["betas"].to_csv(OUTPUT_DIR / "dns_kalman_betas.csv")
    results["fitted_yields"].to_csv(OUTPUT_DIR / "dns_kalman_fitted_yields.csv")
    if "forecast_betas" in results:
        results["forecast_betas"].to_csv(OUTPUT_DIR / "dns_kalman_forecast_betas.csv")
        results["forecast_yields"].to_csv(OUTPUT_DIR / "dns_kalman_forecast_yields.csv")

    print("Dynamic Nelson-Siegel + Kalman Filter completed successfully.")
    print(results["betas"].head())


if __name__ == "__main__":
    main()
