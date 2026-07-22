from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "masi" / "bam_ecb_2004.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_maturity(column_name: str) -> float:
    """Convertit '3M_x' ou '10Y_y' en années."""
    match = re.match(r"^(\d+)([MY])(?:_[xy])?$", column_name)
    if not match:
        raise ValueError(f"Colonne non reconnue : {column_name}")
    value = float(match.group(1))
    unit = match.group(2)
    return value / 12.0 if unit == "M" else value


def nelson_siegel_loadings(maturities: np.ndarray, lambda_val: float) -> np.ndarray:
    """Matrice de design NS (n_maturities × 3)."""
    m = np.asarray(maturities, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        term2 = np.where(m == 0, 1.0, (1 - np.exp(-m / lambda_val)) / (m / lambda_val))
    term3 = term2 - np.exp(-m / lambda_val)
    return np.column_stack([np.ones_like(m), term2, term3])


def kalman_filter(yields: np.ndarray, Lambda: np.ndarray,
                  A: np.ndarray, mu: np.ndarray,
                  Q: np.ndarray, H: np.ndarray):
    """
    Kalman filter complet.
    
    yields  : (T × n_maturities)
    Lambda  : (n_maturities × 3) matrice de design NS
    A       : (3 × 3) matrice de transition VAR(1)
    mu      : (3,) vecteur de moyenne long terme
    Q       : (3 × 3) covariance bruit de processus
    H       : (n_maturities × n_maturities) covariance bruit d'observation
    
    Retourne :
    filtered_states : (T × 3) betas filtrés
    log_likelihood  : float
    """
    T = yields.shape[0]
    n_factors = 3

    # Initialisation — moyenne inconditionnelle et covariance stationnaire
    state_mean = mu.copy()
    state_cov = np.eye(n_factors) * 1.0

    filtered_states = np.zeros((T, n_factors))
    log_likelihood = 0.0

    for t in range(T):
        obs = yields[t]
        valid = ~np.isnan(obs)

        if not np.any(valid):
            filtered_states[t] = state_mean
            continue

        L = Lambda[valid]
        y = obs[valid]
        H_t = H[np.ix_(valid, valid)]

        # ── Prédiction ──
        pred_mean = A @ state_mean + mu
        pred_cov  = A @ state_cov @ A.T + Q

        # ── Innovation ──
        innovation     = y - L @ pred_mean
        innovation_cov = L @ pred_cov @ L.T + H_t

        # ── Log-vraisemblance ──
        try:
            sign, logdet = np.linalg.slogdet(innovation_cov)
            if sign <= 0:
                log_likelihood -= 1e10
            else:
                log_likelihood -= 0.5 * (
                    len(y) * np.log(2 * np.pi)
                    + logdet
                    + innovation @ np.linalg.solve(innovation_cov, innovation)
                )
        except np.linalg.LinAlgError:
            log_likelihood -= 1e10

        # ── Gain de Kalman ──
        K = pred_cov @ L.T @ np.linalg.pinv(innovation_cov)

        # ── Mise à jour ──
        state_mean = pred_mean + K @ innovation
        state_cov  = (np.eye(n_factors) - K @ L) @ pred_cov

        filtered_states[t] = state_mean

    return filtered_states, log_likelihood


def estimate_parameters(yields: np.ndarray, maturities: np.ndarray,
                        lambda_val: float = 0.0609):
    """
    Estimer A, mu, Q, H par maximum de vraisemblance.
    λ fixé (grid search possible séparément).
    """
    Lambda = nelson_siegel_loadings(maturities, lambda_val)
    n_obs = yields.shape[1]

    def neg_log_likelihood(params):
        # Décomposer le vecteur de paramètres
        mu = params[:3]
        # A diagonal pour parcimonie
        a_diag = np.clip(params[3:6], -0.999, 0.999)
        A = np.diag(a_diag)
        # Q et H diagonaux, log-paramétrés pour garantir positivité
        log_q = params[6:9]
        log_h = params[9]
        Q = np.diag(np.exp(log_q))
        H = np.eye(n_obs) * np.exp(log_h)

        _, ll = kalman_filter(yields, Lambda, A, mu, Q, H)
        return -ll

    # Initialisation
    x0 = np.zeros(10)
    x0[:3]  = np.nanmean(yields, axis=0)[:3] if yields.shape[1] >= 3 else 0
    x0[3:6] = [0.95, 0.90, 0.85]   # persistance initiale des betas
    x0[6:9] = [np.log(1e-4)] * 3   # log process noise
    x0[9]   = np.log(1e-3)          # log observation noise

    result = minimize(
        neg_log_likelihood,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    params = result.x
    mu_hat    = params[:3]
    A_hat     = np.diag(np.clip(params[3:6], -0.999, 0.999))
    Q_hat     = np.diag(np.exp(params[6:9]))
    H_hat     = np.eye(n_obs) * np.exp(params[9])

    print(f"Optimisation convergée : {result.success}")
    print(f"Log-vraisemblance : {-result.fun:.4f}")
    print(f"mu  : {mu_hat}")
    print(f"A (diag) : {np.diag(A_hat)}")

    return mu_hat, A_hat, Q_hat, H_hat


def forecast_betas(last_state: np.ndarray, A: np.ndarray,
                   mu: np.ndarray, horizon: int) -> np.ndarray:
    """
    Prévision des betas sur `horizon` pas en avant.
    Retourne (horizon × 3).
    """
    forecasts = np.zeros((horizon, 3))
    state = last_state.copy()
    for h in range(horizon):
        state = A @ state + mu
        forecasts[h] = state
    return forecasts


def load_yield_curve(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charger les taux BAM (colonnes _x)."""
    df = pd.read_csv(path, index_col=0)
    if "Date" in df.columns:
        df = df.set_index("Date")

    cols = [c for c in df.columns if c.endswith("_x")]
    if not cols:
        raise ValueError("Aucune colonne BAM (_x) trouvée.")

    yc = df[cols].copy()
    yc.index = pd.to_datetime(yc.index)
    yc = yc.sort_index()
    yc = yc.apply(pd.to_numeric, errors="coerce")
    yc = yc.dropna(how="all")
    return yc


def main():
    # ── 1. Charger les données ──
    yield_curve = load_yield_curve(DATA_PATH)
    maturities  = np.array([parse_maturity(c) for c in yield_curve.columns])
    yields_arr  = yield_curve.values

    print(f"Données chargées : {yield_curve.shape[0]} jours, {yield_curve.shape[1]} maturités")

    # ── 2. Estimer les paramètres par MLE ──
    print("\nEstimation des paramètres par maximum de vraisemblance...")
    mu, A, Q, H = estimate_parameters(yields_arr, maturities, lambda_val=0.0609)

    # ── 3. Kalman filter sur tout le panel ──
    Lambda = nelson_siegel_loadings(maturities, lambda_val=0.0609)
    filtered_states, ll = kalman_filter(yields_arr, Lambda, A, mu, Q, H)

    beta_df = pd.DataFrame(
        filtered_states,
        index=yield_curve.index,
        columns=["beta0", "beta1", "beta2"]
    )

    # ── 4. Fitted yields ──
    fitted = filtered_states @ Lambda.T
    fitted_df = pd.DataFrame(fitted, index=yield_curve.index, columns=yield_curve.columns)

    # ── 5. RMSE ──
    rmse = np.sqrt(np.nanmean((yields_arr - fitted) ** 2))
    print(f"\nRMSE moyen : {rmse:.6f}")

    # ── 6. Forecast 30 jours ──
    last_state   = filtered_states[-1]
    forecast_arr = forecast_betas(last_state, A, mu, horizon=30)
    forecast_df  = pd.DataFrame(forecast_arr, columns=["beta0", "beta1", "beta2"])
    forecast_yields = forecast_arr @ Lambda.T
    forecast_yields_df = pd.DataFrame(forecast_yields, columns=yield_curve.columns)

    # ── 7. Sauvegarder ──
    beta_df.to_csv(OUTPUT_DIR / "dns_kalman_betas.csv")
    fitted_df.to_csv(OUTPUT_DIR / "dns_kalman_fitted_yields.csv")
    forecast_df.to_csv(OUTPUT_DIR / "dns_kalman_forecast_betas.csv")
    forecast_yields_df.to_csv(OUTPUT_DIR / "dns_kalman_forecast_yields.csv")

    print("\nDNS + Kalman Filter terminé.")
    print(beta_df.tail())


if __name__ == "__main__":
    main()