import pandas as pd 
import numpy as np 
#Chargement des données
def load_data(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors='coerce')
    
    bam_df = df[[c for c in df.columns if c.endswith('_x')]]
    ecb_df = df[[c for c in df.columns if c.endswith('_y')]]
    
    return bam_df, ecb_df

#Parser les maturités
import re

def parse_maturities(columns):
    maturities = []
    for col in columns:
        match = re.match(r'^(\d+)([MY])(?:_[xy])?$', col)
        if not match:
            raise ValueError(f"Colonne non reconnue : {col}")
        value = float(match.group(1))
        unit  = match.group(2)
        maturities.append(value / 12.0 if unit == 'M' else value)
    return np.array(maturities)



#Matrice de design Nelson-Siegel
def nelson_siegel_loadings(maturities, lambda_val=0.0609):
    m = np.asarray(maturities, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        f2 = np.where(
            m == 0,
            1.0,
            (1 - np.exp(-m / lambda_val)) / (m / lambda_val)
        )
    f3 = f2 - np.exp(-m / lambda_val)
    return np.column_stack([np.ones_like(m), f2, f3])


#Kalman Filter

def kalman_filter(yields, Lambda, A, mu, Q, H):
    """
    yields  : (T × n_maturities)
    Lambda  : (n_maturities × 3)
    A       : (3 × 3) matrice de transition
    mu      : (3,) moyenne long terme
    Q       : (3 × 3) covariance bruit de processus
    H       : (n_maturities × n_maturities) covariance bruit observation
    """
    T, n_obs = yields.shape
    n_factors = 3

    # Initialisation
    state_mean = mu.copy()
    state_cov  = np.eye(n_factors)

    filtered_states = np.zeros((T, n_factors))
    log_likelihood  = 0.0

    for t in range(T):
        obs   = yields[t]
        valid = ~np.isnan(obs)

        if not np.any(valid):
            filtered_states[t] = state_mean
            continue

        L   = Lambda[valid]          # (n_valid × 3)
        y   = obs[valid]             # (n_valid,)
        H_t = H[np.ix_(valid, valid)]  # (n_valid × n_valid)

        # ── Étape 1 : Prédiction état ──
        pred_mean = A @ state_mean + mu

        # ── Étape 2 : Prédiction covariance ──
        pred_cov = A @ state_cov @ A.T + Q

        # ── Étape 3 : Innovation ──
        innovation     = y - L @ pred_mean
        innovation_cov = L @ pred_cov @ L.T + H_t

        # ── Log-vraisemblance ──
        sign, logdet = np.linalg.slogdet(innovation_cov)
        if sign > 0:
            log_likelihood -= 0.5 * (
                len(y) * np.log(2 * np.pi)
                + logdet
                + innovation @ np.linalg.solve(innovation_cov, innovation)
            )

        # ── Étape 4 : Gain de Kalman ──
        K = pred_cov @ L.T @ np.linalg.pinv(innovation_cov)

        # ── Étape 5 : Mise à jour ──
        state_mean = pred_mean + K @ innovation
        state_cov  = (np.eye(n_factors) - K @ L) @ pred_cov

        filtered_states[t] = state_mean

    return filtered_states, log_likelihood

#Estimation des paramètres par MLE
from scipy.optimize import minimize

def estimate_parameters(yields, Lambda):
    """
    Estime A (diagonal), mu, Q (diagonal), H (scalaire × I)
    par maximisation de la log-vraisemblance du filtre de Kalman.
    """
    n_obs = yields.shape[1]

    def neg_log_likelihood(params):
        mu    = params[:3]
        a_diag = np.clip(params[3:6], -0.999, 0.999)
        A     = np.diag(a_diag)
        Q     = np.diag(np.exp(params[6:9]))
        H     = np.eye(n_obs) * np.exp(params[9])

        _, ll = kalman_filter(yields, Lambda, A, mu, Q, H)
        return -ll

    # Initialisation
    x0 = np.zeros(10)
    x0[:3]  = np.nanmean(yields, axis=0)[[0, 1, 2]]
    x0[3:6] = [0.95, 0.90, 0.85]
    x0[6:9] = [np.log(1e-4)] * 3
    x0[9]   = np.log(1e-3)

    result = minimize(
        neg_log_likelihood,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    mu_hat = result.x[:3]
    A_hat  = np.diag(np.clip(result.x[3:6], -0.999, 0.999))
    Q_hat  = np.diag(np.exp(result.x[6:9]))
    H_hat  = np.eye(n_obs) * np.exp(result.x[9])

    print(f"Convergence : {result.success}")
    print(f"Log-vraisemblance : {-result.fun:.4f}")
    print(f"mu  : {mu_hat}")
    print(f"A (diag) : {np.diag(A_hat)}")
    print(f"Q (diag) : {np.diag(Q_hat)}")
    print(f"H (scalaire) : {np.exp(result.x[9]):.6f}")

    return mu_hat, A_hat, Q_hat, H_hat

#forecast
def forecast_betas(last_state, A, mu, horizon=30):
    """
    Prédit les betas sur `horizon` jours en avant.
    Retourne (horizon × 3).
    """
    forecasts = np.zeros((horizon, 3))
    state = last_state.copy()
    for h in range(horizon):
        state        = A @ state + mu
        forecasts[h] = state
    return forecasts

#pipeline complet 

from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    # ── 1. Charger les données ──
    bam_df, ecb_df = load_data('./data/masi/bam_ecb_2004.csv')
    maturities     = parse_maturities(bam_df.columns)
    yields_arr     = bam_df.values

    print(f"Données : {bam_df.shape[0]} jours, {bam_df.shape[1]} maturités")
    print(f"Période : {bam_df.index[0].date()} → {bam_df.index[-1].date()}")

    # ── 2. Matrice de design ──
    Lambda = nelson_siegel_loadings(maturities, lambda_val=0.0609)
    print(f"Matrice Lambda : {Lambda.shape}")

    # ── 3. Estimer les paramètres ──
    print("\nEstimation MLE en cours...")
    mu, A, Q, H = estimate_parameters(yields_arr, Lambda)

    # ── 4. Kalman filter sur tout le panel ──
    print("\nFiltre de Kalman en cours...")
    filtered_states, ll = kalman_filter(yields_arr, Lambda, A, mu, Q, H)
    print(f"Log-vraisemblance finale : {ll:.4f}")

    # ── 5. Betas filtrés ──
    beta_df = pd.DataFrame(
        filtered_states,
        index=bam_df.index,
        columns=['beta0', 'beta1', 'beta2']
    )

    # ── 6. Yields reconstruits ──
    fitted_arr = filtered_states @ Lambda.T
    fitted_df  = pd.DataFrame(
        fitted_arr,
        index=bam_df.index,
        columns=bam_df.columns
    )

    # ── 7. RMSE ──
    rmse = np.sqrt(np.nanmean((yields_arr - fitted_arr) ** 2))
    print(f"RMSE moyen : {rmse:.6f}")

    # ── 8. Forecast 30 jours ──
    last_state   = filtered_states[-1]
    forecast_arr = forecast_betas(last_state, A, mu, horizon=30)
    forecast_df  = pd.DataFrame(
        forecast_arr,
        columns=['beta0', 'beta1', 'beta2']
    )
    forecast_yields_arr = forecast_arr @ Lambda.T
    forecast_yields_df  = pd.DataFrame(
        forecast_yields_arr,
        columns=bam_df.columns
    )

    # ── 9. Sauvegarder ──
    beta_df.to_csv(OUTPUT_DIR / 'dns_kalman_betas.csv')
    fitted_df.to_csv(OUTPUT_DIR / 'dns_kalman_fitted_yields.csv')
    forecast_df.to_csv(OUTPUT_DIR / 'dns_kalman_forecast_betas.csv')
    forecast_yields_df.to_csv(OUTPUT_DIR / 'dns_kalman_forecast_yields.csv')

    print("\nRésultats sauvegardés dans outputs/")
    print("\nBetas filtrés (5 dernières lignes) :")
    print(beta_df.tail())
    print("\nForecast betas (30 jours) :")
    print(forecast_df.head())

if __name__ == '__main__':
    main()