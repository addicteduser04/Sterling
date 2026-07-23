import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


# ── 1. Chargement des données ─────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors='coerce')
    bam_df = df[[c for c in df.columns if c.endswith('_x')]]
    ecb_df = df[[c for c in df.columns if c.endswith('_y')]]
    return bam_df, ecb_df


# ── 2. Parser les maturités ───────────────────────────────────────────────────

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


# ── 3. Matrice de design Nelson-Siegel ───────────────────────────────────────

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


# ── 4. Initialisation OLS ────────────────────────────────────────────────────

def initialize_state(yields_arr, Lambda, n_warmup=60):
    betas_warmup = []
    for t in range(min(n_warmup, len(yields_arr))):
        obs   = yields_arr[t]
        valid = ~np.isnan(obs)
        if valid.sum() < 4:
            continue
        beta = np.linalg.lstsq(Lambda[valid], obs[valid], rcond=None)[0]
        betas_warmup.append(beta)

    betas_warmup = np.array(betas_warmup)
    state_mean   = betas_warmup.mean(axis=0)
    state_cov    = np.cov(betas_warmup.T) if len(betas_warmup) > 3 else np.eye(3) * 0.1

    print(f"Initialisation OLS sur {len(betas_warmup)} jours — β(0|0) : {state_mean}")
    return state_mean, state_cov


# ── 5. Paramètres fixes Diebold-Li ───────────────────────────────────────────

def fixed_parameters(n_obs):
    A  = np.diag([0.9977, 0.9767, 0.9011])
    mu = np.zeros(3)
    Q  = np.diag([1e-4, 1e-4, 1e-2])
    H  = np.eye(n_obs) * 1e-2
    return mu, A, Q, H


# ── 6. Estimation de mu ───────────────────────────────────────────────────────

def estimate_mu(yields_arr, Lambda):
    betas = []
    for t in range(len(yields_arr)):
        obs   = yields_arr[t]
        valid = ~np.isnan(obs)
        if valid.sum() < 4:
            continue
        beta = np.linalg.lstsq(Lambda[valid], obs[valid], rcond=None)[0]
        betas.append(beta)
    betas     = np.array(betas)
    beta_mean = betas.mean(axis=0)
    A         = np.diag([0.9977, 0.9767, 0.9011])
    mu        = (np.eye(3) - A) @ beta_mean
    return mu


# ── 7. Extraction betas OLS sur un panel ─────────────────────────────────────

def extract_ols_betas(yields_arr, Lambda):
    """Extrait les betas OLS pour chaque date."""
    betas = np.full((len(yields_arr), 3), np.nan)
    for t in range(len(yields_arr)):
        obs   = yields_arr[t]
        valid = ~np.isnan(obs)
        if valid.sum() < 4:
            continue
        betas[t] = np.linalg.lstsq(Lambda[valid], obs[valid], rcond=None)[0]
    return betas


# ── 8. Kalman Filter (BAM seul) ───────────────────────────────────────────────

def kalman_filter(yields, Lambda, A, mu, Q, H,
                  init_mean=None, init_cov=None):
    T, n_obs  = yields.shape
    n_factors = 3

    state_mean = init_mean.copy() if init_mean is not None else mu.copy()
    state_cov  = init_cov.copy()  if init_cov  is not None else np.eye(n_factors)

    filtered_states  = np.zeros((T, n_factors))
    predicted_states = np.zeros((T, n_factors))
    log_likelihood   = 0.0

    for t in range(T):
        obs   = yields[t]
        valid = ~np.isnan(obs)

        if not np.any(valid):
            filtered_states[t]  = state_mean
            predicted_states[t] = A @ state_mean + mu
            continue

        L   = Lambda[valid]
        y   = obs[valid]
        H_t = H[np.ix_(valid, valid)]

        pred_mean      = A @ state_mean + mu
        pred_cov       = A @ state_cov @ A.T + Q
        innovation     = y - L @ pred_mean
        innovation_cov = L @ pred_cov @ L.T + H_t

        sign, logdet = np.linalg.slogdet(innovation_cov)
        if sign > 0:
            log_likelihood -= 0.5 * (
                len(y) * np.log(2 * np.pi)
                + logdet
                + innovation @ np.linalg.solve(innovation_cov, innovation)
            )

        K          = pred_cov @ L.T @ np.linalg.pinv(innovation_cov)
        state_mean = pred_mean + K @ innovation
        state_cov  = (np.eye(n_factors) - K @ L) @ pred_cov

        filtered_states[t]  = state_mean
        predicted_states[t] = A @ state_mean + mu

    return filtered_states, predicted_states, log_likelihood


# ── 9. Kalman Filter avec ECB comme variable exogène ─────────────────────────

def kalman_filter_with_ecb(yields_bam, ecb_betas_arr, Lambda,
                            A, B, mu, Q, H,
                            init_mean=None, init_cov=None):
    """
    Équation de transition étendue :
    βt_BAM = A × β(t-1)_BAM + B × β(t-1)_ECB + mu + ηt
    """
    T, n_obs  = yields_bam.shape
    n_factors = 3

    state_mean = init_mean.copy() if init_mean is not None else mu.copy()
    state_cov  = init_cov.copy()  if init_cov  is not None else np.eye(n_factors)

    filtered_states  = np.zeros((T, n_factors))
    predicted_states = np.zeros((T, n_factors))
    log_likelihood   = 0.0

    for t in range(T):
        obs   = yields_bam[t]
        valid = ~np.isnan(obs)

        ecb_lag = ecb_betas_arr[t-1] if t > 0 else np.zeros(3)
        if np.any(np.isnan(ecb_lag)):
            ecb_lag = np.zeros(3)

        if not np.any(valid):
            filtered_states[t]  = state_mean
            predicted_states[t] = A @ state_mean + B @ ecb_lag + mu
            continue

        L   = Lambda[valid]
        y   = obs[valid]
        H_t = H[np.ix_(valid, valid)]

        pred_mean      = A @ state_mean + B @ ecb_lag + mu
        pred_cov       = A @ state_cov @ A.T + Q
        innovation     = y - L @ pred_mean
        innovation_cov = L @ pred_cov @ L.T + H_t

        sign, logdet = np.linalg.slogdet(innovation_cov)
        if sign > 0:
            log_likelihood -= 0.5 * (
                len(y) * np.log(2 * np.pi)
                + logdet
                + innovation @ np.linalg.solve(innovation_cov, innovation)
            )

        K          = pred_cov @ L.T @ np.linalg.pinv(innovation_cov)
        state_mean = pred_mean + K @ innovation
        state_cov  = (np.eye(n_factors) - K @ L) @ pred_cov

        filtered_states[t]  = state_mean
        predicted_states[t] = A @ state_mean + B @ ecb_betas_arr[t] + mu

    return filtered_states, predicted_states, log_likelihood


# ── 10. Estimation de B ───────────────────────────────────────────────────────

def estimate_B(bam_betas, ecb_betas):
    """Estime B par OLS : résiduelle BAM ~ B × β_ECB(t-1)"""
    A  = np.diag([0.9977, 0.9767, 0.9011])
    mu = np.zeros(3)
    B  = np.zeros((3, 3))

    for i in range(3):
        residual = bam_betas[1:, i] - (A[i, i] * bam_betas[:-1, i] + mu[i])
        X        = ecb_betas[:-1]
        valid    = ~np.isnan(residual) & ~np.any(np.isnan(X), axis=1)
        if valid.sum() < 10:
            continue
        B[i] = np.linalg.lstsq(X[valid], residual[valid], rcond=None)[0]

    print("\nMatrice B estimée (influence ECB → BAM) :")
    print(f"  β₀_ECB → β₀_BAM : {B[0, 0]:.4f}")
    print(f"  β₁_ECB → β₁_BAM : {B[1, 1]:.4f}")
    print(f"  β₂_ECB → β₂_BAM : {B[2, 2]:.4f}")
    return B


# ── 11. Grid search sur λ ─────────────────────────────────────────────────────

def grid_search_lambda(yields_arr, maturities, A, mu, Q, H,
                       lambda_grid=None):
    if lambda_grid is None:
        lambda_grid = np.arange(0.01, 2.0, 0.05)

    best_lambda = None
    best_rmse   = np.inf
    results     = []

    for lam in lambda_grid:
        Lambda              = nelson_siegel_loadings(maturities, lambda_val=lam)
        init_mean, init_cov = initialize_state(yields_arr, Lambda, n_warmup=60)
        filtered, _, _      = kalman_filter(
            yields_arr, Lambda, A, mu, Q, H,
            init_mean=init_mean, init_cov=init_cov
        )
        fitted = filtered @ Lambda.T
        rmse   = np.sqrt(np.nanmean((yields_arr - fitted) ** 2))
        results.append((lam, rmse))

        if rmse < best_rmse:
            best_rmse   = rmse
            best_lambda = lam

        print(f"λ={lam:.2f}  RMSE={rmse:.6f}", end='\r')

    print(f"\nMeilleur λ : {best_lambda:.4f}  RMSE : {best_rmse:.6f}")

    lams  = [r[0] for r in results]
    rmses = [r[1] for r in results]

    plt.figure(figsize=(10, 4))
    plt.plot(lams, rmses, 'b-o', markersize=3)
    plt.axvline(x=best_lambda, color='red', linestyle='--',
                label=f'λ optimal = {best_lambda:.4f}')
    plt.xlabel('λ')
    plt.ylabel('RMSE')
    plt.title('Grid Search sur λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / 'lambda_grid_search.png', dpi=150, bbox_inches='tight')
    plt.show()

    return best_lambda


# ── 12. Métriques vraies sans leakage ────────────────────────────────────────

def compute_metrics_true(yields_arr, filtered_states, predicted_states, Lambda):
    y_pred_betas   = predicted_states[:-1]
    yields_true    = yields_arr[1:]
    betas_true_ols = extract_ols_betas(yields_true, Lambda)
    errors_betas   = betas_true_ols - y_pred_betas

    rmse_betas = np.sqrt(np.nanmean(errors_betas ** 2, axis=0))
    mae_betas  = np.nanmean(np.abs(errors_betas), axis=0)

    yields_pred   = y_pred_betas @ Lambda.T
    errors_yields = yields_true - yields_pred
    rmse_global   = np.sqrt(np.nanmean(errors_yields ** 2))
    mae_global    = np.nanmean(np.abs(errors_yields))

    print("\n" + "=" * 60)
    print("MÉTRIQUES VRAIES — β prédit vs β OLS sur yields réels t+1")
    print("=" * 60)
    print(f"RMSE global yields : {rmse_global:.6f}")
    print(f"MAE  global yields : {mae_global:.6f}")
    print(f"\n{'Métrique':<10} {'β₀':>10} {'β₁':>10} {'β₂':>10}")
    print("-" * 40)
    print(f"{'RMSE':<10} {rmse_betas[0]:>10.6f} {rmse_betas[1]:>10.6f} {rmse_betas[2]:>10.6f}")
    print(f"{'MAE':<10} {mae_betas[0]:>10.6f} {mae_betas[1]:>10.6f} {mae_betas[2]:>10.6f}")

    return {
        'rmse_global':    rmse_global,
        'mae_global':     mae_global,
        'rmse_betas':     rmse_betas,
        'mae_betas':      mae_betas,
        'betas_true_ols': betas_true_ols
    }


# ── 13. Backtest mensuel — BAM seul ──────────────────────────────────────────

def backtest_monthly(yields_arr, bam_df, Lambda, A, mu, Q, H,
                     start_date='2022-01-01', horizon=22):
    dates        = bam_df.index
    start_idx    = np.searchsorted(dates, pd.Timestamp(start_date))
    month_starts = pd.date_range(
        start=dates[start_idx],
        end=dates[-horizon - 1],
        freq='MS'
    )

    print(f"\nBacktesting BAM seul sur {len(month_starts)} mois...")
    results = []

    for month_start in month_starts:
        cutoff_idx = np.searchsorted(dates, month_start) - 1
        if cutoff_idx < 60:
            continue

        yields_train        = yields_arr[:cutoff_idx + 1]
        init_mean_bt, init_cov_bt = initialize_state(yields_train, Lambda, n_warmup=60)
        filtered_bt, _, _   = kalman_filter(
            yields_train, Lambda, A, mu, Q, H,
            init_mean=init_mean_bt, init_cov=init_cov_bt
        )

        pred_betas = np.zeros((horizon, 3))
        state      = filtered_bt[-1].copy()
        for h in range(horizon):
            state         = A @ state + mu
            pred_betas[h] = state

        future_start  = cutoff_idx + 1
        future_end    = min(future_start + horizon, len(yields_arr))
        n_future      = future_end - future_start
        if n_future < 5:
            continue

        yields_future = yields_arr[future_start:future_end]
        dates_future  = dates[future_start:future_end]
        true_betas    = extract_ols_betas(yields_future, Lambda)
        pred_aligned  = pred_betas[:n_future]
        errors        = true_betas - pred_aligned

        rmse = np.sqrt(np.nanmean(errors ** 2, axis=0))
        mae  = np.nanmean(np.abs(errors), axis=0)

        results.append({
            'month':        month_start,
            'cutoff_date':  dates[cutoff_idx].date(),
            'n_days':       n_future,
            'rmse_beta0':   rmse[0], 'rmse_beta1': rmse[1], 'rmse_beta2': rmse[2],
            'mae_beta0':    mae[0],  'mae_beta1':  mae[1],  'mae_beta2':  mae[2],
            'pred_betas':   pred_aligned,
            'true_betas':   true_betas,
            'dates_future': dates_future[:n_future]
        })

        print(f"  {month_start.strftime('%Y-%m')}  "
              f"RMSE β₀={rmse[0]:.4f}  β₁={rmse[1]:.4f}  β₂={rmse[2]:.4f}")

    return results


# ── 14. Backtest mensuel — BAM + ECB ─────────────────────────────────────────

def backtest_monthly_with_ecb(yields_bam, yields_ecb, bam_df,
                               Lambda, A, B, mu, Q, H,
                               start_date='2022-01-01', horizon=22):
    dates        = bam_df.index
    start_idx    = np.searchsorted(dates, pd.Timestamp(start_date))
    month_starts = pd.date_range(
        start=dates[start_idx],
        end=dates[-horizon - 1],
        freq='MS'
    )

    print(f"\nBacktesting BAM + ECB sur {len(month_starts)} mois...")
    results = []

    for month_start in month_starts:
        cutoff_idx = np.searchsorted(dates, month_start) - 1
        if cutoff_idx < 60:
            continue

        yields_train_bam  = yields_bam[:cutoff_idx + 1]
        yields_train_ecb  = yields_ecb[:cutoff_idx + 1]
        ecb_betas_train   = extract_ols_betas(yields_train_ecb, Lambda)

        init_mean_bt, init_cov_bt = initialize_state(
            yields_train_bam, Lambda, n_warmup=60
        )
        filtered_bt, _, _ = kalman_filter_with_ecb(
            yields_train_bam, ecb_betas_train, Lambda, A, B, mu, Q, H,
            init_mean=init_mean_bt, init_cov=init_cov_bt
        )

        # Forecast — ECB lag constant (dernier ECB connu)
        last_ecb_lag = ecb_betas_train[-1].copy()
        if np.any(np.isnan(last_ecb_lag)):
            last_ecb_lag = np.zeros(3)

        pred_betas = np.zeros((horizon, 3))
        state      = filtered_bt[-1].copy()
        for h in range(horizon):
            state         = A @ state + B @ last_ecb_lag + mu
            pred_betas[h] = state

        future_start  = cutoff_idx + 1
        future_end    = min(future_start + horizon, len(yields_bam))
        n_future      = future_end - future_start
        if n_future < 5:
            continue

        yields_future = yields_bam[future_start:future_end]
        dates_future  = dates[future_start:future_end]
        true_betas    = extract_ols_betas(yields_future, Lambda)
        pred_aligned  = pred_betas[:n_future]
        errors        = true_betas - pred_aligned

        rmse = np.sqrt(np.nanmean(errors ** 2, axis=0))
        mae  = np.nanmean(np.abs(errors), axis=0)

        results.append({
            'month':        month_start,
            'cutoff_date':  dates[cutoff_idx].date(),
            'n_days':       n_future,
            'rmse_beta0':   rmse[0], 'rmse_beta1': rmse[1], 'rmse_beta2': rmse[2],
            'mae_beta0':    mae[0],  'mae_beta1':  mae[1],  'mae_beta2':  mae[2],
            'pred_betas':   pred_aligned,
            'true_betas':   true_betas,
            'dates_future': dates_future[:n_future]
        })

        print(f"  {month_start.strftime('%Y-%m')}  "
              f"RMSE β₀={rmse[0]:.4f}  β₁={rmse[1]:.4f}  β₂={rmse[2]:.4f}")

    return results


# ── 15. Plot backtest ─────────────────────────────────────────────────────────

def plot_backtest_results(results, title_suffix, filename):
    months     = [r['month'] for r in results]
    beta_names  = ['β₀ (Level)', 'β₁ (Slope)', 'β₂ (Curvature)']
    beta_colors = ['#2196F3', '#4CAF50', '#FF5722']

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        f'Backtesting Walk-Forward Mensuel — {title_suffix}\n'
        f'({months[0].strftime("%Y-%m")} → {months[-1].strftime("%Y-%m")})',
        fontsize=14, fontweight='bold'
    )

    for i in range(3):
        rmse_series = [r[f'rmse_beta{i}'] for r in results]
        ax = axes[0, i]
        ax.bar(months, rmse_series, color=beta_colors[i], alpha=0.7, width=20)
        ax.axhline(y=np.mean(rmse_series), color='black', linestyle='--',
                   linewidth=1, label=f'Moyenne = {np.mean(rmse_series):.4f}')
        ax.set_title(f'{beta_names[i]} — RMSE mensuel')
        ax.set_xlabel('Mois')
        ax.set_ylabel('RMSE')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

    last = results[-1]
    for i in range(3):
        ax = axes[1, i]
        ax.plot(last['dates_future'], last['true_betas'][:, i],
                color=beta_colors[i], linewidth=1.5, label='Réel OLS')
        ax.plot(last['dates_future'], last['pred_betas'][:, i],
                color='black', linewidth=1.2, linestyle='--', label='Prédit')
        ax.set_title(f'{beta_names[i]} — Dernier mois ({last["month"].strftime("%Y-%m")})')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

    all_errors = [[], [], []]
    for r in results:
        errors = r['true_betas'] - r['pred_betas']
        for i in range(3):
            valid = errors[:, i][~np.isnan(errors[:, i])]
            all_errors[i].extend(valid.tolist())

    for i in range(3):
        ax = axes[2, i]
        ax.hist(all_errors[i], bins=50, color=beta_colors[i], alpha=0.7, density=True)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.axvline(x=np.mean(all_errors[i]), color='red', linewidth=1.5,
                   linestyle='--', label=f'Biais = {np.mean(all_errors[i]):.4f}')
        rmse_moy = np.mean([r[f'rmse_beta{i}'] for r in results])
        ax.set_title(f'{beta_names[i]} — Distribution erreurs\nRMSE moy = {rmse_moy:.4f}')
        ax.set_xlabel('Erreur')
        ax.set_ylabel('Densité')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Backtest sauvegardé : {filename}")


# ── 16. Plot comparaison BAM seul vs BAM + ECB ───────────────────────────────

def plot_comparison(results_base, results_ecb):
    months_base = [r['month'] for r in results_base]
    months_ecb  = [r['month'] for r in results_ecb]
    beta_names  = ['β₀ (Level)', 'β₁ (Slope)', 'β₂ (Curvature)']
    beta_colors = ['#2196F3', '#4CAF50', '#FF5722']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Comparaison — DNS + Kalman seul vs DNS + Kalman + ECB',
        fontsize=14, fontweight='bold'
    )

    print("\n" + "=" * 65)
    print("COMPARAISON DNS seul vs DNS + ECB")
    print("=" * 65)
    print(f"{'Beta':<10} {'RMSE DNS':>12} {'RMSE ECB':>12} {'Amélioration':>14}")
    print("-" * 50)

    for i in range(3):
        rmse_base = [r[f'rmse_beta{i}'] for r in results_base]
        rmse_ecb  = [r[f'rmse_beta{i}'] for r in results_ecb]
        moy_base  = np.mean(rmse_base)
        moy_ecb   = np.mean(rmse_ecb)
        amelio    = (moy_base - moy_ecb) / moy_base * 100

        print(f"β{i:<9} {moy_base:>12.4f} {moy_ecb:>12.4f} {amelio:>+13.1f}%")

        ax = axes[i]
        ax.plot(months_base, rmse_base, color=beta_colors[i], linewidth=1.5,
                marker='o', markersize=3, label=f'DNS seul (moy={moy_base:.4f})')
        ax.plot(months_ecb, rmse_ecb, color='black', linewidth=1.5,
                linestyle='--', marker='s', markersize=3,
                label=f'DNS + ECB (moy={moy_ecb:.4f})')
        ax.set_title(f'{beta_names[i]}')
        ax.set_xlabel('Mois')
        ax.set_ylabel('RMSE')
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_base_vs_ecb.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparaison sauvegardée.")


# ── 17. Plot validation sans leakage ─────────────────────────────────────────

def plot_true_validation(bam_df, filtered_states, predicted_states,
                         metrics_true, Lambda, best_lambda):
    maturities  = parse_maturities(bam_df.columns)
    dates       = bam_df.index[1:]
    y_pred      = predicted_states[:-1]
    y_true_ols  = metrics_true['betas_true_ols']
    errors      = y_true_ols - y_pred
    beta_names  = ['β₀ (Level)', 'β₁ (Slope)', 'β₂ (Curvature)']
    beta_colors = ['#2196F3', '#4CAF50', '#FF5722']

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('DNS + Kalman — Validation sans leakage\n'
                 '(β prédit vs β OLS sur yields t+1)',
                 fontsize=14, fontweight='bold')

    for i in range(3):
        ax = axes[0, i]
        ax.plot(dates, y_true_ols[:, i], color=beta_colors[i],
                linewidth=0.8, label='OLS réel t+1')
        ax.plot(dates, y_pred[:, i], color='black', linewidth=0.6,
                linestyle='--', alpha=0.7, label='Prédit β(t+1|t)')
        ax.set_title(f'{beta_names[i]} — Prédit vs OLS réel')
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

    for i in range(3):
        ax = axes[1, i]
        ax.plot(dates, errors[:, i], color=beta_colors[i], linewidth=0.6)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.fill_between(dates, errors[:, i], 0, alpha=0.2, color=beta_colors[i])
        ax.set_title(f'{beta_names[i]} — Erreurs\n'
                     f'RMSE={metrics_true["rmse_betas"][i]:.4f}  '
                     f'MAE={metrics_true["mae_betas"][i]:.4f}')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(maturities, bam_df.values[-1], 'o-', color='black',
            linewidth=1.5, markersize=4, label='Observé')
    ax.plot(maturities, Lambda @ filtered_states[-1], 's--', color='#2196F3',
            linewidth=1.5, markersize=4, label='Filtré β(T|T)')
    ax.plot(maturities, Lambda @ predicted_states[-1], '^:', color='#FF5722',
            linewidth=1.5, markersize=4, label='Prédit β(T+1|T)')
    ax.set_title(f'Courbe des taux — Dernier jour ({bam_df.index[-1].date()})')
    ax.set_xlabel('Maturité (années)')
    ax.set_ylabel('Taux (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    for i in range(3):
        ax.scatter(y_true_ols[:, i], y_pred[:, i], alpha=0.2, s=2,
                   color=beta_colors[i], label=beta_names[i])
    lims = [min(y_true_ols.min(), y_pred.min()), max(y_true_ols.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, label='Parfait')
    ax.set_title('Scatter — Prédit vs OLS réel')
    ax.set_xlabel('OLS réel t+1')
    ax.set_ylabel('Prédit β(t+1|t)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    for i in range(3):
        ax.hist(errors[:, i], bins=50, alpha=0.5,
                color=beta_colors[i], label=beta_names[i], density=True)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_title('Distribution des erreurs (vraies)')
    ax.set_xlabel('Erreur')
    ax.set_ylabel('Densité')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dns_kalman_true_validation.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Validation sauvegardée.")


# ── 18. Forecast 1 mois ───────────────────────────────────────────────────────

def forecast_one_month(filtered_states, Lambda, A, mu, bam_df,
                       B=None, last_ecb_beta=None, horizon=22):
    last_state   = filtered_states[-1]
    last_date    = bam_df.index[-1]
    maturities   = parse_maturities(bam_df.columns)
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon
    )

    forecast_betas_arr = np.zeros((horizon, 3))
    state = last_state.copy()
    for h in range(horizon):
        if B is not None and last_ecb_beta is not None:
            state = A @ state + B @ last_ecb_beta + mu
        else:
            state = A @ state + mu
        forecast_betas_arr[h] = state

    forecast_beta_df    = pd.DataFrame(
        forecast_betas_arr, index=future_dates,
        columns=['beta0', 'beta1', 'beta2']
    )
    forecast_yields_arr = forecast_betas_arr @ Lambda.T
    forecast_yields_df  = pd.DataFrame(
        forecast_yields_arr, index=future_dates, columns=bam_df.columns
    )

    label = 'DNS + Kalman + ECB' if B is not None else 'DNS + Kalman'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Prévision {label} — 1 mois\n'
                 f'Depuis le {last_date.date()} → {future_dates[-1].date()}',
                 fontsize=13, fontweight='bold')

    beta_names  = ['β₀ (Level)', 'β₁ (Slope)', 'β₂ (Curvature)']
    beta_colors = ['#2196F3', '#4CAF50', '#FF5722']
    hist_betas  = filtered_states[-130:]
    hist_dates  = bam_df.index[-130:]

    for i in range(3):
        ax = axes[i]
        ax.plot(hist_dates, hist_betas[:, i], color=beta_colors[i],
                linewidth=1.0, label='Historique')
        ax.plot(future_dates, forecast_betas_arr[:, i], color='black',
                linewidth=1.5, linestyle='--', marker='o', markersize=3,
                label='Prévision 1 mois')
        ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=1)
        ax.set_title(f'{beta_names[i]}')
        ax.set_xlabel('Date')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = 'ecb' if B is not None else 'base'
    plt.savefig(OUTPUT_DIR / f'forecast_betas_1month_{suffix}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(maturities, bam_df.values[-1], 'o-', color='black',
            linewidth=2, markersize=5, label=f'Observé ({last_date.date()})')
    ax.plot(maturities, Lambda @ filtered_states[-1], 's--', color='#2196F3',
            linewidth=1.5, markersize=4, label='Filtré β(T|T)')
    colors_f = ['#FF9800', '#E91E63', '#9C27B0']
    for idx, day in enumerate([4, 9, 21]):
        ax.plot(maturities, forecast_yields_arr[day], '^:',
                color=colors_f[idx], linewidth=1.5, markersize=4,
                label=f'Prédit J+{day+1} ({future_dates[day].date()})')
    ax.set_title(f'Courbe des taux — Prévision 1 mois ({label})')
    ax.set_xlabel('Maturité (années)')
    ax.set_ylabel('Taux (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'forecast_curve_1month_{suffix}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    delta = forecast_betas_arr[-1] - filtered_states[-1]
    print(f"\n{'=' * 60}")
    print(f"PRÉVISION 1 MOIS — {label}")
    print(f"{'=' * 60}")
    print(f"Dernier état filtré β(T|T) :")
    print(f"  β₀={filtered_states[-1,0]:.4f}  β₁={filtered_states[-1,1]:.4f}  β₂={filtered_states[-1,2]:.4f}")
    print(f"Prévision β(T+22) :")
    print(f"  β₀={forecast_betas_arr[-1,0]:.4f}  β₁={forecast_betas_arr[-1,1]:.4f}  β₂={forecast_betas_arr[-1,2]:.4f}")
    print(f"Variation prévue :")
    print(f"  Δβ₀={delta[0]:+.4f} → courbe {'monte' if delta[0]>0 else 'descend'}")
    print(f"  Δβ₁={delta[1]:+.4f} → courbe {'se pentifie' if delta[1]>0 else 's aplatit'}")
    print(f"  Δβ₂={delta[2]:+.4f} → courbure {'augmente' if delta[2]>0 else 'diminue'}")

    return forecast_beta_df, forecast_yields_df


# ── 19. Résumé backtest ───────────────────────────────────────────────────────

def print_backtest_summary(results, label=''):
    print(f"\n{'=' * 65}")
    print(f"RÉSUMÉ BACKTEST — {label}")
    print(f"{'=' * 65}")
    for i, beta_name in enumerate(['β₀', 'β₁', 'β₂']):
        rmse_list = [r[f'rmse_beta{i}'] for r in results]
        mae_list  = [r[f'mae_beta{i}']  for r in results]
        print(f"\n{beta_name} :")
        print(f"  RMSE moyen  : {np.mean(rmse_list):.4f}")
        print(f"  RMSE médian : {np.median(rmse_list):.4f}")
        print(f"  RMSE max    : {np.max(rmse_list):.4f}  "
              f"({results[np.argmax(rmse_list)]['month'].strftime('%Y-%m')})")
        print(f"  RMSE min    : {np.min(rmse_list):.4f}  "
              f"({results[np.argmin(rmse_list)]['month'].strftime('%Y-%m')})")
        print(f"  MAE  moyen  : {np.mean(mae_list):.4f}")


# ── 20. Main ──────────────────────────────────────────────────────────────────

def main():
    # 1. Charger les données
    bam_df, ecb_df = load_data('./data/masi/bam_ecb_2004.csv')
    maturities     = parse_maturities(bam_df.columns)
    yields_arr     = bam_df.values
    yields_ecb     = ecb_df.values

    print(f"Données  : {bam_df.shape[0]} jours, {bam_df.shape[1]} maturités")
    print(f"Période  : {bam_df.index[0].date()} → {bam_df.index[-1].date()}")

    # 2. Paramètres fixes
    mu, A, Q, H = fixed_parameters(yields_arr.shape[1])

    # 3. Grid search sur λ
    print("\nGrid search sur λ...")
    best_lambda = grid_search_lambda(yields_arr, maturities, A, mu, Q, H)

    # 4. Lambda optimal + initialisation + mu
    Lambda_opt          = nelson_siegel_loadings(maturities, lambda_val=best_lambda)
    init_mean, init_cov = initialize_state(yields_arr, Lambda_opt, n_warmup=60)
    mu                  = estimate_mu(yields_arr, Lambda_opt)
    print(f"mu estimé : {mu}")

    # 5. Kalman filter BAM seul
    print("\nFiltre de Kalman BAM seul...")
    filtered_states, predicted_states, ll = kalman_filter(
        yields_arr, Lambda_opt, A, mu, Q, H,
        init_mean=init_mean, init_cov=init_cov
    )
    print(f"Log-vraisemblance : {ll:.4f}")

    # 6. Métriques vraies BAM seul
    metrics_true = compute_metrics_true(
        yields_arr, filtered_states, predicted_states, Lambda_opt
    )

    # 7. Validation visuelle BAM seul
    plot_true_validation(
        bam_df, filtered_states, predicted_states,
        metrics_true, Lambda_opt, best_lambda
    )

    # 8. Sauvegarder betas BAM
    beta_df   = pd.DataFrame(filtered_states, index=bam_df.index,
                             columns=['beta0', 'beta1', 'beta2'])
    fitted_df = pd.DataFrame(filtered_states @ Lambda_opt.T,
                             index=bam_df.index, columns=bam_df.columns)
    beta_df.to_csv(OUTPUT_DIR / 'dns_kalman_betas_bam.csv')
    fitted_df.to_csv(OUTPUT_DIR / 'dns_kalman_fitted_yields.csv')

    # 9. Backtest BAM seul
    backtest_base = backtest_monthly(
        yields_arr, bam_df, Lambda_opt, A, mu, Q, H,
        start_date='2022-01-01', horizon=22
    )
    print_backtest_summary(backtest_base, label='DNS + Kalman seul')
    plot_backtest_results(backtest_base, 'DNS + Kalman seul', 'backtest_base.png')

    # 10. Forecast 1 mois BAM seul
    forecast_beta_base, forecast_yields_base = forecast_one_month(
        filtered_states, Lambda_opt, A, mu, bam_df, horizon=22
    )
    forecast_beta_base.to_csv(OUTPUT_DIR / 'forecast_betas_base.csv')
    forecast_yields_base.to_csv(OUTPUT_DIR / 'forecast_yields_base.csv')

    # ── ECB pipeline ──────────────────────────────────────────────────────────

    # 11. Extraire betas ECB par OLS
    print("\nExtraction betas ECB...")
    ecb_betas_arr = extract_ols_betas(yields_ecb, Lambda_opt)
    beta_ecb_df   = pd.DataFrame(ecb_betas_arr, index=ecb_df.index,
                                 columns=['beta0_ecb', 'beta1_ecb', 'beta2_ecb'])
    beta_ecb_df.to_csv(OUTPUT_DIR / 'dns_kalman_betas_ecb.csv')

    # 12. Estimer B
    B = estimate_B(filtered_states, ecb_betas_arr)

    # 13. Kalman filter BAM + ECB
    print("\nFiltre de Kalman BAM + ECB...")
    init_mean_ecb, init_cov_ecb = initialize_state(yields_arr, Lambda_opt, n_warmup=60)
    filtered_ecb, predicted_ecb, ll_ecb = kalman_filter_with_ecb(
        yields_arr, ecb_betas_arr, Lambda_opt, A, B, mu, Q, H,
        init_mean=init_mean_ecb, init_cov=init_cov_ecb
    )
    print(f"Log-vraisemblance ECB : {ll_ecb:.4f}")

    # 14. Métriques vraies BAM + ECB
    metrics_ecb = compute_metrics_true(
        yields_arr, filtered_ecb, predicted_ecb, Lambda_opt
    )

    # 15. Sauvegarder betas BAM + ECB
    beta_ecb_bam_df = pd.DataFrame(filtered_ecb, index=bam_df.index,
                                   columns=['beta0', 'beta1', 'beta2'])
    beta_ecb_bam_df.to_csv(OUTPUT_DIR / 'dns_kalman_betas_bam_ecb.csv')

    # 16. Backtest BAM + ECB
    backtest_ecb = backtest_monthly_with_ecb(
        yields_arr, yields_ecb, bam_df,
        Lambda_opt, A, B, mu, Q, H,
        start_date='2022-01-01', horizon=22
    )
    print_backtest_summary(backtest_ecb, label='DNS + Kalman + ECB')
    plot_backtest_results(backtest_ecb, 'DNS + Kalman + ECB', 'backtest_ecb.png')

    # 17. Comparaison BAM seul vs BAM + ECB
    plot_comparison(backtest_base, backtest_ecb)

    # 18. Forecast 1 mois BAM + ECB
    last_ecb_beta = ecb_betas_arr[-1].copy()
    if np.any(np.isnan(last_ecb_beta)):
        last_ecb_beta = np.zeros(3)

    forecast_beta_ecb, forecast_yields_ecb = forecast_one_month(
        filtered_ecb, Lambda_opt, A, mu, bam_df,
        B=B, last_ecb_beta=last_ecb_beta, horizon=22
    )
    forecast_beta_ecb.to_csv(OUTPUT_DIR / 'forecast_betas_ecb.csv')
    forecast_yields_ecb.to_csv(OUTPUT_DIR / 'forecast_yields_ecb.csv')

    print("\nPipeline complet terminé.")
    print(f"Outputs sauvegardés dans : {OUTPUT_DIR}")


if __name__ == '__main__':
    main()