import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from data_acquisition.data_collect import RatesCollector, EuroRatesCollector

def nelson_siegel(tau, beta0, beta1, beta2, lam):
    """Yield curve NS pour une maturité tau (en années)"""
    factor1 = 1.0
    factor2 = (1 - np.exp(-tau / lam)) / (tau / lam)
    factor3 = factor2 - np.exp(-tau / lam)
    return beta0 + beta1 * factor2 + beta2 * factor3

def fit_ns_single_date(yields_row, maturities, lam_grid=None):
    """
    Fitter NS sur une seule cross-section.
    yields_row : Series avec index = maturités, values = yields
    maturities : array des maturités en années
    """
    # Retirer les NaN
    mask = ~yields_row.isna()
    tau = maturities[mask]
    y = yields_row.values[mask]
    
    if len(tau) < 4:
        return pd.Series({'beta0': np.nan, 'beta1': np.nan, 
                         'beta2': np.nan, 'lambda': np.nan, 'rmse': np.nan})
    
    # Grid search sur lambda, OLS sur les betas
    if lam_grid is None:
        lam_grid = np.arange(0.5, 5.0, 0.1)
    
    best_rmse = np.inf
    best_params = None
    
    for lam in lam_grid:
        # Construire la matrice de design (OLS linéaire sur betas)
        f1 = np.ones(len(tau))
        f2 = (1 - np.exp(-tau / lam)) / (tau / lam)
        f3 = f2 - np.exp(-tau / lam)
        X = np.column_stack([f1, f2, f3])
        
        # OLS : beta = (X'X)^-1 X'y
        try:
            betas = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = X @ betas
            rmse = np.sqrt(np.mean((y - y_hat) ** 2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (*betas, lam, rmse)
        except:
            continue
    
    if best_params is None:
        return pd.Series({'beta0': np.nan, 'beta1': np.nan,
                         'beta2': np.nan, 'lambda': np.nan, 'rmse': np.nan})
    
    return pd.Series({
        'beta0': best_params[0],
        'beta1': best_params[1],
        'beta2': best_params[2],
        'lambda': best_params[3],
        'rmse':   best_params[4]
    })

def fit_ns_panel(df):
    """
    Fitter NS sur tout le panel.
    df : DataFrame avec Date en index, colonnes = maturités (ex: '3M', '1Y', '10Y')
    Retourne un DataFrame avec beta0, beta1, beta2, lambda, rmse par date.
    """
    # Mapping colonnes → maturités en années
    maturity_map = {
        '3M': 0.25, '6M': 0.5, '1Y': 1.0, '2Y': 2.0,
        '3Y': 3.0,  '4Y': 4.0, '5Y': 5.0, '6Y': 6.0,
        '7Y': 7.0,  '8Y': 8.0, '9Y': 9.0, '10Y': 10.0,
        '11Y': 11.0,'12Y': 12.0,'13Y': 13.0,'14Y': 14.0,
        '15Y': 15.0,'16Y': 16.0,'17Y': 17.0,'18Y': 18.0,
        '19Y': 19.0,'20Y': 20.0,'30Y': 30.0
    }
    
    # Garder seulement les colonnes reconnues
    cols = [c for c in df.columns if c in maturity_map]
    maturities = np.array([maturity_map[c] for c in cols])
    
    results = df[cols].apply(
        lambda row: fit_ns_single_date(row, maturities),
        axis=1
    )
    
    return results

# ── Usage ──────────────────────────────────────────────────────────────────────
def calculate_and_save_betas(collector_class, output_path):
    collector = collector_class()
    df = collector.collect_data()

    betas = fit_ns_panel(df)
    betas = betas.set_index(df["Date"])  # Assurer que l'index est correct

    betas.to_csv(output_path)
    print(betas.head(10))
    print(f"\nRMSE moyen : {betas['rmse'].mean():.6f}")
    print(f"Dates avec fit échoué : {betas['beta0'].isna().sum()}")
    return betas

