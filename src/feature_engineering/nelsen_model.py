import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from src.data_collect import RatesCollector, EuroRatesCollector

# Maturities in years: 3mo, 6mo, 1yr, 2yr, 5yr, 10yr, 30yr
maturities = np.array([0.08, 0.4, 0.65, 1.25, 3.82, 5.33, 8.99, 13.08, 19.16, 28.84])
# Observed yields in percentages (or decimals)
market_yields = np.array([2.21, 2.23, 2.28, 2.35, 2.74, 2.89, 3.19, 3.63, 3.74, 4.1])

# 2. Function to calculate Nelson-Siegel factor loadings matrix
def get_factor_loadings(tau, lam):
    """
    Calculates the design matrix X for the linear regression step.
    Column 1: Level loading (always 1)
    Column 2: Slope loading
    Column 3: Curvature loading
    """
    # Avoid division by zero at tau = 0
    tau = np.where(tau == 0, 1e-5, tau)
    
    # Calculate components
    decay_term = (1 - np.exp(-lam * tau)) / (lam * tau)
    hump_term = decay_term - np.exp(-lam * tau)
    
    # Construct the X matrix
    X = np.column_stack((np.ones_like(tau), decay_term, hump_term))
    return X

# 3. Objective function to minimize (Sum of Squared Residuals)
def nelson_siegel_ssr(lam, tau, yields):
    X = get_factor_loadings(tau, lam)
    # Fit OLS without an intercept since column 1 acts as our beta_1 intercept
    model = LinearRegression(fit_intercept=False).fit(X, yields)
    predictions = model.predict(X)
    ssr = np.sum((yields - predictions) ** 2)
    return ssr

# 4. Step 1: Optimize lambda using a bounded scalar search
# Lambda typically ranges between 0.05 and 2.0
result = minimize_scalar(nelson_siegel_ssr, bounds=(0.01, 2.5), args=(maturities, market_yields), method='bounded')
optimal_lambda = result.x # type: ignore

# 5. Step 2: Extract the optimal Betas using OLS with the fixed optimal lambda
X_optimal = get_factor_loadings(maturities, optimal_lambda)
final_model = LinearRegression(fit_intercept=False).fit(X_optimal, market_yields)
beta_1, beta_2, beta_3 = final_model.coef_

# --- Output Results ---
print("--- Nelson-Siegel Calibration Results ---")
print(f"Optimal Lambda (λ): {optimal_lambda:.4f}")
print(f"Beta 1 (Level):     {beta_1:.4f}")
print(f"Beta 2 (Slope):     {beta_2:.4f}")
print(f"Beta 3 (Curvature): {beta_3:.4f}")

# Check model fit quality
fitted_yields = final_model.predict(X_optimal)
print("\n--- Market vs Fitted Yields ---")
for m, o, f in zip(maturities, market_yields, fitted_yields):
    print(f"Maturity: {m:5.2f}Y | Market: {o:.2f}% | Fitted: {f:.2f}% | Error: {o-f:+.4f}%")
