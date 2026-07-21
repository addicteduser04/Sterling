import pandas as pd
import numpy as np

def calculate_nelson_siegel_yields(betas_df, maturities, lambda_val=0.0609):
    """
    Takes a DataFrame of Nelson-Siegel betas and generates a yield curve DataFrame.
    
    Parameters:
    - betas_df: DataFrame containing 'beta0', 'beta1', and 'beta2' columns.
    - maturities: List or array of maturities (e.g., in months or years).
    - lambda_val: The decay parameter (lambda) used in your original NS fitting.
    
    Returns:
    - A DataFrame where the index matches betas_df and columns are the maturities.
    """
    
    # Create an empty DataFrame to store the forecasted yields
    # Index is the dates from your betas forecast
    yields_df = pd.DataFrame(index=betas_df.index)
    
    # Extract the beta series for cleaner math below
    beta0 = betas_df['beta0']
    beta1 = betas_df['beta1']
    beta2 = betas_df['beta2']
    
    for tau in maturities:
        # Prevent division by zero if maturity is 0
        if tau == 0:
            yields_df[tau] = beta0 + beta1
            continue
            
        # The Nelson-Siegel mathematical terms
        term1 = (1 - np.exp(-lambda_val * tau)) / (lambda_val * tau)
        term2 = term1 - np.exp(-lambda_val * tau)
        
        # Calculate the yield for this specific maturity across ALL dates simultaneously
        yields_df[tau] = beta0 + (beta1 * term1) + (beta2 * term2)
        
    return yields_df

# ==========================================
# 2. Example: How to use it with your data
# ==========================================

# Assuming 'forecasted_betas' is the DataFrame you got from your AR model predictions
# It should look something like this:
#             beta0     beta1     beta2
# 2026-08-01  0.035    -0.012     0.005
# 2026-09-01  0.036    -0.011     0.006

# Create some dummy forecast data to test the function
dummy_dates = pd.date_range(start='2026-08-01', periods=5, freq='MS')
forecasted_betas = pd.DataFrame({
    'beta0': [0.035, 0.036, 0.037, 0.036, 0.038],
    'beta1': [-0.012, -0.011, -0.010, -0.008, -0.005],
    'beta2': [0.005, 0.006, 0.004, 0.007, 0.008]
}, index=dummy_dates)

# Define the maturities you want to forecast (e.g., 1 to 30 years)
maturities_list = [1, 2, 3, 5, 7, 10, 20, 30]

# Generate the Yield Curves!
# NOTE: Make sure lambda_val perfectly matches the lambda used when originally generating the betas.
forecasted_yield_curve = calculate_nelson_siegel_yields(
    betas_df=forecasted_betas, 
    maturities=maturities_list, 
    lambda_val=0.0609 
)

print("\n--- Forecasted Yield Curves ---")
print(forecasted_yield_curve)

# Save the final yield curves to CSV
forecasted_yield_curve.to_csv('./data/analysis_results/forecasted_yield_curve.csv')