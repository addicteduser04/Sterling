import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf
import os

# Ensure the output directory exists so savefig doesn't throw a FileNotFoundError
os.makedirs('./data/models/', exist_ok=True)

# 1. Load Data
data = pd.read_csv("./data/preprocessed/betas_nelson_siegel_bam.csv", index_col=0)
data = data.drop(columns='rmse', errors='ignore')

# --- THE DATETIME FIX ---
# Convert index to a proper Datetime format and set the Month-Start frequency
data.index = pd.to_datetime(data.index)
data = data.asfreq('MS') 

def model(colonne):
    # Using dropna() in case .asfreq() introduced NaNs for missing months
    ts_data = data[colonne].dropna() 
    
    # 2. Determine the lag order (p)
    plot_pacf(ts_data, lags=15)
    # Save the PACF plot instead of showing it so the loop doesn't get blocked
    plt.savefig(f'./data/models/PACF_{colonne}.png')
    plt.close() 

    # Let's assume PACF showed significance at lag 1
    p = 1 

    # 3. Fit the AR model
    model_ar = AutoReg(ts_data[:170], lags=p)
    results = model_ar.fit()

    print(f"\n--- Model Summary: {colonne} ---")
    print(results.summary())

    # 4. Make Predictions
    forecast_start = len(ts_data) - 100
    forecast_end = len(ts_data) + 5

    predictions = results.predict(start=forecast_start, end=forecast_end, dynamic=False)
    
    # 5. Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(ts_data.index, ts_data, label='Actual Data')
    plt.plot(predictions.index, predictions, color='red', label='AR Predictions')
    plt.legend()
    plt.title(f"Autoregressive Model AR({p}) - {colonne}")
    
    # Save the figure FIRST, then close it to free up memory
    plt.savefig(f'./data/models/AR_{colonne}.png')
    plt.close() 
    
    # --- FIX: Return the actual predictions series so we can save them ---
    return predictions

# --- FIX: Create a dictionary to hold the predictions ---
all_predictions = {}

# 6. Run the loop
for col in data.columns:
    # Save the returned predictions into our dictionary, using the column name as the key
    all_predictions[col] = model(col)

# --- FIX: Convert to DataFrame and Export ---
# Pandas will automatically align all the dates nicely
df_predictions = pd.DataFrame(all_predictions)
# Save to CSV (Fill in your desired path here!)
df_predictions.to_csv("./data/models/ar.csv")

print("\nPredictions successfully saved!")