

import pandas as pd
import numpy as np
from pathlib import Path
from merge_data import merge_data

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "feature_engineering" / "data.csv"
OUTPUT_FEATURES_PATH = ROOT / "data" / "feature_engineering" / "X.csv"
OUTPUT_TARGETS_PATH = ROOT / "data" / "feature_engineering" / "y.csv"

# Features to lag: all variables except diff columns (those are already derived)
BASE_FEATURES = [
    "beta0",
    "beta1",
    "beta2",
    "Taux directeur",
    "IPC",
    "europe_beta0",
    "europe_beta1",
    "europe_beta2",
]

# Target variables to predict (next month's values)
TARGET_VARS = ["beta0", "beta1", "beta2"]


def build_lagged_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create lagged features and future targets for supervised learning.

    At each time t:
    - Features: values from time t-1 (lag 1) for all BASE_FEATURES
    - Targets: values from time t+1 (shift -1) for TARGET_VARS
    
    Returns:
        X: DataFrame with lagged features (index starts at t=1 because of lag)
        y: DataFrame with future targets aligned to X
    """
    df = df.copy()
    
    # Create lagged features (shift positive = lag = past values)
    X = pd.DataFrame(index=df.index)
    for feat in BASE_FEATURES:
        if feat == 'IPC':
            X[f"{feat}_variation"] = df[feat].pct_change()
        X[f"{feat}_lag1"] = df[feat].shift(1)
    
    # Create future targets (shift negative = look ahead)
    y = pd.DataFrame(index=df.index)
    for target in TARGET_VARS:
        y[f"{target}_t+1"] = df[target].shift(-1)
    
    # Remove rows with NaN (first row has NaN lag, last row has NaN target)
    valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y


def main() -> None:
    # Load preprocessed data
    df = merge_data()
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create lagged dataset
    X, y = build_lagged_dataset(df)
    print(f"\nLagged features X shape: {X.shape}")
    print(f"Targets y shape: {y.shape}")
    
    print("\nFeatures (first 5 rows):")
    print(X.head())
    
    print("\nTargets (first 5 rows):")
    print(y.head())
    
    # Save to CSV
    X.to_csv(OUTPUT_FEATURES_PATH, index=True)
    y.to_csv(OUTPUT_TARGETS_PATH, index=True)
    
    print(f"\nFeatures saved to: {OUTPUT_FEATURES_PATH.name}")
    print(f"Targets saved to: {OUTPUT_TARGETS_PATH.name}")
    
    # Summary statistics
    print("\nFeatures summary:")
    print(X.describe().round(4))
    print("\nTargets summary:")
    print(y.describe().round(4))


if __name__ == "__main__":
    main()
