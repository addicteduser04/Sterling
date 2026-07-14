import pandas as pd 
from scipy.stats import pearsonr
from pathlib import Path

x_path = Path("/home/sifeddine/Documents/Sterling/Sterling/data/feature_engineering/X.csv")
y_path = Path("/home/sifeddine/Documents/Sterling/Sterling/data/feature_engineering/y.csv")

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    return data


def calculate_correlation(x_path, y_path):
    resultats =[]
    X = load_data(x_path)
    y = load_data(y_path)
    for col in X.columns:
        for cible in y.columns:
            correlation, p_value = pearsonr(X[col], y[cible])
            resultats.append({'feature': col, 'cible': cible, 'correlation': correlation, 'p_value': p_value})
    df_resultats = pd.DataFrame(resultats)
   
    return df_resultats

df_results = calculate_correlation(x_path, y_path)
df_results.to_csv("./../data/analysis_results/correlation_results.csv", index=False)
print(df_results.head())