import pandas as pd 
from scipy.stats import pearsonr

from feature_engineering.feature_engineering import build_lagged_dataset
from feature_engineering.merge_data import merge_data

df = merge_data()
X,y = build_lagged_dataset(df)

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    return data


def calculate_correlation(features =X, cibles =y):
    resultats =[]
    for col in X.columns:
        for cible in y.columns:
            correlation, p_value = pearsonr(X[col], y[cible])
            resultats.append({'feature': col, 'cible': cible, 'correlation': correlation, 'p_value': p_value})
    df_resultats = pd.DataFrame(resultats)
   
    return df_resultats

df_results = calculate_correlation(X, y)
df_results.to_csv("./../data/analysis_results/correlation_results.csv", index=False)
print(df_results.head())