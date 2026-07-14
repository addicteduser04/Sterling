import pandas as pd 
from sklearn.preprocessing import StandardScaler
from pathlib import Path


target_cols = ['IPC','Taux directeur','beta0','beta1','beta2','europe_beta0','europe_beta1','europe_beta2']

def differentiate_standarize(file_path,target = target_cols):
    data = pd.read_csv(file_path, index_col=0)
    target_diff_cols = []
    for col in target:
        diff_col = f'{col}_diff'
        data[diff_col] = data[col].pct_change()
        target_diff_cols.append(diff_col)

    scaler = StandardScaler()
    data[target_diff_cols] = scaler.fit_transform(data[target_diff_cols])

    print(data.head())
    data.to_csv('./data/preprocessed/data_standarized.csv', index=True)
    return data

#export differentiate_standarize