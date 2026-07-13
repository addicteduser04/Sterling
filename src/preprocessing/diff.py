import pandas as pd 
from sklearn.preprocessing import StandardScaler
target_cols = ['PIB','IPC','Taux directeur','beta0','beta1','beta2','europe_beta0','europe_beta1','europe_beta2']
data = pd.read_csv("./data/analysis_results/master_data_cleaned.csv", index_col=0)


target_diff_cols = []
for col in target_cols:
    diff_col = f'{col}_diff'
    data[diff_col] = data[col].diff().fillna(0)
    target_diff_cols.append(diff_col)

scaler = StandardScaler()
data[target_diff_cols] = scaler.fit_transform(data[target_diff_cols])

print(data.head())
data.to_csv('./data/feature_engineering/data_standarized.csv', index=True)