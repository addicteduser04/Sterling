import pandas as pd 

target_cols = ['Taux directeur','beta0','beta1','beta2','europe_beta0','europe_beta1','europe_beta2']
data = pd.read_csv("./data/analysis_results/master_data_cleaned.csv", index_col=0)

data = data.drop(columns = ['PIB','IPC'])

for col in target_cols:
    data[f'{col}_diff'] = data[col].diff().fillna(0)
print(data.head())
data.to_csv('./data/feature_engineering/data.csv', index=True)