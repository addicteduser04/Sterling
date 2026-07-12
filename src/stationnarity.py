import pandas as pd
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv("./data/feature_engineering/data.csv", index_col=0)


the_adf_test_results = {}
for column in data.columns:
    result = adfuller(data[column])
    the_adf_test_results[column] = {
        'ADF Statistic': result[0],
        'p-value': result[1]
    }
result_df = pd.DataFrame.from_dict(the_adf_test_results, orient='index')
result_df.to_csv("./data/analysis_results/adf_test_results.csv", index=True)