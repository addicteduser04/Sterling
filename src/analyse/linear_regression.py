import sys
import os

# Adds the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd 
from sklearn.linear_model import LinearRegression
from feature_engineering.feature import build_lagged_dataset
from feature_engineering.merge import merge_data

df = merge_data()
X,y = build_lagged_dataset(df)

resultats =[]

model = LinearRegression()

for feature in X.columns:
    for cible in y.columns:
        a = X[[feature]]
        b = y[cible]
        model.fit(a, b)
        score = model.score(a, b)
        resultats.append({'feature': feature, 'cible': cible, 'score': score})

for cible in y.columns:
    model.fit(X, y)
    score = model.score(X, y)
    resultats.append({'feature': 'all_features', 'cible': cible, 'score': score})

df_resultats = pd.DataFrame(resultats)

print(df_resultats)
df_resultats.to_csv("./data/analysis_results/linear_regression_results.csv", index=False)



