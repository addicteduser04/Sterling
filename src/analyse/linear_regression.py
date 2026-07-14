import pandas as pd 
from sklearn.linear_model import LinearRegression


X = pd.read_csv("./../data/feature_engineering/X.csv", index_col=0)
y = pd.read_csv("./../data/feature_engineering/y.csv", index_col=0)

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
df_resultats.to_csv("./../data/analysis_results/linear_regression_results.csv", index=False)