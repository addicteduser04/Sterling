import pandas as pd 
from sklearn.linear_model import LinearRegression


cibles = ['beta0','beta1','beta2']
features = ['Taux directeur','IPC','PIB','europe_beta0','europe_beta1','europe_beta2']
resultats =[]


data = pd.read_csv("./data/master_data_cleaned.csv")
model = LinearRegression()
for feature in features:
    for cible in cibles:
        X = data[[feature]]
        y = data[cible]
        model.fit(X, y)
        score = model.score(X, y)
        resultats.append({'feature': feature, 'cible': cible, 'score': score})

for cible in cibles:
    X = data[features]
    y = data[cible]
    model.fit(X, y)
    score = model.score(X, y)
    resultats.append({'feature': 'all_features', 'cible': cible, 'score': score})

df_resultats = pd.DataFrame(resultats)

print(df_resultats)
df_resultats.to_csv("./data/linear_regression_results.csv", index=False)