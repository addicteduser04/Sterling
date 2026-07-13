from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

data = pd.read_csv("./data/master_data_cleaned.csv")
cibles = ['beta0','beta1','beta2']
features = ['Taux directeur','IPC','PIB','europe_beta0','europe_beta1','europe_beta2']
resultats = [] 
for cible in cibles:
    X = data[features]
    y = data[cible]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    importance = model.feature_importances_
    for i, feature in enumerate(features):
        resultats.append({'cible': cible, 'feature': feature, 'importance': importance[i]})
df_resultats = pd.DataFrame(resultats)
print(df_resultats)
df_resultats.to_csv("./data/analysis_results/random_forest_results.csv", index=False)