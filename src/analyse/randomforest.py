from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# 1. Load Data
X = pd.read_csv("./data/feature_engineering/X.csv", index_col=0)
cibles = pd.read_csv("./data/feature_engineering/y.csv", index_col=0)

resultats = [] 
result = pd.DataFrame(index=X.index) # Empty DataFrame to hold predictions

# 2. Loop through targets
for cible in cibles.columns:
    y = cibles[cible]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = False)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate and print MSE so it isn't lost
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE for {cible}: {mse}")
    
    # FIX: Create DataFrame with the exact length of y_pred and align to the test index
    # Also, dynamically name the column so they don't overwrite each other
# NEW
    pred_result = pd.DataFrame({
        f"{cible}_actual": y_test, 
        f"{cible}_pred": y_pred
    }, index=y_test.index)    
    # FIX: Use .join() to safely merge on the Date index
    result = result.join(pred_result, how='left').dropna()
    
    # Extract Feature Importances
    importance = model.feature_importances_
    for i, feature in enumerate(X.columns):
        resultats.append({'cible': cible, 'feature': feature, 'importance': importance[i]})

# 3. Save Results
df_resultats = pd.DataFrame(resultats)
print(df_resultats)

# Save feature importances
df_resultats.to_csv("./data/analysis_results/random_forest_results.csv", index=False)

# Optional: You probably want to save your predictions too!
result.to_csv("./data/analysis_results/predictions.csv", index = False)