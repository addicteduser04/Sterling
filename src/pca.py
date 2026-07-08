import pandas as pd 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./data/master_data_cleaned.csv")
features = ['Taux directeur','IPC','PIB','europe_beta0','europe_beta1','europe_beta2']
cibles = ['beta0','beta1','beta2']

X = data[features]
y = data[cibles]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
pca_features = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2', 'PC3'])

print("Transformed Data shape:", pca_df.shape)
print("\nExplained Variance Ratio per component:", pca.explained_variance_ratio_)