from statsmodels.tsa.api import VAR
import pandas as pd
import matplotlib.pyplot as plt

data =pd.read_csv("./data/feature_engineering/data.csv", index_col=0)
data = data.drop(columns = ['Taux directeur_diff','Taux directeur','beta0','beta1','beta2','europe_beta0','europe_beta1','europe_beta2'])
print(data.head())  
model = VAR(data)
lag_selection = model.select_order(maxlags=4)
print(lag_selection.summary())


results = model.fit(maxlags=4, ic='aic')
print(results.summary())


irf = results.irf(10)
irf.plot(orth=True)
plt.suptitle("Fonctions de Réponse Impulsionnelle (IRF)", fontsize=16)
plt.savefig('./data/visualisation/IRF.jpg')

fevd = results.fevd(10)
fevd.plot()
plt.suptitle("Décomposition de la Variance (FEVD)", fontsize=16)
plt.savefig('./data/visualisation/FEVD.jpg')