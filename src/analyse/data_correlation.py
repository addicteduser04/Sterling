from data_collect import RatesCollector, EuroRatesCollector, GDPCollector, CPICollector, TauxDirecteurCollector, BetaCollector
import pandas as pd 
from scipy.stats import pearsonr

def load_data(collector):
    data = collector.collect_data()
    return data

ecb_data = load_data(EuroRatesCollector())
bam_data = load_data(RatesCollector())
pib_data = load_data(GDPCollector())
cpi_data = load_data(CPICollector())
taux_directeur_data = load_data(TauxDirecteurCollector())
beta_bam_data = load_data(BetaCollector())
beta_ecb_data = load_data(BetaCollector(file_path="./data/betas/ecb/betas_nelson_siegel_ecb.csv"))  
beta_ecb_data.columns = ['Date', 'europe_beta0', 'europe_beta1', 'europe_beta2', 'europe_lambda', 'europe_rmse']

cibles = ['beta0','beta1','beta2']
features = ['Taux directeur','IPC','PIB','europe_beta0','europe_beta1','europe_beta2']
resultats =[]
df_master = pd.merge(beta_bam_data, taux_directeur_data, on='Date', how='left').dropna()
df_master = pd.merge(df_master, cpi_data, on='Date', how='left').dropna()
df_master = pd.merge(df_master, pib_data, on='Date', how='left').dropna()
df_master = pd.merge(df_master, beta_ecb_data, on='Date', how='left').dropna()
print(df_master.columns)
for col in features:
    for cible in cibles:
        correlation, p_value = pearsonr(df_master[col], df_master[cible])
        resultats.append({'feature': col, 'cible': cible, 'correlation': correlation, 'p_value': p_value})
df_resultats = pd.DataFrame(resultats)
df_resultats.to_csv("./data/analysis_results/correlation_results.csv", index=False)






df_master.to_csv("./data/analysis_results/master_data.csv", index=False)
df_master = df_master.drop(columns = ['lambda','rmse','europe_lambda','europe_rmse'])
df_master.to_csv("./data/analysis_results/master_data_cleaned.csv", index=False)