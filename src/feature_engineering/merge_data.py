from src.data_acquisition.data_collect import RatesCollector, EuroRatesCollector, CPICollector, TauxDirecteurCollector, BetaCollector
import pandas as pd


def load_data(collector):
    data = collector.collect_data()
    return data

cpi_data = load_data(CPICollector())
taux_directeur_data = load_data(TauxDirecteurCollector())
beta_bam_data = load_data(BetaCollector())
beta_ecb_data = load_data(BetaCollector(file_path="./data/betas/ecb/betas_nelson_siegel_ecb.csv"))  
beta_ecb_data.columns = ['Date', 'europe_beta0', 'europe_beta1', 'europe_beta2', 'europe_lambda', 'europe_rmse']

df_master = pd.merge(beta_bam_data, taux_directeur_data, on='Date', how='left').dropna()
#df_master = pd.merge(df_master, cpi_data, on='Date', how='left').dropna()
df_master = pd.merge(df_master, beta_ecb_data, on='Date', how='left').dropna()
print(df_master.columns)

df_master.to_csv("./data/feature_engineering/merged_data.csv", index=False)
df_master = df_master.drop(columns = ['lambda','rmse','europe_lambda','europe_rmse'])
df_master.to_csv("./data/feature_engineering/merged_data_cleaned.csv", index=False)