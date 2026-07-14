import sys
import os

# Adds the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_acquisition.data_collect import CPICollector, TauxDirecteurCollector, BetaCollector
import pandas as pd


def load_data(collector):
    data = collector.collect_data()
    return data

cpi_data = load_data(CPICollector())
taux_directeur_data = load_data(TauxDirecteurCollector())
beta_bam_data = load_data(BetaCollector())
beta_ecb_data = load_data(BetaCollector(file_path="./data/preprocessed/betas_nelson_siegel_ecb.csv"))  
beta_ecb_data.columns = ['Date', 'europe_beta0', 'europe_beta1', 'europe_beta2', 'europe_lambda', 'europe_rmse']

def merge_data(x = beta_bam_data, y = beta_ecb_data , z = cpi_data, t = taux_directeur_data):
    df_master = pd.merge(x, t, on='Date', how='left').dropna()
    df_master = pd.merge(df_master, z, on='Date', how='left').dropna()
    df_master = pd.merge(df_master, y, on='Date', how='left').dropna()
    print(df_master.columns)

    df_master.to_csv("./data/feature_engineering/merged_data.csv", index=False)
    df_master = df_master.drop(columns = ['lambda','rmse','europe_lambda','europe_rmse'])
    df_master.to_csv("./data/feature_engineering/merged_data_cleaned.csv", index=False)
    return df_master
