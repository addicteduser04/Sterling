from data_acquisition.data_collect import RatesCollector, EuroRatesCollector, GDPCollector, CPICollector, TauxDirecteurCollector, BetaCollector
import pandas as pd 
from scipy.stats import pearsonr

collectors_ = [RatesCollector, EuroRatesCollector, GDPCollector, CPICollector, TauxDirecteurCollector, BetaCollector]

def load_data(collectors):
    data = []
    for collector in collectors:
        df = collector.collect_data()
        data.append(df)
    ecb_beta = collector.collect_data("./data/betas/ecb/betas_nelson_siegel_ecb.csv")
    ecb_beta.columns =  ['Date', 'europe_beta0', 'europe_beta1', 'europe_beta2', 'europe_lambda', 'europe_rmse']
    data.append(df)
    return data

cibles_col = ['beta0','beta1','beta2']
features_col = ['Taux directeur','IPC','PIB','europe_beta0','europe_beta1','europe_beta2']


def merge_data(list):
    df_master = pd.DataFrame()
    for i in len(list):
        df = list[i]
        df_master = pd.merge(df_master, df, on='Date', how='left').dropna()
    df_master.to_csv("./data/analysis_results/master_data.csv", index=False)
    return df_master

def calculate_correlation(cibles = cibles_col, features = features_col, collectors = collectors_):
    resultats =[]
    list = load_data(collectors)
    df_master = merge_data(list)
    for col in features:
        for cible in cibles:
            correlation, p_value = pearsonr(df_master[col], df_master[cible])
            resultats.append({'feature': col, 'cible': cible, 'correlation': correlation, 'p_value': p_value})
    df_resultats = pd.DataFrame(resultats)
    df_resultats.to_csv("./data/analysis_results/correlation_results.csv", index=False)
    return df_resultats