import pandas as pd 
import yfinance as yf
import investpy
data_taux_bam = pd.read_csv('./data/TAUX/processed/taux_bam.csv', index_col=0) 
data_taux_ecb = pd.read_csv('./data/taux_europe/ECB Data Portal 2004.csv', index_col=0)

for col in data_taux_bam.columns:
    if col == "Date":
        continue
    data_taux_bam[col] = data_taux_bam[col]*100

data_final = pd.merge(data_taux_bam, data_taux_ecb, on='Date', how = 'right').dropna()

print(data_final)

#download data for masi 

data_final.to_csv('./data/2004/bam_ecb_2004')