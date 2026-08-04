import pandas as pd 
import yfinance as yf
import investpy
data_taux_bam = pd.read_csv('./data/TAUX/processed/taux_bam.csv', index_col=0) 
data_taux_ecb = pd.read_csv('./data/taux_europe/ECB Data Portal 2004.csv', index_col=0)

# BAM is already stored as a decimal rate.  Keep source units intact here;
# the DNS ingestion layer applies the declared BAM/ECB conversions exactly once.

data_final = pd.merge(data_taux_bam, data_taux_ecb, on='Date', how = 'right').dropna()

print(data_final)

#download data for masi 

data_final.to_csv('./data/2004/bam_ecb_2004')
