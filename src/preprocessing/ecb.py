import pandas as pd 
import numpy as np
from pathlib import Path


datass = pd.read_csv("./data/taux_europe/ECB Data Portal_3M.csv")
date = datass['DATE']
date = pd.to_datetime(date, format='%Y-%m-%d')
YIELDS = ['3M','6M','1Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y','10Y','11Y','12Y','13Y','14Y','15Y','16Y','17Y','18Y','19Y','20Y','30Y']


def to_monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily curves to monthly mean per tenor."""
    monthly = df.resample('ME').mean()
    monthly.index = (monthly.index + pd.offsets.MonthBegin(1)) 
    return monthly

def preprocess_ecb(yields = YIELDS ,date_df = date):
    yield_df = pd.DataFrame(columns=['Date'] + YIELDS)
    yield_df['Date'] = date_df
    for yields in YIELDS:
        file_path = f"./data/taux_europe/ECB Data Portal_{yields}.csv"
        data = pd.read_csv(file_path)
        rate = data.iloc[:, 2]
        yield_df[yields] = rate
    yield_df = yield_df.set_index('Date')
    yield_df.to_csv('./data/taux_europe/ECB Data Portal 2004.csv')
    yield_df = to_monthly_mean(yield_df)
    yield_df.to_csv("./data/taux_europe/ECB Data Portal Monthly.csv", index=True)
    return yield_df

#export preprocess_ecb