import pandas as pd 
from datetime import datetime, date

columns = ['Date','PIB']
data = pd.read_csv("./../data/PIB/Revenu national brut disponible et épargne nationale brute (Trimestrielle Base 2014)_2026-06-25.csv")

def extract_date(quarter : pd.Series):
    new_dates = []
    for str in quarter:
        year = int(str[:4])
        month = int(str[5:]) * 3
        day = 1
        date_obj = date(year, month, day) 
        new_dates.append(date_obj)
    return new_dates

dates = extract_date(data['Quarter'])
data['Date'] = dates
data = data.drop(columns=['Quarter'])
data = data[columns]
print(len(data))
for i in range(len(data)):
    for j in range (1,3):
        ligne = data.iloc[i]
        offset = pd.DateOffset(months=j)
        ligne['Date'] = ligne['Date'] - offset
        ligne['Date'] = date(ligne['Date'].year, ligne['Date'].month, 1)
        data = pd.concat([data, pd.DataFrame([[ligne['Date'], ligne['PIB']]], columns=columns)], ignore_index=True)
data = data.sort_values(by='Date', ignore_index = True)

data.to_csv("./../data/PIB/PIB.csv", index=False)
print(data)
