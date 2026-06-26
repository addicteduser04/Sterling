import pandas as pd 
from datetime import datetime, date

columns = ['DATE','IPC']
data = pd.read_csv("./../data/IPC/Indice des prix à la consommation (Mensuel) (Base 100 2017)_2026-06-25.csv")

def extract_date(dates: pd.Series):
    new_dates = []
    for str in dates:
        year = int(str[:4])
        month = int(str[5:])
        day = 1
        date_obj = date(year, month, day) 
        new_dates.append(date_obj)
    return new_dates

data["DATE"] = extract_date(data["Mois"])
data= data.drop(columns=["Mois"])
data = data.sort_values(by="DATE", ascending=True, ignore_index=True)
data = data[columns]
print(data)
data.to_csv("./../data/IPC/CPI.csv", index=False)