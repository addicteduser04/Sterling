import pandas as pd 
from datetime import datetime, date
from pathlib import Path



def extract_date(dates: pd.Series):
    new_dates = []
    for str in dates:
        year = int(str[:4])
        month = int(str[5:])
        day = 1
        date_obj = date(year, month, day) 
        new_dates.append(date_obj)
    return new_dates



def preprocess_cpi(file_path):
    columns = ['Date','IPC']
    data = pd.read_csv(file_path)
    data["Date"] = extract_date(data["Mois"])
    data= data.drop(columns=["Mois"])
    data = data.sort_values(by="Date", ascending=True, ignore_index=True)
    data = data[columns]
    print(data)
    data.to_csv("~/data/preprocessed/CPI.csv", index=True)
    return data

#export preprocess_cpi