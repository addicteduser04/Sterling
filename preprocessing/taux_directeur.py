import pandas as pd 
from datetime import datetime, date

data = pd.read_csv("data/taux_directeur/taux_directeur.csv")
Date = pd.read_csv("data/IPC/CPI.csv")
Date = Date.drop(columns = ['IPC'])
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data = data.drop(columns = ['Ratio de réserve obligatoire','Rémunération de la réserve'])

def extraire_taux_directeur(data : pd.Series):
    taux_directeur = []
    for str in data:
        str = str.split("%")[0]
        taux_directeur.append(float(str))
    return taux_directeur

taux_directeur = extraire_taux_directeur(data["Taux directeur"])
data["Taux directeur"] = taux_directeur
data = data.sort_values(by = "Date", ignore_index = True)

print(data)

print(Date["DATE"])
