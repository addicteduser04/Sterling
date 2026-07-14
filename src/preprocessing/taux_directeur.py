import pandas as pd 
from datetime import datetime, date
from pathlib import Path

def extraire_taux_directeur(data : pd.Series):
    taux_directeur = []
    for str in data:
        str = str.split("%")[0]
        taux_directeur.append(float(str))
    return taux_directeur


input = Path("../data/taux_directeur/taux_directeur.csv")


def preprocess_td(file_path = input):
    monthly_index = pd.date_range(start='2007-01-01', end='2026-06-01', freq='MS')
    df = pd.DataFrame(index=monthly_index)

    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data = data.drop(columns = ['Ratio de réserve obligatoire','Rémunération de la réserve'])



    taux_directeur = extraire_taux_directeur(data["Taux directeur"])
    data["Taux directeur"] = taux_directeur
    data = data.sort_values(by = "Date", ignore_index = True)

    data['Date'] = (data['Date'] + pd.offsets.MonthBegin(1))

    df = df.join(
        data[['Date', 'Taux directeur']]
            .set_index('Date'),
        how='left')
    df['Taux directeur'] = df['Taux directeur'].ffill()

    df.to_csv("./data/preprocessed/taux_directeur_preprocessed.csv")
    
    return df