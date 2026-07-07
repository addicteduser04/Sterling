import pandas as pd 
from datetime import datetime, date 
from pathlib import Path
import os 
import re 

YIELDS = ['3M','6M','1Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y','10Y','11Y','12Y','13Y','14Y','15Y','16Y','17Y','18Y','19Y','20Y','30Y']
raw_yields = {'13 semaines': '3M', '26 semaines': '6M', '52 semaines': '1Y', '2 ans': '2Y', '5 ans': '5Y', '10 ans': '10Y', '15 ans': '15Y', '20 ans': '20Y', "30 ans": "30Y"}
data = pd.DataFrame(columns = YIELDS)
folder = Path("../data/TAUX")
child = []
child_child = []
#def extract_from_csv(name)

def linear_interpolation(df):
    df = df.sort_index()
    df = df.interpolate(method='linear', axis=0)
    return df


for item in folder.iterdir():
    child.append(item.name)
for name in child:
    # extract_from_csv(name)
    if name == "20250903":
        break
    excel = pd.read_excel(f"../data/TAUX/{name}/excel/curve_{name}_000000.xlsx")
    excel = excel.transpose()
    excel = excel.drop("Tenor")
    excel.index = pd.to_datetime(excel.index, format='%d/%m/%Y',errors='coerce')
    data = pd.concat([data, excel], axis=1)
data.to_csv(f"../data/taux_preprocessed.csv", index=True)    
print(data)

# from 20250903 files structure changed
# what to do next is add a condition: if csv exist, elif excel 

"""
name = "20250903"

folder = Path(f"../data/TAUX/{name}")
for item in folder.iterdir():
    if item.name == "excel":
        for child in item.iterdir():
            curve = child.name
            # Extract excel file
            data = pd.read_excel(f"../data/TAUX/{name}/{item.name}/{curve}")
            print(data)
            for col in data['Tensor']:
                if col in raw_yields.keys():
                    data['Tensor'] = raw_yields[col]
            print(data)
            
            # Transpose the matrix to have dates as index and yields as columns
            data = data.T
            



# add a function that turns 9 columns to 23 (add yields that dont exist)
if not all(yield_ in data.columns for yield_ in YIELDS):
    for yield_ in YIELDS:
        if yield_ not in data.columns:
            data[yield_] = None
"""


