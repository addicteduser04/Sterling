import pandas as pd
import numpy as np 
from abc import ABC, abstractmethod

class DataCollector(ABC):
    @abstractmethod
    def collect_data(self):
        pass

class RatesCollector(DataCollector):
    def __init__(self,file_path = "./data/TAUX/processed/taux_bam_monthly.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        print(data.head())
        if data.empty:
            print("The rates data is empty.")
        return data
    
class TauxDirecteurCollector(DataCollector):
    def __init__(self,file_path = "./data/taux_directeur/taux_directeur_preprocessed.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        print(data.head())
        if data.empty:
            print("The taux directeur data is empty.")
        return data
    
class CPICollector(DataCollector):
    def __init__(self,file_path="./data/IPC/CPI.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        if data.empty:
            print("The CPI data is empty.")
        print(data.head())
        return data
    
class GDPCollector(DataCollector):
    def __init__(self,file_path="./data/PIB/PIB.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        if data.empty:
            print("The PIB data is empty.")
        print(data.head())
        return data
    
class EuroRatesCollector(DataCollector):
    def __init__(self,file_path="./data/taux_europe/ECB Data Portal Monthly.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        if data.empty:
            print("The Euro rates data is empty.")
        print(data.head())
        return data
    
class BetaCollector(DataCollector):
    def __init__(self,file_path="./data/betas/bam/betas_nelson_siegel_bam.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        if data.empty:
            print("The beta data is empty.")
        print(data.head())
        return data

def test_collectors():
    collector = RatesCollector(file_path="./data/TAUX/processed/taux_bam_monthly.csv")

    print(f"Collecting data using {collector.__class__.__name__}...")
    data = collector.collect_data()
    print(f"Data collected:\n{data.head()}\n")
    
    return data


if __name__ == "__main__":
    test_collectors()