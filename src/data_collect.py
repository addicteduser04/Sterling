import pandas as pd
import numpy as np 
from abc import ABC, abstractmethod

class DataCollector(ABC):
    @abstractmethod
    def collect_data(self):
        pass

class RatesCollector(DataCollector):
    def __init__(self,file_path):
        self.file_path = file_path
    def collect_data(self):
        pass
    
class TauxDirecteurCollector(DataCollector):
    def __init__(self,file_path = "./data/taux_directeur/taux_directeur.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        print(data.head())
        return data
    
class CPICollector(DataCollector):
    def __init__(self,file_path="data/IPC/CPI.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        print(data.head())
        return data
    
class GDPCollector(DataCollector):
    def __init__(self,file_path="data/PIB/GDP.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        print(data.head())
        return data
    
class EuroRatesCollector(DataCollector):
    def __init__(self,file_path="./data/taux_europe/ECB Data Portal.csv"):
        self.file_path = file_path
    def collect_data(self):
        data = pd.read_csv(self.file_path)
        print(data.head())
        return data