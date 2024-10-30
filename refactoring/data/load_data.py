import pandas as pd
import numpy as np
import argparse
import os
from refactoring.config.load_params import load_params

class DataLoader():
    def __init__(self, data_path, output_dir, params):
        self.data_path = data_path
        self.output_dir = output_dir
        self.params = params
        self.initialize_params()

    def initialize_params(self):
        self.data_version = self.params['version']

        self.round_decimals = self.params['data_adjusted']['round_decimals']
        self.float_to_int = self.params['data_adjusted']['float_to_int']
        self.int_to_str = self.params['data_adjusted']['int_to_str']

    def load_data(self):
        """
        Carga el archivo CSV en un DataFrame y ajusta los tipos de datos aplicando redondeo, 
        conversión de flotantes a enteros y enteros a cadenas.

        Retorna:
        -------
        self : DataPreprocessor
            Instancia de la clase con los datos cargados y ajustados.
        """
        self.data = pd.read_csv(self.data_path)
        self.features = self.data.columns.to_list()
        
    def _round_columns(self, round_decimals):
        """
        Redondea las columnas especificadas del DataFrame a un número específico de decimales.

        Parámetros:
        ----------
        round_decimals : int, dict, or tuple
            - Si es un entero, redondea todas las columnas numéricas a ese número de decimales.
            - Si es un diccionario, redondea columnas específicas con diferentes precisiones.
            - Si es una tupla, el primer elemento es la lista de columnas a redondear y el segundo es el número de decimales.

        Retorna:
        -------
        self : DataPreprocessor
            Instancia de la clase con los datos ajustados.
        """
        if round_decimals is not None:
            if isinstance(round_decimals, int):
                self.data_ajusted = self.data_ajusted.round(round_decimals)
            elif isinstance(round_decimals, dict):
                for col, decimals in round_decimals.items():
                    self.data_ajusted[col] = self.data_ajusted[col].round(decimals)
            elif isinstance(round_decimals, tuple):
                columns, decimals = round_decimals
                self.data_ajusted[columns] = self.data_ajusted[columns].round(decimals)
        return self
    
    def _convert_float_to_int(self, float_to_int):
        """
        Convierte las columnas especificadas de flotantes a enteros.

        Parámetros:
        ----------
        float_to_int : list or str
            Lista o nombre de las columnas que deben ser convertidas de flotantes a enteros.

        Retorna:
        -------
        self : DataPreprocessor
            Instancia de la clase con los datos ajustados.
        """
        if float_to_int:
            self.data_ajusted[float_to_int] = self.data_ajusted[float_to_int].astype(int)
        return self
    
    def _convert_int_to_str(self, int_to_str):
        """
        Convierte las columnas especificadas de enteros a cadenas de texto.

        Parámetros:
        ----------
        int_to_str : list or str
            Lista o nombre de las columnas que deben ser convertidas de enteros a cadenas de texto.

        Retorna:
        -------
        self : DataPreprocessor
            Instancia de la clase con los datos ajustados.
        """
        if int_to_str:
            self.data_ajusted[int_to_str] = self.data_ajusted[int_to_str].astype(str)
        return self

    def adjust_data_types(self, int_to_str=None, float_to_int=None, round_decimals=None):
        """
        Ajusta los tipos de datos del DataFrame mediante la conversión de tipos y el redondeo de columnas.

        Parámetros:
        ----------
        int_to_str : list or str, opcional
            Lista o nombre de las columnas que deben ser convertidas de enteros a cadenas de texto.
        float_to_int : list or str, opcional
            Lista o nombre de las columnas que deben ser convertidas de flotantes a enteros.
        round_decimals : int, dict, or tuple, opcional
            Si es un entero, redondea todas las columnas numéricas a ese número de decimales.

        Retorna:
        -------
        self : DataPreprocessor
            Conjunto de datos ajustado.
        """
        self.data_ajusted = self.data.copy()
        self._round_columns(round_decimals)
        self._convert_float_to_int(float_to_int)
        self._convert_int_to_str(int_to_str)
        return self
    
    def save_data(self):
        os.makedirs(self.output_dir, exist_ok=True)

        version_suffix = f"_{self.data_version}" if self.data_version is not None else ""
        
        self.data_ajusted.to_csv(f'{self.output_dir}/data{version_suffix}.csv', index=False)
    
    def run(self):
        self.load_data()
        self.adjust_data_types(round_decimals = self.round_decimals,
                               float_to_int = self.float_to_int,
                               int_to_str = self.int_to_str)
        self.save_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load Data')
    parser.add_argument('--data_path', type = str, help = 'Path of raw data')
    parser.add_argument("--output_dir", type = str, help = "Directory for processed data")  
    parser.add_argument("--params", type = str, default = "params.yaml", help="Path to params.yaml") 
    args = parser.parse_args()     

    params = load_params(args.params)

    dl = DataLoader(args.data_path, args.output_dir, params)
    dl.run()