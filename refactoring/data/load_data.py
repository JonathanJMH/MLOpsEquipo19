import pandas as pd
import numpy as np
import argparse
import os
from refactoring.config.load_params import load_params

class DataLoader():
    def __init__(self, data_path, output_dir, params):
        """
        Initializes the DataLoader with paths and parameters.

        Args:
            data_path (str): Path to the input data file.
            output_dir (str): Directory to save processed data.
            params (dict): Dictionary of parameters for data processing.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.params = params
        self.initialize_params()

    def initialize_params(self):
        """
        Initializes specific data transformation parameters from configuration.
        """
        self.data_version = self.params['version']
        self.round_decimals = self.params['data_adjusted']['round_decimals']
        self.float_to_int = self.params['data_adjusted']['float_to_int']
        self.int_to_str = self.params['data_adjusted']['int_to_str']

    def load_data(self):
        """
        Loads the CSV file into a DataFrame and prepares the feature list.

        Returns:
            self : DataLoader
                Instance of the class with loaded data and initialized features.
        """
        self.data = pd.read_csv(self.data_path)
        self.features = self.data.columns.to_list()

    def _round_columns(self, round_decimals):
        """
        Rounds specified columns of the DataFrame to a specific number of decimals.

        Args:
            round_decimals (int, dict, or tuple):
                - If int, rounds all numeric columns to this number of decimals.
                - If dict, rounds specific columns with different precisions.
                - If tuple, rounds specified columns to a given number of decimals.

        Returns:
            self : DataLoader
                Instance of the class with adjusted data.
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
        Converts specified columns from float to integer.

        Args:
            float_to_int (list or str): List or name of columns to convert from float to integer.

        Returns:
            self : DataLoader
                Instance of the class with adjusted data.
        """
        if float_to_int:
            self.data_ajusted[float_to_int] = self.data_ajusted[float_to_int].astype(int)
        return self
    
    def _convert_int_to_str(self, int_to_str):
        """
        Converts specified columns from integer to string.

        Args:
            int_to_str (list or str): List or name of columns to convert from integer to string.

        Returns:
            self : DataLoader
                Instance of the class with adjusted data.
        """
        if int_to_str:
            self.data_ajusted[int_to_str] = self.data_ajusted[int_to_str].astype(str)
        return self

    def adjust_data_types(self, int_to_str=None, float_to_int=None, round_decimals=None):
        """
        Adjusts data types in the DataFrame by converting types and rounding columns.

        Args:
            int_to_str (list or str, optional): List or name of columns to convert from integer to string.
            float_to_int (list or str, optional): List or name of columns to convert from float to integer.
            round_decimals (int, dict, or tuple, optional): Number of decimals for rounding.

        Returns:
            self : DataLoader
                Instance of the class with adjusted data.
        """
        self.data_ajusted = self.data.copy()
        self._round_columns(round_decimals)
        self._convert_float_to_int(float_to_int)
        self._convert_int_to_str(int_to_str)
        return self
    
    def save_data(self):
        """
        Saves the adjusted data to a CSV file in the output directory.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        version_suffix = f"_{self.data_version}" if self.data_version is not None else ""
        self.data_ajusted.to_csv(f'{self.output_dir}/data{version_suffix}.csv', index=False)
    
    def run(self):
        """
        Executes the full data loading, adjusting, and saving process.
        """
        self.load_data()
        self.adjust_data_types(round_decimals=self.round_decimals,
                               float_to_int=self.float_to_int,
                               int_to_str=self.int_to_str)
        self.save_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load Data')
    parser.add_argument('--data_path', type=str, help='Path of raw data')
    parser.add_argument("--output_dir", type=str, help="Directory for processed data")  
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml") 
    args = parser.parse_args()     

    params = load_params(args.params)
    dl = DataLoader(args.data_path, args.output_dir, params)
    dl.run()