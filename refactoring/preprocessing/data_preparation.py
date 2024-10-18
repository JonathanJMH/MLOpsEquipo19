import pandas as pd 
import argparse
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split

from refactoring.config.load_params import load_params

class DataPreprocessor():
    def __init__(self, data_path , output_dir, params, data_version = None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.params = params
        self.data_version = data_version
        self.initialize_params()

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
    
    def initialize_params(self):
        self.round_decimals = self.params['data_adjusted']['round_decimals']
        self.float_to_int = self.params['data_adjusted']['float_to_int']
        self.int_to_str = self.params['data_adjusted']['int_to_str']

        self.numerical_features = self.params['features']['numerical']
        self.categorical_features = self.params['features']['categorical']
        self.ordinal_features = self.params['features']['ordinal']
        self.other_features = self.params['features']['other']
        self.target = self.params['features']['target']
        
        self.test_size = self.params['split_options']['test_size']
        self.val_size = self.params['split_options']['val_size']
        self.ramdom_state = self.params['split_options']['random_state']

        return self
        
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
        self.adjust_data_types(round_decimals = self.round_decimals,
                               float_to_int = self.float_to_int,
                               int_to_str = self.int_to_str)
        return self

    def _create_transformer(self):
        """
        Crea los pipelines de preprocesamiento para diferentes tipos de características 
        (numéricas, categóricas y ordinales).

        Retorna:
        -------
        self : DataPreprocessor
            Instancia con los pipelines de preprocesamiento creados.
        """
        log_pipeline = Pipeline(steps = [
            ('log_transform', FunctionTransformer(func=np.log, validate=False)),
            ('Scaler', StandardScaler())
        ])

        num_pipeline = Pipeline(steps = [
            ('Scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('binary', BinaryEncoder(handle_unknown='ignore'))
        ])

        ord_pipeline = Pipeline(steps=[
            ('ordinal', OrdinalEncoder())
        ])

        self.data_transformer = ColumnTransformer(transformers=[
            ('log', log_pipeline, self.other_features),
            ('num', num_pipeline, self.numerical_features),
            ('cat', cat_pipeline, self.categorical_features),
            ('ord', ord_pipeline, self.ordinal_features)
        ])
        return self
    
    def _split_data(self, test_size=0.2, val_size=0.1, random_state=1):
        X = self.data_ajusted.drop(self.target, axis=1)
        y = self.data_ajusted[self.target]

        # División en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # División en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=random_state, test_size=val_size)

        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def _fit_transform_data(self, X_train, X_test, X_val):
        # Ajustar y transformar los conjuntos de datos
        X_train_preprocessed = self.data_transformer.fit_transform(X_train)
        X_val_preprocessed = self.data_transformer.transform(X_val)
        X_test_preprocessed = self.data_transformer.transform(X_test)
        
        # Obtener los nombres de las nuevas características categóricas
        new_categorical_features = list(self.data_transformer.named_transformers_['cat']['binary'].get_feature_names_out(self.categorical_features))
        
        # Consolidar nombres de características
        feature_names = self.other_features + self.numerical_features + new_categorical_features + self.ordinal_features
        
        # Crear DataFrames para los conjuntos transformados
        X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=feature_names, index=X_train.index)
        X_val_preprocessed = pd.DataFrame(X_val_preprocessed, columns=feature_names, index=X_val.index)
        X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=feature_names, index=X_test.index)
        
        return X_train_preprocessed, X_test_preprocessed, X_val_preprocessed
        
    def save_data(self,X_train, X_test, X_val, y_train, y_test, y_val):
        os.makedirs(self.output_dir, exist_ok=True)

        version_suffix = f"_{self.data_version}" if self.data_version is not None else ""
        
        self.data_ajusted.to_csv(f'{self.output_dir}/data{version_suffix}.csv', index=False)
        X_train.to_csv(f'{self.output_dir}/X_train{version_suffix}.csv', index=False)
        X_test.to_csv(f'{self.output_dir}/X_test{version_suffix}.csv', index=False)
        X_val.to_csv(f'{self.output_dir}/X_val{version_suffix}.csv', index=False)  # Agregado
        y_train.to_csv(f'{self.output_dir}/y_train{version_suffix}.csv', index=False)
        y_test.to_csv(f'{self.output_dir}/y_test{version_suffix}.csv', index=False)
        y_val.to_csv(f'{self.output_dir}/y_val{version_suffix}.csv', index=False)  # Agregado

    
    def preprocess_data(self):
        X_train, X_test, X_val, y_train, y_test, y_val = self._split_data(test_size = self.test_size,
                                                                          val_size = self.val_size,
                                                                          random_state = self.ramdom_state)
        self._create_transformer()
        X_train_processed, X_test_processed, X_val_processed = self._fit_transform_data(X_train = X_train,
                                                                                        X_test = X_test,
                                                                                        X_val = X_val)
        self.save_data(X_train = X_train_processed, 
                       X_test = X_test_processed, 
                       X_val = X_val_processed, 
                       y_train = y_train,
                       y_test = y_test,
                       y_val = y_val)
        return self


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Data')
    parser.add_argument('--data_path', type = str, help = 'Path of raw data')
    parser.add_argument("--output_dir", type = str, help = "Directory for processed data")  
    parser.add_argument("--params", type = str, default = "params.yaml", help="Path to params.yaml") 
    parser.add_argument("--data_version", type = str, default = None, help = "Version of the Processed Data") 
    args = parser.parse_args()     

    params = load_params(args.params)
    data_f = params['data']['filepath']
    data_preprocessor = DataPreprocessor(args.data_path, args.output_dir, params, args.data_version)
    data_preprocessor.load_data()
    data_preprocessor.preprocess_data()


    