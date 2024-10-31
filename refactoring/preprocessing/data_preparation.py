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
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = [None] * 6
        self.data_path = data_path
        self.output_dir = output_dir
        self.params = params
        self.data_version = data_version
        self.initialize_params()
    
    def initialize_params(self):
        self.data_version = self.params['version']

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
        Carga el archivo CSV en un DataFrame.

        Retorna:
        -------
        self : DataPreprocessor
            Instancia de la clase con los datos cargados y ajustados.
        """
        self.data = pd.read_csv(self.data_path)

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
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]

        # División en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # División en entrenamiento y validación
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, random_state=random_state, test_size=val_size)

        return self
    
    def _fit_transform_data(self):
        # Ajustar y transformar los conjuntos de datos
        self.X_train_preprocessed = self.data_transformer.fit_transform(self.X_train)
        self.X_val_preprocessed = self.data_transformer.transform(self.X_val)
        self.X_test_preprocessed = self.data_transformer.transform(self.X_test)
        
        # Obtener los nombres de las nuevas características categóricas
        new_categorical_features = list(self.data_transformer.named_transformers_['cat']['binary'].get_feature_names_out(self.categorical_features))
        
        # Consolidar nombres de características
        self.feature_names = self.other_features + self.numerical_features + new_categorical_features + self.ordinal_features
        
        # Crear DataFrames para los conjuntos transformados
        self.X_train_preprocessed = pd.DataFrame(self.X_train_preprocessed, columns=self.feature_names, index=self.X_train.index)
        self.X_val_preprocessed = pd.DataFrame(self.X_val_preprocessed, columns=self.feature_names, index=self.X_val.index)
        self.X_test_preprocessed = pd.DataFrame(self.X_test_preprocessed, columns=self.feature_names, index=self.X_test.index)
        
        return self
        
    def save_data(self):
        os.makedirs(self.output_dir, exist_ok=True)

        version_suffix = f"_{self.data_version}" if self.data_version is not None else ""
        
        self.X_train_preprocessed.to_csv(f'{self.output_dir}/X_train{version_suffix}.csv', index=False)
        self.X_test_preprocessed.to_csv(f'{self.output_dir}/X_test{version_suffix}.csv', index=False)
        self.X_val_preprocessed.to_csv(f'{self.output_dir}/X_val{version_suffix}.csv', index=False)  # Agregado
        self.y_train.to_csv(f'{self.output_dir}/y_train{version_suffix}.csv', index=False)
        self.y_test.to_csv(f'{self.output_dir}/y_test{version_suffix}.csv', index=False)
        self.y_val.to_csv(f'{self.output_dir}/y_val{version_suffix}.csv', index=False)  # Agregado

    
    def run_data_preprocessing(self):
        self.load_data()
        self._split_data(test_size = self.test_size,
                         val_size = self.val_size,
                         random_state = self.ramdom_state)
        self._create_transformer()
        self._fit_transform_data()
        self.save_data()
        return self


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Data')
    parser.add_argument('--data_path', type = str, help = 'Path of data')
    parser.add_argument("--output_dir", type = str, help = "Directory for processed data")  
    parser.add_argument("--params", type = str, default = "params.yaml", help="Path to params.yaml") 
    args = parser.parse_args()     

    params = load_params(args.params)

    data_preprocessor = DataPreprocessor(args.data_path, args.output_dir, params)
    data_preprocessor.run_data_preprocessing()


    