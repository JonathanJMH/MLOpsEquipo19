import pandas as pd 
import argparse
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split

from refactoring.config.load_params import load_params

class DataPreprocessor():
    def __init__(self, data_path, output_dir, params, data_version=None):
        """
        Initializes the DataPreprocessor instance.

        Parameters:
        -----------
        data_path : str
            Path to the input data file (CSV).
        output_dir : str
            Directory where the processed data will be saved.
        params : dict
            Dictionary containing parameters for data preprocessing.
        data_version : str, optional
            Version of the data (default is None).
        """
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = [None] * 6
        self.data_path = data_path
        self.output_dir = output_dir
        self.params = params
        self.data_version = data_version
        self.initialize_params()
    
    def initialize_params(self):
        """
        Initializes parameters for data preprocessing based on the provided configuration.

        Returns:
        --------
        self : DataPreprocessor
            Instance of the class with parameters initialized.
        """
        self.data_version = self.params['version']
        self.numerical_features = self.params['features']['numerical']
        self.categorical_features = self.params['features']['categorical']
        self.ordinal_features = self.params['features']['ordinal']
        self.other_features = self.params['features']['other']
        self.target = self.params['features']['target']
        
        self.test_size = self.params['split_options']['test_size']
        self.val_size = self.params['split_options']['val_size']
        self.random_state = self.params['split_options']['random_state']

        return self
        
    def load_data(self):
        """
        Loads the CSV file into a DataFrame.

        Returns:
        --------
        self : DataPreprocessor
            Instance of the class with data loaded.
        """
        self.data = pd.read_csv(self.data_path)
        return self

    def _create_transformer(self):
        """
        Creates preprocessing pipelines for different types of features 
        (numerical, categorical, and ordinal).

        Returns:
        --------
        self : DataPreprocessor
            Instance with preprocessing pipelines created.
        """
        log_pipeline = Pipeline(steps=[
            ('log_transform', FunctionTransformer(func=np.log, validate=False)),
            ('Scaler', StandardScaler())
        ])

        num_pipeline = Pipeline(steps=[
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
        """
        Splits the data into training, testing, and validation sets.

        Parameters:
        -----------
        test_size : float
            Proportion of the dataset to include in the test split.
        val_size : float
            Proportion of the training set to include in the validation split.
        random_state : int
            Random seed used for reproducibility.

        Returns:
        --------
        self : DataPreprocessor
            Instance with split data.
        """
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]

        # Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Split into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, random_state=random_state, test_size=val_size)

        return self
    
    def _fit_transform_data(self):
        """
        Fits and transforms the datasets using the defined pipelines.

        Returns:
        --------
        self : DataPreprocessor
            Instance with preprocessed data.
        """
        # Fit and transform training, validation, and test sets
        self.X_train_preprocessed = self.data_transformer.fit_transform(self.X_train)
        self.X_val_preprocessed = self.data_transformer.transform(self.X_val)
        self.X_test_preprocessed = self.data_transformer.transform(self.X_test)
        
        # Retrieve new categorical feature names
        new_categorical_features = list(self.data_transformer.named_transformers_['cat']['binary'].get_feature_names_out(self.categorical_features))
        
        # Consolidate feature names
        self.feature_names = self.other_features + self.numerical_features + new_categorical_features + self.ordinal_features
        
        # Create DataFrames for the transformed sets
        self.X_train_preprocessed = pd.DataFrame(self.X_train_preprocessed, columns=self.feature_names, index=self.X_train.index)
        self.X_val_preprocessed = pd.DataFrame(self.X_val_preprocessed, columns=self.feature_names, index=self.X_val.index)
        self.X_test_preprocessed = pd.DataFrame(self.X_test_preprocessed, columns=self.feature_names, index=self.X_test.index)
        
        return self
        
    def save_data(self):
        """
        Saves the preprocessed datasets to the specified output directory.

        Returns:
        --------
        None
        """
        os.makedirs(self.output_dir, exist_ok=True)

        version_suffix = f"_{self.data_version}" if self.data_version is not None else ""
        
        self.X_train_preprocessed.to_csv(f'{self.output_dir}/X_train{version_suffix}.csv', index=False)
        self.X_test_preprocessed.to_csv(f'{self.output_dir}/X_test{version_suffix}.csv', index=False)
        self.X_val_preprocessed.to_csv(f'{self.output_dir}/X_val{version_suffix}.csv', index=False)
        self.y_train.to_csv(f'{self.output_dir}/y_train{version_suffix}.csv', index=False)
        self.y_test.to_csv(f'{self.output_dir}/y_test{version_suffix}.csv', index=False)
        self.y_val.to_csv(f'{self.output_dir}/y_val{version_suffix}.csv', index=False)

    
    def run_data_preprocessing(self):
        """
        Executes the entire data preprocessing pipeline.

        Returns:
        --------
        self : DataPreprocessor
            Instance with preprocessed data.
        """
        self.load_data()
        self._split_data(test_size=self.test_size,
                         val_size=self.val_size,
                         random_state=self.random_state)
        self._create_transformer()
        self._fit_transform_data()
        self.save_data()
        return self


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Data')
    parser.add_argument('--data_path', type=str, help='Path to the input data')
    parser.add_argument("--output_dir", type=str, help="Directory for saving processed data")  
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml") 
    args = parser.parse_args()     

    params = load_params(args.params)

    data_preprocessor = DataPreprocessor(args.data_path, args.output_dir, params)
    data_preprocessor.run_data_preprocessing()
