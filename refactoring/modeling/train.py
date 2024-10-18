import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import dotenv

from refactoring.config.load_params import load_params

class ModelTrainer:
    def __init__(self, X_train_path, y_train_path, params):
        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.params = params
        self.initialize_params()
        
    def initialize_params(self):
        self.experiment_name = self.params['mlflow']['experiment_name']
        return self

    def load_data(self):
        X_train = pd.read_csv(self.X_train_path)
        y_train = pd.read_csv(self.y_train_path)
        return X_train, y_train 
    
    def train_and_evaluate(self):
        pass

if __name__ == '__main__':
    params = load_params()
    Trainer = ModelTrainer()