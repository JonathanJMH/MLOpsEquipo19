import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score


from refactoring.config.load_params import load_params

class ModelTrainer:
    def __init__(self, X_train_path, y_train_path, params):
        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.params = params
        self.initialize_params()
        
    def initialize_params(self):
        self.version = self.params['version']
        self.version_suffix = f"_{self.version}" if self.version is not None else ""
        self.experiment_name = self.params['mlflow']['experiment_name'] + self.version_suffix
        self.models = self.params['models']
        
        return self

    def load_data(self):
        X_train = pd.read_csv(self.X_train_path)
        y_train = pd.read_csv(self.y_train_path).values.ravel()
        return X_train, y_train 
    
    def train_and_evaluate(self,X_train,y_train):
        best_model = None
        best_score = 0
        best_model_name = ""
        best_params = {}
        best_run_id = None
        
        mlflow.set_experiment(experiment_name = self.experiment_name)
        
        for model_name, param_grid in self.models.items():
            if model_name == 'gradient_boosting':
                model = GradientBoostingClassifier()
            elif model_name == 'logistic_regression':  
                model = LogisticRegression()
            elif model_name == 'svm':
                model = SVC()
            elif model_name == 'random_forest':
                model = RandomForestClassifier()
            elif model_name == 'k_neighbors':
                model = KNeighborsClassifier()
            else:
                print(f"Model {model_name} is not recognized.")
                continue  # Skip unknown models

            grid_search = GridSearchCV(
                estimator = model,
                param_grid = param_grid,
                scoring ='accuracy',
                cv=5,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train,y_train)
            accuracy = grid_search.best_score_

            print(f"{model_name} Best accuracy: {accuracy}")

            run_id = self.log_model_and_metrics(model = grid_search.best_estimator_,
                                                X_train = X_train,
                                                y_train = y_train,
                                                model_name= model_name,
                                                best_params= grid_search.best_params_,
                                                accuracy = accuracy)

            if accuracy > best_score:
                best_score = accuracy
                best_model = grid_search.best_estimator_
                best_model_name = model_name
                best_params = grid_search.best_params_
                best_run_id = run_id
            
        return best_model, best_model_name, best_params, best_run_id
    
    def log_model_and_metrics(self, model, X_train, y_train, model_name, best_params, accuracy):
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=model_name) as run:  # Usa el contexto como 'run'
            run_id = run.info.run_id

            input_example = X_train.head(1)

            mlflow.log_params(best_params)
            mlflow.set_tag('model_name', model_name)

            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train,y_train_pred)
            train_prec = precision_score(y_train,y_train_pred, average='weighted')
            train_rec = recall_score(y_train,y_train_pred, average='weighted')

            mlflow.log_metrics({'accuracy': train_accuracy,
                                'cv_accuracy': accuracy,
                                'precision': train_prec,
                                'recall':train_rec})
            
            mlflow.sklearn.log_model(sk_model = model,
                                     artifact_path = 'model',
                                     input_example = input_example)
            return run_id

    def tag_best_model(self, best_run_id):
        client = mlflow.tracking.MlflowClient()
        client.set_tag(best_run_id, "best_model", "True")
        print(f"Best model run ID: {best_run_id} tagged as best_model.")

    def register_best_model(self, best_run_id, model_name="BestModel"):
        model_uri = f"runs:/{best_run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        print(f"Registered model '{model_name}' with version {result.version}")

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
        print(f"Model '{model_name}' version {result.version} transitioned to 'Production' stage.")

    def save_best_model(self, model):
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/best_model{self.version_suffix}.pkl")

    def run_trainer(self):
        X_train, y_train = self.load_data()
        best_model, best_model_name, best_params, best_run_id = self.train_and_evaluate(X_train = X_train,
                                                                                        y_train = y_train)
        print(f"Best model: {best_model_name} with params: {best_params}")
        self.save_best_model(best_model)

        self.tag_best_model(best_run_id)

        self.register_best_model(best_run_id, model_name="BestModel")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--X_train_path", type=str, help="Path to X_train.csv")
    parser.add_argument("--y_train_path", type=str, help="Path to y_train.csv")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    params = load_params(args.params)
    MLFLOW_TRACKING_URI = params['mlflow']['tracking_uri']
    Trainer = ModelTrainer(args.X_train_path, args.y_train_path, params)
    Trainer.run_trainer()
