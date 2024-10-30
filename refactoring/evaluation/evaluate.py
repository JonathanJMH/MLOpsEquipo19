import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import os
import argparse
import json
import joblib

from refactoring.config.load_params import load_params

class ModelEvaluator:
    def __init__(self, X_test_path, y_test_path, params,model_path):
        self.X_test_path = X_test_path
        self.y_test_path = y_test_path
        self.model_path = model_path
        self.params = params
        self.initialize_params()

    def initialize_params(self):
        self.version = self.params['version']
        self.version_suffix = f"_{self.version}" if self.version is not None else ""
        self.experiment_name = self.params['mlflow']['test_name'] + self.version_suffix

    def load_data(self):
        try:
            X_test = pd.read_csv(self.X_test_path)
            y_test = pd.read_csv(self.y_test_path).values.ravel()
            return X_test, y_test
        except FileNotFoundError as e:
            print(f"Error loading test data: {e}")
            raise

    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            return model
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            raise
        except joblib.JoblibException as e:
            print(f"Error in joblib: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        return accuracy, precision, recall, y_pred

    def log_metrics(self, accuracy, precision, recall):
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)

    def log_predictions(self, y_test, y_pred):
        os.makedirs("data/predictions", exist_ok=True)
        pred_df = pd.DataFrame({"actual": y_test.ravel(), "predicted": y_pred})
        pred_df.to_csv("data/predictions/predictions.csv", index=False)
        mlflow.log_artifact("data/predictions/predictions.csv")

    def save_metrics(self, metrics, output_path):
        with open(output_path, 'w') as f:
            json.dump(metrics, f)

    def ensure_experiment_active(self):
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment:
            if experiment.lifecycle_stage == "deleted":
                client.restore_experiment(experiment.experiment_id)
                print(f"Experiment '{self.experiment_name}' restored.")
            return experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            print(f"Experiment '{self.experiment_name}' created.")
            return experiment_id

    def run_evaluation(self):
        experiment_id = self.ensure_experiment_active()
        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_run():
            X_test, y_test = self.load_data()
            model = self.load_model()

            accuracy, precison, recall, y_pred = self.evaluate_model(model, X_test, y_test)
            self.log_metrics(accuracy, precison, recall)

            self.log_predictions(y_test, y_pred)

            metrics = {
                "accuracy": accuracy,
                "precison": precison,
                "recall": recall
            }
            os.makedirs("metrics", exist_ok=True)
            self.save_metrics(metrics, "metrics/evaluation.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--X_test_path", type=str, help="Path to X_test.csv")
    parser.add_argument("--y_test_path", type=str, help="Path to y_test.csv")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--model_path", type=str, help="Path to the best model file")
    args = parser.parse_args()
    params = load_params(args.params)
    me = ModelEvaluator(args.X_test_path, args.y_test_path, params, args.model_path)
    me.run_evaluation()