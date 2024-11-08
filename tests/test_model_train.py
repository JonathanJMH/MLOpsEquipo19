import numpy as np
import pytest 
import os
from refactoring.config.load_params import load_params
from refactoring.modeling.train import ModelTrainer


# Load test parameters and data schema from the specified configuration file
params = load_params(r'tests/test_config.yaml')
X_train_path = params['paths']['split_X_data'][0] # Path to the 
y_train_path = params['paths']['split_y_data'][0]  # Path to the 

@pytest.fixture
def trainer():
    global X_train_path
    global y_train_path
    global params

    return ModelTrainer(X_train_path,y_train_path,params)

def test_load_data(trainer):
    """
    Test to verify that all expected features of X_train exist whithin it.

    Args:
        trainer (trainer): Temporary directory provided by pytest for storing output files.
    """
    global params

    output_features = params['output_features']  # Expected output features
    X_train, y_train = trainer.load_data()
    assert not X_train.empty, 'No data loaded.'
    assert not y_train.size == 0, 'No data loaded.'

    assert all(item in X_train.columns.tolist() for item in output_features), "Not all features of features are in X_train"

'''def test_train_and_evaluate(trainer):
    X_train, y_train = trainer.load_data()
    best_model, best_model_name, best_params, best_run_id = trainer.train_and_evaluate(
            X_train=X_train,
            y_train=y_train
        )
    
    assert best_model is not None
    assert best_model_name != ""
    assert best_params != ""
    assert best_run_id != ""'''

def test_handle_best_model(trainer):
    # trainer.run_trainer()
    global params
    mlflow_paths = params['paths']['mlflow_runs']
    model_paths = params['paths']['model_path']
    for path in model_paths:
        assert os.path.exists(path), \
            f"The saved file '{path}' does not exist in the directory."
    for path in mlflow_paths:
        assert os.path.isdir(path)
        has_subfolders = any(os.path.isdir(os.path.join(path, item)) for item in os.listdir(path))
        assert has_subfolders, f"La carpeta '{path}' no contiene subcarpetas"
        
if __name__ == "__main__":
    pytest.main([__file__])  # Run the tests
