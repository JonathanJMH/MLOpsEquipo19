import pytest
import pandas as pd 
import numpy as np
from refactoring.config.load_params import load_params
from refactoring.evaluation.evaluate import ModelEvaluator
import os

# Load test parameters and data schema from the specified configuration file
params = load_params(r'tests\test_config.yaml')
X_test_path = params['paths']['split_X_data'][1] # Path to the 
y_test_path = params['paths']['split_y_data'][1]  # Path to the 

@pytest.fixture
def evaluator():
    global X_test_path
    global y_test_path
    global params

    return ModelEvaluator(X_test_path,y_test_path,params)

def test_load_data(evaluator):
    """
    Test to verify that all expected features of X_test exist whithin it.

    Args:
        evaluator (evaluator): Temporary directory provided by pytest for storing output files.
    """
    global params

    output_features = params['output_features']  # Expected output features
    X_test, y_test = evaluator.load_data()
    assert not X_test.empty, 'No data loaded.'
    assert not y_test.size == 0, 'No data loaded.'

    assert all(item in X_test.columns.tolist() for item in output_features), "Not all features of features are in X_test"

def test_save_data(evaluator):
    """
    Test to verify that the test data is saved correctly to a CSV file.

    Args:
        data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    # Check if the CSV file has been saved
    version_suffix = f"_{evaluator.data_version}" if evaluator.data_version is not None else ""
    saved_file_path = f'{evaluator.output_dir}/data{version_suffix}.csv'
    
    # Assert that the saved CSV file exists in the output directory
    assert os.path.exists(saved_file_path), \
        f"The saved file '{saved_file_path}' does not exist."


if __name__ == "__main__":
    pytest.main([__file__])  # Run the tests