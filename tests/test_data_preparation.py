import numpy as np
import pytest 
import os
from refactoring.config.load_params import load_params
from refactoring.preprocessing.data_preparation import DataPreprocessor

# Load test parameters and data schema from the specified configuration file
params = load_params(r'tests\test_config.yaml')
data_path = params['paths']['data']  # Path to the data
data_schema = params['data_schema']  # Schema defining the expected structure and constraints of the data

@pytest.fixture
def preprocessor(tmp_path):
    data_preprocessor_instance = DataPreprocessor(data_path, str(tmp_path), params)
    data_preprocessor_instance.run_data_preprocessing()
    return data_preprocessor_instance

def test_load_data(preprocessor):
    preprocessor.load_data()
    target = params['features']['target']
    data_features = list(data_schema.keys())  # Get the expected features from the schema
    assert not preprocessor.data.empty, 'No Data'
    assert all(item in preprocessor.data.columns.tolist() for item in data_features), "Not all features of input data are in features"
    assert target in preprocessor.data.columns.to_list()

def test_save_data(preprocessor):
    version_suffix = f"_{preprocessor.data_version}" if preprocessor.data_version is not None else ""
    saved_data_names = ['X_train','X_test', 'X_val', 'y_train', 'y_test', 'y_val']
    saved_file_paths = [f'{preprocessor.output_dir}/{names}{version_suffix}.csv' for names in saved_data_names]

    for path in saved_file_paths:
        assert os.path.exists(path), \
            f"The saved file '{path}' does not exist in the output directory."

if __name__ == "__main__":
    pytest.main([__file__])  # Run the tests when the script is executed