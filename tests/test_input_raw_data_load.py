import sys
import os
import pandas as pd
import pytest
from refactoring.config.load_params import load_params
from refactoring.data.load_data import DataLoader

# Load test parameters and data schema from the specified configuration file
params = load_params(r'tests\test_config.yaml')
data_path = params['paths']['raw']  # Path to the raw data
data_schema = params['data_schema']  # Schema defining the expected structure and constraints of the data

@pytest.fixture
def data_loader(tmp_path):
    """
    Fixture to create a DataLoader instance for testing.
    
    Args:
        tmp_path (Path): Temporary directory provided by pytest for storing output files.

    Returns:
        DataLoader: An instance of DataLoader configured with the raw data path and parameters.
    """
    data_loader_instance = DataLoader(data_path, str(tmp_path), params)  # Initialize DataLoader with parameters
    data_loader_instance.run()  # Load the data
    return data_loader_instance  # Return the instance for use in tests

def test_features(data_loader):
    """
    Test to verify that all features defined in the data schema are present in the loaded data.

    Args:
        data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    data_features = list(data_schema.keys())  # Get the expected features from the schema
    assert all(item in data_loader.features for item in data_features), "Not all features of input data are in features"

def test_input_data_ranges(data_loader):
    """
    Test to ensure that the maximum and minimum values of the data columns fall within the defined ranges in the schema.

    Args:
        data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    # Getting the maximum and minimum values for each column
    max_values = data_loader.data.max()
    min_values = data_loader.data.min()
    
    # Ensuring that the maximum and minimum values fall into the expected range
    for feature in data_loader.features:  # Iterate over each feature
        if 'range' in data_schema[feature]:  # Check if the feature has a defined range
            assert max_values[feature] <= data_schema[feature]['range']['max'], \
                f"Max value of '{feature}' ({max_values[feature]}) exceeds the allowed maximum."
            assert min_values[feature] >= data_schema[feature]['range']['min'], \
                f"Min value of '{feature}' ({min_values[feature]}) is below the allowed minimum."

def test_input_data_allowed_values(data_loader):
    """
    Test to verify that the unique values in categorical features of the data are within the allowed values defined in the schema.

    Args:
        data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    for feature in data_loader.features:  # Iterate over each feature
        if 'allowed_values' in data_schema[feature]:  # Check if the feature has allowed values defined
            # Assert that all unique values in the data for the feature are within the allowed values
            assert all(value in data_schema[feature]['allowed_values'] for value in data_loader.data[feature].unique()), \
                f"Some values in '{feature}' are not among the allowed values."

def test_input_data_types(data_loader):
    """
    Test to ensure that the data types of the features match the expected types defined in the schema.

    Args:
        data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    # Getting the data types from each column
    data_types = data_loader.data.dtypes
    
    # Testing compatibility between data types
    for feature in data_loader.features:  # Iterate over each feature
        # Assert that the data type of the feature in the DataFrame matches the expected data type in the schema
        assert data_types[feature] == data_schema[feature]['dtype'], \
            f"Data type of '{feature}' is {data_types[feature]}, expected {data_schema[feature]['dtype']}."

def test_save_data(data_loader):
    """
    Test to verify that the processed data is saved correctly to a CSV file.

    Args:
        data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    # Check if the CSV file has been saved
    version_suffix = f"_{data_loader.data_version}" if data_loader.data_version is not None else ""
    saved_file_path = f'{data_loader.output_dir}/data{version_suffix}.csv'
    
    # Assert that the saved CSV file exists in the output directory
    assert os.path.exists(saved_file_path), \
        f"The saved file '{saved_file_path}' does not exist."

def test_mean_std_dev(data_loader):
    """
    Test to verify that the numeric data mean equals 0 and standard deviation 1 after preprocessing.

    Args:
        data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    for feature in data_loader.features:
        if data_loader.data[feature].dtype in [float, int]:
            mean = data_loader.data[feature].mean()
            std_dev = data_loader.data[feature].std()
        
    assert abs(mean) < 1e-6, f"The mean of '{feature}' is not approximately 0. Found: {mean}"
    assert abs(std_dev - 1) <= 1e-6, f"The mean of '{feature}' is not approximately 0. Found: {std_dev}"



if __name__ == "__main__":
    pytest.main([__file__])  # Run the tests when the script is executed