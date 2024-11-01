import numpy as np
import pytest 
import os
from refactoring.config.load_params import load_params
from refactoring.preprocessing.data_preparation import DataPreprocessor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load test parameters and data schema from the specified configuration file
params = load_params(r'tests\test_config.yaml')
data_path = params['paths']['data']  # Path to the data
data_schema = params['data_schema']  # Schema defining the expected structure and constraints of the data

@pytest.fixture
def preprocessor(tmp_path):
    """
    Fixture to create a DataPreprocessor instance for testing data preprocessing.

    Args:
        tmp_path (Path): Temporary directory provided by pytest for storing output files.

    Returns:
        DataPreprocessor: An instance of DataPreprocessor configured with the data path and parameters.
    """
    data_preprocessor_instance = DataPreprocessor(data_path, str(tmp_path), params)
    data_preprocessor_instance.run_data_preprocessing()  # Execute preprocessing steps
    return data_preprocessor_instance  # Return the instance for use in tests

def test_load_data(preprocessor):
    """
    Test to verify that data is loaded correctly and contains the expected features.

    Args:
        preprocessor (DataPreprocessor): The DataPreprocessor instance used to preprocess the data.
    """
    preprocessor.load_data()  # Load data
    target = params['features']['target']  # Target feature name
    data_features = list(data_schema.keys())  # Expected features from the schema
    
    assert not preprocessor.data.empty, 'No data loaded.'
    assert all(item in preprocessor.data.columns.tolist() for item in data_features), "Not all features of input data are in features"
    assert target in preprocessor.data.columns.tolist(), "Target feature is missing from loaded data."

def test_split_data(preprocessor):
    """
    Test to ensure that data is split into train, test, and validation sets without overlap.

    Args:
        preprocessor (DataPreprocessor): The DataPreprocessor instance used to preprocess the data.
    """
    # Check that each split is not empty
    assert not preprocessor.X_train.empty, 'X_train is empty after splitting'
    assert not preprocessor.X_test.empty, 'X_test is empty after splitting'
    assert not preprocessor.X_val.empty, 'X_val is empty after splitting'
    assert not preprocessor.y_train.empty, 'y_train is empty after splitting'
    assert not preprocessor.y_test.empty, 'y_test is empty after splitting'
    assert not preprocessor.y_val.empty, 'y_val is empty after splitting'
    
    # Verify no overlap in indices across splits
    train_indices = set(preprocessor.X_train.index)
    val_indices = set(preprocessor.X_val.index)
    test_indices = set(preprocessor.X_test.index)
    
    assert train_indices.isdisjoint(val_indices), "X_train and X_val have overlapping indices"
    assert train_indices.isdisjoint(test_indices), "X_train and X_test have overlapping indices"
    assert val_indices.isdisjoint(test_indices), "X_val and X_test have overlapping indices"

def test_create_transformer(preprocessor):
    """
    Test to verify that the data transformer is correctly created and contains the expected pipelines.

    Args:
        preprocessor (DataPreprocessor): The DataPreprocessor instance used to preprocess the data.
    """
    assert isinstance(preprocessor.data_transformer, ColumnTransformer), "data_transformer is not an instance of ColumnTransformer"
    
    # Check that the transformer contains the expected pipelines
    transformer_names = [t[0] for t in preprocessor.data_transformer.transformers]
    assert 'log' in transformer_names, "log_pipeline is missing in data_transformer"
    assert 'num' in transformer_names, "num_pipeline is missing in data_transformer"
    assert 'cat' in transformer_names, "cat_pipeline is missing in data_transformer"
    assert 'ord' in transformer_names, "ord_pipeline is missing in data_transformer"

    # Verify each pipeline type within the transformers
    for name, pipeline, features in preprocessor.data_transformer.transformers:
        assert isinstance(pipeline, Pipeline), f"{name}_pipeline is not an instance of Pipeline"
    
    # Verify that log_pipeline contains the expected steps
    log_pipeline_steps = dict(preprocessor.data_transformer.named_transformers_['log'].steps)
    assert 'log_transform' in log_pipeline_steps, "log_transform is missing in log_pipeline"
    assert 'Scaler' in log_pipeline_steps, "Scaler is missing in log_pipeline"
    
    # Verify that num_pipeline contains a scaler
    num_pipeline_steps = dict(preprocessor.data_transformer.named_transformers_['num'].steps)
    assert 'Scaler' in num_pipeline_steps, "Scaler is missing in num_pipeline"

    # Verify that cat_pipeline contains a binary encoder
    cat_pipeline_steps = dict(preprocessor.data_transformer.named_transformers_['cat'].steps)
    assert 'binary' in cat_pipeline_steps, "BinaryEncoder is missing in cat_pipeline"

    # Verify that ord_pipeline contains an ordinal encoder
    ord_pipeline_steps = dict(preprocessor.data_transformer.named_transformers_['ord'].steps)
    assert 'ordinal' in ord_pipeline_steps, "OrdinalEncoder is missing in ord_pipeline"

def test_traintestval_features(preprocessor):
    """
    Test to ensure that the output features of preprocessed train, test, and validation data match the expected features.

    Args:
        preprocessor (DataPreprocessor): The DataPreprocessor instance used to preprocess the data.
    """
    output_features = params['output_features']  # Expected output features
    traintestval_features = {
        'X_train_features': preprocessor.X_train_preprocessed.columns.tolist(),
        'X_test_features': preprocessor.X_test_preprocessed.columns.tolist(),
        'X_val_features': preprocessor.X_val_preprocessed.columns.tolist()
    }

    for element in traintestval_features.keys():
        assert all(item in traintestval_features[element] for item in output_features), \
            f"Not all features of output data are in {element}"

def test_save_data(preprocessor):
    """
    Test to verify that preprocessed data is saved as CSV files in the output directory.

    Args:
        preprocessor (DataPreprocessor): The DataPreprocessor instance used to preprocess the data.
    """
    version_suffix = f"_{preprocessor.data_version}" if preprocessor.data_version is not None else ""
    saved_data_names = ['X_train','X_test', 'X_val', 'y_train', 'y_test', 'y_val']
    saved_file_paths = [f'{preprocessor.output_dir}/{names}{version_suffix}.csv' for names in saved_data_names]

    for path in saved_file_paths:
        assert os.path.exists(path), \
            f"The saved file '{path}' does not exist in the output directory."

def test_mean_std_dev_num_pip(preprocessor):
    """
    Test to verify that the data from the "num" pipeline mean equals 0 and standard deviation 1 after preprocessing.

    Args:
        preprocessor (DataPreprocessor): The DataPreprocessor instance used to preprocess the data.
    """
    num_feat = preprocessor.numerical_features 
    data_num = preprocessor.data_transformer.named_transformers_["num"].transform(preprocessor.data[num_feat])
    
    means = data_num.mean(axis=0)
    std_devs = data_num.std(axis=0)

    # Check each feature individually and provide detailed failure messages
    for idx, feature in enumerate(num_feat):
        mean = means[idx]
        std_dev = std_devs[idx]
        assert abs(mean) < 1e-6, f"The mean of '{feature}' is not approximately 0. Found: {mean}"
        assert abs(std_dev - 1) < 1e-6, f"The standard deviation of '{feature}' is not approximately 1. Found: {std_dev}"

def test_mean_std_dev_num_pip(preprocessor):
    """
    Test to verify that the data from the "log" pipeline mean equals 0 and standard deviation 1 after preprocessing.

    Args:
        preprocessor (DataPreprocessor): The DataPreprocessor instance used to preprocess the data.
    """
    other_feat = preprocessor.other_features
    data_other = preprocessor.data_transformer.named_transformers_["log"].transform(preprocessor.data[other_feat])

    mean = data_other.mean()
    std_dev = data_other.std()

    assert abs(data_other.mean()) < 1e-6, f"The mean of '{data_other}' is not approximately 0. Found: {mean}"
    assert abs(std_dev - 1) <= 1e-6, f"The standard deviation of '{data_other}' is not approximately 1. Found: {std_dev}"

def test_mean_std_dev(preprocessor):
    """
    Test to verify that the numeric data u.

    Args:
        preprocessor (DataPreprocessor): The DataPreprocessor instance used to preprocess the data.
    """
    other = preprocessor.other_features

    assert other.min() > 0, f"Data entries of '{other}' are below 0"

if __name__ == "__main__":
    pytest.main([__file__])  # Run the tests when the script is executed