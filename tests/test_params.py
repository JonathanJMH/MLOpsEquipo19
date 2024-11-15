import pytest
from pathlib import Path
import os
import yaml
from refactoring.config.load_params import load_params  # Ensure that the function is imported correctly

# Load testing parameters from the specified configuration file
test_params = load_params(r'tests/test_config.yaml')

@pytest.fixture
def yaml_file():
    """
    Fixture to load parameters from a YAML file.

    Returns:
        tuple: A tuple containing the parameters dictionary loaded from the YAML file and the Path object for the YAML file location.
    """
    params_path = test_params['paths']['params']  # Path to YAML file specified in test configuration
    return load_params(params_path), Path(params_path)  # Return both the loaded parameters and file path

def test_path(yaml_file):
    """
    Test to ensure that the YAML file path exists.

    Args:
        yaml_file (tuple): A tuple containing the loaded parameters and the Path object to the YAML file.
    """
    _, path = yaml_file
    assert path.exists(), f"The path {path} does not exist"

def test_load_params_success(yaml_file):
    """
    Test to check that the YAML file is successfully loaded.

    Args:
        yaml_file (tuple): A tuple containing the loaded parameters and the Path object to the YAML file.
    """
    params, _ = yaml_file
    assert params, "Failed to load parameters from YAML file"

def test_load_params_valid_yaml(yaml_file):
    """
    Test to verify that the loaded parameters contain the expected keys.

    Args:
        yaml_file (tuple): A tuple containing the loaded parameters and the Path object to the YAML file.
    """
    params_keys = test_params['params']  # Expected keys from the test configuration
    params, _ = yaml_file
    assert set(params.keys()) == set(params_keys), "The keys in the YAML file do not match the expected keys"

if __name__ == "__main__":
    pytest.main([__file__])  # Run the tests when the script is executed
