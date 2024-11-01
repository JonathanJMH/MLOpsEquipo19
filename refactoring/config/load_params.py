import yaml
import os 

def load_params(filepath="params.yaml"):
    """
    Loads parameters from a YAML file.

    Args:
        filepath (str): Path to the YAML file. Default is 'params.yaml'.

    Returns:
        dict: Parameters loaded from the YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there is an error loading the YAML content.
        ValueError: If the YAML file is empty or has an invalid format.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")

    try:
        with open(filepath, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            if cfg is None:
                raise ValueError("The YAML file is empty or has an invalid format.")
            return cfg
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error loading YAML file: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    try:
        params = load_params("params.yaml")
        print("Loaded parameters:", params)
    except Exception as e:
        print(e)