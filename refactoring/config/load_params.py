import yaml
import os 

def load_params(filepath="params.yaml"):
    """
    Carga los parámetros desde un archivo YAML.

    Args:
        filepath (str): Ruta al archivo YAML. Por defecto es 'params.yaml'.

    Returns:
        dict: Parámetros cargados desde el archivo YAML.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        yaml.YAMLError: Si hay un error al cargar el contenido YAML.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no se encontró.")

    try:
        with open(filepath, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            if cfg is None:
                raise ValueError("El archivo YAML está vacío o no tiene un formato válido.")
            return cfg
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error al cargar el archivo YAML: {e}")
    except Exception as e:
        raise Exception(f"Se produjo un error inesperado: {e}")
    
    # Ejemplo de uso
if __name__ == "__main__":
    try:
        params = load_params("params.yaml")
        print("Parámetros cargados:", params)
    except Exception as e:
        print(e)