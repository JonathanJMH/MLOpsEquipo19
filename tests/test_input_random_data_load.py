# Test with RandomData
import sys
import os
import pandas as pd
import pytest
from refactoring.data.load_data import DataLoader

@pytest.fixture
def setup_data_loader(tmp_path):
    # Crea un archivo CSV temporal para las pruebas
    data = {
        'col1': [1.5, 2.5, 3.5],
        'col2': [4.5, 5.5, 6.5]
    }
    df = pd.DataFrame(data)
    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)

    params = {
        'version': '1.0',
        'data_adjusted': {
            'round_decimals': 1,
            'float_to_int': ['col1'],
            'int_to_str': ['col2']
        }
    }

    output_dir = tmp_path / "output"
    return DataLoader(data_path, output_dir, params)

def test_load_data(setup_data_loader):
    """
    Test to verify that data has been loaded correctly.

    Args:
        setup_data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    dl = setup_data_loader
    dl.load_data()
    assert not dl.data.empty  # Verifica que los datos se cargaron
    assert list(dl.data.columns) == ['col1', 'col2']  # Verifica las columnas

def test_adjust_data_types(setup_data_loader):
    """
    Test to verify that data type has been adjusted correctly.

    Args:
        setup_data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    dl = setup_data_loader
    dl.load_data()
    dl.adjust_data_types(round_decimals=dl.round_decimals, float_to_int=dl.float_to_int, int_to_str=dl.int_to_str)
    
    assert dl.data_ajusted['col1'].dtype == 'int'  # Verifica que col1 fue convertida a entero
    assert dl.data_ajusted['col2'].dtype == 'object'  # Verifica que col2 fue convertida a cadena
    assert dl.data_ajusted['col1'][0] == 1  # Verifica que se redondeó correctamente

def test_save_data(setup_data_loader):
    """
    Test to verify that the processed data is saved correctly to a CSV file.

    Args:
        setup_data_loader (DataLoader): The DataLoader instance used to load the data.
    """
    dl = setup_data_loader
    dl.load_data()
    dl.adjust_data_types(round_decimals=dl.round_decimals, float_to_int=dl.float_to_int, int_to_str=dl.int_to_str)
    dl.save_data()
    
    # Verifica que el archivo CSV se haya guardado
    version_suffix = f"_{dl.data_version}" if dl.data_version is not None else ""
    saved_file_path = dl.output_dir / f"data{version_suffix}.csv"
    assert os.path.exists(saved_file_path)  # Verifica que el archivo existe

    # Lee el archivo guardado y verifica su contenido
    saved_df = pd.read_csv(saved_file_path)
    assert list(saved_df.columns) == ['col1', 'col2']
    assert saved_df['col1'].dtype == 'int64'
    assert saved_df['col2'].dtype == 'float'

if __name__ == "__main__":
    pytest.main([__file__])