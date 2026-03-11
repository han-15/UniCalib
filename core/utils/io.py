import tomllib as tml
import pandas as pd
import h5py
from easydict import EasyDict
from pathlib import Path
from numpy import ndarray
from PIL import Image
from .distributed import master_only
from typing import overload

@overload
def ensure_dir(path: str): ...

@overload
def ensure_dir(path: Path): ...

@master_only
def ensure_dir(path):
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    else:
        assert path.is_dir(), f"'{path}' already exists but is not a directory."

def is_path_exist(*file_parts: str) -> bool:
    """
    Check if a file or directory exists by joining multiple path components using pathlib.

    Args:
        *file_parts (str): Multiple components of the file or directory path.

    Returns:
        bool: True if the file or directory exists, False otherwise.
    """
    path = Path(*file_parts)
    return path.exists()

### toml operation function based on tomllib

def read_toml_file(file_name: str) -> EasyDict:
    """
    Read a TOML file and return its contents as a dictionary.

    Args:
        file_path (str): The path to the TOML file.

    Returns:
        dict[str, any]: The contents of the TOML file as a dictionary.

    """
    with open(file_name, 'rb') as toml_file:
        return EasyDict(tml.load(toml_file))

### h5

def read_h5_file(file_name: str, keys: list[str] | str) -> dict[str, ndarray]:
    """
    Read data from an HDF5 file.

    Args:
        file_name (str): The path to the HDF5 file.
        keys (list[str] | str): The key(s) of the dataset(s) to read from the file. If a single key is provided as a string,
                                it will be converted to a list with a single element.

    Returns:
        dict[str, ndarray]: A dictionary where the keys are the provided dataset keys and the values are the corresponding
                            NumPy arrays containing the data.

    Raises:
        Exception: If there is an error reading the file.

    """
    try:
        keys = [keys] if isinstance(keys, str) else keys
        with h5py.File(file_name, 'r') as hf:
            h5_data = {key: hf[key][:] for key in keys}
        return h5_data
    except Exception as e:
        print(f'File Broken: {file_name}')
        raise e

### Excel operation function based on Pandas

def create_df(data: dict[str, list[any]]) -> pd.DataFrame:
    """
    Create a DataFrame from a dictionary.

    Args:
        data (dict[str, list[any]]): A dictionary where the keys are the column names and the values are lists of data.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the dictionary.

    """
    return pd.DataFrame(data)

def read_csv_file(file_parts: list[str] | str, **kwargs) -> pd.DataFrame:
    """
    Read data from a CSV file.

    Args:
        file_name (str): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to `pd.read_csv`.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.

    Raises:
        Exception: If there is an error reading the file.

    """
    try:
        file_name = Path(*([file_parts] if isinstance(file_parts, str) else file_parts))
        return pd.read_csv(file_name, **kwargs)
    except Exception as e:
        print(f'File Broken: {file_name}')
        raise e

def write_csv_file(data: pd.DataFrame, file_parts: list[str] | str, **kwargs) -> None:
    """
    Write data to a CSV file.

    Args:
        data (pd.DataFrame): The data to write to the CSV file.
        file_parts (str): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to `pd.to_csv`.

    """
    file_name = Path(*([file_parts] if isinstance(file_parts, str) else file_parts))
    data.to_csv(file_name, **kwargs)

### Image operation function based on PIL

def read_image_file(file_name: str) -> Image.Image:
    """
    Load an image from a given file path.

    Args:
        img_path (str): The path to the image file.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    try:
        img = Image.open(file_name)
        return img
    except IOError as e:
        raise IOError(f"Error opening image file {file_name}: {e}")


def read_txt_file(file_path):
    data = {}
    try:
        with open(file_path, 'r') as txt_file:
            for line in txt_file:
                key, value = line.split(':')
                data[key.strip()] = [float(x) for x in value.split()]
        return data
    except Exception as e:
        print(f'Error reading file: {file_path}')
        raise e


def read_matrix_txt_file(file_path):
    """
    Read transformation matrix file (4x4 matrix format).
    
    File format example:
        0.99994381 -0.00781195 0.00716644 2.28199435
        0.00781797 0.99996911 -0.00081181 0.02823589
        -0.00715988 0.00086779 0.99997399 -0.14270000
        0.00000000 0.00000000 0.00000000 1.00000000
    
    Args:
        file_path: file path
        
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    import numpy as np
    try:
        matrix = []
        with open(file_path, 'r') as txt_file:
            for line in txt_file:
                line = line.strip()
                if line:  # skip empty lines
                    values = [float(x) for x in line.split()]
                    if values:  # ensure the line has data
                        matrix.append(values)
        
        # convert to numpy array
        matrix = np.array(matrix, dtype=np.float64)
        
        # validate matrix shape
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            raise ValueError(f"Invalid matrix dimensions: {matrix.shape}, expected at least 3x3")
        
        # if 3x4 matrix, append last row [0, 0, 0, 1]
        if matrix.shape == (3, 4):
            last_row = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
            matrix = np.vstack([matrix, last_row])
        
        return matrix
        
    except Exception as e:
        print(f'Error reading matrix file: {file_path}')
        raise e