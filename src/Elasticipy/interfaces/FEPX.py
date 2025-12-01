from Elasticipy.tensors.second_order import SymmetricSecondOrderTensor, SecondOrderTensor, \
    SkewSymmetricSecondOrderTensor
from Elasticipy.tensors.stress_strain import StressTensor, StrainTensor
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

DTYPES={'stress':StressTensor,
        'strain':StrainTensor,
        'strain_el':StrainTensor,
        'strain_pl':StrainTensor,
        'velgrad':SecondOrderTensor,
        'defrate':SymmetricSecondOrderTensor,
        'defrate_pl':SymmetricSecondOrderTensor,
        'spinrate':SkewSymmetricSecondOrderTensor,
        }


def _list_valid_filenames(folder, startswith='strain'):
    file_list = os.listdir(folder)
    pattern = r'{}\.step\d+'.format(startswith)
    return [f for f in file_list if re.fullmatch(pattern, f)]

def from_step_file(file, dtype=None):
    """
    Import data from a single step file given by FEPX.

    The type of returns is inferred from the data one wants to parse
    (according the `FEPX documentation <https://fepx.info/doc/output.html>`_ ).

    Parameters
    ----------
    file : str
        Path to the file to read
    dtype : type, optional
        If provided, sets the type of returned array. It can be:
          - float
          - SecondOrderTensor
          - SymmetricSecondOrderTensor
          - SkewSymmetricSecondOrderTensor
          - stressTensor
          - strainTensor

    Returns
    -------
    SecondOrderTensor or numpy.ndarray
        Array of second-order tensors built from the read data. The array will be of shape (n,), where n is the number
        of elements in the mesh.
    """
    data = pd.read_csv(file, header=None, sep=' ')
    array = data.to_numpy()
    base_name = os.path.splitext(os.path.basename(file))[0]
    if dtype is None:
        if base_name in DTYPES:
            dtype = DTYPES[base_name]
        else:
            dtype = float
    if issubclass(dtype,SymmetricSecondOrderTensor):
        return dtype.from_Voigt(array, voigt_map=[1,1,1,1,1,1])
    elif dtype == SkewSymmetricSecondOrderTensor:
        zeros = np.zeros(array.shape[0])
        mat = np.array([[ zeros,         array[:, 0],   array[:, 1]],
                        [-array[:, 0],  zeros,          array[:, 2]],
                        [-array[:, 1], -array[:, 2],    zeros     ]]).transpose((2, 0, 1))
        return SkewSymmetricSecondOrderTensor(mat)
    elif dtype == SecondOrderTensor:
        length = array.shape[0]
        return SecondOrderTensor(array.reshape((length,3,3)))
    elif dtype == float:
        return array



def from_results_folder(folder, dtype=None):
    """
    Import all data of a given field from FEPX results folder.

    The type of returns is inferred from the data one wants to parse
    (according the `FEPX documentation <https://fepx.info/doc/output.html>`_ ).

    Parameters
    ----------
    folder : str
        Path to the folder to read the results from
    dtype : type, optional
        If provided, sets the type of returned array. It can be:
          - float
          - SecondOrderTensor
          - SymmetricSecondOrderTensor
          - SkewSymmetricSecondOrderTensor
          - StressTensor
          - StrainTensor

    Returns
    -------
    SecondOrderTensor or numpy.ndarray
        Array of second-order tensors built from the read data. The array will be of shape (m, n), where m is the number
        of time increment n is the number of elements in the mesh.
    """
    dir_path = Path(folder)
    folder_name = dir_path.name
    if not dir_path.is_dir():
        raise ValueError(f"{folder} is not a valid directory.")
    constructor = None
    array = []
    for file in dir_path.iterdir():
        if file.is_file() and file.name.startswith(folder_name):
            data_file = from_step_file(str(file), dtype=dtype)
            if constructor is None:
                constructor = type(data_file)
            elif constructor != type(data_file):
                raise ValueError('The types of data contained in {} seem to be inconsistent.'.format(folder))
            array.append(data_file)
    if constructor == np.ndarray:
        return np.stack(array, axis=0)
    else:
        return constructor.stack(array)
