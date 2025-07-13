from Elasticipy.tensors.second_order import SymmetricSecondOrderTensor, SecondOrderTensor, \
    SkewSymmetricSecondOrderTensor
from Elasticipy.tensors.stress_strain import StressTensor, StrainTensor
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


def _list_valid_filenames(folder, startswith='strain'):
    file_list = os.listdir(folder)
    pattern = r'{}\.step\d+'.format(startswith)
    return [f for f in file_list if re.fullmatch(pattern, f)]

def from_step_file(file, dtype=None):
    """
    Import data from a single step file given by FEPX.

    The type of data is inferred from the file basename ("strain.stepX" -> StrainTensor, "stress.stepX" -> StressTensor
    etc.)

    Parameters
    ----------
    file : str
        Path to the file to read
    dtype : str, optional
        If provided, force sets the type of returned array. It can be:
          - None (let the function infer the dtype from the data)
          - SecondOrderTensor
          - SymmetricSecondOrderTensor
          - SkewSymmetricSecondOrderTensor
          - stressTensor
          - strainTensor
          - float
          - int

    Returns
    -------
    SecondOrderTensor
        Array of second-order tensors built from the read data. The array will be of shape (n,), where n is the number
        of elements in the mesh.
    """
    data = pd.read_csv(file, header=None, sep=' ')
    array = data.to_numpy()
    base_name = os.path.splitext(os.path.basename(file))[0]
    if isinstance(dtype, str):
        dtype = re.sub('[^A-Za-z0-9]+', '', dtype).lower()
    if base_name == 'strain' or dtype=='straintensor':
        return StrainTensor.from_Voigt(array, voigt_map=[1,1,1,1,1,1])
    elif base_name == 'stress' or dtype=='stresstensor':
        return StressTensor.from_Voigt(array)
    else:
        n_compo = array.shape[1]
        if n_compo == 3:
            if dtype is None:
                raise ValueError('I cannot automatically infer the dtype from the data. Use dtype option to set whether'
                                 ' it is a float array or a skew-symmetric second-order tensor array.')
            elif dtype == 'skewsymmetricsecondordertensor':
                zeros = np.zeros(array.shape[0])
                mat = np.array([[ zeros,         array[:, 0],   array[:, 1]],
                                [-array[:, 0],  zeros,          array[:, 2]],
                                [-array[:, 1], -array[:, 2],    zeros     ]]).transpose((2, 0, 1))
                return SkewSymmetricSecondOrderTensor(mat)
            elif dtype == 'float':
                return array
        elif n_compo == 6:
            return SymmetricSecondOrderTensor.from_Voigt(array)
        elif n_compo == 9:
            mat = np.array([[array[:,0], array[:,1], array[:,2]],
                            [array[:,3], array[:,4], array[:,5]],
                            [array[:,6], array[:,7], array[:,8]]]).transpose((2,0,1))
            return SecondOrderTensor(mat)


def from_results_folder(folder):
    """
    Import all result data (all steps) from a given FEPX's results folder

    Parameters
    ----------
    folder : str
        Path to the results folder

    Returns
    -------
    SecondOrderTensor
        Array of second-order tensors built from the read data. The array will be of shape (m,n), where m is the number
        of steps and n is the number of elements in the mesh.
    """
    dir_path = Path(folder)
    folder_name = dir_path.name
    if not dir_path.is_dir():
        raise ValueError(f"{folder} is not a valid directory.")
    dtype = None
    array = []
    for file in dir_path.iterdir():
        if file.is_file() and file.name.startswith(folder_name):
            data_file = from_step_file(str(file))
            if dtype is None:
                dtype = type(data_file)
            elif dtype != type(data_file):
                raise ValueError('The types of data contained in {} seem to be inconsistent.'.format(folder))
            array.append(from_step_file(file))
    return dtype.stack(array)
