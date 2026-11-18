from elasticipy.tensors.second_order import SymmetricSecondOrderTensor, SecondOrderTensor, \
    SkewSymmetricSecondOrderTensor
from elasticipy.tensors.stress_strain import StressTensor, StrainTensor, StrainRateTensor
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

# Dictionary mapping FEPX variable names to their corresponding data types
# Based on FEPX documentation: https://fepx.info/doc/output.html#elemental-results
#
# Note: FEPX writes tensor components (not engineering shears) in Voigt notation:
# - Symmetric tensors: t11, t22, t33, t23, t13, t12 (6 components)
# - General tensors: all 9 components L11, L12, L13, L21, L22, L23, L31, L32, L33
# - Skew-symmetric tensors: w12, w13, w23 (3 components)
# - Vectors: v1, v2, v3 (3 components)
# - Scalars: single value per element

DTYPES = {
    # Symmetric second-order tensors (6 components)
    'stress': StressTensor,           # Cauchy stress [MPa]
    'strain': StrainTensor,           # Total strain [-]
    'strain_el': StrainTensor,        # Elastic strain [-]
    'strain_pl': StrainTensor,        # Plastic strain [-] (in crystal frame, written in sample frame)
    'strain_pl_sample': StrainTensor, # Plastic strain [-] (in sample frame)
    'defrate': StrainRateTensor,       # Deformation rate d = sym(L) [1/s]
    'defrate_pl': StrainRateTensor,    # Plastic deformation rate [1/s]
    'dmg': SymmetricSecondOrderTensor, # Damage tensor [1/mm²] (in crystal frame)
    'dmgrate': SymmetricSecondOrderTensor, # Damage rate [1/(mm² s)]
    
    # General second-order tensors (9 components)
    'velgrad': SecondOrderTensor,     # Velocity gradient L [1/s]
    
    # Skew-symmetric second-order tensors (3 components: w12, w13, w23)
    'spinrate': SkewSymmetricSecondOrderTensor,  # Plastic spin rate [1/s]
    
    # Vectors (3 components)
    # Note: These will be returned as numpy arrays with shape (n, 3)
    'coo': np.ndarray,                # Coordinates [mm] - nodal result
    'disp': np.ndarray,               # Displacements [mm] - nodal result
    'vel': np.ndarray,                # Velocities [mm/s] - nodal result
    'ori': np.ndarray,               # Crystal orientation (format depends on input)
    'rotrate': np.ndarray,            # Lattice rotation rate [1/s] (rotation vector: ω1, ω2, ω3)
    'rotrate_spin': np.ndarray,       # Spin rotation rate [1/s] (rotation vector)
    'rotrate_slip': np.ndarray,       # Slip rotation rate [1/s] (rotation vector)
    
    # Face results (3 components: Fx, Fy, Fz)
    'forces': np.ndarray,             # Face forces [N] - face result
    
    # Scalars (1 component per element)
    # Note: These will be returned as numpy arrays with shape (n,)
    'stress_eq': float,              # Equivalent (von Mises) stress [MPa]
    'strain_eq': float,               # Equivalent total strain [-]
    'strain_el_eq': float,            # Equivalent elastic strain [-]
    'strain_pl_eq': float,            # Equivalent plastic strain [-]
    'defrate_eq': float,              # Equivalent deformation rate [1/s]
    'defrate_pl_eq': float,           # Equivalent plastic deformation rate [1/s]
    'work': float,                    # Work per unit volume [MPa = mJ/mm³]
    'work_pl': float,                 # Plastic work per unit volume [MPa]
    'workrate': float,                 # Work rate σ:d [MPa/s]
    'workrate_pl': float,              # Plastic work rate [MPa/s]
    
    # Crystal plasticity results (slip system values)
    # These have variable number of components (one per slip system)
    # Will be returned as numpy arrays
    'crss': np.ndarray,               # Slip-system strengths τcrit [MPa]
    'rss': np.ndarray,                # Resolved shear stresses [MPa]
    'sliprate': np.ndarray,           # Slip rates γ̇ [1/s]
    'slip': np.ndarray,               # Accumulated slip γ [-]
    'strain_pl_sample_eq': float,     # Equivalent plastic strain of strain_pl_sample [-]
    
    # Kocks-Mecking hardening (Crystal Plasticity)
    'km_rho': float,                  # Network dislocation density ρ [1/mm²]
    'km_k2': float,                   # Dynamic-recovery coefficient k2 [-]
    'km_rhorate': float,              # Rate of network dislocation density ρ̇ [1/(mm² s)]
    
    # Convergence (global iterations)
    'convergence': np.ndarray,        # Convergence data (special format)
}


def _list_valid_filenames(folder, startswith='strain'):
    file_list = os.listdir(folder)
    pattern = r'{}\.step\d+'.format(startswith)
    matching_files = [f for f in file_list if re.fullmatch(pattern, f)]
    
    # Sort files by step number extracted from filename
    def extract_step_number(filename):
        match = re.search(r'\.step(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    matching_files.sort(key=extract_step_number)
    return matching_files

def from_step_file(file, dtype=None):
    """
    Import data from a single step file given by FEPX.

    The type of returns is inferred from the data one wants to parse
    (according the `FEPX documentation <https://fepx.info/doc/output.html>`_ ).

    Parameters
    ----------
    file : str
        Path to the file to read
    dtype : ``type``, optional
        If provided, sets the type of returned array. It can be:
          - float
          - numpy.ndarray
          - SecondOrderTensor
          - SymmetricSecondOrderTensor
          - SkewSymmetricSecondOrderTensor
          - StressTensor
          - StrainTensor
          - StrainRateTensor

    Returns
    -------
    SecondOrderTensor, numpy.ndarray, or float array
        Array built from the read data. The shape depends on the data type:
        - Tensors: shape (n,) where n is the number of elements
        - Vectors: shape (n, 3)
        - Scalars: shape (n,)
    """
    data = pd.read_csv(file, header=None, sep=' ')
    array = data.to_numpy()
    base_name = os.path.splitext(os.path.basename(file))[0]
    
    # Determine dtype from base_name if not provided
    if dtype is None:
        if base_name in DTYPES:
            dtype = DTYPES[base_name]
        else:
            # Auto-detect dtype based on number of columns
            n_cols = array.shape[1]
            if n_cols == 1:
                dtype = float  # Scalar
            elif n_cols == 3:
                dtype = np.ndarray  # Vector
            elif n_cols == 6:
                dtype = SymmetricSecondOrderTensor  # Symmetric tensor
            elif n_cols == 9:
                dtype = SecondOrderTensor  # General tensor
            else:
                dtype = np.ndarray  # Unknown, return as array
    
    # Handle different data types
    if dtype == float or dtype == np.float64 or dtype == np.float32:
        # Scalar data: return 1D array
        if array.shape[1] == 1:
            return array[:, 0]
        else:
            return array
    elif dtype == np.ndarray:
        # Return as numpy array (vectors, orientations, etc.)
        return array
    elif issubclass(dtype, SymmetricSecondOrderTensor):
        # Symmetric tensor (6 components)
        # FEPX writes tensor components (not engineering shears), so use mapping [1,1,1,1,1,1]
        mapping = [1, 1, 1, 1, 1, 1]  # https://fepx.info/doc/output.html#elemental-results
        return dtype.from_Voigt(array, voigt_map=mapping)
    elif issubclass(dtype, SkewSymmetricSecondOrderTensor):
        # Skew-symmetric tensor (3 components: w12, w13, w23)
        zeros = np.zeros(array.shape[0])
        mat = np.array([[ zeros,         array[:, 0],   array[:, 1]],
                        [-array[:, 0],  zeros,          array[:, 2]],
                        [-array[:, 1], -array[:, 2],    zeros     ]]).transpose((2, 0, 1))
        return SkewSymmetricSecondOrderTensor(mat)
    elif issubclass(dtype, SecondOrderTensor):
        # General second-order tensor (9 components)
        length = array.shape[0]
        return SecondOrderTensor(array.reshape((length, 3, 3)))
    else:
        # Fallback: return as numpy array
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
    dtype : ``type``, optional
        If provided, sets the type of returned array. It can be:
          - float
          - numpy.ndarray
          - SecondOrderTensor
          - SymmetricSecondOrderTensor
          - SkewSymmetricSecondOrderTensor
          - StressTensor
          - StrainTensor
          - StrainRateTensor

    Returns
    -------
    SecondOrderTensor, numpy.ndarray, or float array
        Array built from the read data. The shape depends on the data type:
        - Tensors: shape (m, n) where m is the number of steps, n is the number of elements
        - Vectors: shape (m, n, 3)
        - Scalars: shape (m, n)
    """
    dir_path = Path(folder)
    folder_name = dir_path.name
    if not dir_path.is_dir():
        raise ValueError(f"{folder} is not a valid directory.")
    
    # Collect all matching files and sort them by step number
    matching_files = []
    for file in dir_path.iterdir():
        if file.is_file() and file.name.startswith(folder_name):
            matching_files.append(file)
    
    # Sort files by step number extracted from filename
    # Filename pattern: <folder_name>.step<number>
    def extract_step_number(filename):
        # Extract the step number from filename like "strain.step10"
        match = re.search(r'\.step(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    matching_files.sort(key=lambda f: extract_step_number(f.name))
    
    constructor = None
    array = []
    
    for file in matching_files:
        data_file = from_step_file(str(file), dtype=dtype)
        if constructor is None:
            constructor = type(data_file)
        elif constructor != type(data_file):
            raise ValueError('The types of data contained in {} seem to be inconsistent.'.format(folder))
        array.append(data_file)
    
    # Stack the results along the first axis (time steps)
    if constructor == np.ndarray:
        # For numpy arrays, use np.stack
        return np.stack(array, axis=0)
    elif hasattr(constructor, 'stack'):
        # For tensor classes that have a stack method
        return constructor.stack(array)
    else:
        # Fallback: try to stack as numpy array
        return np.stack(array, axis=0)
