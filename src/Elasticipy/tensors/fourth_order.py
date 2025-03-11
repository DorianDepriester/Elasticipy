import numpy as np
from Elasticipy.tensors.second_order import SymmetricSecondOrderTensor, rotation_to_matrix, is_orix_rotation, \
    SecondOrderTensor, ALPHABET
from scipy.spatial.transform import Rotation
from copy import deepcopy

a = np.sqrt(2)
KELVIN_MAPPING_MATRIX = np.array([[1, 1, 1, a, a, a],
                                  [1, 1, 1, a, a, a],
                                  [1, 1, 1, a, a, a],
                                  [a, a, a, 2, 2, 2],
                                  [a, a, a, 2, 2, 2],
                                  [a, a, a, 2, 2, 2], ])


def voigt_indices(i, j):
    """
    Translate the two-index notation to one-index notation

    Parameters
    ----------
    i : int or np.ndarray
        First index
    j : int or np.ndarray
        Second index

    Returns
    -------
    Index in the vector of length 6
    """
    voigt_mat = np.array([[0, 5, 4],
                          [5, 1, 3],
                          [4, 3, 2]])
    return voigt_mat[i, j]


def unvoigt_index(i):
    """
    Translate the one-index notation to two-index notation

    Parameters
    ----------
    i : int or np.ndarray
        Index to translate
    """
    inverse_voigt_mat = np.array([[0, 0],
                                  [1, 1],
                                  [2, 2],
                                  [1, 2],
                                  [0, 2],
                                  [0, 1]])
    return inverse_voigt_mat[i]

def rotate_tensor(full_tensor, r):
    """
    Rotate a (full) fourth-order tensor.

    Parameters
    ----------
    full_tensor : numpy.ndarray
        array of shape (3,3,3,3) or (...,3,3,3,3) containing all the components
    r : scipy.spatial.Rotation or orix.quaternion.Rotation
        Rotation, or set of rotations, to apply

    Returns
    -------
    numpy.ndarray
        Rotated tensor. If r is an array, the corresponding axes will be added as first axes in the result array.
    """
    rot_mat = rotation_to_matrix(r)
    str_ein = '...im,...jn,...ko,...lp,...mnop->...ijkl'
    return np.einsum(str_ein, rot_mat, rot_mat, rot_mat, rot_mat, full_tensor)

class FourthOrderTensor:
    """
    Template class for manipulating symmetric fourth-order tensors.

    Attributes
    ----------
    matrix : np.ndarray
        (6,6) matrix gathering all the components of the tensor, using the Voigt notation.
    """
    tensor_name = '4th-order'

    def __init__(self, M, mapping='Kelvin'):
        """
        Construct of stiffness tensor from a (6,6) matrix.

        The input matrix must be symmetric, otherwise an error is thrown (except if check_symmetry==False, see below)

        Parameters
        ----------
        M : np.ndarray
            (6,6) matrix corresponding to the stiffness tensor, written using the Voigt notation, or array of shape
            (3,3,3,3).
        mapping : str or list of list, or numpy/ndarray, optional
            Mapping convention to translate the (3,3,3,3) array to (6,6) matrix
        """
        if isinstance(mapping, (list, tuple, np.ndarray)):
            self.mapping_matrix = mapping
            self.mapping_name = 'custom'
        else:
            mapping = mapping.capitalize()
            self.mapping_name = mapping
            if mapping == 'Kelvin':
                self.mapping_matrix = KELVIN_MAPPING_MATRIX
            elif mapping == 'Voigt':
                self.mapping_matrix = np.ones((6,6))
            else:
                raise ValueError('The mapping to use can be either "Kelvin", "Voigt", or a (6,6) matrix.')

        M = np.asarray(M)
        if M.shape[-2:] == (6, 6):
            matrix = M
        elif M.shape[-4:] == (3, 3, 3, 3):
            matrix = self._full_to_matrix(M)
        else:
            raise ValueError('The input matrix must of shape (6,6)')


        self.matrix = matrix
        for i in range(0, 6):
            for j in range(0, 6):
                def getter(obj, I=i, J=j):
                    return obj.matrix[...,I, J]

                getter.__doc__ = f"Returns the ({i + 1},{j + 1}) component of the {self.tensor_name} matrix."
                component_name = 'C{}{}'.format(i + 1, j + 1)
                setattr(self.__class__, component_name, property(getter))  # Dynamically create the property

    def __repr__(self):
        if (self.ndim == 0) or ((self.ndim==1) and self.shape[0]<5):
            msg = '{} tensor (in {} mapping):\n'.format(self.tensor_name, self.mapping_name)
            msg += self.matrix.__str__()
        else:
            msg = '{} tensor array of shape {}'.format(self.tensor_name, self.shape)
        return msg

    @property
    def shape(self):
        """
        Return the shape of the tensor array
        Returns
        -------
        tuple
            Shape of the tensor array
        """
        *shape, _, _ = self.matrix.shape
        return tuple(shape)

    def full_tensor(self):
        """
        Returns the full (unvoigted) tensor, as a [3, 3, 3, 3] array

        Returns
        -------
        np.ndarray
            Full tensor (4-index notation)
        """
        i, j, k, ell = np.indices((3, 3, 3, 3))
        ij = voigt_indices(i, j)
        kl = voigt_indices(k, ell)
        m = self.matrix[..., ij, kl] / self.mapping_matrix[ij, kl]
        return m

    def flatten(self):
        """
        Flatten the tensor

        If the tensor has (m,n,o...,r) orientations, the flattened tensor will have m*n*o*...*r orientations

        Returns
        -------
        SymmetricFourthOrderTensor
            Flattened tensor
        """
        shape = self.shape
        if shape:
            t2 = deepcopy(self)
            p = (np.prod(self.shape), 6, 6)
            t2.matrix = self.matrix.reshape(p)
            return t2
        else:
            return self

    def _full_to_matrix(self, full_tensor):
        kl, ij = np.indices((6, 6))
        i, j = unvoigt_index(ij).T
        k, ell = unvoigt_index(kl).T
        return full_tensor[..., i, j, k, ell] * self.mapping_matrix[ij, kl]

    def rotate(self, rotation):
        """
        Apply a single rotation to a tensor, and return its component into the rotated frame.

        Parameters
        ----------
        rotation : Rotation or orix.quaternion.rotation.Rotation
            Rotation to apply

        Returns
        -------
        SymmetricFourthOrderTensor
            Rotated tensor
        """
        t2 = deepcopy(self)
        rotated_tensor = rotate_tensor(self.full_tensor(), rotation)
        t2.matrix = self._full_to_matrix(rotated_tensor)
        return t2

    @property
    def ndim(self):
        """
        Returns the dimensionality of the tensor (number of dimensions in the orientation array)

        Returns
        -------
        int
            Number of dimensions
        """
        shape = self.shape
        if shape:
            return len(shape)
        else:
            return 0

    def mean(self, axis=None):
        """
        Compute the mean value of the tensor T

        Parameters
        ----------
        axis : int or list of int or tuple of int, optional
            axis along which to compute the mean. If None, the mean is computed on the flattened tensor

        Returns
        -------
        numpy.ndarray
            If no axis is given, the result will be of shape (3,3,3,3).
            Otherwise, if T.ndim=m, and len(axis)=n, the returned value will be of shape (...,3,3,3,3), with ndim=m-n+4
        """
        t2 = deepcopy(self)
        if axis is None:
            axis = tuple([i for i in range(0,self.ndim)])
        t2.matrix = np.mean(self.matrix, axis=axis)
        return t2

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            if other.shape == (6, 6):
                mat = self.matrix + other
            elif other.shape == (3, 3, 3, 3):
                mat = self._full_to_matrix(self.full_tensor() + other)
            else:
                raise ValueError('The input argument must be either a 6x6 matrix or a (3,3,3,3) array.')
        elif isinstance(other, FourthOrderTensor):
            if type(other) == type(self):
                mat = self.matrix + other.matrix
            else:
                raise ValueError('The two tensors to add must be of the same class.')
        else:
            raise ValueError('I don''t know how to add {} with {}.'.format(type(self), type(other)))
        return self.__class__(mat)

    def __sub__(self, other):
        if isinstance(other, FourthOrderTensor):
            return self.__add__(-other.matrix)
        else:
            return self.__add__(-other)

    def ddot(self, other, mode='pair'):
        """
        Perform tensor product contracted twice (":") between two fourth-order tensors

        Parameters
        ----------
        other : FourthOrderTensor or SecondOrderTensor
            Right-hand side of ":" symbol
        mode : str, optional
            If mode=="pair", the tensors must be broadcastable, and the tensor product are performed on the last axes.
            If mode=="cross", all cross-combinations are considered.

        Returns
        -------
        FourthOrderTensor or numpy.ndarray
         If both the tensors are 0D (no orientation), the return value will be of type SymmetricTensor
         Otherwise, the return value will be the full tensor, of shape (...,3,3,3,3).
        """
        if isinstance(other, FourthOrderTensor):
            if self.ndim == 0 and other.ndim == 0:
                return FourthOrderTensor(np.einsum('ijmn,nmkl->ijkl', self.full_tensor(), other.full_tensor()))
            else:
                if mode == 'pair':
                    ein_str = '...ijmn,...nmkl->...ijkl'
                else:
                    ndim_0 = self.ndim
                    ndim_1 = other.ndim
                    indices_0 = ALPHABET[:ndim_0]
                    indices_1 = ALPHABET[:ndim_1].upper()
                    indices_2 = indices_0 + indices_1
                    ein_str = indices_0 + 'wxXY,' + indices_1 + 'YXyz->' + indices_2 + 'wxyz'
                matrix = np.einsum(ein_str, self.full_tensor(), other.full_tensor())
                return FourthOrderTensor(matrix)
        elif isinstance(other, SecondOrderTensor):
            if self.ndim == 0 and other.ndim == 0:
                return SymmetricSecondOrderTensor(np.einsum('ijkl,kl->ij', self.full_tensor(), other.matrix))
            else:
                if mode == 'pair':
                    ein_str = '...ijkl,...kl->...ij'
                else:
                    ndim_0 = self.ndim
                    ndim_1 = other.ndim
                    indices_0 = ALPHABET[:ndim_0]
                    indices_1 = ALPHABET[:ndim_1].upper()
                    indices_2 = indices_0 + indices_1
                    ein_str = indices_0 + 'wxXY,' + indices_1 + 'XY->' + indices_2 + 'wx'
                matrix = np.einsum(ein_str, self.full_tensor(), other.matrix)
                return SecondOrderTensor(matrix)


    def __mul__(self, other):
        if isinstance(other, (SymmetricFourthOrderTensor, SymmetricSecondOrderTensor)):
            return self.ddot(other)
        elif isinstance(other, np.ndarray):
            shape = other.shape
            if other.shape == self.shape[-len(shape):]:
                matrix = self.matrix * other[...,np.newaxis, np.newaxis]
                return self.__class__(matrix)
            else:
                raise ValueError('The arrays to multiply could not be broadcasted with shapes {} and {}'.format(self.shape, other.shape[:-2]))
        elif isinstance(other, Rotation) or is_orix_rotation(other):
            return self.rotate(other)
        else:
            return self.__class__(self.matrix * other)

    def transpose_array(self):
        """
        Transpose the orientations of the tensor array

        Returns
        -------
        FourthOrderTensor
            The same tensor, but with transposed axes
        """
        ndim = self.ndim
        if ndim==0 or ndim==1:
            return self
        else:
            new_axes = tuple(range(ndim))[::-1] + (ndim, ndim + 1)
            transposed_matrix = self.matrix.transpose(new_axes)
            return self.__class__(transposed_matrix)

    def __rmul__(self, other):
        if isinstance(other, (Rotation, float, int, np.number)) or is_orix_rotation(other):
            return self * other
        else:
            raise NotImplementedError('A fourth order tensor can be left-multiplied by rotations or scalar only.')

    def __truediv__(self, other):
        if isinstance(other, (float, int, np.number)):
            return self.__class__(self.matrix / other)
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, FourthOrderTensor):
            return np.all(self.matrix == other.matrix, axis=(-1,-2))
        elif isinstance(other, (float, int)) or (isinstance(other, np.ndarray) and other.shape[-2:] == (6, 6)):
            return np.all(self.matrix == other, axis=(-1,-2))
        else:
            raise NotImplementedError('The element to compare with must be a fourth-order tensor '
                                      'or an array of shape (6,6).')

    def __getitem__(self, item):
        if self.ndim:
            sub_mat= self.matrix[item]
            if sub_mat.shape[-2:] != (6,6):
                raise IndexError('Too many indices for tensor array: array is {}-dimensional, but {} were provided'.format(self.ndim, len(item)))
            else:
                return self.__class__(sub_mat)
        else:
            raise IndexError('A single tensor cannot be subindexed')

    def __setitem__(self, index, value):
        if isinstance(value, np.ndarray):
            if value.shape[-2:] == (6,6):
                self.matrix[index] = value
            elif value.shape[-4:] == (3,3,3,3):
                submatrix = self._full_to_matrix(value)
                self.matrix[index] = submatrix
            else:
                return ValueError('The R.h.s must be either of shape (...,6,6) or (...,3,3,3,3)')
        elif isinstance(value, FourthOrderTensor):
            self.matrix[index] = value.matrix / value.mapping_matrix * self.mapping_matrix
        else:
            raise NotImplementedError('The r.h.s must be either an ndarray or an object of class {}'.format(self.__class__))

    @classmethod
    def identity(cls, shape=(), return_full_tensor=False):
        """
        Create a 4th-order identity tensor

        Parameters
        ----------
        shape : int or tuple, optional
            Shape of the tensor to create
        return_full_tensor : bool, optional
            If True, return the full tensor as a (3,3,3,3) or a (...,3,3,3,3) array. Otherwise, the tensor is returned
            as a SymmetricTensor object.

        Returns
        -------
        numpy.ndarray or SymmetricTensor
            Identity tensor
        """
        eye = np.eye(3)
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape):
            for n in shape:
                eye = np.repeat(eye[np.newaxis,...], n, axis=0)
        a = np.einsum('...ik,...jl->...ijkl', eye, eye)
        b = np.einsum('...il,...jk->...ijkl', eye, eye)
        full = 0.5*(a + b)
        if return_full_tensor:
            return full
        else:
            return cls(full)

    def inv(self):
        """
        Invert the tensor. The inverted tensors inherits the properties (if any)

        Returns
        -------
        FourthOrderTensor
            Inverse tensor
        """
        t2 = deepcopy(self)
        new_matrix = np.linalg.inv(self.matrix)
        t2.matrix = new_matrix
        return t2

    @classmethod
    def zeros(cls, shape=()):
        """
        Create a fourth-order tensor populated with zeros

        Parameters
        ----------
        shape : int or tuple, optional
            Shape of the tensor to create
        Returns
        -------
        FourthOrderTensor
        """
        if isinstance(shape, int):
            shape = (shape, 6, 6)
        else:
            shape = shape + (6,6)
        zeros = np.zeros(shape)
        return cls(zeros)

class SymmetricFourthOrderTensor(FourthOrderTensor):
    tensor_name = 'Symmetric 4th-order'

    def __init__(self, M, check_symmetry=True, force_symmetry=False, **kwargs):
        """
        Construct of symmetric fourth-order tensor from a (6,6) matrix.

        The input matrix must be symmetric, otherwise an error is thrown (except if check_symmetry==False, see below)

        Parameters
        ----------
        M : np.ndarray
            (6,6) matrix corresponding to the stiffness tensor, written using the Voigt notation, or array of shape
            (3,3,3,3).
        check_symmetry : bool, optional
            Whether to check or not that the input matrix is symmetric.
        force_symmetry : bool, optional
            If true, the major symmetry of the tensor is forces
        """
        super().__init__(M, **kwargs)
        if force_symmetry:
            self.matrix = 0.5*(self.matrix + self.matrix.swapaxes(-1,-2))
        elif check_symmetry and not np.all(np.isclose(self.matrix, self.matrix.swapaxes(-1,-2))):
            raise ValueError('The input matrix must be symmetric')