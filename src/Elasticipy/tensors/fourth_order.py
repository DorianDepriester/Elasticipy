import numpy as np
from Elasticipy.tensors.second_order import SymmetricSecondOrderTensor, rotation_to_matrix, is_orix_rotation, \
    SecondOrderTensor, ALPHABET
from scipy.spatial.transform import Rotation
from copy import deepcopy
from Elasticipy.tensors.mapping import KelvinMapping, VoigtMapping

kelvin_mapping = KelvinMapping()

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

def _rotate_tensor(full_tensor, r):
    rot_mat = rotation_to_matrix(r)
    str_ein = '...im,...jn,...ko,...lp,...mnop->...ijkl'
    return np.einsum(str_ein, rot_mat, rot_mat, rot_mat, rot_mat, full_tensor)

def _isotropic_matrix(C11, C12, C44):
    C11 = np.asarray(C11)
    C12 = np.asarray(C12)
    C44 = np.asarray(C44)
    shape = np.broadcast_shapes(C11.shape, C12.shape, C44.shape)
    matrix = np.zeros(shape=shape + (6, 6))
    matrix[..., 0, 0] = C11
    matrix[..., 1, 1] = C11
    matrix[..., 2, 2] = C11
    matrix[..., 0, 1] = matrix[..., 1, 0] = C12
    matrix[..., 0, 2] = matrix[..., 2, 0] = C12
    matrix[..., 1, 2] = matrix[..., 2, 1] = C12
    matrix[..., 3, 3] = C44
    matrix[..., 4, 4] = C44
    matrix[..., 5, 5] = C44
    return matrix

class FourthOrderTensor:
    """
    Template class for manipulating symmetric fourth-order tensors.
    """
    _tensor_name = '4th-order'

    def _array_to_Kelvin(self, matrix):
        return matrix / self.mapping.matrix * kelvin_mapping.matrix

    def __init__(self, M, mapping=kelvin_mapping, check_minor_symmetry=True, force_minor_symmetry=False):
        """
        Construct of Fourth-order tensor with minor symmetry.

        Parameters
        ----------
        M : np.ndarray or FourthOrderTensor
            (...,6,6) matrix corresponding to the stiffness tensor, written using the Voigt notation, or array of shape
            (...,3,3,3,3).
        mapping : MappingConvention, optional
            Mapping convention to translate the (3,3,3,3) array to (6,6) matrix.
        check_minor_symmetry : bool, optional
            If true (default), check that the input array have minor symmetries (see Notes). Only used if an array of
            shape (...,3,3,3,3) is passed.
        force_minor_symmetry :
            Ensure that the tensor displays minor symmetry.

        Notes
        -----
        The minor symmetry is defined so that:

        .. math::

            M_{ijkl}=M_{jikl}=M_{jilk}=M_{ijlk}

        Given a generic 4th-order tensor T, the corresponding matrix with respect to Kelvin convention is:

        .. math::

            T =
            \\begin{bmatrix}
                T_{1111}          & T_{1122}          & T_{1133}            & \\sqrt{2}T_{1123} & \\sqrt{2}T_{1113} & \\sqrt{2}T_{1112}\\\\
                T_{2211}          & T_{2222}          & T_{2233}            & \\sqrt{2}T_{2223} & \\sqrt{2}T_{2213} & \\sqrt{2}T_{2212}\\\\
                T_{3311}          & T_{3322}          & T_{3333}            & \\sqrt{2}T_{3323} & \\sqrt{2}T_{3313} & \\sqrt{2}T_{3312}\\\\
                \\sqrt{2}T_{2311} & \\sqrt{2}T_{2322}   & \\sqrt{2}T_{2333} & 2T_{2323}         & 2T_{2313}         & 2T_{2312}\\\\
                \\sqrt{2}T_{1311} & \\sqrt{2}T_{1322}   & \\sqrt{2}T_{1333} & 2T_{423}          & 2T_{1313}         & 2T_{1312}\\\\
                \\sqrt{2}T_{1211} & \\sqrt{2}T_{1222}   & \\sqrt{2}T_{1233} & 2T_{1223}         & 2T_{1223}         & 2T_{1212}\\\\
            \\end{bmatrix}

        Examples
        --------
        Consider a Fourth-order tensor, whose Kelvin matrix is:

        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> import numpy as np
        >>> mat = np.array([[100, 200, 300, 0, 0, 0],
        ...                 [-200, 100, 50, 0, 0, 0],
        ...                 [-300, -50, 100, 0, 0, 0],
        ...                 [0, 0, 0, 150, 0, 0],
        ...                 [0, 0, 0, 0, 150, 0],
        ...                 [0, 0, 0, 0, 0, 150]])
        >>> T = FourthOrderTensor(mat)
        >>> print(T)
        4th-order tensor (in Kelvin mapping):
        [[ 100.  200.  300.    0.    0.    0.]
         [-200.  100.   50.    0.    0.    0.]
         [-300.  -50.  100.    0.    0.    0.]
         [   0.    0.    0.  150.    0.    0.]
         [   0.    0.    0.    0.  150.    0.]
         [   0.    0.    0.    0.    0.  150.]]

        If one wants to evaluate the tensor as a (full) (3,3,3,3) array:

        >>> T_array = T.full_tensor

        For instance:

        >>> T_array[0,0,0,0]
        100.0

        whereas

        >>> T_array[0,1,0,1] # Corresponds to T_{66}/2
        75.0

        The half factor comes from the Kelvin mapping convention (see Notes). One can also use the Voigt mapping to
        avoid this:

        >>> from Elasticipy.tensors.mapping import VoigtMapping
        >>> T_voigt = FourthOrderTensor(mat, mapping=VoigtMapping())
        >>> print(T_voigt)
        4th-order tensor (in Voigt mapping):
        [[ 100.  200.  300.    0.    0.    0.]
         [-200.  100.   50.    0.    0.    0.]
         [-300.  -50.  100.    0.    0.    0.]
         [   0.    0.    0.  150.    0.    0.]
         [   0.    0.    0.    0.  150.    0.]
         [   0.    0.    0.    0.    0.  150.]]

        Although T and T_voigt appear to be the same, note that they are not expressed using the same mapping
        convention. Indeed:

        >>> T_voigt.full_tensor[0,0,0,0]
        100.0

        whereas

        >>> T_voigt.full_tensor[0,1,0,1]
        150.0

        Alternatively, the differences can be checked with:

        >>> T == T_voigt
        False

        Conversely, let consider the following Voigt matrix:

        >>> mat = np.array([[100, 200, 300, 0, 0, 0],
        ...                 [-200, 100, 50, 0, 0, 0],
        ...                 [-300, -50, 100, 0, 0, 0],
        ...                 [0, 0, 0, 75, 0, 0],
        ...                 [0, 0, 0, 0, 75, 0],
        ...                 [0, 0, 0, 0, 0, 75]])
        >>> T_voigt2 = FourthOrderTensor(mat, mapping=VoigtMapping())
        >>> print(T_voigt2)
        4th-order tensor (in Voigt mapping):
        [[ 100.  200.  300.    0.    0.    0.]
         [-200.  100.   50.    0.    0.    0.]
         [-300.  -50.  100.    0.    0.    0.]
         [   0.    0.    0.   75.    0.    0.]
         [   0.    0.    0.    0.   75.    0.]
         [   0.    0.    0.    0.    0.   75.]]

        Although T and T_voigt2 are not written using the same mapping, we can compare them:

        >>> T == T_voigt2 # Same tensors, but different mapping
        True

        whereas

        >>> T == T_voigt  # Different tensors, but same mapping
        False

        This property comes from the fact that the comparison is made independently of the underlying mapping convention.
        """
        if isinstance(mapping, str):
            if mapping.lower() == 'voigt':
                mapping = VoigtMapping()
            elif mapping.lower() == 'kelvin':
                mapping = kelvin_mapping
            else:
                raise ValueError('Mapping must be either "voigt" or "kelvin"')
        self.mapping=mapping
        if isinstance(M, FourthOrderTensor):
            self._matrix = M._matrix
        else:
            M = np.asarray(M)
            if M.shape[-2:] == (6, 6):
                matrix = self._array_to_Kelvin(M)
            elif M.shape[-4:] == (3, 3, 3, 3):
                Mijlk = np.swapaxes(M, -1, -2)
                Mjikl = np.swapaxes(M, -3, -4)
                Mjilk = np.swapaxes(Mjikl, -1, -2)
                if force_minor_symmetry:
                    M = 0.25 * (M + Mijlk + Mjikl + Mjilk)
                elif check_minor_symmetry:
                    symmetry = np.all(M == Mijlk) and np.all(M == Mjikl) and np.all(M == Mjilk)
                    if not symmetry:
                        raise ValueError('The input array does not have minor symmetry')
                matrix = self._full_to_matrix(M)
            else:
                raise ValueError('The input matrix must of shape (...,6,6) or (...,3,3,3,3)')
            self._matrix = matrix
        for i in range(0, 6):
            for j in range(0, 6):
                def getter(obj, I=i, J=j):
                    new_matrix = obj._matrix / kelvin_mapping.matrix * self.mapping.matrix
                    return new_matrix[...,I, J]

                getter.__doc__ = f"Returns the ({i + 1},{j + 1}) component of the {self._tensor_name} matrix."
                component_name = 'C{}{}'.format(i + 1, j + 1)
                setattr(self.__class__, component_name, property(getter))  # Dynamically create the property

    def __repr__(self):
        if (self.ndim == 0) or ((self.ndim==1) and self.shape[0]<5):
            msg = '{} tensor (in {} mapping):\n'.format(self._tensor_name, self.mapping.name)
            matrix = self.matrix(self.mapping)
            msg += matrix.__str__()
        else:
            msg = '{} tensor array of shape {}'.format(self._tensor_name, self.shape)
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
        *shape, _, _ = self._matrix.shape
        return tuple(shape)

    @property
    def full_tensor(self):
        """
        Returns the full (unvoigted) tensor as a (3, 3, 3, 3) or (..., 3, 3, 3, 3) array

        Returns
        -------
        np.ndarray
            Full tensor (4-index notation)

        Examples
        --------
        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> I = FourthOrderTensor.eye() # 4th order identity tensor
        >>> print(I)
        4th-order tensor (in Kelvin mapping):
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]

        >>> I_full = I.full_tensor
        >>> type(I_full)
        <class 'numpy.ndarray'>
        >>> I_full.shape
        (3, 3, 3, 3)

        When working on tensor arrays, the shape of the resulting numpy array will change accordlingly. E.g.:

        >>> I_array = FourthOrderTensor.eye(shape=(5,6)) # Array of 4th order identity tensor
        >>> I_array.full_tensor.shape
        (5, 6, 3, 3, 3, 3)
        """
        i, j, k, ell = np.indices((3, 3, 3, 3))
        ij = voigt_indices(i, j)
        kl = voigt_indices(k, ell)
        matrix = self._matrix / kelvin_mapping.matrix
        m = matrix[..., ij, kl]
        return m

    def flatten(self):
        """
        Flatten the tensor

        If the tensor array is of shape (m,n,o...,r), the flattened array will be of shape (m*n*o*...*r,).

        Returns
        -------
        SymmetricFourthOrderTensor
            Flattened tensor

        Examples
        --------
        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> T = FourthOrderTensor.rand(shape=(5,6))
        >>> T
        4th-order tensor array of shape (5, 6)
        >>> T.flatten()
        4th-order tensor array of shape (30,)
        """
        shape = self.shape
        if shape:
            t2 = deepcopy(self)
            p = (np.prod(self.shape), 6, 6)
            t2._matrix = self._matrix.reshape(p)
            return t2
        else:
            return self

    def _full_to_matrix(self, full_tensor):
        kl, ij = np.indices((6, 6))
        i, j = unvoigt_index(ij).T
        k, ell = unvoigt_index(kl).T
        return full_tensor[..., i, j, k, ell] * kelvin_mapping.matrix[ij, kl]

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

        Examples
        --------
        Let start from a given tensor, (say ones):

        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> T = FourthOrderTensor.ones()
        >>> T
        4th-order tensor (in Kelvin mapping):
        [[1.         1.         1.         1.41421356 1.41421356 1.41421356]
         [1.         1.         1.         1.41421356 1.41421356 1.41421356]
         [1.         1.         1.         1.41421356 1.41421356 1.41421356]
         [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
         [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
         [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]]

        Define a rotation. E.g.:

        >>> from scipy.spatial.transform import Rotation
        >>> g = Rotation.from_euler('X', 90, degrees=True)

        Then , apply rotation:

        >>> Trotated = T.rotate(g)
        >>> Trotated
        4th-order tensor (in Kelvin mapping):
        [[ 1.          1.          1.         -1.41421356  1.41421356 -1.41421356]
         [ 1.          1.          1.         -1.41421356  1.41421356 -1.41421356]
         [ 1.          1.          1.         -1.41421356  1.41421356 -1.41421356]
         [-1.41421356 -1.41421356 -1.41421356  2.         -2.          2.        ]
         [ 1.41421356  1.41421356  1.41421356 -2.          2.         -2.        ]
         [-1.41421356 -1.41421356 -1.41421356  2.         -2.          2.        ]]

        Actually, a more simple syntax is:

        >>> T * g
        4th-order tensor (in Kelvin mapping):
        [[ 1.          1.          1.         -1.41421356  1.41421356 -1.41421356]
         [ 1.          1.          1.         -1.41421356  1.41421356 -1.41421356]
         [ 1.          1.          1.         -1.41421356  1.41421356 -1.41421356]
         [-1.41421356 -1.41421356 -1.41421356  2.         -2.          2.        ]
         [ 1.41421356  1.41421356  1.41421356 -2.          2.         -2.        ]
         [-1.41421356 -1.41421356 -1.41421356  2.         -2.          2.        ]]

        Obviously, the original tensor can be retrieved by applying the reverse rotation:

        >>> Trotated * g.inv()
        4th-order tensor (in Kelvin mapping):
        [[1.         1.         1.         1.41421356 1.41421356 1.41421356]
         [1.         1.         1.         1.41421356 1.41421356 1.41421356]
         [1.         1.         1.         1.41421356 1.41421356 1.41421356]
         [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
         [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
         [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]]

        If ``g`` is composed of multiple rotations, this will result in a tensor array, corresponding to each rotation:

        >>> import numpy as np
        >>> theta = np.linspace(0, 90, 100)
        >>> g = Rotation.from_euler('X', theta, degrees=True)
        >>> Trotated = T * g
        >>> Trotated
        4th-order tensor array of shape (100,)
        """
        t2 = deepcopy(self)
        rotated_tensor = _rotate_tensor(self.full_tensor, rotation)
        t2._matrix = self._full_to_matrix(rotated_tensor)
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

        Examples
        --------
        Create a random tensor array of shape (5,6):

        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> T = FourthOrderTensor.rand(shape=(5,6))
        >>> Overall_mean = T.mean()
        >>> Overall_mean.shape
        ()
        >>> Overall_mean # doctest: +SKIP
        4th-order tensor (in Kelvin mapping):
        [[0.514295   0.52259217 0.42899181 0.77148692 0.64073221 0.73211491]
         [0.49422678 0.43718365 0.40786118 0.8170971  0.68435571 0.67262655]
         [0.48753674 0.51142541 0.44650454 0.76310921 0.67724973 0.69430165]
         [0.53946846 0.75101474 0.73578098 1.04338905 1.21598419 0.99489014]
         [0.75354555 0.61193555 0.82341479 1.11197826 0.89183143 1.20986243]
         [0.66078807 0.70126535 0.63719147 0.87567139 1.05671229 1.03004098]]

         >>> axis_0_mean = T.mean(axis=0)
         >>> axis_0_mean.shape
         (6,)
         >>> axis_1_mean = T.mean(axis=1)
         >>> axis_1_mean.shape
         (5,)
        """
        t2 = deepcopy(self)
        if axis is None:
            axis = tuple([i for i in range(0,self.ndim)])
        t2._matrix = np.mean(self._matrix, axis=axis)
        return t2

    def __add__(self, other):
        new_tensor = deepcopy(self)
        if isinstance(other, np.ndarray):
            if other.shape[-2:] == (6, 6):
                mat = self._matrix + self._array_to_Kelvin(other)
            elif other.shape == (3, 3, 3, 3):
                mat = self._full_to_matrix(self.full_tensor + other)
            else:
                raise ValueError('The input argument must be either a 6x6 matrix or a (3,3,3,3) array.')
        elif isinstance(other, FourthOrderTensor):
            if type(other) == type(self):
                mat = self._matrix + other._matrix
            else:
                raise ValueError('The two tensors to add must be of the same class.')
        else:
            raise ValueError('I don''t know how to add {} with {}.'.format(type(self), type(other)))
        new_tensor._matrix = mat
        return new_tensor

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        t = deepcopy(self)
        t._matrix = -t._matrix
        return t

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
        FourthOrderTensor
            Result from double-contraction

        Examples
        --------
        First, let consider two random arrays of Fourth-order tensors:

        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> T1 = FourthOrderTensor.rand(shape=(2,3))
        >>> T2 = FourthOrderTensor.rand(shape=3)
        >>> T1T2_pair = T1.ddot(T2)
        >>> T1T2_pair
        4th-order tensor array of shape (2, 3)

        whereas:

        >>> T1T2_cross = T1.ddot(T2, mode='cross')
        >>> T1T2_cross
        4th-order tensor array of shape (2, 3, 3)

        The command above is equivalent (but way faster) to:

        >>> T1T2_cross_loop = FourthOrderTensor.zeros(shape=(2,3,3))
        >>> for i in range(2):
        ...     for j in range(3):
        ...         for k in range(3):
        ...             T1T2_cross_loop[i,j,k] = T1[i,j].ddot(T2[k])

        One can check that the results are consistent with:

        >>> T1T2_cross_loop == T1T2_cross
        array([[[ True,  True,  True],
                [ True,  True,  True],
                [ True,  True,  True]],
        <BLANKLINE>
               [[ True,  True,  True],
                [ True,  True,  True],
                [ True,  True,  True]]])

        """
        if isinstance(other, FourthOrderTensor):
            if self.ndim == 0 and other.ndim == 0:
                return FourthOrderTensor(np.einsum('ijmn,nmkl->ijkl', self.full_tensor, other.full_tensor))
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
                matrix = np.einsum(ein_str, self.full_tensor, other.full_tensor)
                return FourthOrderTensor(matrix)
        elif isinstance(other, SecondOrderTensor):
            if self.ndim == 0 and other.ndim == 0:
                return SymmetricSecondOrderTensor(np.einsum('ijkl,kl->ij', self.full_tensor, other.matrix))
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
                matrix = np.einsum(ein_str, self.full_tensor, other.matrix)
                return SecondOrderTensor(matrix)

    @classmethod
    def rand(cls, shape=None, **kwargs):
        """
        Populate a Fourth-order tensor with random values in half-open interval [0.0, 1.0).

        Parameters
        ----------
        shape : tuple or int, optional
            Set the shape of the tensor array. If None, the returned tensor will be single.
        kwargs
            Keyword arguments passed to the Fourth-order tensor constructor.

        Returns
        -------
        FourthOrderTensor
            Fourth-order tensor
        """
        if shape is None:
            shape = (6,6)
        elif isinstance(shape, int):
            shape = (shape, 6, 6)
        else:
            shape = tuple(shape) + (6,6)
        mat = np.random.random_sample(shape)
        t = FourthOrderTensor(mat, **kwargs)
        t._matrix = t._matrix * t.mapping.matrix
        return t

    def __mul__(self, other):
        if isinstance(other, (FourthOrderTensor, SecondOrderTensor)):
            return self.ddot(other)
        elif isinstance(other, np.ndarray):
            shape = other.shape
            if other.shape == self.shape[-len(shape):]:
                matrix = self._matrix * other[...,np.newaxis, np.newaxis]
                return self.__class__(matrix)
            else:
                raise ValueError('The arrays to multiply could not be broadcasted with shapes {} and {}'.format(self.shape, other.shape[:-2]))
        elif isinstance(other, Rotation) or is_orix_rotation(other):
            return self.rotate(other)
        else:
            new_tensor = deepcopy(self)
            new_tensor._matrix = self._matrix * other
            return new_tensor

    def __truediv__(self, other):
        if isinstance(other, (SecondOrderTensor, FourthOrderTensor)):
            return self * other.inv()
        else:
            return self * (1 / other)


    def transpose_array(self):
        """
        Transpose the orientations of the tensor array

        Returns
        -------
        FourthOrderTensor
            The same tensor, but with transposed axes

        Examples
        --------
        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> A = FourthOrderTensor.rand(shape=(3,4))
        >>> A.transpose_array()
        4th-order tensor array of shape (4, 3)
        """
        ndim = self.ndim
        if ndim==0 or ndim==1:
            return self
        else:
            new_array = deepcopy(self)
            new_axes = tuple(range(ndim))[::-1] + (ndim, ndim + 1)
            new_array._matrix = self._matrix.transpose(new_axes)
            return new_array

    def __rmul__(self, other):
        if isinstance(other, (Rotation, float, int, np.number)) or is_orix_rotation(other):
            return self * other
        else:
            raise NotImplementedError('A fourth order tensor can be left-multiplied by rotations or scalar only.')

    def __eq__(self, other):
        if isinstance(other, FourthOrderTensor):
            return np.all(self._matrix == other._matrix, axis=(-1, -2))
        elif isinstance(other, (float, int)) or (isinstance(other, np.ndarray) and other.shape[-2:] == (6, 6)):
            return np.all(self._matrix == other, axis=(-1, -2))
        else:
            raise NotImplementedError('The element to compare with must be a fourth-order tensor '
                                      'or an array of shape (6,6).')

    def __getitem__(self, item):
        if self.ndim:
            sub_tensor = deepcopy(self)
            sub_tensor._matrix = self._matrix[item]
            return sub_tensor
        else:
            raise IndexError('A single tensor cannot be subindexed')

    def __setitem__(self, index, value):
        if isinstance(value, np.ndarray):
            if value.shape[-2:] == (6,6):
                self._matrix[index] = value / self.mapping.matrix * kelvin_mapping.matrix
            elif value.shape[-4:] == (3,3,3,3):
                submatrix = self._full_to_matrix(value)
                self._matrix[index] = submatrix
            else:
                return ValueError('The R.h.s must be either of shape (...,6,6) or (...,3,3,3,3)')
        elif isinstance(value, FourthOrderTensor):
            self._matrix[index] = value._matrix / value.mapping.matrix * self.mapping.matrix
        else:
            raise NotImplementedError('The r.h.s must be either an ndarray or an object of class {}'.format(self.__class__))


    @classmethod
    def identity(cls, **kwargs):
        """
        Construct the Fourth-order identity tensor.

        This is actually an alias for eye().

        Parameters
        ----------
        kwargs
            Keyword arguments passed to the Fourth-order tensor constructor.

        Returns
        -------
        FourthOrderTensor

        See Also
        --------
        eye : Fourth-order identity tensor
        """
        return cls.eye(**kwargs)

    @classmethod
    def _broadcast_matrix(cls, M, shape=None, **kwargs):
        if shape is None:
            new_shape = M.shape
        elif isinstance(shape, int):
            new_shape = (shape,) + M.shape
        else:
            new_shape = shape + M.shape
        M_repeat = np.broadcast_to(M, new_shape)
        t = cls(M_repeat, **kwargs)
        t._matrix = t._matrix * t.mapping.matrix / kelvin_mapping.matrix
        return t

    @classmethod
    def eye(cls, shape=(), **kwargs):
        """
        Create a 4th-order identity tensor.

        See notes for definition.

        Parameters
        ----------
        shape : int or tuple, optional
            Shape of the tensor to create
        mapping : Kelvin mapping, optional
            Mapping convention to use. Must be either Kelvin or Voigt.

        Returns
        -------
        FourthOrderTensor or SymmetricFourthOrderTensor
            Identity tensor

        Notes
        -----

        The Fourth-order identity tensor is defined as:

        .. math::

            I_{ijkl} = \\frac12\\left( \\delta_{ik}\\delta_{jl} + \\delta_{il}\\delta_{jk}\\right)

        Examples
        --------
        Create a (single) identity tensor:

        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> I = FourthOrderTensor.eye()
        >>> print(I)
        4th-order tensor (in Kelvin mapping):
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]

        Alternatively, one can use another mapping convention, e.g. Voigt:

        >>> from Elasticipy.tensors.mapping import VoigtMapping
        >>> Iv = FourthOrderTensor.eye(mapping=VoigtMapping())
        >>> print(Iv)
        4th-order tensor (in Voigt mapping):
        [[1.  0.  0.  0.  0.  0. ]
         [0.  1.  0.  0.  0.  0. ]
         [0.  0.  1.  0.  0.  0. ]
         [0.  0.  0.  0.5 0.  0. ]
         [0.  0.  0.  0.  0.5 0. ]
         [0.  0.  0.  0.  0.  0.5]]

        Still, we have:

        >>> I == Iv
        True

        as they correspond to the same tensor, but expressed as a matrix with different mapping conventions. Indeed,
        one can check that:

        >>> import numpy as np
        >>> np.array_equal(I.full_tensor, Iv.full_tensor)
        True
        """
        return cls._broadcast_matrix(np.eye(6), shape=shape, **kwargs)

    @classmethod
    def ones(cls, shape=None, **kwargs):
        """
        Create a 4th-order tensor full of ones.

        Parameters
        ----------
        shape : int or tuple, optional
            Shape of the tensor to create
        kwargs
            keyword arguments passed to the constructor

        Returns
        -------
        FourthOrderTensor

        Examples
        --------
        >>> tensor_of_ones = FourthOrderTensor.ones()
        >>> tensor_of_ones
        4th-order tensor (in Kelvin mapping):
        [[1.         1.         1.         1.41421356 1.41421356 1.41421356]
         [1.         1.         1.         1.41421356 1.41421356 1.41421356]
         [1.         1.         1.         1.41421356 1.41421356 1.41421356]
         [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
         [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
         [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]]

        At first sight, the tensor may appear not full of ones at all, but the representation above uses the Kelvin
        mapping convention. Indeed, one can check that the full tensor is actually full of ones. E.g.:

        >>> tensor_of_ones.full_tensor[0,1,0,2]
        1.0

        Alternatively, the Voigt mapping convention may help figuring it out:

        >>> from Elasticipy.tensors.mapping import VoigtMapping
        >>> tensor_of_ones_voigt = FourthOrderTensor.ones(mapping=VoigtMapping())
        >>> tensor_of_ones_voigt
        4th-order tensor (in Voigt mapping):
        [[1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1. 1.]]

        although both tensors are actually the same:

        >>> tensor_of_ones == tensor_of_ones_voigt
        True
        """
        return cls._broadcast_matrix(kelvin_mapping.matrix, shape=shape, **kwargs)

    @classmethod
    def identity_spherical_part(cls, shape=(), **kwargs):
        """
        Return the spherical part of the identity tensor.

        See Notes for mathematical definition.

        Parameters
        ----------
        shape : tuple of int, optional
            Shape of the tensor to create
        kwargs
            Keyword arguments passed to the Fourth-order tensor constructor.

        Returns
        -------
        FourthOrderTensor

        See Also
        --------
        identity_tensor : return the identity tensor
        identity_deviatoric_part : return the deviatoric part of the identity tensor

        Notes
        -----
        The spherical part of the identity tensor is defined as:

        .. math::

            J_{ijkl} = \\frac13 \\delta_{ij}\\delta_{kl}

        Examples
        --------
        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> J = FourthOrderTensor.identity_spherical_part()
        >>> print(J)
        4th-order tensor (in Kelvin mapping):
        [[0.33333333 0.33333333 0.33333333 0.         0.         0.        ]
         [0.33333333 0.33333333 0.33333333 0.         0.         0.        ]
         [0.33333333 0.33333333 0.33333333 0.         0.         0.        ]
         [0.         0.         0.         0.         0.         0.        ]
         [0.         0.         0.         0.         0.         0.        ]
         [0.         0.         0.         0.         0.         0.        ]]

        On can check that J has zero deviatoric part:

        >>> J.deviatoric_part()
        4th-order tensor (in Kelvin mapping):
        [[2.77555756e-17 2.77555756e-17 2.77555756e-17 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [2.77555756e-17 2.77555756e-17 2.77555756e-17 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [2.77555756e-17 2.77555756e-17 2.77555756e-17 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
          0.00000000e+00 0.00000000e+00]]
        """
        A = np.zeros((6, 6))
        A[:3, :3] = 1 / 3
        return cls._broadcast_matrix(A, shape=shape, **kwargs)

    @classmethod
    def identity_deviatoric_part(cls, **kwargs):
        """
        Return the deviatoric part of the identity tensor.

        See notes for the mathematical definition.

        Parameters
        ----------
        kwargs
            keyword arguments passed to eye constructor

        Returns
        -------
        FourthOrderTensor or SymmetricTensor

        See Also
        --------
        identity_tensor : return the identity tensor
        identity_spherical_part : return the spherical part of the identity tensor

        Notes
        -----
        The deviatoric part of the identity tensor is defined as:

        .. math::

            K = I - J

        where :math:`I` and :math:`J` denote the identity and the deviatoric part of the identity tensor, respectively.

        Examples
        --------
        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> K = FourthOrderTensor.identity_deviatoric_part()
        >>> print(K)
        4th-order tensor (in Kelvin mapping):
        [[ 0.66666667 -0.33333333 -0.33333333  0.          0.          0.        ]
         [-0.33333333  0.66666667 -0.33333333  0.          0.          0.        ]
         [-0.33333333 -0.33333333  0.66666667  0.          0.          0.        ]
         [ 0.          0.          0.          1.          0.          0.        ]
         [ 0.          0.          0.          0.          1.          0.        ]
         [ 0.          0.          0.          0.          0.          1.        ]]

        One can check that K has zero spherical part:

        >>> print(K.spherical_part())
        4th-order tensor (in Kelvin mapping):
        [[2.77555756e-17 2.77555756e-17 2.77555756e-17 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [2.77555756e-17 2.77555756e-17 2.77555756e-17 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [2.77555756e-17 2.77555756e-17 2.77555756e-17 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
          0.00000000e+00 0.00000000e+00]
         [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
          0.00000000e+00 0.00000000e+00]]
        """
        I = FourthOrderTensor.identity(**kwargs)
        J = FourthOrderTensor.identity_spherical_part(**kwargs)
        return I-J

    def spherical_part(self):
        """
        Return the spherical part of the tensor

        Returns
        -------
        FourthOrderTensor
            Spherical part of the tensor

        See Also
        --------
        identity_tensor : return the identity tensor
        deviatoric_part : return the deviatoric part of the tensor
        """
        I = self.identity_spherical_part(shape=self.shape)
        return I.ddot(self)

    def deviatoric_part(self):
        """
        Return the deviatoric part of the tensor

        Returns
        -------
        FourthOrderTensor
            Deviatoric part of the tensor

        See Also
        --------
        identity_tensor : return the identity tensor
        spherical_part : return the spherical part of the tensor
        """
        K = self.identity_deviatoric_part(shape=self.shape)
        return K.ddot(self)

    def inv(self):
        """
        Invert the tensor. The inverted tensors inherits the properties (if any)

        Returns
        -------
        FourthOrderTensor
            Inverse tensor

        Examples
        --------
        Let consider a random Fourth-order tensor:

        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> T = FourthOrderTensor.rand()
        >>> print(T) # doctest: +SKIP

        >>> Tinv = T.inv()
        >>> print(Tinv) # doctest: +SKIP

        One can check that ``T.ddot(Tinv)`` and ``Tinv.ddot(T)`` are really close to the identity tensor:

        >>> I = FourthOrderTensor.eye()
        >>> (T.ddot(Tinv) - I) * 1e16 # doctest: +SKIP
        4th-order tensor (in Kelvin mapping):
        [[ 1.00000000e+00  0.00000000e+00  1.11022302e-16  0.00000000e+00
           3.14018492e-16  0.00000000e+00]
         [-2.22044605e-16  1.00000000e+00  2.77555756e-17 -1.25607397e-15
           2.35513869e-16 -7.85046229e-17]
         [ 1.55431223e-15  0.00000000e+00  1.00000000e+00 -3.14018492e-16
           0.00000000e+00 -4.71027738e-16]
         [-6.28036983e-16 -4.39625888e-15  0.00000000e+00  1.00000000e+00
           0.00000000e+00  1.11022302e-16]
         [-1.25607397e-15 -4.39625888e-15  0.00000000e+00  2.55351296e-15
           1.00000000e+00 -1.66533454e-16]
         [-5.88784672e-16 -1.17756934e-15 -2.62499833e-16  5.96744876e-16
           1.24900090e-16  1.00000000e+00]]

        >>> (Tinv.ddot(T) - I) * 1e16 # doctest: +SKIP
        4th-order tensor (in Kelvin mapping):
        [[ 1.00000000e+00 -1.33226763e-15 -3.99680289e-15 -6.90840682e-15
          -7.53644380e-15 -2.51214793e-15]
         [ 2.33146835e-15  1.00000000e+00  1.55431223e-15 -7.85046229e-16
           1.41308321e-15  9.42055475e-16]
         [ 3.88578059e-16  1.11022302e-16  1.00000000e+00 -3.92523115e-16
          -1.57009246e-16  1.57009246e-16]
         [-5.88784672e-17 -1.86448479e-16 -1.47196168e-16  1.00000000e+00
          -2.49800181e-16 -2.08166817e-16]
         [ 5.10280049e-16  7.85046229e-17 -7.85046229e-17  1.27675648e-15
           1.00000000e+00  7.77156117e-16]
         [-7.85046229e-16 -6.28036983e-16  6.28036983e-16  2.44249065e-15
           3.55271368e-15  1.00000000e+00]]

        This function obvisouly also works for tensor arrays. E.g.:

        >>> T = FourthOrderTensor.rand(shape=(5,3))
        >>> Tinv = T.inv()
        >>> Tinv.shape
        (5, 3)

        Again, one can check that ``T.ddot(Tinv)`` is close to the array of identity tensors:

        >>> import numpy as np
        >>> I = FourthOrderTensor.eye(shape=(5,3))
        >>> np.max(T.ddot(Tinv).matrix() - I.matrix())  # doctest: +SKIP
        5.906386491005833e-14
        """
        matrix_inv = np.linalg.inv(self._matrix)
        t = self.__class__(matrix_inv, mapping=kelvin_mapping)
        t.mapping = self.mapping.mapping_inverse
        return t

    @classmethod
    def zeros(cls, shape=(), **kwargs):
        """
        Create a fourth-order tensor populated with zeros

        Parameters
        ----------
        shape : int or tuple, optional
            Shape of the tensor to create
        kwargs
            Keyword arguments passed to the FourthOrderTensor constructor

        Returns
        -------
        FourthOrderTensor

        Examples
        --------
        The single-valued null 4th order tensor is just:

        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> FourthOrderTensor.zeros()
        4th-order tensor (in Kelvin mapping):
        [[0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0.]]

        One can also create an array of such tensors:

        >>> zeros_tensor = FourthOrderTensor.zeros(shape=3)

        and check that it populated with zeros:

        >>> zeros_tensor == 0.
        array([ True,  True,  True])
        """
        if isinstance(shape, int):
            shape = (shape, 6, 6)
        else:
            shape = shape + (6,6)
        zeros = np.zeros(shape)
        return cls(zeros, **kwargs)

    def matrix(self, mapping_convention=None):
        """
        Returns the components of the tensor as a matrix.

        Parameters
        ----------
        mapping_convention : VoigtMapping, optional
            Mapping convention to use for the returned matrix. If not provided, that of the tensor is used.

        Returns
        -------
        numpy.ndarray
            Components of the tensor as a matrix

        Examples
        --------
        Create an identity 4th-order tensor:

        >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
        >>> t = FourthOrderTensor.eye()

        Its matrix with respect to Kelvin mapping is:

        >>> t.matrix()
        array([[1., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0., 1.]])

        whereas, when using the Voigt mapping, we have:

        >>> from Elasticipy.tensors.mapping import VoigtMapping
        >>> t.matrix(mapping_convention=VoigtMapping())
        array([[1. , 0. , 0. , 0. , 0. , 0. ],
               [0. , 1. , 0. , 0. , 0. , 0. ],
               [0. , 0. , 1. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0.5, 0. , 0. ],
               [0. , 0. , 0. , 0. , 0.5, 0. ],
               [0. , 0. , 0. , 0. , 0. , 0.5]])

        For stiffness tensors, the default mapping convention is Voigt, so that:

        >>> from Elasticipy.tensors.elasticity import StiffnessTensor, ComplianceTensor
        >>> StiffnessTensor.eye().matrix()
        array([[1. , 0. , 0. , 0. , 0. , 0. ],
               [0. , 1. , 0. , 0. , 0. , 0. ],
               [0. , 0. , 1. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0.5, 0. , 0. ],
               [0. , 0. , 0. , 0. , 0.5, 0. ],
               [0. , 0. , 0. , 0. , 0. , 0.5]])

        whereas for compliance tensor, the default mapping convention gives:

        >>> ComplianceTensor.eye().matrix()
        array([[1., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 2., 0., 0.],
               [0., 0., 0., 0., 2., 0.],
               [0., 0., 0., 0., 0., 2.]])
        """
        matrix = self._matrix
        if mapping_convention is None:
            mapping_convention = self.mapping
        elif isinstance(mapping_convention, str):
            if mapping_convention.lower() == 'voigt':
                mapping_convention = VoigtMapping()
            elif mapping_convention.lower() == 'kelvin':
                mapping_convention = kelvin_mapping
            else:
                raise ValueError('Mapping convention must be either Kelvin or Voigt')
        return matrix / kelvin_mapping.matrix * mapping_convention.matrix

    def copy(self):
        """Create a copy of the tensor"""
        a = deepcopy(self)
        return a

class SymmetricFourthOrderTensor(FourthOrderTensor):
    _tensor_name = 'Symmetric 4th-order'

    def __init__(self, M, check_symmetries=True, force_symmetries=False, **kwargs):
        """
        Construct a fully symmetric fourth-order tensor from a (...,6,6) or a (...,3,3,3,3) array.

        The input matrix must be symmetric, otherwise an error is thrown (except if ``check_symmetry==False``, see
        below)

        Parameters
        ----------
        M : np.ndarray or FourthOrderTensor
            (6,6) matrix corresponding to the stiffness tensor, or slices of (6,6) matrices or array of shape
            (...,3,3,3,3).
        check_symmetries : bool, optional
            Whether to check or not that the tensor to built displays both major and minor symmetries (see Notes).
        force_symmetries : bool, optional
            If true, ensure that the tensor displays both minor and major symmetries.

        Notes
        -----
        The major symmetry is defined so that:

        .. math::

            M_{ijkl}=M_{klij}

        whereas the minor symmetry is:

        .. math::

            M_{ijkl}=M_{jikl}=M_{jilk}=M_{ijlk}
        """
        super().__init__(M, check_minor_symmetry=check_symmetries, force_minor_symmetry=force_symmetries, **kwargs)
        if force_symmetries:
            self._matrix = 0.5 * (self._matrix + self._matrix.swapaxes(-1, -2))
        elif check_symmetries and not np.all(np.isclose(self._matrix, self._matrix.swapaxes(-1, -2))):
            raise ValueError('The input matrix must be symmetric')

    def linear_invariants(self):
        """
        Compute the linear invariants of the tensor, or tensor array.

        If the object is a tensor array, the linear invariants are returned as arrays of each invariant. See notes for
        the actual definitions.

        Returns
        -------
        A1 : float or np.ndarray
            First linear invariant
        A2 : float or np.ndarray
            Second linear invariant

        See Also
        --------
        quadratic_invariants : compute the quadratic invariants of a fourth-order tensor

        Notes
        -----
        The linear invariants are:

        .. math::

            A_1=C_{ijij}

            A_2=C_{iijj}

        """
        t = self.full_tensor
        A1 = np.einsum('...ijij->...',t)
        A2 = np.einsum('...iijj->...',t)
        return A1, A2

    def quadratic_invariants(self):
        """
        Compute the quadratic invariants of the tensor, or tensor array.

        If the object is a tensor array, the returned values are arrays of each invariant. See notes for definitions.

        Returns
        -------
        B1, B2, B3, B4, B5 : float or np.ndarray

        See Also
        --------
        linear_invariants : compute the linear invariants of a Fourth-order tensor

        Notes
        -----
        The quadratic invariants are defined as [Norris]_:

        .. math::

            B_1 = C_{ijkl}C_{ijkl}

            B_2 = C_{iikl}C_{jjkl}

            B_3 = C_{iikl}C_{jkjl}

            B_4 = C_{kiil}C_{kjjl}

            B_5 = C_{ijkl}C_{ikjl}

        References
        ----------
        .. [Norris] Norris, A. N. (22 May 2007). "Quadratic invariants of elastic moduli". The Quarterly Journal of Mechanics
         and Applied Mathematics. 60 (3): 367–389. doi:10.1093/qjmam/hbm007

        """
        t = self.full_tensor
        B1 = np.einsum('...ijkl,...ijkl->...', t, t)
        B2 = np.einsum('...iikl,...jjkl->...', t, t)
        B3 = np.einsum('...iikl,...jkjl->...', t, t)
        B4 = np.einsum('...kiil,...kjjl->...', t, t)
        B5 = np.einsum('...ijkl,...ikjl->...', t, t)
        return B1, B2, B3, B4, B5

    def infinite_random_average(self):
        """
        Compute the average of the tensor, assuming that an infinite number of random orientations is applied.

        Returns
        -------
        FourthOrderTensor
            Average tensor or tensor array. The returned array will be of the same shape as the input object.
        """
        new_tensor = deepcopy(self)
        matrix = self._matrix / kelvin_mapping.matrix
        A = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
        B = matrix[..., 0, 1] + matrix[..., 0, 2] + matrix[..., 1, 2]
        C = matrix[..., 3, 3] + matrix[..., 4, 4] + matrix[..., 5, 5]
        C11 = 1/5  * A + 2/15 * B + 4/15 * C
        C12 = 1/15 * A + 4/15 * B - 2/15 * C
        C44 = 1/15 * A - 1/15 * B + 1/5 * C
        new_matrix = _isotropic_matrix(C11, C12, C44)
        new_tensor._matrix = new_matrix * kelvin_mapping.matrix
        return new_tensor

    @classmethod
    def rand(cls, shape=None, **kwargs):
        t1 = super().rand(shape)
        return cls(t1, force_symmetries=True, **kwargs)

