import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
ALPHABET = 'abcdefghijklmnopqrstuv'

class _MatrixProxy:
    def __init__(self, matrix):
        self.matrix = matrix

    def __getitem__(self, args):
        sub = self.matrix[(...,) + (args if isinstance(args, tuple) else (args,))]
        if sub.shape == ():
            return float(sub)
        else:
            return sub

    def __setitem__(self, args, value):
        self.matrix[(...,) + (args if isinstance(args, tuple) else (args,))] = value

def _tensor_from_direction_magnitude(u, v, magnitude):
    if np.asarray(u).shape != (3,):
        raise ValueError('u must be 3D vector.')
    if np.asarray(v).shape != (3,):
        raise ValueError('v must be 3D vector.')
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    direction_matrix = np.outer(u, v)
    if np.asarray(magnitude).ndim:
        return np.einsum('ij,...p->...pij', direction_matrix, magnitude)
    else:
        return magnitude * direction_matrix

def _transpose_matrix(matrix):
    return np.swapaxes(matrix, -1, -2)

def _symmetric_part(matrix):
    return 0.5 * (matrix + _transpose_matrix(matrix))

def _orientation_shape(g):
    if is_orix_rotation(g):
        return g.shape
    else:
        return (len(g),)

def _is_single_rotation(rotation):
    if isinstance(rotation, Rotation):
        return rotation.single
    elif is_orix_rotation(rotation):
        return rotation.size == 1
    else:
        raise TypeError('The input argument must be of class scipy.transform.Rotation or '
                        'orix.quaternion.rotation.Rotation')

_voigt_numbering = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]

def _unmap(array, mapping_convention):
    array = np.asarray(array)
    shape = array.shape
    if shape and (shape[-1] == 6):
        new_shape = shape[:-1] + (3, 3)
        unmapped_matrix = np.zeros(new_shape)
        for i in range(6):
            unmapped_matrix[..., _voigt_numbering[i][0], _voigt_numbering[i][1]] = array[..., i] / mapping_convention[i]
        return unmapped_matrix
    else:
        raise ValueError("array must be of shape (6,) or (...,6) with Voigt vector")

def _map(matrix, mapping_convention):
    shape = matrix.shape[:-2] + (6,)
    array = np.zeros(shape)
    for i in range(6):
        j, k = _voigt_numbering[i]
        array[...,i] = matrix[...,j,k]
    return array * mapping_convention

def filldraw_circle(ax, center, radius, color, fill=False, alpha=1.):
    theta = np.linspace(0, 2 * np.pi, 500)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    if fill:
        ax.fill(x, y, color=color, alpha=alpha)
    else:
        ax.plot(x, y, color=color)

kelvin_mapping = [1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)]

class SecondOrderTensor:
    """
    Template class for manipulation of second order tensors or arrays of second order tensors

    Attributes
    ----------
    matrix : np.ndarray
        (...,3,3) matrix storing all the components of the tensor

    """
    name = 'Second-order tensor'
    "Name to use when printing the tensor"

    def __init__(self, matrix):
        """
        Create an array of second-order tensors.

        The input argument can be:
            - an array of shape (3,3) defining all the components of the tensor;
            - a stack of matrices, that is an array of shape (...,3,3).

        Parameters
        ----------
        matrix : list or np.ndarray
            (3,3) matrix, stack of (3,3) matrices
        """
        matrix = np.array(matrix)
        shape = matrix.shape
        if len(shape) > 1 and shape[-2:] == (3, 3):
            self.matrix = matrix
        else:
            raise ValueError('The input matrix must be of shape (3,3) or (...,3,3)')

    def __repr__(self):
        s = self.name + '\n'
        if self.shape:
            s += 'Shape={}'.format(self.shape)
        else:
            s += self.matrix.__str__()
        return s

    def __getitem__(self, index):
        return self.__class__(self.matrix[index])

    def __setitem__(self, index, value):
        if isinstance(value, (float, np.ndarray)):
            self.matrix[index] = value
        elif type(value) == self.__class__:
            self.matrix[index] = value.matrix
        else:
            raise NotImplementedError('The r.h.s must be either float, a ndarray or an object of class {}'.format(self.__class__))

    def __add__(self, other):
        if type(self) == type(other):
            return self.__class__(self.matrix + other.matrix)
        elif isinstance(other, (int, float, np.ndarray)):
            mat = self.matrix + other
            if isinstance(self, SkewSymmetricSecondOrderTensor):
                return SecondOrderTensor(mat)
            else:
                return self.__class__(mat)
        elif isinstance(other, SecondOrderTensor):
            return SecondOrderTensor(self.matrix + other.matrix)
        else:
            raise NotImplementedError('The element to add must be a number, a numpy.ndarray or a tensor.')

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if type(self) == type(other):
            return self.__class__(self.matrix - other.matrix)
        elif isinstance(other, (int, float, np.ndarray)):
            return self.__class__(self.matrix - other)
        else:
            raise NotImplementedError('The element to subtract must be a number, a numpy ndarray or a tensor.')

    def __neg__(self):
        return self.__class__(-self.matrix)

    def __rsub__(self, other):
        return -self + other

    @property
    def shape(self):
        """
        Return the shape of the tensor array

        Returns
        -------
        tuple
            Shape of array

        See Also
        --------
        ndim : number of dimensions
        """
        *shape, _, _ = self.matrix.shape
        return tuple(shape)

    @property
    def ndim(self):
        """
        Return the number of dimensions of the tensor array

        Returns
        -------
        int
            number of dimensions

        See Also
        --------
        shape : shape of tensor array
        """
        return len(self.shape)

    @property
    def C(self):
        """
        Return tensor components

        For instance T.C[i,j] returns all the (i,j)-th components of each tensor in the array.

        Returns
        -------
        np.ndarray
            Tensor components
        """
        return _MatrixProxy(self.matrix)

    def eig(self):
        """
        Compute the eigenvalues and eigenvectors of the tensor

        Returns
        -------
        lambda : np.ndarray
            Eigenvalues of each tensor.
        v : np.ndarray
            Eigenvectors of teach tensor.

        See Also
        --------
        eigvals : return only the eigenvalues (without directions)
        principal_directions : return only the principal directions (without eigenvalues)
        """
        return np.linalg.eig(self.matrix)

    def eigvals(self):
        """
        Compute the eigenvalues of the tensor, without computing the associated eigenvectors

        Returns
        -------
        numpy.ndarray
            Eigenvalues

        See Also
        --------
        eig : compute the eigenvalues and the eigenvector
        """
        return np.linalg.eigvals(self.matrix)

    def principal_directions(self):
        """
        Principal directions of the tensors

        Returns
        -------
        np.ndarray
            Principal directions of each tensor of the tensor array

        See Also
        --------
        eig : Return both eigenvalues and corresponding principal directions
        """
        return self.eig()[1]

    @property
    def I1(self):
        """
        First invariant of the tensor (trace)

        Returns
        -------
        np.ndarray or float
            First invariant(s) of the tensor(s)

        See Also
        --------
        I2 : Second invariant of the tensors
        I3 : Third invariant of the tensors (det)
        """
        return self.matrix.trace(axis1=-1, axis2=-2)

    @property
    def I2(self):
        """
        Second invariant of the tensor

        For a matrix M, it is defined as::

            I_2 = 0.5 * ( np.trace(M)**2 + np.trace(np.matmul(M, M.T)) )

        Returns
        -------
        np.array or float
            Second invariant(s) of the tensor(s)

        See Also
        --------
        I1 : First invariant of the tensors (trace)
        I3 : Third invariant of the tensors (det)
        """
        a = self.I1**2
        b = np.matmul(self.matrix, self._transpose_tensor()).trace(axis1=-1, axis2=-2)
        return 0.5 * (a - b)

    @property
    def I3(self):
        """
        Third invariant of the tensor (determinant)

        Returns
        -------
        np.array or float
            Third invariant(s) of the tensor(s)

        See Also
        --------
        I1 : First invariant of the tensors (trace)
        I2 : Second invariant of the tensors
        """
        return np.linalg.det(self.matrix)

    @property
    def J1(self):
        """
        First invariant of the deviatoric part of the stress tensor. It is always zeros, as the deviatoric part as null
        trace.

        Returns
        -------
        float or np.ndarray
            zero(s)
        """
        if self.shape:
            return np.zeros(self.shape)
        else:
            return 0.0

    @property
    def J2(self):
        """
        Second invariant of the deviatoric part of the tensor.

        Returns
        -------
        float or np.ndarray
            J2 invariant
        """
        return -self.deviatoric_part().I2

    @property
    def J3(self):
        """
        Third invariant of the deviatoric part of the tensor.

        Returns
        -------
        float or np.ndarray
            J3 invariant
        """
        return self.deviatoric_part().I3

    def Lode_angle(self, degrees=False):
        """
        Computes the Lode angle of the tensor.

        The returned value is defined from the positive cosine (see Notes).

        Parameters
        ----------
        degrees : bool, optional
            Whether to return the angle in degrees or not

        Returns
        -------
        float or numpy.ndarray

        See Also
        --------
        J2 : Second invariant of the deviatoric part
        J3 : Third invariant of the deviatoric part

        Notes
        -----
        The Lode angle is defined such that:

        .. math::

            \\cos(3\\theta)= \\frac{J_3}{2}\\left(\\frac{3}{J_2}\\right)^{3/2}
        """
        J2 = np.atleast_1d(self.J2)
        J3 = np.atleast_1d(self.J3)
        non_hydro =  J2 !=0.
        cosine = np.ones(shape=J3.shape) * np.nan
        cosine[non_hydro] = J3[non_hydro] / 2 * (3 / J2[non_hydro] )**(3 / 2)
        if degrees:
            theta = np.arccos(cosine) * 60 / np.pi
        else:
            theta = np.arccos(cosine) / 3
        if self.shape:
            return theta
        else:
            return theta[0]

    def trace(self):
        """
        Return the traces of the tensor array

        Returns
        -------
        np.ndarray or float
            traces of each tensor of the tensor array

        See Also
        --------
        I1 : First invariant of the tensors (trace)
        I2 : Second invariant of the tensors
        I3 : Third invariant of the tensors (det)
        """
        return self.I1

    def __mul__(self, B):
        """
        Element-wise matrix multiplication of arrays of tensors. Each tensor of the resulting tensor array is computed
        as the matrix product of the tensor components.

        Parameters
        ----------
        B : SecondOrderTensor or np.ndarray or Rotation or float
            If B is a numpy array, we must have::

                B.shape == (..., 3, 3)

        Returns
        -------
            Array of tensors populated with element-wise matrix multiplication.

        See Also
        --------
        matmul : matrix-like multiplication of tensor arrays
        """
        if isinstance(B, SecondOrderTensor):
            return self.dot(B, mode='pair')
        elif isinstance(B, Rotation) or is_orix_rotation(B):
            return self.rotate(B, mode='pair')
        elif isinstance(B, (float, int)):
            return self.__class__(self.matrix * B)
        elif isinstance(B, np.ndarray):
            if B.shape == self.shape:
                new_matrix = np.einsum('...ij,...->...ij', self.matrix, B)
                return self.__class__(new_matrix)
            elif B.shape == self.matrix.shape:
                return self.__class__(np.matmul(self.matrix, B))
            else:
                err_msg = 'For a tensor of shape {}, the input argument must be an array of shape {} or {}'.format(
                    self.shape, self.shape, self.shape + (3, 3))
                raise ValueError(err_msg)
        else:
            raise ValueError('The input argument must be a tensor, an ndarray, a rotation or a scalar value.')

    def rotate(self, rotation, mode='pair'):
        """
        Apply rotation(s) to the tensor(s).

        The rotations can be applied element-wise, or on each cross-combination (see below).

        Parameters
        ----------
        rotation : scipy.spatial.Rotation or orix.quaternion.Rotation
        mode : str, optional
            If 'pair', the rotations are applied element wise. Broadcasting rule applies.
            If 'cross', all the possible combinations are considered. If ``C=A.rotate(rot)``, then
            ``C.shape==A.shape + rot.shape``.

        Returns
        -------
        SecondOrderTensor
        """
        if self.shape == ():
            ein_str = '...li,...kj,lk->...ij'
        elif _is_single_rotation(rotation):
            ein_str = 'li,kj,...lk->...ij'
        else:
            if mode=='pair':
                ein_str = '...li,...kj,...lk->...ij'
            elif mode=='cross':
                ndim_0 = self.ndim
                ndim_1 = len(_orientation_shape(rotation))
                indices_self = ALPHABET[:ndim_0]
                indices_g = ALPHABET[:ndim_1].upper()
                indices_res = indices_self + indices_g
                ein_str = indices_g + 'zw,' + indices_g + 'yx,' + indices_self + 'zy->' + indices_res + 'wx'
            else:
                raise ValueError('Invalid mode. It can be "cross" or "pair".')
        g_mat = rotation_to_matrix(rotation)
        matrix = np.einsum(ein_str, g_mat, g_mat, self.matrix)
        return self.__class__(matrix)

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return self.__mul__(other)
        else:
            raise NotImplementedError('Left multiplication is only implemented for scalar values.')

    def __truediv__(self, other):
        new_mat = np.zeros(self.matrix.shape)
        non_zero = np.any(self.matrix, axis=(-1, -2))
        if isinstance(other, (float, int)):
            new_mat[non_zero] = self.matrix[non_zero] / other # Hack to force 0/0 = 0
        elif isinstance(other, np.ndarray) and (self.shape == other.shape):
            new_mat[non_zero] = np.einsum('pij,p->pij', self.matrix[non_zero], 1/other[non_zero])
            return self.__class__(new_mat)
        else:
            raise NotImplementedError('Tensors can only be divided by scalar values or by arrays of the same shape.')
        return self.__class__(new_mat)

    def __eq__(self, other):
        """
        Check whether the tensors in the tensor array are equal

        Parameters
        ----------
        other : SecondOrderTensor or np.ndarray
            Tensor to compare with

        Returns
        -------
        numpy.ndarray
            True element is True if the corresponding tensors are equal.
        """
        if isinstance(other, SecondOrderTensor):
            return self == other.matrix
        elif isinstance(other, np.ndarray):
            if (other.shape == (3,3)) or (other.shape == self.shape + (3,3)):
                return np.all(self.matrix == other, axis=(-2, -1))
            else:
                raise ValueError('The value to compare must be an array of shape {} or {}'.format(self.shape, self.shape + (3,3)))

    def dot(self, other, mode='pair'):
        """
        Perform contraction product ("dot product") between tensor.

        On tensor arrays, the product contraction can be performed element-wise, or considering all cross-combinations
        (see below).

        Parameters
        ----------
        other : SecondOrderTensor
            tensor or tensor array to compute the product from
        mode : str, optional
            If 'pair' (default), the contraction products of tensor arrays are applied element-wise. Broadcasting rule
            applies. If 'cross', all combinations of contraction product are considered. If ``C=A.dot(B,mode='cross')``,
            then ``C.shape==A.shape + B.shape``.

        Returns
        -------
        SecondOrderTensor

        Examples
        --------
        >>> from Elasticipy.tensors.second_order import SecondOrderTensor
        >>> A=SecondOrderTensor.rand(10)
        >>> B=SecondOrderTensor.rand(10)
        >>> AB_pair = A.dot(B)
        >>> AB_pair.shape
        (10,)

        >>> AB_cross = A.dot(B, mode='cross')
        >>> AB_cross.shape
        (10, 10)

        We can for instance check that:

        >>> AB_pair[5] == A[5].dot(B[5])
        True

        and:

        >>> AB_cross[0,1] == A[0].dot(B[1])
        True

        See Also
        --------
        ddot : Double-contraction product
        """
        if self.shape == ():
            ein_str = 'ik,...kj->...ij'
        else:
            if mode=='pair':
                ein_str = '...ik,...kj->...ij'
            elif mode=='cross':
                ndim_0 = self.ndim
                ndim_1 = other.ndim
                indices_0 = ALPHABET[:ndim_0]
                indices_1 = ALPHABET[:ndim_1].upper()
                indices_2 = indices_0 + indices_1
                ein_str = indices_0 + 'ik,' + indices_1 + 'kj->' + indices_2 + 'ij'
            else:
                raise ValueError('Invalid mode. Use "pair" or "cross".')
        matrix = np.einsum(ein_str, self.matrix, other.matrix)
        return SecondOrderTensor(matrix)


    def matmul(self, other):
        """
        Perform matrix-like product between tensor arrays. Each "product" is a matrix product between
        the tensor components.

        If A.shape=(a1, a2, ..., an) and B.shape=(b1, b2, ..., bn), with C=A.matmul(B), we have::

            C.shape = (a1, a2, ..., an, b1, b2, ..., bn)

        and::

            C[i,j,k,...,p,q,r...] = np.matmul(A[i,j,k,...], B[p,q,r,...])

        Parameters
        ----------
        other : SecondOrderTensor or np.ndarray or Rotation
            Tensor array or rotation to right-multiply by. If Rotation is provided, the rotations are applied on each
            tensor.

        Returns
        -------
        SecondOrderTensor
            Tensor array

        See Also
        --------
        __mul__ : Element-wise matrix product
        """
        warnings.warn(
            'matmul() is deprecated and will be removed in a future version. Use dot(tensor,mode="cross") or '
            'rotate(rotation,mode="cross") instead.',
            DeprecationWarning,
            stacklevel=2)
        if isinstance(other, SecondOrderTensor):
            return self.dot(other, mode='cross')
        elif isinstance(other, Rotation) or is_orix_rotation(Rotation):
            return self.rotate(other, mode='cross')
        else:
            raise ValueError('The input argument must be either a rotation or a SecondOrderTensor')

    def transpose_array(self):
        """
        Transpose the array of tensors

        If A is a tensor array of shape [s1, s2, ..., sn], A.T is of shape [sn, ..., s2, s1].

        Returns
        -------
        SecondOrderTensor
            Transposed array

        See Also
        --------
        T : transpose the tensor array (not the components)
        """
        if self.ndim < 2:
            return self
        else:
            matrix = self.matrix
            ndim = matrix.ndim
            new_axes = np.hstack((ndim - 3 - np.arange(ndim - 2), -2, -1))
            transposed_arr = np.transpose(matrix, new_axes)
            return self.__class__(transposed_arr)

    @property
    def T(self):
        """
        Transpose the array of tensors.

        It is actually an alias for transpose_array()

        Returns
        -------
        SecondOrderTensor
            Transposed array
        """
        return self.transpose_array()

    def _transpose_tensor(self):
        return _transpose_matrix(self.matrix)

    def transpose_tensor(self):
        """
        Transpose of tensors of the tensor array

        Returns
        -------
        SecondOrderTensor
            Array of transposed tensors of the tensor array

        See Also
        --------
        transpose_array : transpose the array (not the components)
        """
        return self.__class__(self._transpose_tensor())

    def ddot(self, other, mode='pair'):
        """
        Double dot product (contraction of tensor product, usually denoted ":") of two tensors.

        For two tensors whose matrices are M1 and M2::

            M1.ddot(M2) == np.trace(np.matmul(M1, M2))

        Parameters
        ----------
        other : SecondOrderTensor or np.ndarray
            Tensor or tensor array to multiply by before contraction.
        mode : str, optional
            If "pair", the dot products are performed element-wise before contraction. Broadcasting rule applies.
            If "cross", all the cross-combinations are computed, increasing the dimensionality.
            If ``C=A.ddot(B, mode='cross')``, then ``C.shape = A.shape + B.shape``.


        Returns
        -------
        float or np.ndarray
            Result of double dot product

        See Also
        --------
        dot : contraction product ("dot product") between tensor.

        """
        tensor_prod = self.transpose_tensor().dot(other, mode=mode)
        return tensor_prod.trace()

    def _flatten(self):
        if self.shape:
            new_len = np.prod(self.shape)
            return np.reshape(self.matrix, (new_len, 3, 3))
        else:
            return self.matrix

    def _stats(self, fun, axis=None):
        if axis is None:
            flat_mat = self._flatten()
            new_matrix = fun(flat_mat, axis=0)
        else:
            if axis < 0:
                axis += -2
            if (axis > self.ndim - 1) or (axis < -self.ndim - 2):
                raise ValueError('The axis index is out of bounds for tensor array of shape {}'.format(self.shape))
            new_matrix = fun(self.matrix, axis=axis)
        return self.__class__(new_matrix)

    def flatten(self):
        """
        Flatten the array of tensors.

        If T is of shape [s1, s2, ..., sn], the shape for T.flatten() is [s1*s2*...*sn].

        Returns
        -------
        SecondOrderTensor
            Flattened array (vector) of tensor

        See Also
        --------
        ndim : number of dimensions of the tensor array
        shape : shape of the tensor array
        reshape : reshape a tensor array
        """
        return self.__class__(self._flatten())

    def reshape(self, shape, **kwargs):
        """
        Reshape the array of tensors

        Parameters
        ----------
        shape : tuple
            New shape of the array
        kwargs : dict
            Keyword arguments passed to numpy.reshape()

        Returns
        -------
        SecondOrderTensor
            Reshaped array

        See Also
        --------
        flatten : flatten an array to 1D
        """
        new_matrix = self.matrix.reshape(shape + (3,3,), **kwargs)
        return self.__class__(new_matrix)

    def mean(self, axis=None):
        """
        Arithmetic mean value

        Parameters
        ----------
        axis : int or None, default None
            Axis to compute the mean along with.
            If None, returns the overall mean (mean of flattened array)

        Returns
        -------
        SecondOrderTensor
            Mean tensor

        See Also
        --------
        std : Standard deviation
        min : Minimum value
        max : Maximum value
        """
        if self.ndim:
            return self._stats(np.mean, axis=axis)
        else:
            return self

    def std(self, axis=None):
        """
        Standard deviation

        Parameters
        ----------
        axis : int or None, default None
            Axis to compute standard deviation along with.
            If None, returns the overall standard deviation (std of flattened array)

        Returns
        -------
        SecondOrderTensor
            Tensor of standard deviation

        See Also
        --------
        mean : Mean value
        min : Minimum value
        max : Maximum value
          """
        if self.ndim:
            return self._stats(np.std, axis=axis)
        else:
            return self.__class__(np.zeros((3, 3)))

    def min(self, axis=None):
        """
        Minimum value

        Parameters
        ----------
        axis : int or None, default None
           Axis to compute minimum along with.
           If None, returns the overall minimum (min of flattened array)

        Returns
        -------
        SecondOrderTensor
           Minimum value of tensors

        See Also
        --------
        max : Maximum value
        mean : Mean value
        std : Standard deviation
        """
        if self.ndim:
            return self._stats(np.min, axis=axis)
        else:
            return self

    def max(self, axis=None):
        """
        Maximum value

        Parameters
        ----------
        axis : int or None, default None
            Axis to compute maximum along with.
            If None, returns the overall maximum (max of flattened array)

        Returns
        -------
        SecondOrderTensor
            Maximum value of tensors

        See Also
        --------
        min : Minimum value
        mean : Mean value
        std : Standard deviation
        """
        if self.ndim:
            return self._stats(np.max, axis=axis)
        else:
            return self

    def _symmetric_part(self):
        return 0.5 * (self.matrix + self._transpose_tensor())

    def symmetric_part(self):
        """
        Symmetric part of the tensor

        Returns
        -------
        SymmetricSecondOrderTensor
            Symmetric tensor

        See Also
        --------
        skewPart : Skew-symmetric part of the tensor
        """
        return SymmetricSecondOrderTensor(self._symmetric_part())

    def skew_part(self):
        """
        Skew-symmetric part of the tensor

        Returns
        -------
        SkewSymmetricSecondOrderTensor
            Skew-symmetric tensor
        """
        new_mat = 0.5 * (self.matrix - self._transpose_tensor())
        return SkewSymmetricSecondOrderTensor(new_mat)

    def spherical_part(self):
        """
        Spherical (hydrostatic) part of the tensor

        Returns
        -------
        self
            Spherical part

        See Also
        --------
        I1 : compute the first invariant of the tensor
        deviatoricPart : deviatoric the part of the tensor
        """
        s = self.I1 / 3
        return self.eye(self.shape)*s

    def deviatoric_part(self):
        """
        Deviatoric part of the tensor

        Returns
        -------
        self

        See Also
        --------
        sphericalPart : spherical part of the tensor
        """
        return self - self.spherical_part()

    @classmethod
    def eye(cls, shape=()):
        """
        Create an array of tensors populated with identity matrices

        Parameters
        ----------
        shape : tuple or int, default ()
            If not provided, it just creates a single identity tensor. Otherwise, the tensor array will be of the
            specified shape.

        Returns
        -------
        cls
            Array of identity tensors

        See Also
        --------
        ones : creates an array of tensors full of ones
        zeros : creates an array full of zero tensors
        """
        if isinstance(shape, int):
            matrix_shape = (shape, 3, 3)
        else:
            matrix_shape = shape + (3, 3,)
        eye = np.zeros(matrix_shape)
        eye[..., np.arange(3), np.arange(3)] = 1
        return cls(eye)

    @classmethod
    def ones(cls, shape=()):
        """
        Create an array of tensors populated with matrices of full of ones.

        Parameters
        ----------
        shape : tuple or int, default ()
            If not provided, it just creates a single tensor of ones. Otherwise, the tensor array will be of the
            specified shape.

        Returns
        -------
        cls
            Array of ones tensors

        See Also
        --------
        eye : creates an array of identity tensors
        zeros : creates an array full of zero tensors
        """
        if isinstance(shape, int):
            matrix_shape = (shape, 3, 3)
        else:
            matrix_shape = shape + (3, 3,)
        ones = np.ones(matrix_shape)
        return cls(ones)

    @classmethod
    def zeros(cls, shape=()):
        """
        Create an array of tensors populated with matrices full of zeros.

        Parameters
        ----------
        shape : tuple or int, default ()
            If not provided, it just creates a single tensor of ones. Otherwise, the tensor array will be of the
            specified shape.

        Returns
        -------
        cls
            Array of ones tensors

        See Also
        --------
        eye : creates an array of identity tensors
        ones : creates an array of tensors full of ones
        """
        if isinstance(shape, int):
            matrix_shape = (shape, 3, 3)
        else:
            matrix_shape = shape + (3, 3,)
        zeros = np.zeros(matrix_shape)
        return cls(zeros)

    @classmethod
    def tensile(cls, u, magnitude):
        """
        Create an array of tensors corresponding to tensile state along a given direction.

        Parameters
        ----------
        u : np.ndarray or list
            Tensile direction. Must be a 3D vector.
        magnitude : float or np.ndarray or list
            Magnitude of the tensile state to consider. If a list or an array is provided, the shape of the tensor array
            will be of the same shape as magnitude.
        Returns
        -------
        cls
            tensor or tensor array
        """
        mat = _tensor_from_direction_magnitude(u, u, magnitude)
        return cls(mat)

    @classmethod
    def rand(cls, shape=None, seed=None):
        """
        Generate a tensor array, populated with random uniform values in [0,1).

        Parameters
        ----------
        shape : int or tuple, optional
            Shape of the tensor array. If not provided, a single tensor is returned
        seed : int, optional
            Sets the seed for random generation. Useful to ensure reproducibility

        Returns
        -------
        cls
            Tensor or tensor array of uniform random value

        See Also
        --------
        randn : Generate a random sample of tensors whose components follows a normal distribution

        Examples
        --------
        Generate a single random tensor:

        >>> from Elasticipy.tensors.second_order import SecondOrderTensor as tensor
        >>> tensor.rand(seed=123)
        Second-order tensor
        [[0.68235186 0.05382102 0.22035987]
         [0.18437181 0.1759059  0.81209451]
         [0.923345   0.2765744  0.81975456]]

        Now try with tensor array:
        >>> t = tensor.rand(shape=(100,50))
        >>> t.shape
        (100,50)
        """
        if shape is None:
            shape = (3,3)
        elif isinstance(shape, int):
            shape = (shape, 3, 3)
        else:
            shape = shape + (3,3)
        rng = np.random.default_rng(seed)
        a = rng.random(shape)
        if issubclass(cls, SymmetricSecondOrderTensor):
            a = _symmetric_part(a)
        return cls(a)

    def inv(self):
        """Compute the reciprocal (inverse) tensor"""
        return SecondOrderTensor(np.linalg.inv(self.matrix))

    @classmethod
    def randn(cls, mean=np.zeros((3,3)), std=np.ones((3,3)), shape=None, seed=None):
        """
        Generate a tensor array, populated with components follow a normal distribution.

        Parameters
        ----------
        mean : list of numpy.ndarray, optional
            (3,3) matrix providing the mean values of the components.
        std : list of numpy.ndarray, optional
            (3,3) matrix providing the standard deviations of the components.
        shape : tuple, optional
            Shape of the tensor array
        seed : int, optional
            Sets the seed for random generation. Useful to ensure reproducibility

        Returns
        -------
        cls
            Tensor or tensor array of normal random value
        """
        if shape is None:
            new_shape = (3,3)
        else:
            new_shape = shape + (3,3)
        rng = np.random.default_rng(seed)
        mat = np.zeros(new_shape)
        mean = np.asarray(mean)
        std = np.asarray(std)
        for i in range(0,3):
            for j in range(0,3):
                mat[...,i,j] = rng.normal(mean[i,j], std[i,j], shape)
        if issubclass(cls, SymmetricSecondOrderTensor):
            mat = _symmetric_part(mat)
        return cls(mat)

    @classmethod
    def shear(cls, u, v, magnitude):
        """
        Create an array of tensors corresponding to shear state along two orthogonal directions.

        Parameters
        ----------
        u : np.ndarray or list
            First direction. Must be a 3D vector.
        v : np.ndarray or list
            Second direction. Must be a 3D vector.
        magnitude : float or np.ndarray or list
            Magnitude of the shear state to consider. If a list or an array is provided, the shape of the tensor array
            will be of the same shape as magnitude.
        Returns
        -------
        cls
            tensor or tensor array
        """
        if np.abs(np.dot(u, v)) > 1e-5:
            raise ValueError("u and v must be orthogonal")
        mat = _tensor_from_direction_magnitude(u, v, magnitude)
        return cls(mat)

    def div(self, axes=None, spacing=1.):
        """
        Compute the divergence vector of the tensor array, along given axes.

        If the tensor has n dimensions, the divergence vector will be computed along its m first axes, with
        m = min(n, 3), except if specified in the ``axes`` parameter (see below).

        Parameters
        ----------
        axes : list of int, tuple of int, int or None, default None
            Indices of axes along which to compute the divergence vector. If None (default), the m first axes of the
            array will be used to compute the derivatives.
        spacing : float or np.ndarray or list, default 1.
            Spacing between samples the in each direction. If a scalar value is provided, the spacing is assumed equal
            in each direction. If an array or a list is provided, spacing[i] must return the spacing along the i-th
            axis (spacing[i] can be float or np.ndarray).

        Returns
        -------
        np.ndarray
            Divergence vector of the tensor array. If the tensor array is of shape (m,n,...,q), the divergence vector
            will be of shape (m,n,...,q,3).

        Notes
        -----
        The divergence of a tensor field :math:`\\mathbf{t}(\\mathbf{x})` is defined as:

        .. math::

            [\\nabla\\cdot\\mathbf{t}]_i = \\frac{\\partial t_{ij}}{\\partial x_j}

        The main application of this operator is for balance of linear momentum of stress tensor:

        .. math::

            \\rho \\mathbf{\\gamma} = \\nabla\\cdot\\mathbf{\\sigma} + \\rho\\mathbf{b}

        where :math:`\\mathbf{\\sigma}` is the stress tensor, :math:`\\mathbf{\\gamma}` is the acceleration,
        :math:`\\mathbf{b}` is the body force density and :math:`\\rho` is the mass density.

        In this function, the derivatives are computed with ``numpy.grad`` function.

        Examples
        --------
        First, we build an array of tensile stress with evenly spaced magnitude:

        >>> from Elasticipy.tensors.stress_strain import StressTensor
        >>> magnitude = [0,1,2,3,4]
        >>> s = StressTensor.tensile([1,0,0],magnitude)
        >>> s.div()
        array([[1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.],
               [1., 0., 0.]])

        We now create a stress tensor whose components follows this:

        .. math::

            \\sigma_{xx} = 2x+3y^2
            \\sigma_{yy} = y^2+z
            \\sigma_{xy} = xy


        To do this, we consider a regular grid of 0.1 in a unit cube:

        >>> import numpy as np
        >>> spacing = 0.1
        >>> x,y,z=np.meshgrid(np.arange(0,1,spacing),np.arange(0,1,spacing),np.arange(0,1,spacing), indexing='ij')
        >>> s_xx = 2*x + 3*y**2
        >>> s_yy = y**2+z
        >>> s_xy = x*y
        >>> s = StressTensor.tensile([1,0,0],s_xx) + StressTensor.tensile([0,1,0],s_yy) + StressTensor.shear([1,0,0],[0,1,0],s_xy)
        >>> div = s.div(spacing = 0.1)

        As we work here in 3D, the result is of shape (10,10,10,3):

        >>> div.shape
        (10, 10, 10, 3)

        For instance, the divergence at x=y=z=0 is:

        >>> div[0,0,0,:]
        array([2. , 0.1, 0. ])
        """
        ndim = min(self.ndim, 3)    # Even if the array has more than 3Ds, we restrict to 3D
        if isinstance(spacing, (float, int)):
            spacing = [spacing, spacing, spacing]
        if axes is None:
            axes = range(ndim)
        elif isinstance(axes, int):
            axes = (axes,)
        elif not isinstance(axes, (tuple, list)):
            raise TypeError("axes must be int, tuple of int, or list of int.")
        if len(axes) > ndim:
            error_msg = ("The number of axes must be less or equal to the number of dimensions ({}), "
                         "and cannot exceed 3").format(self.ndim)
            raise ValueError(error_msg)
        else:
            ndim = len(axes)
        if max(axes) >= ndim:
            raise IndexError("axes index must be in range of dimensions ({})".format(self.ndim))
        div = np.zeros(self.shape + (3,))
        for dim in range(0, ndim):
            div += np.gradient(self.C[:,dim], spacing[dim], axis=axes[dim])
        return div

    def save(self, file, **kwargs):
        """
        Save the tensor array as binary file (.npy format).

        This function uses numpy.save function.

        Parameters
        ----------
        file : file, str or pathlib.Path
            File or filename to which the tensor is saved.
        kwargs : dict
            Keyword arguments passed to numpy.save()

        See Also
        --------
        load_from_npy : load a tensor array from a numpy file
        """
        np.save(file, self.matrix, **kwargs)

    @classmethod
    def load_from_npy(cls, file, **kwargs):
        """
        Load a tensor array for .npy file.

        This function uses numpy.load()

        Parameters
        ----------
        file : file, str or pathlib.Path
            File to read to create the array
        kwargs : dict
            Keyword arguments passed to numpy.load()

        Returns
        -------
        SecondOrderTensor
            Tensor array

        See Also
        --------
        save : save the tensor array as a numpy file
        """
        matrix = np.load(file, **kwargs)
        if matrix.shape[-2:] != (3,3):
            raise ValueError('The shape of the array to load must be (...,3,3).')
        else:
            return cls(matrix)

    def save_as_txt(self, file, name_prefix='', **kwargs):
        """
        Save the tensor array to human-readable text file.

        The array must be 1D. The i-th row of the file will provide the components of the i-th tensor in of the array.
        This function uses pandas.DataFrame.to_csv().

        Parameters
        ----------
        file : file or str
            File to dump tensor components to.
        name_prefix : str, optional
            Prefix to add for naming the columns. For instance, name_prefix='E' will result in columns named E11, E12,
            E13 etc.
        kwargs : dict
            Keyword arguments passed to pandas.DataFrame.to_csv()
        """
        if self.ndim > 1:
            raise ValueError('The array must be flatten before getting dumped to text file.')
        else:
            d = dict()
            for i in range(3):
                if isinstance(self, SkewSymmetricSecondOrderTensor):
                    r = range(i+1, 3)   # If the tensor is skew-symmetric, there is no need to save the full matrix
                elif isinstance(self, SymmetricSecondOrderTensor):
                    r = range(i, 3)     # Idem, except that we also need the diagonal
                else:
                    r =range(3)
                for j in r:
                    key = name_prefix + '{}{}'.format(i+1, j+1)
                    d[key] = self.C[i,j]
            df = pd.DataFrame(d)
            df.to_csv(file, index=False, **kwargs)

    @classmethod
    def load_from_txt(cls, file, name_prefix='', **kwargs):
        """
        Load a tensor array from text file.

        Parameters
        ----------
        file : str or file
            Textfile to read the components from.
        name_prefix : str, optional
            Prefix to add to each column when parsing the file. For instance, with name_prefix='E', the function will
            look for columns names E11, E12, E13 etc.

        Returns
        -------
        SecondOrderTensor
            Flat (1D) tensor constructed from the values given in the text file
        """
        df = pd.read_csv(file, **kwargs)
        matrix = np.zeros((len(df), 3, 3))
        for i in range(3):
            if cls is SkewSymmetricSecondOrderTensor:
                r = range(i+1, 3)
            elif cls is SymmetricSecondOrderTensor:
                r = range(i, 3)
            else:
                r= range(3)
            for j in r:
                key = name_prefix + '{}{}'.format(i + 1, j + 1)
                matrix[:, i, j] = df[key]
        return cls(matrix)

    def to_pymatgen(self):
        """
        Convert the second order object into an object compatible with pymatgen.

        The object to use must be either a single tensor, or a flat tensor array. In the latter case, the output will be
        a list of pymatgen's tensors.

        Returns
        -------
        pymatgen.analysis.elasticity.Strain, pymatgen.analysis.elasticity.Stress, pymatgen.core.tensors.Tensor or list
            The type of output depends on the type of object to use:
                - if the object is of class StrainTensor, the output will be of class pymatgen.analysis.elasticity.Strain
                - if the object is of class StressTensor, the output will be of class pymatgen.analysis.elasticity.Stress
                - otherwise, the output will be of class pymatgen.core.tensors.Tensor

        See Also
        --------
        flatten : Converts a tensor array to 1D tensor array
        """
        try:
            from Elasticipy.tensors.stress_strain import StrainTensor, StressTensor
            if isinstance(self, StrainTensor):
                from pymatgen.analysis.elasticity import Strain as Constructor
            elif isinstance(self, StressTensor):
                from pymatgen.analysis.elasticity import Stress as Constructor
            else:
                from pymatgen.core.tensors import Tensor as Constructor
        except ImportError:
            raise ModuleNotFoundError('Module pymatgen is required for this function.')
        if self.ndim > 1:
            raise ValueError('The array must be flattened (1D tensor array) before converting to pytmatgen.')
        if self.shape:
            return [Constructor(self[i].matrix) for i in range(self.shape[0])]
        else:
            return Constructor(self.matrix)

    @classmethod
    def stack(cls, arrays, axis=0):
        """
        Stack tensor arrays along the specified axis.

        Parameters
        ----------
        arrays : list of SecondOrderTensor or tuple of SecondOrderTensor
            List of tensor to stack together
        axis : int, optional
            Axis along which to stack

        Returns
        -------
        SecondOrderTensor
            Stacked tensor array

        Examples
        --------
        >>> from Elasticipy.tensors.second_order import SecondOrderTensor
        >>> import numpy as np
        >>> a = SecondOrderTensor.rand(shape=3)
        >>> b = SecondOrderTensor.rand(shape=3)
        >>> c = SecondOrderTensor.stack((a, b))
        >>> c.shape
        (2, 3)
        >>> np.all(c[0] == a)
        True
        >>> np.all(c[1] == b)
        True

        >>> a = SecondOrderTensor.rand(shape=(3, 4))
        >>> b = SecondOrderTensor.rand(shape=(3, 4))
        >>> c = SecondOrderTensor.stack((a, b), axis=1)
        >>> c.shape
        (3, 2, 4)
        >>> np.all(c[:,0,:] == a)
        True
        >>> np.all(c[:,1,:] == b)
        True
        """
        mat_array = [a.matrix for a in arrays]
        if axis<0:
            axis = axis - 2
        mat_stacked = np.stack(mat_array, axis=axis)
        return cls(mat_stacked)


class SymmetricSecondOrderTensor(SecondOrderTensor):
    _voigt_map = [1, 1, 1, 1, 1, 1]
    "List of factors to use for building a tensor from Voigt vector(s)"

    name = 'Symmetric second-order tensor'

    def __init__(self, mat, force_symmetry=False):
        """
        Create a symmetric second-order tensor

        Parameters
        ----------
        mat : list or numpy.ndarray
            matrix or array to construct the symmetric tensor. It must be symmetric with respect to the two last indices
            (mat[...,i,j]=mat[...,j,i]), or composed of slices of upper-diagonal matrices (mat[i,j]=0 for each i>j).
        force_symmetry : bool, optional
            If true, the symmetric part of the matrix will be used. It is mainly meant for debugging purpose.

        Examples
        --------
        We can create a symmetric tensor by privoding the full matrix, as long it is symmetric:

        >>> from Elasticipy.tensors.second_order import SymmetricSecondOrderTensor
        >>> a = SymmetricSecondOrderTensor([[11, 12, 13],[12, 22, 23],[13, 23, 33]])
        >>> print(a)
        Symmetric second-order tensor
        [[11. 12. 13.]
         [12. 22. 23.]
         [13. 23. 33.]]


        Alternatively, we can pass the upper-diagonal part only:

        >>> b = SymmetricSecondOrderTensor([[11, 12, 13],[0, 22, 23],[0, 0, 33]])

        and check that a==b:

        >>> a==b
        True
        """
        if isinstance(mat, SecondOrderTensor):
            mat = mat.matrix
        else:
            mat = np.asarray(mat, dtype=float)
        mat_transposed = _transpose_matrix(mat)
        if np.all(np.isclose(mat, mat_transposed)) or force_symmetry:
            # The input matrix is symmetric
            super().__init__(0.5 * (mat + mat_transposed))
        elif np.all(mat[..., np.tril_indices(3, k=-1)[0], np.tril_indices(3, k=-1)[1]] == 0):
            # The input matrix is upper-diagonal
            lower_diagonal = np.zeros_like(mat)
            triu_indices = np.triu_indices(3,1)
            lower_diagonal[..., triu_indices[0], triu_indices[1]] = mat[..., triu_indices[0], triu_indices[1]]
            super().__init__(mat + _transpose_matrix(lower_diagonal))
        else:
            raise ValueError('The input array must be either slices of symmetric matrices, of slices of upper-diagonal '
                             'matrices.')

    @classmethod
    def from_Voigt(cls, array, voigt_map=None):
        """
        Construct a SymmetricSecondOrderTensor from a Voigt vector, or slices of Voigt vectors.

        If the array is of shape (6,), a single tensor is returned. If the array is of shape (m,n,o,...,6), the tensor
        will be of shape (m,n,o,...).

        Parameters
        ----------
        array : np.ndarray or list
            array to build the SymmetricSecondOrderTensor from. We must have array.ndim>0 and array.shape[-1]==6.

        voigt_map : list or tuple, optional
            6-lenght list of factors to use for mapping. If None (default), the default Voigt map of the constructor is
            used.

        Returns
        -------
        SymmetricSecondOrderTensor

        See Also
        --------
        from_Kelvin : Construct a tensor from vector(s) following the Kelvin notation

        Examples
        --------
        >>> from Elasticipy.tensors.second_order import SymmetricSecondOrderTensor
        >>> SymmetricSecondOrderTensor.from_Voigt([11, 22, 33, 23, 13, 12])
        Symmetric second-order tensor
        [[11. 12. 13.]
         [12. 22. 23.]
         [13. 23. 33.]]
        """
        if voigt_map is None:
            voigt_map = cls._voigt_map
        matrix = _unmap(array, voigt_map)
        return cls(matrix)

    def to_Voigt(self):
        """
        Convert the tensor to vector, or slices of vector, following the Voigt convention.
        
        If the tensor array has shape (m,n,...), the result will be of shape (m,n,...,6).
        
        Returns
        -------
        numpy.ndarray
            Voigt vector summarizing the components
        """
        return _map(self.matrix, self._voigt_map)
    
    @classmethod
    def from_Kelvin(cls, array):
        """
        Build a tensor from the Kelvin vector, or slices of Kelvin vectors

        Parameters
        ----------
        array : np.ndarray or list
            Vectors, or slices of vectors, consisting in components following the Kelvin convention
        Returns
        -------
        SymmetricSecondOrderTensor

        See Also
        --------
        from_Voigt : construct a tensor from vector(s) following the Voigt notation
        to_Kelvin : convert the tensor to vector(s) following the Kelvin convention
        """
        matrix = _unmap(array, kelvin_mapping)
        return cls(matrix)
    
    def to_Kelvin(self):
        """
        Convert the tensor to vector, or slices of vector, following the Kelvin(-Mandel) convention.

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        from_Kelvin : Construct a tensor from vector(s) following the Kelvin convention
        to_Voigt : Convert the tensor to vector(s) following the Voigt convention
        """
        return _map(self.matrix, kelvin_mapping)

    def eig(self):
        """
        Compute the principal values (eigenvalues) and principal direction (eigenvectors) of the tensor, sorted in
        descending order of principal values

        Returns
        -------
        numpy.ndarray
            Principal values
        numpy.ndarray
            Principal directions

        See Also
        --------
        eigvals : compute the principal values only
        """
        eigvals, eigdir = np.linalg.eigh(self.matrix)
        return eigvals[..., ::-1], eigdir[..., :, ::-1]

    def eigvals(self):
        """
        Compute the principal values (eigenvalues), sorted in descending order.

        Returns
        -------
        numpy.ndarray
            Principal values

        See Also
        --------
        eig : return the principal values and principal directions
        """
        eigvals = np.linalg.eigvalsh(self.matrix)
        return np.flip(eigvals,axis=-1)

    def draw_Mohr_circles(self):
        """
        Draw the Mohr circles of the symmetric second-order tensor

        Given a tensor, the Mohr circles are meant to visually illustrate the possible shear components one can find
        when randomly rotating the tensor. See `here <https://en.wikipedia.org/wiki/Mohr%27s_circle>`_ for details.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Matplotlib figure containing the plot.
        ax : matplotlib.axes._axes.Axes
            The Matplotlib axes of the plot.
        """
        c,b,a = self.eigvals()

        # Sizes and locations of circles
        r1 = (c - b) / 2
        r2 = (b - a) / 2
        r3 = (c - a) / 2
        center1 = ((b + c) /2, 0)
        center2 = ((a + b) /2, 0)
        center3 = ((a + c) /2, 0)

        fig, ax = plt.subplots()
        filldraw_circle(ax, center1, r1, 'skyblue')
        filldraw_circle(ax, center2, r2, 'lightgreen')
        filldraw_circle(ax, center3, r3, 'red')
        filldraw_circle(ax, center3, r3, 'red', fill=True, alpha=0.2)
        filldraw_circle(ax, center1, r1, 'white', fill=True)
        filldraw_circle(ax, center2, r2, 'white', fill=True)
        ax.set_aspect('equal')
        ax.set_xlabel(f"Normal")
        ax.set_ylabel(f"Shear")
        ax.grid(True)
        xticks = (a,b,c, center1[0], center2[0])
        ax.set_xticks(np.unique(xticks))
        yticks = (-r3, -r2, -r1, 0, r1, r2, r3)
        ax.set_yticks(np.unique(yticks))

        return fig, ax


class SkewSymmetricSecondOrderTensor(SecondOrderTensor):
    name = 'Skew-symmetric second-order tensor'

    def __init__(self, mat, force_skew_symmetry=False):
        """Class constructor for skew-symmetric second-order tensors

        Parameters
        ----------
        mat : list or numpy.ndarray
            Input matrix, or slices of matrices. Each matrix should be skew-symmetric, or have zero-component on lower -
            diagonal part (including the diagonal).

        Examples
        --------
        One can construct a skew-symmetric tensor by providing the full skew-symmetric matrix:

        >>> from Elasticipy.tensors.second_order import SkewSymmetricSecondOrderTensor
        >>> a = SkewSymmetricSecondOrderTensor([[0, 12, 13],[-12, 0, 23],[-13, -23, 0]])
        >>> print(a)
        Skew-symmetric second-order tensor
        [[  0.  12.  13.]
         [-12.   0.  23.]
         [-13. -23.   0.]]

        Alternatively, one can pass the upper-diagonal part only:

        >>> b = SkewSymmetricSecondOrderTensor([[0, 12, 13],[0, 0, 23],[0, 0, 0]])

        and check that a==b:

        >>> a==b
        True

        """
        mat = np.asarray(mat, dtype=float)
        mat_transposed = _transpose_matrix(mat)
        if np.all(np.isclose(mat, -mat_transposed)) or force_skew_symmetry:
            # The input matrix is symmetric
            super().__init__(0.5 * (mat - mat_transposed))
        elif np.all(mat[..., np.tril_indices(3)[0], np.tril_indices(3)[1]] == 0):
            # The input matrix is upper-diagonal
            super().__init__(mat - mat_transposed)
        else:
            raise ValueError('The input array must be either slices of skew-symmetric matrices, of slices of upper-'
                             'diagonal matrices with zero-diagonal.')


def rotation_to_matrix(rotation, return_transpose=False):
    """
    Converts a rotation to slices of matrices

    Parameters
    ----------
    rotation : scipy.spatial.Rotation or orix.quaternion.Rotation
        Object to convert
    return_transpose : bool, optional
        If true, it will also return the transpose matrix as a 2nd output argument
    Returns
    -------
    numpy.ndarray or tuple
        Rotation matrices
    """
    if isinstance(rotation, Rotation):
        matrix = rotation.as_matrix()
    elif is_orix_rotation(rotation):
        inv_rotation = ~rotation
        matrix = inv_rotation.to_matrix()
        if matrix.shape == (1,3,3):
            matrix = matrix[0]
    else:
        raise TypeError('The input argument must be of class scipy.transform.Rotation or '
                        'orix.quaternion.rotation.Rotation')
    if return_transpose:
        return matrix, _transpose_matrix(matrix)
    else:
        return matrix


def is_orix_rotation(other):
    """
    Check whether the argument is a rotation from Orix by looking at the existing methods.

    Parameters
    ----------
    other : any
        object to test
    Returns
    -------
    bool
        True if other.to_matrix() exists
    """
    return hasattr(other, "to_matrix") and callable(getattr(other, "to_matrix"))
