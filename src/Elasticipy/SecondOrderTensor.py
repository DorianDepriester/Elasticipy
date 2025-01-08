import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


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
        Eigenvalues of the tensor

        Returns
        -------
        lambda : np.ndarray
            Eigenvalues of each tensor.
        v : np.ndarray
            Eigenvectors of teach tensor.

        See Also
        --------
        principalDirections : return only the principal directions (without eigenvalues)
        """
        return np.linalg.eig(self.matrix)

    def principalDirections(self):
        """
        Principal directions of the tensors

        Returns
        -------
        np.nparray
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
        b = np.matmul(self.matrix, self._transposeTensor()).trace(axis1=-1, axis2=-2)
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
            new_mat = np.matmul(self.matrix, B.matrix)
            return SecondOrderTensor(new_mat)
        elif isinstance(B, Rotation):
            rotation_matrices = B.as_matrix()
            transpose_matrices = B.inv().as_matrix()
            new_matrix = np.matmul(np.matmul(transpose_matrices, self.matrix), rotation_matrices)
            # In case of rotation, the property of the transformed tensor is kept
            return self.__class__(new_matrix)
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

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return self.__mul__(other)
        else:
            raise NotImplementedError('Left multiplication is only implemented for scalar values.')

    def __eq__(self, other) -> np.ndarray:
        """
        Check whether the tensors in the tensor array are equal

        Parameters
        ----------
        other : SecondOrderTensor or np.ndarray
            Tensor to compare with

        Returns
        -------
        np.array of bool
            True element is True if the corresponding tensors are equal.
        """
        if isinstance(other, SecondOrderTensor):
            return self == other.matrix
        elif isinstance(other, np.ndarray):
            if (other.shape == (3,3)) or (other.shape == self.shape + (3,3)):
                return np.all(self.matrix == other, axis=(-2, -1))
            else:
                raise ValueError('The value to compare must be an array of shape {} or {}'.format(self.shape, self.shape + (3,3)))

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
        if isinstance(other, SecondOrderTensor):
            other_matrix = other.matrix
        elif isinstance(other, Rotation):
            other_matrix = other.as_matrix()
        else:
            other_matrix = other
        matrix = self.matrix
        shape_matrix = matrix.shape[:-2]
        shape_other = other_matrix.shape[:-2]
        extra_dim_matrix = len(shape_other)
        extra_dim_other = len(shape_matrix)
        matrix_expanded = matrix.reshape(shape_matrix + (1,) * extra_dim_other + (3, 3))
        other_expanded = other_matrix.reshape((1,) * extra_dim_matrix + shape_other + (3, 3))
        if isinstance(other, Rotation):
            other_expanded_t = np.swapaxes(other_expanded, -1, -2)
            new_mat = np.matmul(np.matmul(other_expanded_t, matrix_expanded), other_expanded)
            return self.__class__(np.squeeze(new_mat))
        else:
            new_mat = np.matmul(matrix_expanded, other_expanded)
            return SecondOrderTensor(np.squeeze(new_mat))

    def transposeArray(self):
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

        It is actually an alias for transposeArray()

        Returns
        -------
        SecondOrderTensor
            Transposed array
        """
        return self.transposeArray()

    def _transposeTensor(self):
        return _transpose_matrix(self.matrix)

    def transposeTensor(self):
        """
        Transpose of tensors of the tensor array

        Returns
        -------
        SecondOrderTensor
            Array of transposed tensors of the tensor array

        See Also
        --------
        Transpose : transpose the array (not the components)
        """
        return self.__class__(self._transposeTensor())

    def ddot(self, other):
        """
        Double dot product (contraction of tensor product, usually denoted ":") of two tensors.

        For two tensors whose matrices are M1 and M2::

            M1.ddot(M2) == np.trace(np.matmul(M1, M2))

        Parameters
        ----------
        other : SecondOrderTensor or np.ndarray
            Tensor or tensor array to multiply by before contraction.

        Returns
        -------
        float or np.ndarray
            Result of double dot product

        See Also
        --------
        matmul : matrix-like product between two tensor arrays.

        """
        tensor_prod = self.transposeTensor()*other
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
        return 0.5 * (self.matrix + self._transposeTensor())

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
        SecondOrderTensor
            Skew-symmetric tensor
        """
        new_mat = 0.5 * (self.matrix - self._transposeTensor())
        return SkewSymmetricSecondOrderTensor(new_mat)

    def spherical_part(self):
        """
        Spherical (hydrostatic) part of the tensor

        Returns
        -------
        SecondOrderTensor
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
        SecondOrderTensor

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
        SecondOrderTensor
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
        SecondOrderTensor
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
        SecondOrderTensor
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
        SecondOrderTensor
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
        shape : tuple, optional
            Shape of the tensor array. If not provided, a single tensor is returned
        seed : int, optional
            Sets the seed for random generation. Useful to ensure reproducibility

        Returns
        -------
        SecondOrderTensor
            Tensor or tensor array of uniform random value

        See Also
        --------
        randn : Generate a random sample of tensors whose components follows a normal distribution

        Examples
        --------
        Generate a single random tensor:

        >>> from Elasticipy.SecondOrderTensor import SecondOrderTensor as tensor
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
        SecondOrderTensor
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
        SecondOrderTensor
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
            from Elasticipy.StressStrainTensors import StrainTensor, StressTensor
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

class SymmetricSecondOrderTensor(SecondOrderTensor):
    voigt_map = [1, 1, 1, 1, 1, 1]
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

        >>> from Elasticipy.SecondOrderTensor import SymmetricSecondOrderTensor
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
    def from_Voigt(cls, array):
        """
        Construct a SymmetricSecondOrderTensor from a Voigt vector, or slices of Voigt vectors.

        If the array is of shape (6,), a single tensor is returned. If the array is of shape (m,n,o,...,6), the tensor
        will be of shape (m,n,o,...).

        Parameters
        ----------
        array : np.ndarray or list
            array to build the SymmetricSecondOrderTensor from. We must have array.ndim>0 and array.shape[-1]==6.
        Returns
        -------
        SymmetricSecondOrderTensor

        Examples
        --------
        >>> from Elasticipy.SecondOrderTensor import SymmetricSecondOrderTensor
        >>> SymmetricSecondOrderTensor.from_Voigt([11, 22, 33, 23, 13, 12])
        Symmetric second-order tensor
        [[11. 12. 13.]
         [12. 22. 23.]
         [13. 23. 33.]]

        """
        array = np.asarray(array)
        shape = array.shape
        if shape and (shape[-1] == 6):
            new_shape = shape[:-1] + (3, 3)
            unvoigted_matrix = np.zeros(new_shape)
            voigt = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
            for i in range(6):
                unvoigted_matrix[..., voigt[i][0], voigt[i][1]] = array[..., i] / cls.voigt_map[i]
            return cls(unvoigted_matrix)
        else:
            raise ValueError("array must be of shape (6,) or (...,6) with Voigt vector")


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

        >>> from Elasticipy.SecondOrderTensor import SkewSymmetricSecondOrderTensor
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