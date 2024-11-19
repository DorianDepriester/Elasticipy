import numpy as np
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

    voigt_map = [1, 1, 1, 1, 1, 1]
    "List of factors to use for building a tensor from Voigt vector(s)"

    def __init__(self, matrix):
        """
        Create an array of second-order tensors.

        The input argument can be:
            - an array of shape (3,3) defining all the component of the tensor;
            - a stack of matrices, that is an array of shape (...,3,3);
            - an array of shape (6);
            - a stack of vectors of lenghts 6, that is an array of shape (...,6).
        In the two last cases, it is assumed that the Voigt numbering convention is used.

        Parameters
        ----------
        matrix : list or np.ndarray
            (3,3) matrix, stack of (3,3) matrices, 6-length vector or stack of 6-length vectors
        """
        matrix = np.array(matrix)
        shape = matrix.shape
        if shape and (shape[-1] == 6):
            new_shape = shape[:-1] + (3, 3)
            unvoigted_matrix = np.zeros(new_shape)
            voigt = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
            for i in range(6):
                unvoigted_matrix[..., voigt[i][0], voigt[i][1]] = matrix[..., i]/self.voigt_map[i]
                unvoigted_matrix[..., voigt[i][1], voigt[i][0]] = matrix[..., i]/self.voigt_map[i]
            self.matrix = unvoigted_matrix
        elif len(shape) > 1 and shape[-2:] == (3, 3):
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
            raise ValueError('The r.h.s must be either float, a ndarray or an object of class {}'.format(self.__class__))

    def __add__(self, other):
        if type(self) == type(other):
            return self.__class__(self.matrix + other.matrix)
        elif isinstance(other, (int, float, np.ndarray)):
            return self.__class__(self.matrix + other)
        else:
            raise ValueError('The element to add must be a number, a ndarray or of the same class.')

    def __sub__(self, other):
        if type(self) == type(other):
            return self.__class__(self.matrix - other.matrix)
        elif isinstance(other, (int, float, np.ndarray)):
            return self.__class__(self.matrix - other)
        else:
            raise ValueError('The element to subtract must be a number, a ndarray or of the same class.')

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
        float
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
            return SecondOrderTensor(np.matmul(self.matrix, B.matrix))
        elif isinstance(B, Rotation):
            rotation_matrices = B.as_matrix()
            transpose_matrices = np.swapaxes(rotation_matrices, -1, -2)
            new_matrix = np.matmul(np.matmul(transpose_matrices, self.matrix), rotation_matrices)
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
        else:
            new_mat = np.matmul(matrix_expanded, other_expanded)
        return self.__class__(np.squeeze(new_mat))

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
        return np.swapaxes(self.matrix, -1, -2)

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
        tensor_prod = self*other
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
        """
        return self.__class__(self._flatten())

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

    def symmetricPart(self):
        """
        Symmetric part of the tensor

        Returns
        -------
        SecondOrderTensor
            Symmetric tensor

        See Also
        --------
        skewPart : Skew-symmetric part of the tensor
        """
        new_mat = 0.5 * (self.matrix + self._transposeTensor())
        return self.__class__(new_mat)

    def skewPart(self):
        """
        Skew-symmetric part of the tensor

        Returns
        -------
        SecondOrderTensor
            Skew-symmetric tensor
        """
        new_mat = 0.5 * (self.matrix - self._transposeTensor())
        return self.__class__(new_mat)

    def sphericalPart(self):
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

    def deviatoricPart(self):
        """
        Deviatoric part of the tensor

        Returns
        -------
        SecondOrderTensor

        See Also
        --------
        sphericalPart : spherical part of the tensor
        """
        return self - self.sphericalPart()

    @classmethod
    def eye(cls, shape=()):
        """
        Create an array of tensors populated with identity matrices

        Parameters
        ----------
        shape : list or int, default ()
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
        shape : list or int, default ()
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
        shape : list or int, default ()
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
