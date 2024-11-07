import numpy as np
from scipy.spatial.transform import Rotation


class SecondOrderTensor:
    name = 'Second-order tensor'

    def __init__(self, matrix):
        matrix = np.atleast_2d(matrix)
        shape = matrix.shape
        if len(shape) < 2 or shape[-2:] != (3, 3):
            raise ValueError('The input matrix must be of shape (3,3) or (...,3,3)')
        self.matrix = matrix

    def __repr__(self):
        s = self.name + '\n'
        if self.shape:
            s += 'Shape={}'.format(self.shape)
        else:
            s += self.matrix.__str__()
        return s

    def __getitem__(self, index):
        return self.__class__(self.matrix[index])

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
        *shape, _, _ = self.matrix.shape
        return shape

    @property
    def ndim(self):
        return len(self.shape)

    def C(self, i, j):
        return self.matrix.T[j, i].T

    def eig(self):
        return np.linalg.eig(self.matrix)

    def principalDirections(self):
        return self.eig()[1]

    def firstInvariant(self):
        return self.matrix.trace(axis1=-1, axis2=-2)

    def secondInvariant(self):
        a = self.matrix.trace(axis1=-1, axis2=-2)**2
        b = np.linalg.matrix_power(self.matrix, 2).trace(axis1=-1, axis2=-2)
        return 0.5 * (a - b)

    def thirdInvariant(self):
        return np.linalg.det(self.matrix)

    def trace(self):
        return self.firstInvariant()

    def __mul__(self, other):
        """
        Element-wise matrix multiplication of arrays of tensors. The two tensors must be of the same shape, resulting in
        a tensor with the same shape.

        For instance, if T1.shape=T2.shape=[m,n],
        with T3=T1*T2, we have:
        T3[i, j] == np.matmul(T1[i, j], T2[i, j]) for i=0...m and j=0...n.

        Parameters
        ----------
        other : SecondOrderTensor or np.ndarray or Rotation
            If other is a SecondOrderTensor, it must of the same shape.
            If T2 is a numpy array, we must have:
                T2.shape == T1.shape + (3, 3)

        Returns
        -------
            Array of tensors populated with element-wise matrix multiplication, with the same shape as the input arguments.
        """
        if isinstance(other, SecondOrderTensor):
            other_matrix = other.matrix
        elif isinstance(other, Rotation):
            return self.matmul(other)
        else:
            other_matrix = other
        if other_matrix.shape != self.matrix.shape:
            return ValueError('The two tensor arrays must be of the same shape.')
        else:
            return SecondOrderTensor(np.matmul(self.matrix, other_matrix))

    def matmul(self, other):
        """
        Perform matrix-like multiplication between tensor arrays. Each "product" is a matrix product between
        the tensor components.

        If A.shape=(a1, a2, ..., an) and B.shape=(b1, b2, ..., bn), with C=A.matmul(B), we have:
            C.shape = (a1, a2, ..., an, b1, b2, ..., bn)
        and
            C[i,j,k,...,p,q,r...] = np.matmul(A[i,j,k,...], B[p,q,r,...])

        Parameters
        ----------
        other : SecondOrderTensor or np.ndarray or scipy.spatial.transform.Rotation
            Tensor array or rotation to right-multiply by.

        Returns
        -------
        Tensor array

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
            ndim = other_matrix.ndim
            new_axes = np.hstack((np.arange(ndim - 2), -1, -2))
            other_expanded_t = other_matrix.transpose(new_axes)
            new_mat = np.matmul(np.matmul(other_expanded_t, matrix_expanded), other_expanded)
        else:
            new_mat = np.matmul(matrix_expanded, other_expanded)
        return self.__class__(new_mat)

    @property
    def T(self):
        matrix = self.matrix
        ndim = matrix.ndim
        new_axes = np.hstack((ndim-3-np.arange(ndim-2), -2, -1))
        transposed_arr = np.transpose(matrix, new_axes)
        return self.__class__(transposed_arr)

    def ddot(self, other):
        """
        Double dot product (contraction of tensor product) of two tensors, usually denoted ":". For two tensors whose
        matrices are M1 and M2:
            M1.ddot(M2) == np.trace(np.matmul(M1, M2))

        Parameters
        ----------
        other : SecondOrderTensor or np.ndarray
            Tensor or tensor array to multiply by before contraction

        Returns
        -------
        float or np.ndarray
            Result of double dot product

        """
        tensor_prod = self*other
        return tensor_prod.trace()

    def rotate(self, rotation_matrix):
        ndim = rotation_matrix.ndim
        new_axes = np.hstack((np.arange(ndim - 2), -1, -2))
        rot_mat_transpose = rotation_matrix.transpose(new_axes)
        new_mat = np.matmul(np.matmul(rot_mat_transpose, self.matrix), rotation_matrix)
        return self.__class__(new_mat)


class StrainTensor(SecondOrderTensor):
    name = 'Strain tensor'

    def principalStrains(self):
        return self.eig()[0]

    def volumetricChange(self):
        return self.firstInvariant()


class StressTensor(SecondOrderTensor):
    name = 'Stress tensor'

    def principalStresses(self):
        return self.eig()[0]

    def vonMises(self):
        p = (self.C(0, 0) - self.C(1, 1))**2 + (self.C(0, 0) - self.C(2, 2))**2 + (self.C(1, 1) - self.C(2, 2))**2 + \
            6*self.C(0, 1)**2 + 6*self.C(0, 2)**2 + 6*self.C(1, 2)**2
        return np.sqrt(0.5*p)

    def Tresca(self):
        ps = self.principalStresses().T
        return np.max(np.real(ps), axis=0) - np.min(np.real(ps), axis=0)

    def hydrostaticPressure(self):
        return -self.firstInvariant()/3

    def deviatoricStress(self):
        eye = np.zeros(self.matrix.shape)
        eye[..., np.arange(3), np.arange(3)] = 1
        new_mat = self.matrix + self.hydrostaticPressure()*eye
        return StressTensor(new_mat)
