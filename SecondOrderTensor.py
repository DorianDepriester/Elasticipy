import numpy as np


class SecondOrderTensor:
    name = 'Second-order tensor'

    def __init__(self, matrix):
        shape = matrix.shape
        if len(shape) < 2 or shape[-2:] != (3, 3):
            raise ValueError('The input matrix must of shape (3,3) or (...,3,3)')
        self.matrix = matrix

    def __repr__(self):
        s = self.name + '\n'
        if self.shape:
            s += 'Shape={}'.format(self.shape)
        else:
            s += self.matrix.__str__()
        return s

    def __getitem__(self, index):
        if len(index) > self.ndim:
            raise IndexError('Too many indices for tensor of shape {}'.format(self.shape))
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
            raise ValueError('The element to substract must be a number, a ndarray or of the same class.')

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

    def __mul__(self, other):
        if isinstance(other, SecondOrderTensor):
            return SecondOrderTensor(np.matmul(self.matrix, other.matrix))

    @property
    def T(self):
        matrix = self.matrix
        ndim = matrix.ndim
        new_axes = np.hstack((ndim-3-np.arange(ndim-2), -2, -1))
        transposed_arr = np.transpose(matrix, new_axes)
        return self.__class__(transposed_arr)

    def dot(self, other):
        if isinstance(other, SecondOrderTensor):
            other_matrix = other.matrix
        else:
            other_matrix = other
        shape_other = other_matrix.shape
        n1 = self.ndim
        n2 = len(shape_other)
        if (n2 < 2) or (shape_other[-1] != 3) or (shape_other[-2] != 3):
            raise ValueError('The array to dot by must be of shape (...,3,3)')
        shape_other = shape_other[:-2]
        ein_str = [['ik,jk->ij',     'ik,njk->nij',   'ik,npjk->npij'],
                   ['nik,jk->nij',   'nik,njk->ij',   'nik,npjk->pij'],
                   ['mnik,jk->mnij', 'mnik,njk->mij', 'mnik,npjk->mpij']]
        new_mat = np.einsum(ein_str[n1][n2], self.matrix, other_matrix)
        return self.__class__(new_mat)

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
