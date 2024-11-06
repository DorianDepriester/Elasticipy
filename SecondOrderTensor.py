import numpy as np


class SecondOrderTensor:
    name = 'Second-order tensor'

    def __init__(self, matrix):
        self.matrix = matrix

    def __repr__(self):
        s = self.name + '\n'
        if self.shape:
            s += 'Shape={}'.format(self.shape)
        else:
            s += self.matrix.__str__()
        return s

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
            raise ValueError('The element to add must be a number, a ndarray or of the same class.')

    @property
    def shape(self):
        *shape, _, _ = self.matrix.shape
        return shape

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

    def volumetricChange(self):
        return self.firstInvariant()

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
        if (len(shape_other) < 2) or (shape_other[-1] != 3) or (shape_other[-2] != 3):
            raise ValueError('The array to dot by must be of shape (...,3,3)')
        shape_other = shape_other[:-2]
        if len(self.shape) == 0:
            if len(shape_other) == 0:
                new_mat = np.matmul(self.matrix, other_matrix)
            elif len(shape_other) == 1:
                new_mat = np.einsum('ik,njk->nij', self.matrix, other_matrix)
            else:
                new_mat = np.einsum('ik,npjk->npij', self.matrix, other_matrix)
        elif len(self.shape) == 1:
            if len(shape_other) == 0:
                new_mat = np.einsum('nik,jk->nij', self.matrix, other_matrix)
            elif len(shape_other) == 1:
                new_mat = np.einsum('nik,njk->ij', self.matrix, other_matrix)
            else:
                new_mat = np.einsum('nik,npjk->pij', self.matrix, other_matrix)
        else:
            if len(shape_other) == 0:
                new_mat = np.einsum('mnik,jk->mnij', self.matrix, other_matrix)
            elif len(shape_other) == 1:
                new_mat = np.einsum('mnik,njk->mij', self.matrix, other_matrix)
            else:
                new_mat = np.einsum('mnik,npjk->mpij', self.matrix, other_matrix)
        return SecondOrderTensor(new_mat)


class StrainTensor(SecondOrderTensor):
    name = 'Strain tensor'

    def principalStrains(self):
        return self.eig()[0]


class StressTensor(SecondOrderTensor):
    name = 'Stress tensor'

    def principalStresses(self):
        return self.eig()[0]

    def vonMises(self):
        p = (self.C(0, 0) - self.C(1, 1))**2 + (self.C(0, 0) - self.C(2, 2))**2 + (self.C(1, 1) - self.C(1, 1))**2 + \
            6*self.C(0, 1)**2 + 6*self.C(0, 2)**2 + 6*self.C(1, 2)**2
        return np.sqrt(0.5*p)

    def Tresca(self):
        ps = self.principalStresses().T
        return np.max(ps, axis=-1) - np.min(ps, axis=-1)

    def hydrostaticPressure(self):
        return -self.firstInvariant()/3

    def deviatoricStress(self):
        eye = np.zeros(self.matrix.shape)
        eye[..., np.arange(3), np.arange(3)] = 1
        new_mat = self.matrix + self.hydrostaticPressure()*eye
        return StressTensor(new_mat)