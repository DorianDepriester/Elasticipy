import unittest
from Elasticipy.ThermalExpansion import ThermalExpansionTensor as ThEx
from scipy.spatial.transform import Rotation
import numpy as np

coeff = np.array([[11, 12, 13], [12, 22, 23], [13, 23, 33]])
alpha = ThEx(coeff)

class TestThermalExpansion(unittest.TestCase):
    def test_constructor_isotropic(self):
        coeff = 23e-6
        alpha = ThEx.isotropic(coeff)
        temp = 25
        eps = alpha * temp
        np.testing.assert_almost_equal(eps.matrix, np.eye(3) * coeff * temp)

    def test_monoclinic_mul(self):
        n=50
        rotations = Rotation.random(n)
        alphas = alpha * rotations
        temp = np.linspace(0,10,n)
        eps = alphas * temp
        assert eps.shape == (n,)
        for i in range(n):
            rot_mat = rotations[i].as_matrix()
            rotated_strain_matrix = np.matmul(rot_mat.T, np.matmul(coeff, rot_mat)) * temp[i]
            np.testing.assert_almost_equal(eps[i].matrix, rotated_strain_matrix)

    def test_monoclinic_matmul(self):
        m, n= 50, 100
        rotations = Rotation.random(m)
        alphas = alpha * rotations
        temp = np.linspace(0,10,n)
        eps = alphas.matmul(temp)
        assert eps.shape == (m, n)
        for i in range(m):
            rot_mat = rotations[i].as_matrix()
            rotated_tensor_matrix = np.matmul(rot_mat.T, np.matmul(coeff, rot_mat))
            for j in range(n):
                np.testing.assert_almost_equal(eps[i,j].matrix, rotated_tensor_matrix * temp[j])


if __name__ == '__main__':
    unittest.main()

