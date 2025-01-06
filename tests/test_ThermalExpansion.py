import unittest
from Elasticipy.ThermalExpansion import ThermalExpansionTensor as ThEx
from scipy.spatial.transform import Rotation
import numpy as np

coeff = np.array([[11, 12, 13], [12, 22, 23], [13, 23, 33]])
alpha = ThEx(coeff)

class TestThermalExpansion(unittest.TestCase):
    def test_transverse(self):
        alpha = ThEx.transverse_isotropic(alpha_11=11, alpha_33=33)
        np.testing.assert_array_equal(alpha.matrix, np.diag([11, 11, 33]))

    def test_orthotropic(self):
        alpha_orthotropic = ThEx.orthotropic(11, 22, 33)
        alpha_orthorhombic= ThEx.orthorhombic(11, 22, 33)
        matrix = np.diag([11., 22., 33.])
        np.testing.assert_array_equal(alpha_orthotropic.matrix, matrix)
        np.testing.assert_array_equal(alpha_orthorhombic.matrix, matrix)

    def test_triclinic(self):
        alpha = ThEx.triclinic(alpha_11=11, alpha_12=12, alpha_13=13, alpha_22=22, alpha_23=23, alpha_33=33)
        matrix = np.array([[11., 12., 13.],[12., 22., 23.],[13., 23., 33.]])
        np.testing.assert_array_equal(alpha.matrix, matrix)

    def test_monoclinic(self):
        # Check monoclinic with X-Y mirror plane
        alpha = ThEx.monoclinic(alpha_11=11, alpha_22=22, alpha_33=33, alpha_12=12)
        np.testing.assert_array_equal(alpha.matrix, np.array([[11, 12, 0],[12, 22, 0],[0, 0, 33]], dtype=np.float64))

        # Check monoclinic with X-Z mirror plane
        alpha = ThEx.monoclinic(alpha_11=11, alpha_22=22, alpha_33=33, alpha_13=13)
        np.testing.assert_array_equal(alpha.matrix, np.array([[11, 0, 13],[0, 22, 0],[13, 0, 33]], dtype=np.float64))

        # Check error if alpha_12 and alpha_13 are provided
        with self.assertRaises(ValueError) as context:
            ThEx.monoclinic(alpha_11=11, alpha_22=22, alpha_33=33, alpha_12=12, alpha_13=13)
        self.assertEqual(str(context.exception), 'alpha_13 and alpha_12 cannot be used together.')

        # Check error if none of alpha_12 and alpha_13 are provided
        with self.assertRaises(ValueError) as context:
            ThEx.monoclinic(alpha_11=11, alpha_22=22, alpha_33=33)
        self.assertEqual(str(context.exception), 'Either alpha_13 or alpha_12 must be provided.')

    def test_volumetric(self):
        matrix = np.random.random((3,3))
        matrix = matrix + matrix.T
        alpha = ThEx(matrix)
        assert np.trace(matrix) == alpha.volumetric_coefficient

    def test_matmul_rotations(self):
        m = 50
        rotations = Rotation.random(m)
        alphas = alpha.matmul(rotations)
        for i in range(m):
            rot_mat = np.matmul(rotations[i].as_matrix().T, np.matmul(alpha.matrix, rotations[i].as_matrix()))
            np.testing.assert_almost_equal(rot_mat, alphas[i].matrix)

    def test_mul(self):
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

    def test_matmul_array(self):
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

    def test_constructor_isotropic(self):
        coeff = 23e-6
        alpha = ThEx.isotropic(coeff)
        temp = 25
        eps = alpha * temp
        np.testing.assert_almost_equal(eps.matrix, np.eye(3) * coeff * temp)

if __name__ == '__main__':
    unittest.main()

