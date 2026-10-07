import unittest
from elasticipy.tensors.finite_strain import DeformationGradient
import numpy as np

from elasticipy.tensors.second_order import SymmetricSecondOrderTensor
from elasticipy.tensors.stress_strain import StrainTensor


class TestDeformationGradient(unittest.TestCase):
    def test_polar(self):
        F = DeformationGradient.rand()
        R, U = F.polar()
        assert U == F.U
        np.testing.assert_array_equal(R, F.R)

        R, V = F.polar(side='left')
        assert V == F.V
        np.testing.assert_array_equal(R, F.R)

    def test_right_CauchyGreen(self):
        F = DeformationGradient.rand()
        C = F.C
        U = F.U
        np.testing.assert_array_almost_equal(C.matrix, U.dot(U).matrix)

    def test_left_CauchyGreen(self):
        F = DeformationGradient.rand()
        B = F.B
        V = F.V
        np.testing.assert_array_almost_equal(B.matrix, V.dot(V).matrix)

    def test_green_lagrange(self):
        shape = (5, 3)
        F = DeformationGradient.rand(shape=shape)
        E = F.E
        assert isinstance(E, SymmetricSecondOrderTensor)
        assert E.shape == F.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                Fij = F[i,j].matrix
                Eij = 0.5*(Fij.T @ Fij - np.eye(3))
                np.testing.assert_array_almost_equal(E[i,j].matrix, Eij)

    def test_negative_J(self):
        with self.assertRaises(ValueError) as context:
            _ = DeformationGradient.diag([-1,2,3])
        self.assertEqual(str(context.exception), "The determinant of the deformation tensor must be positive.")

    def test_small_strain(self):
        gamma = 0.1
        F = DeformationGradient.eye() + np.array([[0, 0.1, 0],[0, 0, 0], [0, 0, 0]])
        eps = F.small_strain()
        assert isinstance(eps, StrainTensor)
        assert eps == StrainTensor.shear([1, 0, 0], [0, 1, 0], gamma/2)

    def test_tensile(self):
        u = [1, 0, 0]
        mag = 0.5
        F = DeformationGradient.tensile(u, mag)
        np.testing.assert_array_equal(F.matrix, np.diag([1.5, 1, 1]))

    def test_shear(self):
        u = [1, 0, 0]
        v = [0, 1, 0]
        mag = 0.5
        F = DeformationGradient.shear(u, v, mag)
        F_the = np.eye(3)
        F_the[0,1] = mag
        np.testing.assert_array_equal(F.matrix, F_the)

    def test_isochoric_tensile(self):
        F = DeformationGradient.isochoric_tensile([1,0,0], 1)
        assert F == DeformationGradient.eye()
        assert F.volumetric_strain() == 0.0

        F = DeformationGradient.isochoric_tensile([0,1,0], 100)
        np.testing.assert_array_equal(F.matrix, np.diag([0.1, 100, 0.1]))

        F = DeformationGradient.isochoric_tensile([1,1,0], [1,2,3,4,5])
        np.testing.assert_array_almost_equal(F.J, np.ones(5))
        np.testing.assert_array_almost_equal(F.volumetric_strain(), np.zeros(5))

    def test_elongation(self):
        F = DeformationGradient.diag([2, 1, 1])
        assert F.elongation([1, 0, 0]) == 1.0
        assert F.elongation([0, 1, 0]) == 0.0
        assert F.elongation([0, 0, 1]) == 0.0

if __name__ == '__main__':
    unittest.main()