import unittest
from elasticipy.tensors.finite_strain import DeformationGradient
import numpy as np

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
        F = DeformationGradient.rand(shape=(5,3))
        C = F.C
        U = F.U
        np.testing.assert_array_almost_equal(C.matrix, U.dot(U).matrix)

    def test_left_CauchyGreen(self):
        F = DeformationGradient.rand(shape=(5,3))
        B = F.B
        V = F.V
        np.testing.assert_array_almost_equal(B.matrix, V.dot(V).matrix)

if __name__ == '__main__':
    unittest.main()