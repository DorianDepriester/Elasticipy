import unittest
from Elasticipy.tensors.fourth_order import FourthOrderTensor, SymmetricFourthOrderTensor
import numpy as np


class TestFourthOrderTensor(unittest.TestCase):
    def test_multidimensionalArrayTensors(self):
        m = 5
        a = np.random.random((m, 6, 6))
        T = FourthOrderTensor(a)
        np.testing.assert_array_almost_equal(a, T.matrix)
        T2 = FourthOrderTensor(T.full_tensor())
        np.testing.assert_array_almost_equal(a, T2.matrix)

    def test_inversion(self):
        m = 5
        a = np.random.random((m, 6, 6))
        a = a + np.swapaxes(a, axis1=-1, axis2=-2)
        T = FourthOrderTensor(a)
        Tinv = T.inv()
        TTinv = Tinv.ddot(T)
        eye = SymmetricFourthOrderTensor.identity(m)
        for i in range(m):
            np.testing.assert_array_almost_equal(TTinv[i].full_tensor(), eye[i].full_tensor())


class TestSymmetricFourthOrderTensor(unittest.TestCase):
    def test_inversion(self):
        m = 5
        a = np.random.random((m, 6, 6))
        T = SymmetricFourthOrderTensor(a, force_symmetry=True)
        Tinv = T.inv()
        TTinv = Tinv.ddot(T)
        eye = SymmetricFourthOrderTensor.identity(m)
        for i in range(m):
            np.testing.assert_array_almost_equal(TTinv[i].full_tensor(), eye[i].full_tensor())

if __name__ == '__main__':
    unittest.main()
