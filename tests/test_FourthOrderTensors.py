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
        T = FourthOrderTensor(a)
        Tinv = T.inv()
        TTinv = Tinv.ddot(T)
        eye = FourthOrderTensor.identity(m)
        for i in range(m):
            np.testing.assert_array_almost_equal(TTinv[i].full_tensor(), eye[i].full_tensor())

    def test_mult(self):
        m, n, o = 5, 4, 3
        a = np.random.random((m,n,o,6,6))
        a = FourthOrderTensor(a)
        b = 5
        ab = a * b
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    np.testing.assert_array_equal(ab[i,j,k].matrix, a[i,j,k].matrix * b)

        b = np.random.random((n,o))
        ab = a * b
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    np.testing.assert_array_equal(ab[i,j,k].matrix, a[i,j,k].matrix * b[j,k])


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
