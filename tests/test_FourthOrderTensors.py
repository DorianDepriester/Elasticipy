import unittest
from Elasticipy.tensors.fourth_order import FourthOrderTensor, SymmetricFourthOrderTensor
import numpy as np

from Elasticipy.tensors.mapping import VoigtMapping
from Elasticipy.tensors.second_order import SecondOrderTensor


class TestFourthOrderTensor(unittest.TestCase):
    def test_multidimensionalArrayTensors(self):
        m = 5
        a = np.random.random((m, 6, 6))
        T = FourthOrderTensor(a)
        np.testing.assert_array_almost_equal(a, T._matrix)
        T2 = FourthOrderTensor(T.full_tensor)
        np.testing.assert_array_almost_equal(a, T2._matrix)

    def test_nonsymmetry(self):
        a = np.random.random((3,3,3,3))
        with self.assertRaises(ValueError) as context:
            _ = FourthOrderTensor(a)
        self.assertEqual(str(context.exception), 'The input array does not have minor symmetry')
        T = FourthOrderTensor(a, force_minor_symmetry=True)
        Tfull = T.full_tensor
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        b = 0.25 * (a[i,j,k,l] + a[j,i,k,l] + a[i,j,l,k] + a[j,i,l,k])
                        np.testing.assert_array_almost_equal(Tfull[i,j,k,l], b)

    def test_inversion(self):
        m = 5
        T = FourthOrderTensor.rand(shape=m)
        assert T._matrix.shape == (m, 6,6)
        Tinv = T.inv()
        TTinv = Tinv.ddot(T)
        eye = FourthOrderTensor.identity(shape=m)
        for i in range(m):
            np.testing.assert_array_almost_equal(TTinv[i].full_tensor, eye[i].full_tensor)

    def test_mult(self):
        m, n, o = 5, 4, 3
        a = FourthOrderTensor.rand(shape=(m,n,o))
        b = 5
        ab = a * b
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    np.testing.assert_array_almost_equal(ab[i,j,k]._matrix, a[i,j,k]._matrix * b)

        b = np.random.random((n,o))
        ab = a * b
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    np.testing.assert_array_almost_equal(ab[i,j,k]._matrix, a[i,j,k]._matrix * b[j,k])

    def test_zeros_setitem(self):
        m, n = 4, 5
        t = FourthOrderTensor.zeros()
        assert t.shape == ()
        assert np.all(t.full_tensor==0.)

        t = FourthOrderTensor.zeros(n)
        assert t.shape == (n,)

        t = FourthOrderTensor.zeros((m,n))
        assert t.shape == (m, n)

        t[1,3] = np.ones((6,6))
        for i in range(m):
            for j in range(n):
                if (i == 1) and (j == 3):
                    assert np.all(t[i,j] == 1.)
                else:
                    assert np.all(t[i, j] == 0.)

        t0 = t == 0.
        t0_th = np.ones((m, n))
        t0_th[1,3] = 0.
        assert np.all(t0== t0_th)

    def test_div(self):
        m, n, o = 5, 4, 3
        a = FourthOrderTensor.rand(shape=(m,n,o))
        a_div_a = a / a
        np.testing.assert_array_almost_equal(a_div_a.full_tensor, FourthOrderTensor.identity(shape=(m,n,o)).full_tensor)

        half_a = a / 2
        np.testing.assert_array_almost_equal(half_a.full_tensor, a.full_tensor/2)

        b = SecondOrderTensor.rand(shape=(4,3))
        a_div_b = a / b
        np.testing.assert_array_almost_equal(a_div_b.matrix, (a * b.inv()).matrix)

    def test_inconsistent_mapping(self):
        t1 = FourthOrderTensor.rand()
        t2 = FourthOrderTensor.rand(mapping=VoigtMapping())
        t3 = t1 + t2
        assert t3.mapping.name == t1.mapping.name
        np.testing.assert_array_equal(t3._matrix, t1._matrix + t2._matrix)
        t4 = t1.ddot(t2)
        assert t4.mapping.name == t1.mapping.name
        np.testing.assert_array_almost_equal(t4._matrix, np.matmul(t1._matrix, t2._matrix))

    def test_copy(self):
        t1 = FourthOrderTensor.rand()
        t2 = t1.copy()
        assert t2 == t1
        t2.mapping = VoigtMapping()
        assert t2 == t1
        assert not np.all(t1.matrix() == t2.matrix())

    def test_identity(self):
        I = FourthOrderTensor.identity()
        Ifull = I.full_tensor
        eye = np.eye(3)
        a = np.einsum('ik,jl->ijkl', eye, eye)
        b = np.einsum('il,jk->ijkl', eye, eye)
        np.testing.assert_array_equal(Ifull, (a + b) / 2)
        A = FourthOrderTensor.rand()
        IA = I.ddot(A)
        AI = A.ddot(I)
        np.testing.assert_array_almost_equal(A.matrix(), IA.matrix())
        np.testing.assert_array_almost_equal(A.matrix(), AI.matrix())

    def test_identity_spherical_part(self):
        Jfull = FourthOrderTensor.identity_spherical_part().full_tensor
        eye = np.eye(3)
        np.testing.assert_array_equal(Jfull, np.einsum('ij,kl->ijkl',eye, eye) / 3)

    def test_spherical_deviatoric_parts(self):
        A = FourthOrderTensor.rand()
        AJ = A.spherical_part()
        AJJ = AJ.spherical_part()
        np.testing.assert_array_almost_equal(AJ.matrix(), AJJ.matrix())
        AK = A.deviatoric_part()
        AKK = AK.deviatoric_part()
        np.testing.assert_array_almost_equal(AK.matrix(), AKK.matrix())
        AKJ = AK.spherical_part()
        AJK = AJ.deviatoric_part()
        np.testing.assert_array_almost_equal(AKJ.matrix(), np.zeros((6,6)))
        np.testing.assert_array_almost_equal(AJK.matrix(), np.zeros((6, 6)))

    def test_rand(self):
        shapes = [(), (3,2)]
        for shape in shapes:
            t = FourthOrderTensor.rand(shape=shape)
            assert t.shape == shape
            assert np.all(t.full_tensor >=0.)
            assert np.all(t.full_tensor < 1.)

    def test_ones(self):
        A = FourthOrderTensor.ones()
        np.testing.assert_array_almost_equal(A.full_tensor, np.ones((3,3,3,3)))
        Av = FourthOrderTensor.ones(mapping=VoigtMapping())
        np.testing.assert_array_almost_equal(Av.full_tensor, np.ones((3, 3, 3, 3)))
        A2d = FourthOrderTensor.ones(shape=(3,4))
        np.testing.assert_array_almost_equal(A2d.full_tensor, np.ones((3,4,3, 3, 3, 3)))


class TestSymmetricFourthOrderTensor(unittest.TestCase):
    def test_inversion(self):
        m = 5
        T = SymmetricFourthOrderTensor.rand(shape=m)
        Tinv = T.inv()
        TTinv = Tinv.ddot(T)
        eye = SymmetricFourthOrderTensor.identity(shape=m)
        for i in range(m):
            np.testing.assert_array_almost_equal(TTinv[i].full_tensor, eye[i].full_tensor)



if __name__ == '__main__':
    unittest.main()
