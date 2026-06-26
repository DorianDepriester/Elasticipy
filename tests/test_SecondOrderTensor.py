import unittest
from elasticipy.tensors.second_order import SecondOrderTensor, SymmetricSecondOrderTensor, SkewSymmetricSecondOrderTensor
import numpy as np
from pytest import approx


class TestSecondOrderTensor(unittest.TestCase):
    def test_set_item(self):
        """Check setting a tensor in a tensor array"""
        stress = SecondOrderTensor.zeros((3, 3))
        stress[0,0] = np.ones(3)
        matrix = np.zeros((3, 3, 3, 3))
        matrix[0, 0, :, :] = 1
        np.testing.assert_array_equal(stress.matrix, matrix)

    def test_add_sub_mult_strain(self):
        """Check addition, subtraction and float multiplication of tensors"""
        shape = (3,3,3)
        a = SecondOrderTensor.ones(shape)
        b = 2 * SecondOrderTensor.ones(shape)
        c = 3 * SecondOrderTensor.ones(shape)
        d = a + b - c + 5 - 5 - a
        np.testing.assert_array_equal(d.matrix, -np.ones(shape + (3,3)))

    def test_flatten(self):
        """Check flattening of a tensor array"""
        shape = (3,3,3)
        a = SecondOrderTensor.rand(shape=shape)
        a_flat = a.flatten()

        # Fist, check that the shapes are consistent
        assert a_flat.shape == np.prod(shape)

        # Then, check out each element
        for p in range(0, np.prod(shape)):
            i, j, k = np.unravel_index(p, shape)
            np.testing.assert_array_equal(a_flat[p].matrix, a[i,j,k].matrix)

    def test_ddot(self):
        """
        Test the ddot method.
        """
        shape = (4, 3, 2)
        tens1 = SecondOrderTensor.rand(shape)
        tens2 = SecondOrderTensor.rand(shape[1:])   # Force tens2 to have a different shape, just to check broadcasting
        ddot = tens1.ddot(tens2)
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    ddot_th = np.trace(np.matmul(tens1.matrix[i,j,k].T, tens2.matrix[j,k]))
                    assert ddot_th == approx(ddot[i, j, k])

    def test_randn(self):
        """Test normal random distribution"""
        shape = (50, 40, 30)
        mean = np.random.random((3,3))
        std = np.random.random((3,3))
        t = SecondOrderTensor.randn(mean=mean, std=std, shape=shape)
        tmean = t.mean()
        tstd = t.std()
        tol = 1e-5
        np.testing.assert_array_almost_equal(tmean.matrix, mean, decimal=tol)
        np.testing.assert_array_almost_equal(tstd.matrix, std, decimal=tol)

    def test_rand(self):
        """Test uniform random generation"""
        # Test two ways to define a rand tensor
        shape = (5,4)
        seed = 1324 # Ensure reproducibility
        t1 = SecondOrderTensor.rand(shape=shape, seed=seed)
        rng = np.random.default_rng(seed)
        t2 = SecondOrderTensor(rng.random(shape + (3,3)))
        assert np.all(t1==t2)

    def test_inv(self):
        """Test inverse method"""
        shape = (3,4)
        t = SecondOrderTensor.rand(shape=shape)
        tinv = t.inv()
        for i in range(shape[0]):
            for j in range(shape[1]):
                np.testing.assert_array_almost_equal(tinv[i,j].matrix, np.linalg.inv(t.matrix[i,j]))

    def test_stack(self):
        size = 5
        a = SecondOrderTensor.rand(shape=size)
        b = SecondOrderTensor.rand(shape=size)
        c = SecondOrderTensor.stack((a, b))
        assert c.shape == (2, size)
        assert np.all(c[0] == a) and np.all(c[1] == b)
        c2 = SecondOrderTensor.stack((a,b), axis=1)
        assert c2.shape == (size, 2)
        assert np.all(c2[:,0] == a) and np.all(c2[:,1] == b)
        c3 = SecondOrderTensor.stack((a,b), axis=-1)
        assert np.all(c3 == c2)

class TestSymmetricSecondOrderTensor(unittest.TestCase):
    def test_inv(self):
        shape = (3,4)
        t = SymmetricSecondOrderTensor.rand(shape=shape)
        tinv = t.inv()
        for i in range(shape[0]):
            for j in range(shape[1]):
                np.testing.assert_array_almost_equal(tinv[i,j].matrix, np.linalg.inv(t.matrix[i,j]))

if __name__ == '__main__':
    unittest.main()
