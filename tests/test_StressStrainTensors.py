import unittest
from pytest import approx
import numpy as np
from scipy.spatial.transform import Rotation

from Elasticipy.FourthOrderTensor import StiffnessTensor
import Elasticipy.StressStrainTensors as Tensors


Cmat = [[231, 127, 104, 0, -18, 0],
        [127, 240, 131, 0, 1, 0],
        [104, 131, 175, 0, -3, 0],
        [0, 0, 0, 81, 0, 3],
        [-18, 1, -3, 0, 11, 0],
        [0, 0, 0, 3, 0, 85]]
C = StiffnessTensor(Cmat)


class TestStrainStressTensors(unittest.TestCase):
    def test_mult_by_stiffness(self):
        tensile_dir = [1, 0, 0]
        stress = Tensors.StressTensor([[1, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]])
        strain = C.inv()*stress
        eps_xx = strain.C[0, 0]
        eps_yy = strain.C[1, 1]
        eps_zz = strain.C[2, 2]
        E = C.Young_modulus.eval(tensile_dir)
        nu_y = C.Poisson_ratio.eval(tensile_dir, [0, 1, 0])
        nu_z = C.Poisson_ratio.eval(tensile_dir, [0, 0, 1])
        assert eps_xx == approx(1/E)
        assert eps_yy == approx(-nu_y / E)
        assert eps_zz == approx(-nu_z / E)

    def test_rotate_tensor(self, n_oris=10):
        random_tensor = Tensors.SecondOrderTensor(np.random.random((3, 3)))
        random_oris = Rotation.random(n_oris)
        eps_rotated = random_tensor * random_oris
        for i in range(n_oris):
            rot_mat = random_oris.as_matrix()[i, :, :]
            eps_matrix_th = np.matmul(np.matmul(rot_mat.T, random_tensor.matrix), rot_mat)
            np.testing.assert_almost_equal(eps_matrix_th, eps_rotated[i].matrix)

    def test_transpose_array(self, shape=(1, 2, 3)):
        random_matrix = np.random.random(shape + (3, 3))
        random_tensor = Tensors.SecondOrderTensor(random_matrix)
        transposed_tensor = random_tensor.transposeArray()
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    np.testing.assert_array_equal(random_matrix[i, j, k], transposed_tensor[k, j, i].matrix)

    def test_mul(self, shape=(4, 5)):
        shape = shape + (3, 3)
        matrix1 = np.random.random(shape)
        matrix2 = np.random.random(shape)
        tensor_prod = Tensors.SecondOrderTensor(matrix1) * Tensors.SecondOrderTensor(matrix2)
        for i in range(shape[0]):
            for j in range(shape[1]):
                mat_prod = np.matmul(matrix1[i, j], matrix2[i, j])
                np.testing.assert_array_equal(tensor_prod[i, j].matrix, mat_prod)

    def test_matmul(self, length1=3, length2=4):
        matrix1 = np.random.random((length1, 3, 3))
        matrix2 = np.random.random((length2, 3, 3))
        rand_tensor1 = Tensors.SecondOrderTensor(matrix1)
        rand_tensor2 = Tensors.SecondOrderTensor(matrix2)
        cross_prod_tensor = rand_tensor1.matmul(rand_tensor2)
        for i in range(0, length1):
            for j in range(0, length2):
                mat_prod = np.matmul(matrix1[i], matrix2[j])
                np.testing.assert_array_equal(cross_prod_tensor[i, j].matrix, mat_prod)

    def test_statistics(self, shape=(5, 4, 3, 2)):
        matrix = np.random.random(shape + (3, 3))
        tensor = Tensors.SecondOrderTensor(matrix)
        std = tensor.std()
        # First, check T.std()
        for i in range(0, 3):
            for j in range(0, 3):
                Cij = matrix[..., i, j].flatten()
                assert np.std(Cij) == approx(std.C[i, j])

        # Then, check T.std(axis=...)
        for i in range(0, len(shape)):
            np.testing.assert_array_equal(tensor.std(axis=i).matrix, np.std(matrix, axis=i))

    def test_ddot(self, shape=(4, 3, 2)):
        matrix1 = np.random.random(shape + (3, 3))
        matrix2 = np.random.random(shape + (3, 3))
        tens1 = Tensors.SecondOrderTensor(matrix1)
        tens2 = Tensors.SecondOrderTensor(matrix2)
        ddot = tens1.ddot(tens2)
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    ddot_th = np.trace(np.matmul(matrix1[i, j, k], matrix2[i, j, k]))
                    self.assertEqual(ddot_th, ddot[i, j, k])

    def test_vonMises_Tresca(self):
        matrix = np.zeros((3, 3, 3))
        matrix[0, 0, 0] = 1  # Simple tension
        matrix[1, 1, 0] = matrix[1, 0, 1] = 1  # Simple shear
        matrix[2, np.arange(3), np.arange(3)] = -1  # Hydrostatic pressure
        stress = Tensors.StressTensor(matrix)

        vM_stress = stress.vonMises()
        vm_th = np.array([1, np.sqrt(3), 0.0])
        np.testing.assert_array_equal(vM_stress, vm_th)

        Tresca_stress = stress.Tresca()
        Tresca_th = np.array([1, 2, 0.0])
        np.testing.assert_array_equal(Tresca_stress, Tresca_th)


if __name__ == '__main__':
    unittest.main()
