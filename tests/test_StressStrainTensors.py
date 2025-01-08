import unittest
from pytest import approx
import numpy as np
from scipy.spatial.transform import Rotation

from Elasticipy.FourthOrderTensor import StiffnessTensor
import Elasticipy.StressStrainTensors as Tensors
from Elasticipy.SecondOrderTensor import SecondOrderTensor, SymmetricSecondOrderTensor, SkewSymmetricSecondOrderTensor
from Elasticipy.StressStrainTensors import StrainTensor, StressTensor
from pymatgen.analysis.elasticity import Strain as mgStrain, Stress as mgStress

Cmat = [[231, 127, 104, 0, -18, 0],
        [127, 240, 131, 0, 1, 0],
        [104, 131, 175, 0, -3, 0],
        [0, 0, 0, 81, 0, 3],
        [-18, 1, -3, 0, 11, 0],
        [0, 0, 0, 3, 0, 85]]
C = StiffnessTensor(Cmat)


class TestStressStrainTensors(unittest.TestCase):
    def test_mult_by_stiffness(self):
        """
        Test Stiffness/Strain tensors product C*eps (which stands for C:E)
        """
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

    def test_rotate_tensor(self, n_oris=100):
        """
        Test the rotation of a tensor

        Parameters
        ----------
        n_oris : int
            Number of random orientations to use
        """
        random_tensor = SecondOrderTensor(np.random.random((3, 3)))
        random_oris = Rotation.random(n_oris)
        eps_rotated = random_tensor * random_oris
        for i in range(n_oris):
            rot_mat = random_oris[i].as_matrix()
            eps_matrix_th = np.matmul(np.matmul(rot_mat.T, random_tensor.matrix), rot_mat)
            np.testing.assert_almost_equal(eps_matrix_th, eps_rotated[i].matrix)

    def test_transpose_array(self):
        """
        Test transposing a tensor array
        """
        shape = (1, 2, 3)
        random_matrix = np.random.random(shape + (3, 3))
        random_tensor = SecondOrderTensor(random_matrix)
        transposed_tensor = random_tensor.T
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    np.testing.assert_array_equal(random_matrix[i, j, k], transposed_tensor[k, j, i].matrix)

    def test_transpose_tensor(self):
        """
        Test transposing a tensor array
        """
        shape = (2, 3, 4)
        tensor = SecondOrderTensor(np.random.random(shape + (3, 3)))
        tensor_transposed = tensor.transposeTensor()
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    np.testing.assert_array_equal(tensor_transposed[i, j, k].matrix, tensor[i, j, k].matrix.T)


    def test_mul(self):
        """
        Test the element-wise product of tensors.
        """
        # First, multiply a SymmetricSecondOrderTensor with another one, and expect a matrix product between each
        # sliced matrix
        shape = (4, 5)
        shape = shape + (3, 3)
        matrix1 = np.random.random(shape)
        matrix2 = np.random.random(shape)
        tensor_prod = SecondOrderTensor(matrix1) * SecondOrderTensor(matrix2)
        for i in range(shape[0]):
            for j in range(shape[1]):
                mat_prod = np.matmul(matrix1[i, j], matrix2[i, j])
                np.testing.assert_array_equal(tensor_prod[i, j].matrix, mat_prod)

        # Now, multiply a SymmetricSecondOrderTensor with an array of the same shape, and expect an element-wise
        # multiplication between the sliced matrix of the tensor and the values of the array
        t = SecondOrderTensor(matrix1)
        random_array = np.random.random(t.shape)
        tensor_prod = t * random_array
        for i in range(shape[0]):
            for j in range(shape[1]):
                matrix = tensor_prod[i, j].matrix
                np.testing.assert_array_equal(matrix, t.matrix[i,j,:] * random_array[i,j])

    def test_matmul(self, length1=3, length2=4):
        """
        Test the matrix-like product of tensor arrays

        Parameters
        ----------
        length1 : int
            Length of the first array
        length2 : int
            Length of the second array
        """
        matrix1 = np.random.random((length1, 3, 3))
        matrix2 = np.random.random((length2, 3, 3))
        rand_tensor1 = SecondOrderTensor(matrix1)
        rand_tensor2 = SecondOrderTensor(matrix2)
        cross_prod_tensor = rand_tensor1.matmul(rand_tensor2)
        for i in range(0, length1):
            for j in range(0, length2):
                mat_prod = np.matmul(matrix1[i], matrix2[j])
                np.testing.assert_array_equal(cross_prod_tensor[i, j].matrix, mat_prod)

    def test_matmul_rotation(self):
        m, n = 5, 100
        random_tensor = SecondOrderTensor(np.random.random((m,) + (3, 3)))
        random_oris = Rotation.random(n)
        array = random_tensor.matmul(random_oris)
        assert array.shape == (m, n)
        for i in range(m):
            for j in range(n):
                rot_mat = random_oris[j].as_matrix()
                matrix = np.matmul(np.matmul(rot_mat.T, random_tensor[i].matrix), rot_mat)
                np.testing.assert_almost_equal(matrix, array[i,j].matrix)

    def test_statistics(self):
        """
        Test the std, min and max functions for tensor arrays.
        """
        shape = (5, 4, 3, 2)
        matrix = np.random.random(shape + (3, 3))
        tensor = SecondOrderTensor(matrix)
        mini = tensor.min()
        maxi = tensor.max()
        std = tensor.std()
        # First, check T.std()
        for i in range(0, 3):
            for j in range(0, 3):
                Cij = matrix[..., i, j].flatten()
                assert np.std(Cij) == approx(std.C[i, j])
                assert np.min(Cij) == approx(mini.C[i, j])
                assert np.max(Cij) == approx(maxi.C[i, j])
        # Then, check T.std(axis=...)
        for i in range(0, len(shape)):
            np.testing.assert_array_equal(tensor.std(axis=i).matrix, np.std(matrix, axis=i))
            np.testing.assert_array_equal(tensor.min(axis=i).matrix, np.min(matrix, axis=i))
            np.testing.assert_array_equal(tensor.max(axis=i).matrix, np.max(matrix, axis=i))

        # Now, check for single value tensors
        tensor = tensor[0,0,0,0]
        np.testing.assert_array_equal(tensor.min().matrix, tensor.matrix)
        np.testing.assert_array_equal(tensor.max().matrix, tensor.matrix)
        np.testing.assert_array_equal(tensor.std().matrix, np.zeros((3,3)))

    def test_ddot(self):
        """
        Test the ddot method.
        """
        shape = (4, 3, 2)
        matrix1 = np.random.random(shape + (3, 3))
        matrix2 = np.random.random(shape + (3, 3))
        tens1 = SecondOrderTensor(matrix1)
        tens2 = SecondOrderTensor(matrix2)
        ddot = tens1.ddot(tens2)
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    ddot_th = np.trace(np.matmul(matrix1[i,j,k].T, matrix2[i,j,k]))
                    self.assertEqual(ddot_th, ddot[i, j, k])

    def test_vonMises_Tresca(self):
        """
        Check that the Tresca and von Mises methods work well for simple tension, simple shear and hydrostatic
        pressure.
        """
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

    def test_rotation_stiffness(self, ):
        """
        Check that the two ways to compute stress from a rotated stiffness tensor are consistent.
        """
        n_strain = 50
        n_ori = 100
        matrix = np.random.random((n_strain, 3, 3))
        eps = Tensors.StrainTensor(matrix, force_symmetry=True)
        ori = Rotation.random(n_ori)
        C_rotated = C * ori
        sigma = C_rotated * eps

        # Rotate stress and stress by their own
        eps_rot = eps.matmul(ori)
        sigma_rot2 = C * eps_rot
        sigma2 = sigma_rot2 * ori.inv()
        np.testing.assert_almost_equal(sigma.matrix, sigma2.transposeArray().matrix)

    def test_multidimensional_tensors(self, ):
        """
        Check that the shape of (C * rotations) * eps is (m, p, r, ...) if rotation.shape=(p, q,...) and
        len(rotations)=m.
        """
        shape_strain = (5, 4, 3)
        n_ori = 100
        strain = Tensors.StrainTensor.ones(shape_strain)
        ori = Rotation.random(n_ori)
        C_rotated = C * ori
        stress = C_rotated * strain
        self.assertEqual(stress.shape, (n_ori,) + shape_strain)

    def test_Voigt_notation_strain(self):
        """
        Check that the strain tensor can be reconstructed from Voigt vectors.
        Returns
        """
        a = np.random.random((3, 4, 6))
        strain = Tensors.StrainTensor.from_Voigt(a)
        for i in range(0, 3):
            for j in range(0, 4):
                for k in range(0, 6):
                    if k<3:
                        assert a[i,j,k] == strain[i,j].C[k,k]
                    elif k==3:
                        assert a[i,j,k] == 2*strain[i,j].C[1,2]
                    elif k==4:
                        assert a[i,j,k] == 2*strain[i,j].C[0,2]
                    else:
                        assert a[i,j,k] == 2*strain[i,j].C[0,1]

    def test_Voigt_notation_stress(self):
        """
        Check that the stress tensor can be reconstructed from Voigt vectors.
        """
        a = np.random.random((3, 4 , 6))
        stress = Tensors.StressTensor.from_Voigt(a)
        for i in range(0, 3):
            for j in range(0, 4):
                for k in range(0, 6):
                    if k<3:
                        assert a[i,j,k] == stress[i,j].C[k,k]
                    elif k==3:
                        assert a[i,j,k] == stress[i,j].C[1,2]
                    elif k==4:
                        assert a[i,j,k] == stress[i,j].C[0,2]
                    else:
                        assert a[i,j,k] == stress[i,j].C[0,1]

    def test_tensile_stress(self):
        """Check that a stress tensor can be defined for tensile state"""
        n = 10
        sigma_11 = np.linspace(0,1, n)
        stress = Tensors.StressTensor.tensile([1,0,0], sigma_11)
        for i in range(0, n):
            stress_i = np.diag([sigma_11[i], 0, 0])
            np.testing.assert_array_equal(stress[i].matrix, stress_i)

    def test_shear_stress(self):
        """Check that a stress tensor can be defined for shear state"""
        n = 10
        sigma_12 = np.linspace(0,1, n)
        stress = Tensors.StressTensor.shear([1,0,0], [0,1,0], sigma_12)
        for i in range(0, n):
            stress_i = np.zeros((3,3))
            stress_i[0,1] = stress_i[1,0] = sigma_12[i]
            np.testing.assert_array_equal(stress[i].matrix, stress_i)

        # Now check if error is thrown if the two vectors are not orthogonal
        with self.assertRaises(ValueError) as context:
            Tensors.StrainTensor.shear([1,0,0], [1,1,0], 0.1)
        self.assertEqual(str(context.exception), 'u and v must be orthogonal')

    def test_set_item(self):
        """Check setting a tensor in a tensor array"""
        stress = Tensors.StressTensor.zeros((3, 3))
        stress[0,0] = np.ones(3)
        matrix = np.zeros((3, 3, 3, 3))
        matrix[0, 0, :, :] = 1
        np.testing.assert_array_equal(stress.matrix, matrix)

    def test_add_sub_mult_strain(self):
        """Check addition, subtraction and float multiplication of tensors"""
        shape = (3,3,3)
        a = Tensors.StrainTensor.ones(shape)
        b = 2 * Tensors.StrainTensor.ones(shape)
        c = 3 * Tensors.StrainTensor.ones(shape)
        d = a + b - c + 5 - 5
        np.testing.assert_array_equal(d.matrix, np.zeros(shape + (3,3)))

    def test_flatten(self):
        """Check flattening of a tensor array"""
        shape = (3,3,3)
        matrix = np.random.random(shape + (3,3))
        a = SecondOrderTensor(matrix)
        a_flat = a.flatten()

        # Fist, check that the shapes are consistent
        assert a_flat.shape == np.prod(shape)

        # Then, check out each element
        for p in range(0, np.prod(shape)):
            i, j, k = np.unravel_index(p, shape)
            np.testing.assert_array_equal(a_flat[p].matrix, a[i,j,k].matrix)

    def test_symmetric_skew_parts(self):
        """Check the values returned by the symmetric and skew parts of a tensor"""
        shape = (2,3,4)
        a = SecondOrderTensor(np.random.random(shape + (3,3)))
        a_symm = a.symmetric_part()
        a_skew = a.skew_part()
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    matrix = a[i,j,k].matrix
                    np.testing.assert_array_equal(2 * a_symm[i, j, k].matrix, matrix + matrix.T )
                    np.testing.assert_array_equal(2 * a_skew[i, j, k].matrix, matrix - matrix.T)

    def test_equality(self):
        """Test the == operator"""
        # Test equality for two tensors of the same shape
        shape = (3,4,5)
        a = SecondOrderTensor(np.random.random(shape + (3,3)))
        b = SecondOrderTensor(np.random.random(shape + (3,3)))
        a[0,1,2] = b[0,1,2]
        is_equal = a == b
        assert is_equal.shape == shape
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    assert is_equal[i ,j, k] == np.all(a[i, j, k,:,:].matrix == b[i, j, k,:,:].matrix, axis=(-2,-1))

        # Test equality for an array of tensors, and a single tensor
        c = a[2, 1, 0]
        is_equal = a == c
        assert is_equal.shape == shape
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    assert is_equal[i ,j, k] == np.all(a[i, j, k,:,:].matrix == c.matrix, axis=(-2,-1))

        # Now test inconsistent shapes
        shape2 = (3,4,5,6)
        d = SecondOrderTensor(np.random.random(shape2 + (3, 3)))
        expected_error = 'The value to compare must be an array of shape {} or {}'.format(shape, shape + (3,3))
        with self.assertRaises(ValueError) as context:
            _ = a == d
        self.assertEqual(str(context.exception), expected_error)

    def test_divergence(self):
        """Test the divergence operator"""
        spacing = [0.1, 0.2, 0.3]
        x = np.arange(0, 1, spacing[0])
        y = np.arange(0, 1, spacing[1])
        z = np.arange(0, 1, spacing[2])
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        shape = x.shape
        a_11, b_11, c_11 = 2., 3., -1.
        a_12, b_12, c_12 = 4, -1., 2.
        a_23, b_23, c_23 = 1, -1, 3.

        s = SecondOrderTensor.zeros(shape=shape)
        s.C[0, 0] = a_11 * x + b_11 * y + c_11 * z
        s.C[0, 1] = a_12 * x + b_12 * y + c_12 * z
        s.C[1, 2] = a_23 * x + b_23 * y + c_23 * z
        div_space = s.div(spacing=spacing)
        div_nonspaced = s.div()
        div_uniaxial = s.div(axes=0)
        div_biaxial = s.div(axes=(0,1))
        expected_div_spaced = [a_11 + b_12, c_23, 0]
        expected_div_nonspaced = [a_11 * spacing[0] + b_12 * spacing[1], c_23 * spacing[2], 0]
        expected_div_uniaxial = [a_11 * spacing[0], 0, 0]
        expected_div_biaxial = [a_11 * spacing[0] + b_12 * spacing[1], 0, 0]
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    np.testing.assert_almost_equal(div_space[i, j, k], expected_div_spaced)
                    np.testing.assert_almost_equal(div_nonspaced[i, j, k], expected_div_nonspaced)
                    np.testing.assert_almost_equal(div_uniaxial[i, j, k], expected_div_uniaxial)
                    np.testing.assert_almost_equal(div_biaxial[i, j, k], expected_div_biaxial)


    def test_reshape(self):
        """Test reshaping and 'un-reshaping' an array of tensors"""
        init_shape = (6, 4, 5)
        new_shape = (30, 4)
        a = np.random.random(init_shape + (3, 3))
        t = SecondOrderTensor(a)
        t_reshaped = t.reshape(new_shape)
        assert t_reshaped.shape == new_shape
        t_reshaped_back = t_reshaped.reshape(init_shape)
        for i in range(0, init_shape[0]):
            for j in range(0, init_shape[1]):
                for k in range(0, init_shape[2]):
                    old_mat = a[i,j,k]
                    new_mat = t_reshaped_back[i,j,k].matrix
                    np.testing.assert_array_equal(old_mat, new_mat)

    def test_save_load_tensor(self):
        """Test save and load data to/from file"""

        # First, check with consistent shape
        a = np.random.random((5, 4, 3, 3))
        t = SecondOrderTensor(a)
        file_name = 'test_save.npy'
        t.save(file_name)
        t2 = t.load_from_npy(file_name)
        np.testing.assert_array_equal(t.matrix, t2.matrix)

        # Now try with inconsistent shape
        np.save(file_name, a[...,0])
        expected_error = 'The shape of the array to load must be (...,3,3).'
        with self.assertRaises(ValueError) as context:
            _ = SecondOrderTensor.load_from_npy(file_name)
        self.assertEqual(str(context.exception), expected_error)

    def test_save_load_csv(self):
        """Test saving and reading a CSV file"""
        # Check that when exporting/importing, we get the same tensor
        a = np.random.random((10, 3, 3))
        t = SecondOrderTensor(a)
        file_name = 'test_textfile.txt'
        t.save_as_txt(file_name)
        t2 = SecondOrderTensor.load_from_txt(file_name)
        np.testing.assert_array_almost_equal(t.matrix, t2.matrix)

        # Try with non-flatten tensor
        a = np.random.random((5, 3, 3, 3))
        t = SecondOrderTensor(a)
        expected_error = 'The array must be flatten before getting dumped to text file.'
        with self.assertRaises(ValueError) as context:
            t.save_as_txt(file_name)
        self.assertEqual(str(context.exception), expected_error)

        # Now try with a symmetric tensor
        a = np.random.random((10, 3, 3))
        a = a + np.swapaxes(a, -1, -2)
        t = SymmetricSecondOrderTensor(a)
        file_name = 'test_symmetric_textfile.txt'
        t.save_as_txt(file_name)
        t2 = SymmetricSecondOrderTensor.load_from_txt(file_name)
        np.testing.assert_array_almost_equal(t.matrix, t2.matrix)


    def test_symmetric_tensor_constructor(self):
        """Test constructor for symmetric second Order tensors"""

        # When a symmetric matrix is passed to the constructor
        mat = np.random.random((3,3))
        sym_mat = mat + mat.T
        t = SymmetricSecondOrderTensor(sym_mat)
        np.testing.assert_array_equal(t.matrix, sym_mat)

        # When an upper-diagonal matrix is passed to the constructor
        upper_mat = sym_mat
        upper_mat[np.tril_indices(3, -1)] = 0 # Set lower part to zero
        t2 = SymmetricSecondOrderTensor(upper_mat)
        assert t == t2

        # Expect error in any other case
        expected_error = ('The input array must be either slices of symmetric matrices, of slices of upper-diagonal '
                          'matrices.')
        with self.assertRaises(ValueError) as context:
            _ = SymmetricSecondOrderTensor(mat)
        self.assertEqual(str(context.exception), expected_error)

    def test_skew_symmetric_tensor_constructor(self):
        """Test constructor for symmetric second Order tensors"""

        # When a symmetric matrix is passed to the constructor
        mat = np.random.random((3,3))
        skew_sym_mat = mat - mat.T
        t = SkewSymmetricSecondOrderTensor(skew_sym_mat)
        np.testing.assert_array_equal(t.matrix, skew_sym_mat)

        # When an upper-diagonal matrix is passed to the constructor
        upper_mat = skew_sym_mat
        upper_mat[np.tril_indices(3)] = 0 # Set lower part to zero
        t2 = SkewSymmetricSecondOrderTensor(upper_mat)
        assert t == t2

        # Expect error in any other case
        expected_error = ('The input array must be either slices of symmetric matrices, of slices of upper-diagonal '
                          'matrices.')
        with self.assertRaises(ValueError) as context:
            _ = SymmetricSecondOrderTensor(mat)
        self.assertEqual(str(context.exception), expected_error)

    def test_add_sub_skew_symmetric_tensor(self):
        """Test basic operations between symmetric and skew-symmetric tensor, and check that the output classes are
        consistent."""
        mat = np.random.random((3,3))
        strain = StrainTensor(mat + mat.T)
        spin = SkewSymmetricSecondOrderTensor(mat - mat.T)

        # Check with consistent add.sub
        assert isinstance(strain + 2 * strain, StrainTensor)
        assert isinstance(spin + 2 * spin, SkewSymmetricSecondOrderTensor)

        # Check with inconsistent classes
        a = strain + 2 * spin
        assert (isinstance(a, SecondOrderTensor) and not isinstance(a, SymmetricSecondOrderTensor) and
                not isinstance(a, SkewSymmetricSecondOrderTensor))

        # Check negative values
        assert isinstance(-strain, StrainTensor)
        assert isinstance(-spin, SkewSymmetricSecondOrderTensor)

        # Check when adding a scaler value
        b = spin + 5
        assert (isinstance(b, SecondOrderTensor) and not isinstance(b, SymmetricSecondOrderTensor) and
                not isinstance(b, SkewSymmetricSecondOrderTensor))
        assert isinstance(strain + 5, StrainTensor)

        # Check when multiplying the tensor
        c = strain * spin
        assert (isinstance(c, SecondOrderTensor) and not isinstance(c, SymmetricSecondOrderTensor) and
                not isinstance(c, SkewSymmetricSecondOrderTensor))

        # Now check with rotations
        rotations = Rotation.random(100)
        assert isinstance(strain * rotations, StrainTensor)
        assert isinstance(spin * rotations, SkewSymmetricSecondOrderTensor)

    def test_to_pymatgen(self):
        """Test convertion from Elasticipy to pymatgen"""
        # First, try with a single tensor
        a = np.random.random((3, 3))
        a_sym = a + a.T
        strain = StrainTensor(a_sym)
        Strain_pymatgen = strain.to_pymatgen()
        np.testing.assert_array_equal(Strain_pymatgen.__array__(), a_sym)
        assert isinstance(Strain_pymatgen, mgStrain)

        # Now try with a 1D array
        n = 10
        b = np.random.random((n, 3, 3))
        b_sym = b + np.swapaxes(b, -1, -2)
        strain = StrainTensor(b_sym)
        Strain_pymatgen = strain.to_pymatgen()
        for i in range(n):
            np.testing.assert_array_equal(Strain_pymatgen[i].__array__(), b_sym[i])

        # Finally, try with a multidimensional array
        c = np.random.random((n, n, 3, 3))
        c_sym = c + np.swapaxes(c, -1, -2)
        strain = StrainTensor(c_sym)
        expected_error = 'The array must be flattened (1D tensor array) before converting to pytmatgen.'
        with self.assertRaises(ValueError) as context:
            _ = strain.to_pymatgen()
        self.assertEqual(str(context.exception), expected_error)

        # Now check that this also works for stress
        stress = StressTensor(a_sym)
        stress_pymatgen = stress.to_pymatgen()
        assert isinstance(stress_pymatgen, mgStress)

    def test_rand(self):
        """Test uniform random generation"""
        # Test two ways to define a rand tensor
        shape = (5,4)
        seed = 1324 # Ensure reproducibility
        t1 = SecondOrderTensor.rand(shape=shape, seed=seed)
        rng = np.random.default_rng(seed)
        t2 = SecondOrderTensor(rng.random(shape + (3,3)))
        assert np.all(t1==t2)

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

    def test_inv(self):
        """Test inverse method"""
        shape = (3,4)
        matrix = np.random.random(shape + (3,3))
        t = SecondOrderTensor(matrix)
        tinv = t.inv()
        for i in range(shape[0]):
            for j in range(shape[1]):
                np.testing.assert_array_almost_equal(tinv[i,j].matrix, np.linalg.inv(matrix[i,j]))

if __name__ == '__main__':
    unittest.main()
