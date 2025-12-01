import unittest

import numpy as np
from pytest import approx
import os
import pandas as pd

from Elasticipy.tensors.elasticity import StiffnessTensor, ComplianceTensor
from scipy.spatial.transform import Rotation
from Elasticipy.tensors.elasticity import _indices2str
from Elasticipy.crystal_symmetries import SYMMETRIES
from Elasticipy.tensors.stress_strain import StressTensor, StrainTensor
from pymatgen.analysis.elasticity import elastic as mg
from orix.quaternion import Rotation as orix_rot
from Elasticipy.tensors.mapping import KelvinMapping, VoigtMapping
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'MaterialsProject.json')
data_base = pd.read_json(file_path)
rotations = Rotation.random(10000)

def variant_selection(symmetry, variant_name):
    for variant_group in symmetry.keys():
        elements = [elem.strip() for elem in variant_group.split(",")]
        if variant_name in elements:
            return symmetry[variant_group]
    return None

Smat = np.array([[8, -3, -2, 0, 14, 0],
                 [-3, 8, -5, 0, -8, 0],
                 [-2, -5, 10, 0, 0, 0],
                 [0, 0, 0, 12, 0, 0],
                 [14, -8, 0, 0, 116, 0],
                 [0, 0, 0, 0, 0, 12]])/1000
S = ComplianceTensor(Smat)


def crystal_symmetry_tester(symmetry_name, cls='stiffness', variant=None):
    symmetry = SYMMETRIES[symmetry_name]
    if variant is None:
        materials_of_interest = data_base[data_base.symmetry == symmetry_name]
        required_fields = symmetry.required
    else:
        materials_of_interest = data_base[data_base.point_group == variant]
        variant = variant_selection(symmetry, variant)
        required_fields = variant.required
    for index, row in materials_of_interest.iterrows():
        matrix = np.array(row['C'])
        if cls=='stiffness':
            class_constructor = StiffnessTensor
            prefix = 'C'
        else:
            class_constructor = ComplianceTensor
            prefix = 'S'
            matrix = np.linalg.inv(matrix)*1000
        kwargs = dict()
        for indices in required_fields:
            component_name = prefix + _indices2str(indices)
            kwargs[component_name] = matrix[tuple(indices)]
        constructor = getattr(class_constructor, symmetry_name.lower())
        C = constructor(**kwargs)
        assert np.all(C.matrix() == approx(matrix, rel=0.5))
        return C


class TestComplianceTensor(unittest.TestCase):
    def test_young_modulus_eval(self):
        E = S.Young_modulus
        E_xyz = E.eval(np.eye(3))
        for i in range(3):
            self.assertEqual(E_xyz[i], 1/Smat[i, i])

    def test_young_modulus_stats(self):
        E = S.Young_modulus
        assert E.mean(method='exact') == approx(101.994)
        assert E.std() == approx(48.48264174566468)
        assert E.mean(method='trapezoid') == approx(101.9855, rel=1e-3)

    def test_shear_modulus_eval(self):
        G = S.shear_modulus
        u = [[0, 1, 0], [1, 0, 0], [1, 0, 0]]
        v = [[0, 0, 1], [0, 0, 1], [0, 1, 0]]
        G_xyz = G.eval(u, v)
        for i in range(3):
            self.assertEqual(G_xyz[i],  1/Smat[i+3, i+3])

    def test_Poisson_ratio_eval(self):
        nu = S.Poisson_ratio
        u = [[0, 1, 0], [1, 0, 0], [1, 0, 0]]
        v = [[0, 0, 1], [0, 0, 1], [0, 1, 0]]
        nu_xyz = nu.eval(u, v)
        nu_xyz_th = [0.625, 0.25, 0.375]
        for i in range(3):
            self.assertEqual(nu_xyz[i],  nu_xyz_th[i])

    def test_shear_modulus_mini_maxi(self):
        G = S.shear_modulus
        G_min, _ = G.min()
        G_max, _ = G.max()
        assert G_min == approx(8.47165)
        assert G_max == approx(83.3333)

    def test_unvoigt(self):
        lame1, lame2 = 1, 2
        C = StiffnessTensor.isotropic(lame1=lame1, lame2=lame2)
        C_full = C.full_tensor
        eye = np.eye(3)
        A = np.einsum('ij,kl->ijkl', eye, eye)
        C_th = lame1 * A + 2 * lame2 * StiffnessTensor.eye().full_tensor
        np.testing.assert_almost_equal(C_th, C_full)

    def test_averages(self):
        averages = [S.Voigt_average(), S.Reuss_average(), S.Hill_average()]
        E_mean_th = [151.738, 75.76, 114.45]
        G_mean_th = [55.653, 26.596, 41.124]
        nu_mean_th = [0.36325, 0.42424242, 0.3915]
        for i, average in enumerate(averages):
            assert approx(average.Young_modulus.eval([1,0,0]), rel=1e-4) == E_mean_th[i]
            assert approx(average.shear_modulus.eval([1,0,0],[0,1,0]), rel=1e-4) == G_mean_th[i]
            assert approx(average.Poisson_ratio.eval([1,0,0],[0,1,0]), rel=1e-4) == nu_mean_th[i]

    def test_wave_velocity(self, E=210, nu=0.3, rho=7.8):
        C = StiffnessTensor.isotropic(E=E, nu=nu)
        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        cp, cs_1, cs_2 = C.wave_velocity(rho)
        assert approx(cp.mean()) == np.sqrt(M / rho)
        G = C.shear_modulus.mean()
        assert approx(cs_2.mean()) == np.sqrt(G / rho)
        assert approx(cs_1.mean()) == np.sqrt(G / rho)

    def test_symmetry(self):
        S = np.random.random((6, 6))
        with self.assertRaises(ValueError) as context:
            _ = ComplianceTensor(S)
        self.assertEqual(str(context.exception), 'The input matrix must be symmetric')

    def test_positive_definite(self):
        S = np.array([
            [2, -1, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0],
            [0, -1, 2, -1, 0, 0],
            [0, 0, -1, 2, -1, 0],
            [0, 0, 0, -1, 2, -1],
            [0, 0, 0, 0, -1, 0]
        ])
        with self.assertRaises(ValueError) as context:
            _ = ComplianceTensor(S)
        eig_vals = np.array([3.47221969, 2.23865632, 1.46407147, 0.80765321, 0.26960962, -0.2522103])
        expected_error = 'The input matrix is not definite positive (eigenvalues: {})'.format(eig_vals)
        self.assertEqual(str(context.exception), expected_error)

    def test_input_matrix(self):
        for n in range(1,8):
            m = np.ones(shape=(n,n))
            if n != 6:
                with self.assertRaises(ValueError) as context:
                    _ = ComplianceTensor(m)
                self.assertEqual(str(context.exception), 'The input matrix must of shape (...,6,6) or (...,3,3,3,3)')

    def test_full_tensor_as_input(self):
        a = ComplianceTensor.isotropic(E=210, nu=0.3)
        b = ComplianceTensor(a.full_tensor)
        np.testing.assert_array_almost_equal(a._matrix, b._matrix)


    def test_component(self):
#        S = ComplianceTensor(Smat)
        assert S.C11 == Smat[0, 0]
        assert S.C12 == Smat[0, 1]
        assert S.C13 == Smat[0, 2]
        assert S.C14 == Smat[0, 3]
        assert S.C15 == Smat[0, 4]
        assert S.C16 == Smat[0, 5]
        assert S.C22 == Smat[1, 1]
        assert S.C23 == Smat[1, 2]
        assert S.C24 == Smat[1, 3]
        assert S.C25 == Smat[1, 4]
        assert S.C26 == Smat[1, 5]
        assert S.C32 == Smat[2, 1]
        assert S.C33 == Smat[2, 2]
        assert S.C34 == Smat[2, 3]
        assert S.C35 == Smat[2, 4]
        assert S.C36 == Smat[2, 5]
        assert S.C44 == Smat[3, 3]
        assert S.C45 == Smat[3, 4]
        assert S.C46 == Smat[3, 5]
        assert S.C55 == Smat[4, 4]
        assert S.C56 == Smat[4, 5]
        assert S.C66 == Smat[5, 5]
        docstring = ComplianceTensor.C12.__doc__
        assert docstring == 'Returns the (1,2) component of the Compliance matrix.'

    def test_bulk_modulus(self):
        E, nu = 210, 0.3
        Siso= ComplianceTensor.isotropic(E=E, nu=nu)
        assert Siso.bulk_modulus == approx(E / (3 * (1-2 * nu)))

    def test_mean(self):
        m, n = 5, 10
        orix_rotations = orix_rot.random((m, n))
        S_rotated = S * orix_rotations
        S_mean_0 = S_rotated.mean(axis=0)
        for i in range(n):
            np.testing.assert_array_almost_equal(S_mean_0[i]._matrix, S_rotated[:, i].Reuss_average()._matrix)
        S_mean_1 = S_rotated.mean(axis=1)
        for i in range(m):
            np.testing.assert_array_almost_equal(S_mean_1[i]._matrix, S_rotated[i, :].Reuss_average()._matrix)
        S_mean = S_rotated.mean()
        np.testing.assert_array_almost_equal(S_mean._matrix, S_rotated.flatten().Reuss_average()._matrix)
        np.testing.assert_array_almost_equal(S_mean._matrix, S_rotated.Reuss_average()._matrix)

    def test_to_from_Kelvin(self):
        matrix = S.to_Kelvin()
        S2 = ComplianceTensor.from_Kelvin(matrix)
        np.testing.assert_array_almost_equal(S._matrix, S2._matrix)

    def test_matrix(self):
        S2 = ComplianceTensor(Smat)
        np.testing.assert_array_equal(S2._matrix / KelvinMapping().matrix, Smat / VoigtMapping(tensor='compliance').matrix)

    def test_full_tensor(self):
        S_full = S.full_tensor
        S2 = ComplianceTensor(S_full)
        assert S == S2
        np.testing.assert_array_equal(S.full_tensor, S2.full_tensor)

    def test_repr(self):
        """Test printing out the tensor"""
        str = S.__repr__()
        assert str == ('Compliance tensor (in Voigt mapping):\n'
                       '[[ 0.008 -0.003 -0.002  0.     0.014  0.   ]\n'
                       ' [-0.003  0.008 -0.005  0.    -0.008  0.   ]\n'
                       ' [-0.002 -0.005  0.01   0.     0.     0.   ]\n'
                       ' [ 0.     0.     0.     0.012  0.     0.   ]\n'
                       ' [ 0.014 -0.008  0.     0.     0.116  0.   ]\n'
                       ' [ 0.     0.     0.     0.     0.     0.012]]')

        S_rotated = S * rotations
        assert S_rotated.__repr__() ==  'Compliance tensor array of shape ({},)'.format(len(rotations))

    def test_multidimensional_attributes(self):
        S_rotated = S * rotations
        shape = (len(rotations),)
        for attr in ['Young_modulus', 'shear_modulus', 'linear_compressibility', 'Poisson_ratio']:
            a = getattr(S_rotated, attr, 0)
            assert isinstance(a, np.ndarray)
            assert a.shape == shape
            assert getattr(S_rotated[0], attr, 0) == a[0]
        with self.assertRaises(ValueError) as context:
            _ = S_rotated.wave_velocity(1.)
        expected = 'This function is not suitable for tensor array. Consider subscripting (e.g. C[0].wave_velocity()).'
        self.assertEqual(str(context.exception), expected)


    def test_voigt_reuss_axis(self):
        m,n = 5,6
        orientations = orix_rot.random((m,n))
        S_rotated = S * orientations
        Sv_0 = S_rotated.Voigt_average(axis=0)
        Sr_0 = S_rotated.Reuss_average(axis=0)
        assert Sv_0.shape == (n,)
        assert Sr_0.shape == (n,)
        for i in range(n):
            np.testing.assert_array_almost_equal(Sv_0[i]._matrix, S_rotated[:, i].Voigt_average()._matrix)
            np.testing.assert_array_almost_equal(Sr_0[i]._matrix, S_rotated[:, i].Reuss_average()._matrix)
        Sv_1 = S_rotated.Voigt_average(axis=1)
        Sr_1 = S_rotated.Reuss_average(axis=1)
        assert Sv_1.shape == (m,)
        assert Sr_1.shape == (m,)
        for i in range(m):
            np.testing.assert_array_almost_equal( Sv_1[i]._matrix , S_rotated[i, :].Voigt_average()._matrix)
            np.testing.assert_array_almost_equal( Sr_1[i]._matrix , S_rotated[i, :].Reuss_average()._matrix)

        np.testing.assert_array_almost_equal(Sv_0.Voigt_average()._matrix, S_rotated.Voigt_average()._matrix)
        np.testing.assert_array_almost_equal(Sv_1.Voigt_average()._matrix, S_rotated.Voigt_average()._matrix)
        np.testing.assert_array_almost_equal(Sr_0.Reuss_average()._matrix, S_rotated.Reuss_average()._matrix)
        np.testing.assert_array_almost_equal(Sr_1.Reuss_average()._matrix, S_rotated.Reuss_average()._matrix)

    def test_linear_invariants(self):
        lin_inv = S.linear_invariants()
        S_rotated = S * rotations
        for i, inv_rotated_i in enumerate(S_rotated.linear_invariants()):
            np.testing.assert_array_almost_equal(inv_rotated_i, lin_inv[i])

    def test_quadratic_invariants(self):
        lin_inv = S.quadratic_invariants()
        S_rotated = S * rotations
        for i, inv_rotated_i in enumerate(S_rotated.quadratic_invariants()):
            np.testing.assert_array_almost_equal(inv_rotated_i, lin_inv[i])

    def test_deprecated_path(self):
        expected_warn = ("The module 'Elasticipy.FourthOrderTensor' is deprecated and will be removed in a future "
                            "release. Please use 'Elasticipy.tensors.elasticity' instead.")
        with self.assertWarns(DeprecationWarning) as context:
            from Elasticipy.FourthOrderTensor import ComplianceTensor, StiffnessTensor
        self.assertEqual(str(context.warning), expected_warn)


class TestStiffnessConstructor(unittest.TestCase):
    def test_averages(self):
        """Check that the Voigt, Reuss and Hill averages are consistent with those provided by MP."""
        rel = 5e-2
        for index, row in data_base.iterrows():
            matrix = row['C']
            C = StiffnessTensor(matrix)
            C_rotated = C * rotations
            for method in ('voigt', 'reuss', 'hill', 'dummy'):
                if method == 'dummy':
                    with self.assertRaises(NotImplementedError) as context:
                        _ = C_rotated.average(method)
                    self.assertEqual(str(context.exception), 'Only Voigt, Reus, and Hill are implemented.')
                else:
                    Gavg = C.average(method).shear_modulus.mean(n_evals=10000)
                    assert row['G' + method] == approx(Gavg, rel=rel)
                    Gavg = C_rotated.average(method).shear_modulus.mean(n_evals=10000)
                    assert row['G' + method] == approx(Gavg, rel=rel)

    def test_isotropic(self):
        E, nu = 210000, 0.28
        C = StiffnessTensor.isotropic(E=E, nu=nu)
        G = C.shear_modulus.mean()
        K = C.bulk_modulus
        lame1 = C.lame1
        lame2 = C.lame2
        assert approx(G) == E / (1+nu) /2
        input_args1 = {'E': E, 'nu': nu, 'K':K, 'lame1':lame1, 'lame2':lame2}
        input_args2 = {'E': E, 'nu': nu, 'K':K, 'lame1':lame1, 'lame2':lame2}
        for k1 in input_args1.keys():
            del input_args2[k1]
            for k2 in input_args2.keys():
                kwargs = {k1: input_args1[k1], k2: input_args2[k2]}
                C_combo = StiffnessTensor.isotropic(**kwargs)
                np.testing.assert_array_almost_equal(C.matrix(), C_combo.matrix())

        with self.assertRaises(ValueError) as context:
            _ = StiffnessTensor.isotropic(E=E, nu=nu, G=G)
        self.assertEqual(str(context.exception), "Exactly two values are required among E, nu, G, K, lame1 and lame2.")
        with self.assertRaises(ValueError) as context:
            _ = StiffnessTensor.isotropic(G=G, lame2=G)
        self.assertEqual(str(context.exception), "G and lame2 cannot be provided together.")
        assert approx(C.Young_modulus.mean()) == E
        assert C.is_isotropic()
        assert C.is_cubic()
        assert C.is_tetragonal()

    def test_stiffness_cubic(self):
        """Check that all symmetries in stiffness are well taken into account for cubic case"""
        C = crystal_symmetry_tester('Cubic')
        shape = (50,100)
        C_rotated = C * (orix_rot.random(shape))
        is_cubic = C_rotated.is_cubic()
        assert is_cubic.shape == shape
        assert np.all(is_cubic)
        assert np.all(C.is_tetragonal())
        assert not np.any(C.is_isotropic())

    def test_stiffness_hexagonal(self):
        """Check that all symmetries in stiffness are well taken into account for hexagonal case"""
        C = crystal_symmetry_tester('Hexagonal')
        assert not C.is_tetragonal()
        assert not np.any(C.is_cubic())

    def test_stiffness_trigonal(self):
        """Check that all symmetries in stiffness are well taken into account for trigonal case"""
        C = crystal_symmetry_tester('Trigonal', variant='32')
        C_rotated = C * rotations
        assert not np.any(C_rotated.is_tetragonal())
        C = crystal_symmetry_tester('Trigonal', variant='-3')
        C_rotated = C * rotations
        assert not np.any(C_rotated.is_tetragonal())

    def test_stiffness_tetragonal(self):
        """Check that all symmetries in stiffness are well taken into account for tetragonal case"""
        C = crystal_symmetry_tester('Tetragonal', variant='-42m')
        C_rotated = C * rotations
        assert np.all(C_rotated.is_tetragonal())
        C = crystal_symmetry_tester('Tetragonal', variant='-4')
        C_rotated = C * rotations
        assert np.all(C_rotated.is_tetragonal())

    def test_stiffness_orthorhombic(self):
        """Check that all symmetries in stiffness are well taken into account for orthorhombic case"""
        crystal_symmetry_tester('Orthorhombic')

    def test_stiffness_monoclinic(self):
        """Check that all symmetries in stiffness are well taken into account for monoclinic case"""
        crystal_symmetry_tester('Monoclinic', variant='Diad || y')

    def test_compliance_cubic(self):
        """Check that all symmetries in compliance are well taken into account for cubic case"""
        crystal_symmetry_tester('Cubic', cls='compliance')

    def test_compliance_hexagonal(self):
        """Check that all symmetries in compliance are well taken into account for hexagonal case"""
        crystal_symmetry_tester('Hexagonal', cls='compliance')

    def test_compliance_trigonal(self):
        """Check that all symmetries in compliance are well taken into account for trigonal case"""
        crystal_symmetry_tester('Trigonal', variant='32', cls='compliance')
        crystal_symmetry_tester('Trigonal', variant='-3', cls='compliance')

    def test_compliance_tetragonal(self):
        """Check that all symmetries in compliance are well taken into account for tetragonal case"""
        crystal_symmetry_tester('Tetragonal', variant='-42m', cls='compliance')
        crystal_symmetry_tester('Tetragonal', variant='-4', cls='compliance')

    def test_compliance_orthorhombic(self):
        """Check that all symmetries in compliance are well taken into account for orthorhombic case"""
        crystal_symmetry_tester('Orthorhombic', cls='compliance')

    def test_compliance_monoclinic(self):
        """Check that all symmetries in compliance are well taken into account for monoclinic case"""
        crystal_symmetry_tester('Monoclinic', variant='Diad || y', cls='compliance')

    def test_young_modulus_eval(self):
        """Check that the Young modulus is given somehow given by the compliance tensor"""
        E = S.Young_modulus
        E_xyz = E.eval(np.eye(3))
        for i in range(3):
            self.assertEqual(E_xyz[i], 1/Smat[i, i])

    def test_young_modulus_stats(self):
        """Test statistics for Young moduli"""
        E = S.Young_modulus
        assert E.mean(method='exact') == approx(101.994)
        assert E.std() == approx(48.48264174566468)
        assert E.mean() == approx(101.9942123)

    def test_shear_modulus_eval(self):
        """Test shear moduli estimations"""
        G = S.shear_modulus
        u = [[0, 1, 0], [1, 0, 0], [1, 0, 0]]
        v = [[0, 0, 1], [0, 0, 1], [0, 1, 0]]
        G_xyz = G.eval(u, v)
        for i in range(3):
            self.assertEqual(G_xyz[i],  1/Smat[i+3, i+3])

    def test_Poisson_ratio_eval(self):
        """Test Poisson ration estimations"""
        nu = S.Poisson_ratio
        u = [[0, 1, 0], [1, 0, 0], [1, 0, 0]]
        v = [[0, 0, 1], [0, 0, 1], [0, 1, 0]]
        nu_xyz = nu.eval(u, v)
        nu_xyz_th = [0.625, 0.25, 0.375]
        for i in range(3):
            self.assertEqual(nu_xyz[i],  nu_xyz_th[i])

    def test_shear_modulus_mini_maxi(self):
        """Test shear min/max"""
        G = S.shear_modulus
        G_min, _ = G.min()
        G_max, _ = G.max()
        assert G_min == approx(8.47165)
        assert G_max == approx(83.3333)

    def test_unvoigt(self):
        """Test if the isotropic second-order tensor is well reconstructed"""
        lame1, lame2 = 1, 2
        C = StiffnessTensor.isotropic(lame1=lame1, lame2=lame2)
        C_full = C.full_tensor
        eye = np.eye(3)
        A = np.einsum('ij,kl->ijkl', eye, eye)
        B = np.einsum('ik,jl->ijkl', eye, eye)
        C = np.einsum('il,kj->ijkl', eye, eye)
        C_th = lame1 * A + lame2 * (B + C)
        np.testing.assert_almost_equal(C_th, C_full)

    def test_wave_velocity(self):
        """Test computation of wave velocities against simple isotropic case"""
        E = 210
        nu = 0.3
        rho = 7.8
        C = StiffnessTensor.isotropic(E=E, nu=nu)
        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        cp, cs_1, cs_2 = C.wave_velocity(rho)
        assert approx(cp.mean()) == np.sqrt(M / rho)
        G = C.shear_modulus.mean()
        assert approx(cs_2.mean()) == np.sqrt(G / rho)
        assert approx(cs_1.mean()) == np.sqrt(G / rho)

    def test_monoclinic(self):
        """Test constructor for monoclinic symmetry"""
        common_arguments = {'C11':11, 'C12':12, 'C13':13, 'C22':22, 'C23':23, 'C33':33, 'C44':44, 'C55':55, 'C66':66}

        # Check for Diad||y
        C = StiffnessTensor.monoclinic(**common_arguments, C16=16, C26=26, C36=36, C45=45)
        matrix = np.array([[11, 12, 13, 0, 0, 16],
                           [12, 22, 23, 0, 0, 26],
                           [13, 23, 33, 0, 0, 36],
                           [0,  0,  0, 44, 45, 0],
                           [0,  0,  0, 45, 55, 0],
                           [16, 26, 36, 0, 0, 66]], dtype=np.float64)
        np.testing.assert_array_almost_equal(matrix, C.matrix())

        # Check for Diad||z
        C = StiffnessTensor.monoclinic(**common_arguments, C15=15, C25=25, C35=35, C46=46)
        matrix = np.array([[11, 12, 13, 0, 15, 0],
                           [12, 22, 23, 0,  25, 0],
                           [13, 23, 33, 0,  35, 0],
                           [0,  0,  0,  44, 0, 46],
                           [15, 25, 35, 0,  55, 0],
                           [0,  0,  0,  46, 0, 66]], dtype=np.float64)
        np.testing.assert_array_almost_equal(matrix, C.matrix())

        # Check ambiguous cases
        expected_error = "'Ambiguous diad. Provide either C15, C25, C35 and C46; or C16, C26, C36 and C45'"
        with self.assertRaises(KeyError) as context:
            C = StiffnessTensor.monoclinic(**common_arguments,
                                          C15=15, C25=25, C35=35, C46=46, C16=16, C26=26, C36=36, C45=45)
        self.assertEqual(str(context.exception), expected_error)

        expected_error = ("'For monoclinic symmetry, one should provide either C15, C25, C35 and C46, "
                          "or C16, C26, C36 and C45.'")
        with self.assertRaises(KeyError) as context:
            C = StiffnessTensor.monoclinic(**common_arguments)
        self.assertEqual(str(context.exception), expected_error)

    def test_write_read_tensor(self):
        """Test export and import stiffness tensor to text file"""
        filename = 'C_tmp.txt'
        C = StiffnessTensor.isotropic(E=210, nu=0.3)
        C.save_to_txt(filename)
        C2 = StiffnessTensor.from_txt_file(filename)
        np.testing.assert_allclose(C2._matrix, C._matrix, atol=1e-2)

        # Now the same with phase name
        C = StiffnessTensor.isotropic(E=210, nu=0.3, phase_name='Steel')
        C.save_to_txt(filename)
        C2 = StiffnessTensor.from_txt_file(filename)
        np.testing.assert_allclose(C2._matrix, C._matrix, atol=1e-2)
        assert C2.phase_name == 'Steel'


    def test_equality(self):
        """Test == operator"""
        C1 = StiffnessTensor.isotropic(E=210000, nu=0.3)
        C2 = StiffnessTensor.isotropic(E=210000, nu=0.3)
        assert C1 == C2
        assert C1 == C2._matrix

    def test_add_sub(self):
        """Test addition and subtraction of tensors"""
        C1 = StiffnessTensor.isotropic(E=200, nu=0.3)
        C2 = StiffnessTensor.isotropic(E=100, nu=0.3)
        C_plus = C1 + C2
        assert C_plus.Young_modulus.mean() == approx(300)
        C_minus = C1 - C2
        assert C_minus.Young_modulus.mean() == approx(100)
        C_minus = C1 - C2.matrix()
        assert C_minus.Young_modulus.mean() == approx(100)
        C_plus_full = C1 + C2.full_tensor
        assert C_plus_full == C_plus

        with self.assertRaises(ValueError) as context:
            _ = C1 + C2.inv()
        self.assertEqual(str(context.exception), 'The two tensors to add must be of the same class.')

    def test_mul_rmul(self):
        C = StiffnessTensor.isotropic(E=200, nu=0.3)
        C1 = C * 2
        C2 = 2 * C
        assert C1 == C2

    def test_div(self):
        C = StiffnessTensor.isotropic(E=200, nu=0.3)
        Cdiv = C/2
        np.testing.assert_array_almost_equal(Cdiv._matrix, C._matrix / 2)

    def test_weighted_average(self):
        """Test averaging two phases"""
        E1 = 100
        E2 = 200
        C1 = StiffnessTensor.isotropic(E=E1, nu=0.3)
        C2 = StiffnessTensor.isotropic(E=E2, nu=0.3)
        Cv = StiffnessTensor.weighted_average((C1, C2), [0.5, 0.5], method='Voigt')
        Cr = StiffnessTensor.weighted_average((C1, C2), [0.5, 0.5], method='Reuss')
        Ch = StiffnessTensor.weighted_average((C1, C2), [0.5, 0.5], method='Hill')
        E_voigt = (E1 + E2) / 2
        E_reuss = 2 / (1/E1 + 1/E2)
        assert Cv.Young_modulus.mean() == approx(E_voigt)
        assert Cr.Young_modulus.mean() == approx(E_reuss)
        assert Ch.Young_modulus.mean() == approx(E_voigt/2 + E_reuss/2)

    def test_orthotropic(self):
        """Check if the engineering constants are well retrieved in the orthotropic case"""
        Ex, Ey, Ez = 100., 200., 300.
        nu_yx, nu_zy, nu_zx = 0.2, 0.3, 0.4
        G_xy, G_xz, G_yz = 50., 60., 70.
        C = StiffnessTensor.orthotropic(Ex=Ex, Ey=Ey, Ez=Ez, nu_yx=nu_yx, nu_zx=nu_zx, nu_zy=nu_zy,
                                        Gxy=G_xy, Gxz=G_xz, Gyz=G_yz)
        E = C.Young_modulus
        assert E.eval([1,0,0]) == approx(Ex)
        assert E.eval([0,1,0]) == approx(Ey)
        assert E.eval([0,0,1]) == approx(Ez)
        G = C.shear_modulus
        assert G.eval([1,0,0], [0,1,0]) == approx(G_xy)
        assert G.eval([1, 0, 0], [0, 0, 1]) == approx(G_xz)
        assert G.eval([0, 1, 0], [0, 0, 1]) == approx(G_yz)
        nu = C.Poisson_ratio
        assert nu.eval([0,1,0], [1,0,0]) == approx(nu_yx)
        assert nu.eval([0, 0, 1], [0, 1, 0]) == approx(nu_zy)
        assert nu.eval([0, 0, 1], [1, 0, 0]) == approx(nu_zx)

        # Now check with "inverted" Poisson ratio
        nu_xy = nu.eval([1, 0, 0], [0, 1, 0])
        nu_xz = nu.eval([1, 0, 0], [0, 0, 1])
        nu_yz = nu.eval([0, 1, 0], [0, 0, 1])
        C2 = StiffnessTensor.orthotropic(Ex=Ex, Ey=Ey, Ez=Ez, nu_xy=nu_xy, nu_xz=nu_xz, nu_yz=nu_yz,
                                        Gxy=G_xy, Gxz=G_xz, Gyz=G_yz)
        np.testing.assert_array_almost_equal(C._matrix, C2._matrix)

        # Now try passing both nu_yz and nu_zy
        with self.assertRaises(ValueError) as context:
            StiffnessTensor.orthotropic(Ex=Ex, Ey=Ey, Ez=Ez, nu_xy=nu_xy, nu_xz=nu_xz, nu_yz=nu_yz, nu_zy=nu_zy,
                                        Gxy=G_xy, Gxz=G_xz, Gyz=G_yz)
        self.assertEqual(str(context.exception), 'Either nu_yz or nu_zy must be provided')

        # Check that the Poisson ratios are consistent with tensile test
        stress = StressTensor.tensile([1,0,0],1)
        strain = C.inv()*stress
        assert strain.C[1,1]/strain.C[0,0] == approx(-nu_xy)
        assert strain.C[2,2]/strain.C[0,0] == approx(-nu_xz)
        stress = StressTensor.tensile([0,1,0],1)
        strain = C.inv()*stress
        assert strain.C[0,0]/strain.C[1,1] == approx(-nu_yx)
        assert strain.C[2,2]/strain.C[1,1] == approx(-nu_yz)

    def test_transverse_isotropic(self):
        """Check if the engineering constants are well retrieved in the transverse-isotropic case"""
        Ex, Ez = 100., 200.
        nu_yx, nu_zx = 0.2, 0.3
        Gxz = 80
        C = StiffnessTensor.transverse_isotropic(Ex=Ex, Ez=Ez, nu_yx=nu_yx, nu_zx=nu_zx, Gxz=Gxz)
        E = C.Young_modulus
        assert E.eval([1,0,0]) == approx(Ex)
        assert E.eval([0,1,0]) == approx(Ex)
        assert E.eval([0,0,1]) == approx(Ez)
        G = C.shear_modulus
        assert G.eval([1, 0, 0], [0, 0, 1]) == approx(Gxz)
        nu = C.Poisson_ratio
        assert nu.eval([0, 1, 0], [1, 0, 0]) == approx(nu_yx)
        assert nu.eval([0, 0, 1], [0, 1, 0]) == approx(nu_zx)
        assert nu.eval([0, 0, 1], [1, 0, 0]) == approx(nu_zx)

        # Try passing the opposite Poisson ratios
        nu_xy = nu.eval([1, 0, 0], [0, 1, 0])
        nu_xz = nu.eval([1, 0, 0], [0, 0, 1])
        C2 = StiffnessTensor.transverse_isotropic(Ex=Ex, Ez=Ez, nu_xy=nu_xy, nu_xz=nu_xz, Gxz=Gxz)
        np.testing.assert_array_almost_equal(C._matrix, C2._matrix)

        # Now try passing both nu_xy and nu_yx
        with self.assertRaises(ValueError) as context:
            StiffnessTensor.transverse_isotropic(Ex=Ex, Ez=Ez, nu_yx=nu_yx, nu_zx=nu_zx, Gxz=Gxz, nu_xy=nu_xy)
        self.assertEqual(str(context.exception), 'Either nu_xy or nu_yx must be provided')

    def test_straining_energy(self):
        """Test if the elastic energies are consistent."""
        matrix = np.random.random((3,3))
        stress = StressTensor(matrix + matrix.T)
        strain = S * stress
        e1 = stress.elastic_energy(strain)
        e2 = strain.elastic_energy(stress)
        np.testing.assert_approx_equal(e1, e2)


    def test_to_pymatgen(self):
        """Test exporting stiffness and compliance to pymatgen format"""
        C = S.inv()
        Cvrh = C.Hill_average()
        Cvrh_pymatgen = mg.ElasticTensor(Cvrh.full_tensor)
        C_pymatgen = mg.ElasticTensor(C.full_tensor)
        assert Cvrh_pymatgen.y_mod == approx(Cvrh.Young_modulus.mean()*1e9)
        assert C_pymatgen.g_vrh == approx(Cvrh.shear_modulus.mean())

        S_pymatgen = S.to_pymatgen()
        np.testing.assert_array_almost_equal(S_pymatgen.voigt, np.linalg.inv(C_pymatgen.voigt))

    @unittest.skipIf(sys.version_info < (3, 11), "Python < 3.11")
    def test_MaterialsProject(self):
        """Test import from the Materials Project"""
        # Try with cubic Cu
        C = StiffnessTensor.from_MP("mp-30")
        C_Cu = StiffnessTensor.cubic(C11=186, C12=134, C44=77)
        np.testing.assert_array_almost_equal(C._matrix, C_Cu._matrix)

        # Now try with a list of entries
        Cs = StiffnessTensor.from_MP(("mp-30", "mp-1048"))
        assert len(Cs) == 2
        np.testing.assert_array_almost_equal(Cs[0]._matrix, C_Cu._matrix)


    def test_getitem(self):
        """Test indexing of stiffness tensor"""
        S_rotated = S * rotations
        S1 = S_rotated[0]
        S2 = S * rotations[0]
        np.testing.assert_array_almost_equal(S1._matrix, S2._matrix)
        expected_error = 'A single tensor cannot be subindexed'
        with self.assertRaises(IndexError) as context:
            _ = S[0]
        self.assertEqual(str(context.exception), expected_error)

    def test_symmetry(self):
        S = np.random.random((6, 6))
        with self.assertRaises(ValueError) as context:
            _ = StiffnessTensor(S)
        self.assertEqual(str(context.exception), 'The input matrix must be symmetric')

    def test_positive_definite(self):
        S = np.array([
            [2, -1, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0],
            [0, -1, 2, -1, 0, 0],
            [0, 0, -1, 2, -1, 0],
            [0, 0, 0, -1, 2, -1],
            [0, 0, 0, 0, -1, 0]
        ])
        with self.assertRaises(ValueError) as context:
            _ = StiffnessTensor(S)
        eig_vals = np.linalg.eigvals(S * KelvinMapping().matrix)
        expected_error = 'The input matrix is not definite positive (eigenvalues: {})'.format(eig_vals)
        self.assertEqual(str(context.exception), expected_error)

    def test_Zener_universal_anisotropy(self):
        C11, C12, C44 = 173, 33, 18
        C = StiffnessTensor.cubic(C11=C11, C12=C12, C44=C44)
        Z = C.Zener_ratio()
        A = C.universal_anisotropy
        assert 6/5 * (Z**0.5 - Z**(-0.5))**2 == approx(A)
        C_cub_iso = StiffnessTensor.cubic(C11=C11, C12=C12, C44=(C11-C12)/2)
        assert C_cub_iso.Zener_ratio() == 1.0
        Ciso = StiffnessTensor.isotropic(E=210, nu=0.3)
        assert Ciso.Zener_ratio() == 1.0
        assert Ciso.universal_anisotropy < 1e-10
        Cmono = S.inv()
        with self.assertRaises(ValueError) as context:
            Cmono.Zener_ratio()
        self.assertEqual(str(context.exception), 'The tensor does not seem to have cubic symmetry within the given tolerance (0.0001)')
        C_rot = C*rotations[0]
        Zrot = C_rot.Zener_ratio()
        assert Zrot == approx(Z)

        C_rot = C * rotations
        assert C_rot.universal_anisotropy.shape == (len(rotations),)
        np.testing.assert_array_almost_equal(C_rot.universal_anisotropy, A)

    def test_orix(self):
        # Orix allows multidimensional arrays of rotations
        m, n= 5,6
        orix_rotations = orix_rot.random((m,n))
        C = S.inv()
        C_rotated = C * orix_rotations
        C_rotated_full = C_rotated.full_tensor
        for i in range(m):
            for j in range(n):
                inv_rotation = ~orix_rotations
                rot_mat = inv_rotation.to_matrix()[i,j]
                tensor_i = np.einsum('im,jn,ko,lp,mnop -> ijkl', rot_mat, rot_mat, rot_mat, rot_mat, C.full_tensor)
                np.testing.assert_array_almost_equal(C_rotated_full[i,j], tensor_i)

        # Check that the result is consistent with scipy.Rotation
        orix_rotations = orix_rotations.transpose().flatten()   # orix seems to use the opposite order from numpy for flattening
        C_rotated_flat = C_rotated.flatten()
        euler = orix_rotations.to_euler()
        scipy_rotations = Rotation.from_euler('ZXZ', euler)
        C_rotated_scipy = C * scipy_rotations
        C_rotated_full_scipy = C_rotated_scipy.full_tensor
        np.testing.assert_array_almost_equal(C_rotated_full_scipy, C_rotated_flat.full_tensor)


    def test_linear_compressibility(self):
        E, nu = 210, 0.3
        Ciso = StiffnessTensor.isotropic(E=E, nu=nu)
        beta = Ciso.linear_compressibility.mean()
        assert Ciso.bulk_modulus == approx(1 / (3*beta))

    def test_full_tensor_as_input(self):
        a = StiffnessTensor.isotropic(E=210, nu=0.3)
        b = StiffnessTensor(a.full_tensor)
        assert a == b

    def test_component(self):
        C = S.inv()
        Cmat = C._matrix / KelvinMapping().matrix
        assert C.C11 == Cmat[0, 0]
        assert C.C12 == Cmat[0, 1]
        assert C.C13 == Cmat[0, 2]
        assert C.C14 == Cmat[0, 3]
        assert C.C15 == Cmat[0, 4]
        assert C.C16 == Cmat[0, 5]
        assert C.C22 == Cmat[1, 1]
        assert C.C23 == Cmat[1, 2]
        assert C.C24 == Cmat[1, 3]
        assert C.C25 == Cmat[1, 4]
        assert C.C26 == Cmat[1, 5]
        assert C.C32 == Cmat[2, 1]
        assert C.C33 == Cmat[2, 2]
        assert C.C34 == Cmat[2, 3]
        assert C.C35 == Cmat[2, 4]
        assert C.C36 == Cmat[2, 5]
        assert C.C44 == Cmat[3, 3]
        assert C.C45 == Cmat[3, 4]
        assert C.C46 == Cmat[3, 5]
        assert C.C55 == Cmat[4, 4]
        assert C.C56 == Cmat[4, 5]
        assert C.C66 == Cmat[5, 5]
        docstring = StiffnessTensor.C12.__doc__
        assert docstring == 'Returns the (1,2) component of the Stiffness matrix.'

    def test_bulk_modulus(self):
        E, nu = 210, 0.3
        Ciso= StiffnessTensor.isotropic(E=E, nu=nu)
        assert Ciso.bulk_modulus == approx(E / (3 * (1-2 * nu)))

        b = S.bulk_modulus
        C = S.inv()
        C_rotated = C * rotations
        b_r = C_rotated.bulk_modulus
        assert np.allclose(b, b_r)

    def test_broadcasting(self):
        C11, C12, C44 = 173, 33, 18
        m,n,o = 9,7,5
        rot_0d = orix_rot.random()
        rot_1d = orix_rot.random(o)
        rot_2d = orix_rot.random((n, o))
        rot_3d = orix_rot.random((m, n, o))
        C_0d = StiffnessTensor.cubic(C11=C11, C12=C12, C44=C44)
        C_rotated_0d = C_0d * rot_0d
        C_rotated_1d = C_0d * rot_1d
        C_rotated_2d = C_0d * rot_2d
        C_rotated_3d = C_0d * rot_3d
        strain_0d = StrainTensor.rand()
        strain_1d = StrainTensor.rand((o,))
        strain_2d = StrainTensor.rand((n, o))
        strain_3d = StrainTensor.rand((m, n, o))

        stress = C_rotated_0d * strain_1d
        for i in range(o):
            assert stress[i] == C_rotated_0d * strain_1d[i]
        stress = C_rotated_1d * strain_0d
        for i in range(o):
            np.testing.assert_array_almost_equal(stress[i].matrix, (C_rotated_1d[i] * strain_0d).matrix)
        stress = C_rotated_1d * strain_1d
        for i in range(o):
            np.testing.assert_array_almost_equal(stress[i].matrix, (C_rotated_1d[i] * strain_1d[i]).matrix)
        stress = C_rotated_2d * strain_0d
        for i in range(n):
            for j in range(o):
                np.testing.assert_array_almost_equal(stress[i, j].matrix, (C_rotated_2d[i, j] * strain_0d).matrix)
        stress = C_rotated_2d * strain_1d
        for i in range(n):
            for j in range(o):
                np.testing.assert_array_almost_equal(stress[i,j].matrix, (C_rotated_2d[i,j] * strain_1d[j]).matrix)
        stress = C_rotated_2d * strain_2d
        for i in range(n):
            for j in range(o):
                np.testing.assert_array_almost_equal(stress[i,j].matrix, (C_rotated_2d[i,j] * strain_2d[i,j]).matrix)
        stress = C_rotated_3d * strain_0d
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    np.testing.assert_array_almost_equal(stress[i,j,k].matrix, (C_rotated_3d[i,j,k] * strain_0d).matrix)
        stress = C_rotated_3d * strain_1d
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    np.testing.assert_array_almost_equal(stress[i, j, k].matrix,
                                                         (C_rotated_3d[i, j, k] * strain_1d[k]).matrix)
        stress = C_rotated_3d * strain_2d
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    np.testing.assert_array_almost_equal(stress[i, j, k].matrix,
                                                         (C_rotated_3d[i, j, k] * strain_2d[j,k]).matrix)
        stress = C_rotated_3d * strain_3d
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    np.testing.assert_array_almost_equal(stress[i, j, k].matrix,
                                                         (C_rotated_3d[i, j, k] * strain_3d[i,j,k]).matrix)

    def test_transpose_array(self):
        m, n, o = 5, 6, 7
        rot_2d = orix_rot.random((m, n, o))
        C11, C12, C44 = 173, 33, 18
        C = StiffnessTensor.cubic(C11=C11, C12=C12, C44=C44)
        C_rotated = C * rot_2d
        C_rotated_T = C_rotated.transpose_array()
        assert C_rotated_T.shape == (o, n, m)
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    np.testing.assert_array_almost_equal(C_rotated_T[k,j,i].full_tensor, C_rotated[i,j,k].full_tensor)


    def test_eigenstiffness_eigencompliance(self):
        C11, C12, C44 = 22, 12, 44
        C = StiffnessTensor.cubic(C11=C11, C12=C12, C44=C44)
        eigen_stiffnesses = C.eig_stiffnesses
        eigen_stiffnesses_th = [C11 + 2 * C12, C11 - C12, 2 * C44] # 10.1111/j.1365-2478.2011.01049.x
        for e in eigen_stiffnesses:
            assert np.any(np.isclose(e, eigen_stiffnesses_th))
        eigen_strains = C.eig_strains
        for i, e in enumerate(eigen_strains.T):
            strain = StrainTensor.from_Kelvin(e)
            stress = C * strain
            np.testing.assert_array_almost_equal(stress.matrix, strain.matrix * eigen_stiffnesses[i])

        S = C.inv()
        eigen_compliances = S.eig_compliances
        eigen_compliances_th = 1 / np.array([C11 + 2 * C12, C11 - C12, 2 * C44])  # 10.1111/j.1365-2478.2011.01049.x
        for e in eigen_compliances:
            assert np.any(np.isclose(e, eigen_compliances_th))
        eigen_stresses = S.eig_stresses
        for i, e in enumerate(eigen_stresses.T):
            stress = StressTensor.from_Kelvin(e)
            strain = S * stress
            np.testing.assert_array_almost_equal(strain.matrix, stress.matrix * eigen_compliances[i])

        np.testing.assert_array_almost_equal(eigen_stiffnesses, np.sort(S.eig_stiffnesses))
        np.testing.assert_array_almost_equal(eigen_compliances, np.sort(C.eig_compliances))

    def test_to_from_Kelvin(self):
        C = S.inv()
        matrix = C.to_Kelvin()
        C2 = StiffnessTensor.from_Kelvin(matrix)
        np.testing.assert_array_almost_equal(C._matrix, C2._matrix)

    def test_ddot(self):
        # Check inverse
        C1 = StiffnessTensor.cubic(C11=22, C12=12, C44=44)
        C2 = StiffnessTensor.isotropic(E=210000, nu=0.3)
        np.testing.assert_almost_equal(C1.ddot(C1.inv()).full_tensor, StiffnessTensor.identity().full_tensor)

        # Check product between tensor arrays
        m = 5
        rot_1d = orix_rot.random(m)
        C1_rot = C1 * rot_1d
        C2_rot = C1 * rot_1d

        C1C2_rot = C1_rot * C2
        assert C1C2_rot.shape == (m,)
        for i in range(m):
            np.testing.assert_array_almost_equal(C1C2_rot[i].full_tensor, np.einsum('ijmn,nmkl->ijkl', C1_rot[i].full_tensor, C2.full_tensor))

        C1C2_rotrot = C1_rot * C2_rot
        assert C1C2_rot.shape == (m,)
        for i in range(m):
            np.testing.assert_array_almost_equal(C1C2_rotrot[i].full_tensor, np.einsum('ijmn,nmkl->ijkl', C1_rot[i].full_tensor,
                                                                        C2_rot[i].full_tensor))

        C1C2_rotrot_cross = C1_rot.ddot(C2_rot, mode='cross')
        assert C1C2_rotrot_cross.shape == (m, m)
        for i in range(m):
            for j in range(m):
                np.testing.assert_array_almost_equal(C1C2_rotrot_cross[i,j].full_tensor, np.einsum('ijmn,nmkl->ijkl', C1_rot[i].full_tensor,
                                                                           C2_rot[j].full_tensor))

    def test_full_tensor(self):
        C=S.inv()
        C_full = C.full_tensor
        C2 = StiffnessTensor(C_full)
        np.testing.assert_array_almost_equal(C._matrix, C2._matrix)
        np.testing.assert_array_almost_equal(C.full_tensor, C2.full_tensor)

    def test_voigt_reuss_axis(self):
        m,n = 5,6
        C=S.inv()
        orientations = orix_rot.random((m,n))
        C_rotated = C * orientations
        Cv_0 = C_rotated.Voigt_average(axis=0)
        Cr_0 = C_rotated.Reuss_average(axis=0)
        assert Cv_0.shape == (n,)
        assert Cr_0.shape == (n,)
        for i in range(n):
            assert Cv_0[i] == C_rotated[:, i].Voigt_average()
            assert Cr_0[i] == C_rotated[:, i].Reuss_average()
        Cv_1 = C_rotated.Voigt_average(axis=1)
        Cr_1 = C_rotated.Reuss_average(axis=1)
        assert Cv_1.shape == (m,)
        assert Cr_1.shape == (m,)
        for i in range(m):
            assert Cv_1[i] == C_rotated[i, :].Voigt_average()
            assert Cr_1[i] == C_rotated[i, :].Reuss_average()

        np.testing.assert_array_almost_equal(Cv_0.Voigt_average().matrix(), C_rotated.Voigt_average().matrix())
        np.testing.assert_array_almost_equal(Cv_1.Voigt_average().matrix(), C_rotated.Voigt_average().matrix())
        np.testing.assert_array_almost_equal(Cr_0.Reuss_average().matrix(), C_rotated.Reuss_average().matrix())
        np.testing.assert_array_almost_equal(Cr_1.Reuss_average().matrix(), C_rotated.Reuss_average().matrix())

    def test_linear_invariants(self):
        C = S.inv()
        lin_inv = C.linear_invariants()
        C_rotated = C * rotations
        for i, inv_rotated_i in enumerate(C_rotated.linear_invariants()):
            np.testing.assert_array_almost_equal(inv_rotated_i, lin_inv[i])

    def test_quadratic_invariants(self):
        C = S.inv()
        quad_inv = C.quadratic_invariants()
        C_rotated = C * rotations
        for i, inv_rotated_i in enumerate(C_rotated.quadratic_invariants()):
            np.testing.assert_array_almost_equal(inv_rotated_i, quad_inv[i])

if __name__ == '__main__':
    unittest.main()
