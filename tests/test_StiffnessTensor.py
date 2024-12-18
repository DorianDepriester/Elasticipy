import unittest
import numpy as np
from pytest import approx
import os
import pandas as pd
from Elasticipy.FourthOrderTensor import StiffnessTensor, ComplianceTensor
from scipy.spatial.transform import Rotation
from Elasticipy.FourthOrderTensor import _indices2str
from Elasticipy.CrystalSymmetries import SYMMETRIES


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
        else:
            class_constructor = ComplianceTensor
            matrix = np.linalg.inv(matrix)*1000
        kwargs = dict()
        for indices in required_fields:
            component_name = 'C' + _indices2str(indices)
            kwargs[component_name] = matrix[*indices]
        constructor = getattr(class_constructor, symmetry_name.lower())
        C = constructor(**kwargs)
        assert np.all(C.matrix == approx(matrix, rel=0.5))


class TestComplianceTensor(unittest.TestCase):
    def test_young_modulus_eval(self):
        E = S.Young_modulus
        E_xyz = E.eval(np.eye(3))
        for i in range(3):
            self.assertEqual(E_xyz[i], 1/Smat[i, i])

    def test_young_modulus_stats(self):
        E = S.Young_modulus
        assert E.mean() == approx(101.994)
        assert E.std() == approx(48.48065)

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
        C = StiffnessTensor.fromCrystalSymmetry(C11=lame1 + 2 * lame2,
                                                C12=lame1, symmetry='isotropic')
        C_full = C.full_tensor()
        eye = np.eye(3)
        A = np.einsum('ij,kl->ijkl', eye, eye)
        B = np.einsum('ik,jl->ijkl', eye, eye)
        C = np.einsum('il,kj->ijkl', eye, eye)
        C_th = lame1 * A + lame2 * (B + C)
        np.testing.assert_almost_equal(C_th, C_full)

    def test_averages(self):
        averages = [S.Voigt_average(), S.Reuss_average(), S.Hill_average()]
        E_mean_th = [151.738, 75.76, 114.45]
        G_mean_th = [55.653, 26.596, 41.124]
        nu_mean_th = [0.36325, 0.4242, 0.3915]
        for i, average in enumerate(averages):
            assert approx(average.Young_modulus.mean(), rel=1e-4) == E_mean_th[i]
            assert approx(average.shear_modulus.mean(), rel=1e-4) == G_mean_th[i]
            assert approx(average.Poisson_ratio.mean(), rel=1e-4) == nu_mean_th[i]

    def test_isotropic(self, E=210000, nu=0.28):
        C = StiffnessTensor.isotropic(E=E, nu=nu)
        G = C.shear_modulus.mean()
        assert approx(G) == E / (1+nu) /2
        C = StiffnessTensor.isotropic(E=E, lame2=G)
        assert approx(C.Poisson_ratio.mean()) == nu
        C = StiffnessTensor.isotropic(lame2=G, nu=nu)
        assert approx(C.Young_modulus.mean()) == E

    def test_wave_velocity(self, E=210, nu=0.3, rho=7.8):
        C = StiffnessTensor.isotropic(E=E, nu=nu)
        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        cp, cs_1, cs_2 = C.wave_velocity(rho)
        assert approx(cp.mean()) == np.sqrt(M / rho)
        G = C.shear_modulus.mean()
        assert approx(cs_2.mean()) == np.sqrt(G / rho)
        assert approx(cs_1.mean()) == np.sqrt(G / rho)


class TestStiffnessConstructor(unittest.TestCase):
    def test_averages(self):
        rel = 5e-2
        for index, row in data_base.iterrows():
            matrix = row['C']
            symmetry = row['symmetry']
            C = StiffnessTensor(matrix, symmetry=symmetry)
            Gvoigt = C.Voigt_average().shear_modulus.mean()
            Greuss = C.Reuss_average().shear_modulus.mean()
            Gvrh = C.Hill_average().shear_modulus.mean()
            assert row['Gvoigt'] == approx(Gvoigt, rel=rel)
            assert row['Greuss'] == approx(Greuss, rel=rel)
            assert row['Gvrh'] == approx(Gvrh, rel=rel)

            C_rotated = C * rotations
            Gvoigt = C_rotated.Voigt_average().shear_modulus.mean()
            Greuss = C_rotated.Reuss_average().shear_modulus.mean()
            Gvrh = C_rotated.Hill_average().shear_modulus.mean()
            assert row['Gvoigt'] == approx(Gvoigt, rel=rel)
            assert row['Greuss'] == approx(Greuss, rel=rel)
            assert row['Gvrh'] == approx(Gvrh, rel=rel)

    def test_stiffness_cubic(self):
        crystal_symmetry_tester('Cubic')

    def test_stiffness_hexagonal(self):
        crystal_symmetry_tester('Hexagonal')

    def test_stiffness_trigonal(self):
        crystal_symmetry_tester('Trigonal', variant='32')
        crystal_symmetry_tester('Trigonal', variant='-3')

    def test_stiffness_tetragonal(self):
        crystal_symmetry_tester('Tetragonal', variant='-42m')
        crystal_symmetry_tester('Tetragonal', variant='-4')

    def test_stiffness_orthorhombic(self):
        crystal_symmetry_tester('Orthorhombic')

    def test_stiffness_monoclinic(self):
        crystal_symmetry_tester('Monoclinic', variant='Diad || y')

    def test_compliance_cubic(self):
        crystal_symmetry_tester('Cubic', cls='compliance')

    def test_compliance_hexagonal(self):
        crystal_symmetry_tester('Hexagonal', cls='compliance')

    def test_compliance_trigonal(self):
        crystal_symmetry_tester('Trigonal', variant='32', cls='compliance')
        crystal_symmetry_tester('Trigonal', variant='-3', cls='compliance')

    def test_compliance_tetragonal(self):
        crystal_symmetry_tester('Tetragonal', variant='-42m', cls='compliance')
        crystal_symmetry_tester('Tetragonal', variant='-4', cls='compliance')

    def test_compliance_orthorhombic(self):
        crystal_symmetry_tester('Orthorhombic', cls='compliance')

    def test_compliance_monoclinic(self):
        crystal_symmetry_tester('Monoclinic', variant='Diad || y', cls='compliance')

    def test_young_modulus_eval(self):
        E = S.Young_modulus
        E_xyz = E.eval(np.eye(3))
        for i in range(3):
            self.assertEqual(E_xyz[i], 1/Smat[i, i])

    def test_young_modulus_stats(self):
        E = S.Young_modulus
        assert E.mean() == approx(101.994)
        assert E.std() == approx(48.48065)

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
        C = StiffnessTensor.fromCrystalSymmetry(C11=lame1 + 2 * lame2,
                                                C12=lame1, symmetry='isotropic')
        C_full = C.full_tensor()
        eye = np.eye(3)
        A = np.einsum('ij,kl->ijkl', eye, eye)
        B = np.einsum('ik,jl->ijkl', eye, eye)
        C = np.einsum('il,kj->ijkl', eye, eye)
        C_th = lame1 * A + lame2 * (B + C)
        np.testing.assert_almost_equal(C_th, C_full)

    def test_isotropic(self, E=210000, nu=0.28):
        C = StiffnessTensor.isotropic(E=E, nu=nu)
        G = C.shear_modulus.mean()
        assert approx(G) == E / (1+nu) /2
        C = StiffnessTensor.isotropic(E=E, lame2=G)
        assert approx(C.Poisson_ratio.mean()) == nu
        C = StiffnessTensor.isotropic(lame2=G, nu=nu)
        assert approx(C.Young_modulus.mean()) == E

    def test_wave_velocity(self, E=210, nu=0.3, rho=7.8):
        C = StiffnessTensor.isotropic(E=E, nu=nu)
        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        cp, cs_1, cs_2 = C.wave_velocity(rho)
        assert approx(cp.mean()) == np.sqrt(M / rho)
        G = C.shear_modulus.mean()
        assert approx(cs_2.mean()) == np.sqrt(G / rho)
        assert approx(cs_1.mean()) == np.sqrt(G / rho)

    def test_len(self):
        C = StiffnessTensor.isotropic(E=210000, nu=0.3)
        assert len(C) == 1
        assert len(C * rotations[0]) == 1
        assert len(C * rotations) == 10000

    def test_monoclinic(self):
        common_arguments = {'C11':11, 'C12':12, 'C13':13, 'C22':22, 'C23':23, 'C33':33, 'C44':44, 'C55':55, 'C66':66}

        # Check for Diad||y
        C = StiffnessTensor.monoclinic(**common_arguments, C16=16, C26=26, C36=36, C45=45)
        matrix = np.array([[11, 12, 13, 0, 0, 16],
                           [12, 22, 23, 0, 0, 26],
                           [13, 23, 33, 0, 0, 36],
                           [0,  0,  0, 44, 45, 0],
                           [0,  0,  0, 45, 55, 0],
                           [16, 26, 36, 0, 0, 66]], dtype=np.float64)
        np.testing.assert_array_equal(matrix, C.matrix)

        # Check for Diad||z
        C = StiffnessTensor.monoclinic(**common_arguments, C15=15, C25=25, C35=35, C46=46)
        matrix = np.array([[11, 12, 13, 0, 15, 0],
                           [12, 22, 23, 0,  25, 0],
                           [13, 23, 33, 0,  35, 0],
                           [0,  0,  0,  44, 0, 46],
                           [15, 25, 35, 0,  55, 0],
                           [0,  0,  0,  46, 0, 66]], dtype=np.float64)
        np.testing.assert_array_equal(matrix, C.matrix)

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
        C = StiffnessTensor.isotropic(E=210, nu=0.3)
        filename = 'C_tmp.txt'
        C.save_to_txt(filename)
        C2 = StiffnessTensor.from_txt_file(filename)
        np.testing.assert_allclose(C2.matrix, C.matrix, atol=1e-2)

    def test_equality(self):
        C1 = StiffnessTensor.isotropic(E=210000, nu=0.3)
        C2 = StiffnessTensor.isotropic(E=210000, nu=0.3)
        assert C1 == C2
        assert C1 == C2.matrix

    def test_add_sub(self):
        C1 = StiffnessTensor.isotropic(E=200, nu=0.3)
        C2 = StiffnessTensor.isotropic(E=100, nu=0.3)
        C_plus = C1 + C2
        assert C_plus.Young_modulus.mean() == approx(300)
        C_minus = C1 - C2
        assert C_minus.Young_modulus.mean() == approx(100)
        C_plus_full = C1 + C2.full_tensor()
        assert C_plus_full == C_plus

    def test_repr(self):
        C = StiffnessTensor.isotropic(E=210000, nu=0.3)
        str = C.__repr__()
        assert str == ('Stiffness tensor (in Voigt notation):\n'
                       '[[282692.30769231 121153.84615385 121153.84615385      0.\n'
                       '       0.              0.        ]\n'
                       ' [121153.84615385 282692.30769231 121153.84615385      0.\n'
                       '       0.              0.        ]\n'
                       ' [121153.84615385 121153.84615385 282692.30769231      0.\n'
                       '       0.              0.        ]\n'
                       ' [     0.              0.              0.          80769.23076923\n'
                       '       0.              0.        ]\n'
                       ' [     0.              0.              0.              0.\n'
                       '   80769.23076923      0.        ]\n'
                       ' [     0.              0.              0.              0.\n'
                       '       0.          80769.23076923]]\nSymmetry: isotropic')

    def test_weighted_average(self):
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

if __name__ == '__main__':
    unittest.main()
