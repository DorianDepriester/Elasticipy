import unittest
from pytest import approx
import pandas as pd
import numpy as np

from Elasticipy.FourthOrderTensor import StiffnessTensor, ComplianceTensor
from scipy.spatial.transform import Rotation
from Elasticipy.FourthOrderTensor import _indices2str
from Elasticipy.CrystalSymmetries import SYMMETRIES

data_base = pd.read_json('MaterialsProject.json')
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

if __name__ == '__main__':
    unittest.main()
