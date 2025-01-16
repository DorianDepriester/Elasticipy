import unittest
import numpy as np
from pytest import approx
import os
import pandas as pd

from Elasticipy.FourthOrderTensor import StiffnessTensor, ComplianceTensor
from scipy.spatial.transform import Rotation
from Elasticipy.FourthOrderTensor import _indices2str
from Elasticipy.CrystalSymmetries import SYMMETRIES
from Elasticipy.StressStrainTensors import StressTensor
from pymatgen.analysis.elasticity import elastic as mg

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
            kwargs[component_name] = matrix[tuple(indices)]
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
        nu_mean_th = [0.36325, 0.42424242, 0.3915]
        for i, average in enumerate(averages):
            assert approx(average.Young_modulus.eval([1,0,0]), rel=1e-4) == E_mean_th[i]
            assert approx(average.shear_modulus.eval([1,0,0],[0,1,0]), rel=1e-4) == G_mean_th[i]
            assert approx(average.Poisson_ratio.eval([1,0,0],[0,1,0]), rel=1e-4) == nu_mean_th[i]

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
        eig_vals = np.linalg.eigvals(S)
        expected_error = 'The input matrix is not definite positive (eigenvalues: {})'.format(eig_vals)
        self.assertEqual(str(context.exception), expected_error)


class TestStiffnessConstructor(unittest.TestCase):
    def test_averages(self):
        """Check that the Voigt, Reuss and Hill averages are consistent with those provided by MP."""
        rel = 5e-2
        for index, row in data_base.iterrows():
            matrix = row['C']
            symmetry = row['symmetry']
            C = StiffnessTensor(matrix, symmetry=symmetry)
            C_rotated = C * rotations
            for method in ('voigt', 'reuss', 'hill'):
                Gavg = C.average(method).shear_modulus.mean()
                assert row['G' + method] == approx(Gavg, rel=rel)
                Gavg = C_rotated.average(method).shear_modulus.mean()
                assert row['G' + method] == approx(Gavg, rel=rel)

    def test_stiffness_cubic(self):
        """Check that all symmetries in stiffness are well taken into account for cubic case"""
        crystal_symmetry_tester('Cubic')

    def test_stiffness_hexagonal(self):
        """Check that all symmetries in stiffness are well taken into account for hexagonal case"""
        crystal_symmetry_tester('Hexagonal')

    def test_stiffness_trigonal(self):
        """Check that all symmetries in stiffness are well taken into account for trigonal case"""
        crystal_symmetry_tester('Trigonal', variant='32')
        crystal_symmetry_tester('Trigonal', variant='-3')

    def test_stiffness_tetragonal(self):
        """Check that all symmetries in stiffness are well taken into account for tetragonal case"""
        crystal_symmetry_tester('Tetragonal', variant='-42m')
        crystal_symmetry_tester('Tetragonal', variant='-4')

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
        C = StiffnessTensor.fromCrystalSymmetry(C11=lame1 + 2 * lame2,
                                                C12=lame1, symmetry='isotropic')
        C_full = C.full_tensor()
        eye = np.eye(3)
        A = np.einsum('ij,kl->ijkl', eye, eye)
        B = np.einsum('ik,jl->ijkl', eye, eye)
        C = np.einsum('il,kj->ijkl', eye, eye)
        C_th = lame1 * A + lame2 * (B + C)
        np.testing.assert_almost_equal(C_th, C_full)

    def test_isotropic(self):
        """Test if we correctly retrieve engineering constants in the isotropic case"""
        E = 210000
        nu = 0.28
        C = StiffnessTensor.isotropic(E=E, nu=nu)
        G = C.shear_modulus.mean()
        assert approx(G) == E / (1+nu) /2
        C = StiffnessTensor.isotropic(E=E, lame2=G)
        assert approx(C.Poisson_ratio.mean()) == nu
        C = StiffnessTensor.isotropic(lame2=G, nu=nu)
        assert approx(C.Young_modulus.mean()) == E

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

    def test_len(self):
        """Test len(ght) operator"""
        C = StiffnessTensor.isotropic(E=210000, nu=0.3)
        assert len(C) == 1
        assert len(C * rotations[0]) == 1
        assert len(C * rotations) == 10000

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
        """Test export and import stiffness tensor to text file"""
        C = StiffnessTensor.isotropic(E=210, nu=0.3)
        filename = 'C_tmp.txt'
        C.save_to_txt(filename)
        C2 = StiffnessTensor.from_txt_file(filename)
        np.testing.assert_allclose(C2.matrix, C.matrix, atol=1e-2)

    def test_equality(self):
        """Test == operator"""
        C1 = StiffnessTensor.isotropic(E=210000, nu=0.3)
        C2 = StiffnessTensor.isotropic(E=210000, nu=0.3)
        assert C1 == C2
        assert C1 == C2.matrix

    def test_add_sub(self):
        """Test addition and subtraction of tensors"""
        C1 = StiffnessTensor.isotropic(E=200, nu=0.3)
        C2 = StiffnessTensor.isotropic(E=100, nu=0.3)
        C_plus = C1 + C2
        assert C_plus.Young_modulus.mean() == approx(300)
        C_minus = C1 - C2
        assert C_minus.Young_modulus.mean() == approx(100)
        C_plus_full = C1 + C2.full_tensor()
        assert C_plus_full == C_plus

    def test_repr(self):
        """Test printing out the tensor"""
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
        assert nu.eval([0,1,0], [1,0,0]) == approx(nu_yx)
        assert nu.eval([0, 0, 1], [0, 1, 0]) == approx(nu_zx)
        assert nu.eval([0, 0, 1], [1, 0, 0]) == approx(nu_zx)

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
        Cvrh_pymatgen = mg.ElasticTensor(Cvrh.full_tensor())
        C_pymatgen = mg.ElasticTensor(C.full_tensor())
        assert Cvrh_pymatgen.y_mod == approx(Cvrh.Young_modulus.mean()*1e9)
        assert C_pymatgen.g_vrh == approx(Cvrh.shear_modulus.mean())

        S_pymatgen = S.to_pymatgen()
        np.testing.assert_array_almost_equal(S_pymatgen.voigt, np.linalg.inv(C_pymatgen.voigt))

    def test_MaterialsProject(self):
        """Test import from the Materials Project"""
        # Try with cubic Cu
        C = StiffnessTensor.from_MP("mp-30")
        C_Cu = StiffnessTensor.cubic(C11=186, C12=134, C44=77)
        np.testing.assert_array_almost_equal(C.matrix, C_Cu.matrix)

    def test_getitem(self):
        """Test indexing of stiffness tensor"""
        S_rotated = S * rotations
        S1 = S_rotated[0]
        S2 = S * rotations[0]
        np.testing.assert_array_almost_equal(S1.matrix, S2.matrix)
        expected_error = 'The tensor has no orientation, therefore it cannot be indexed.'
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
        eig_vals = np.linalg.eigvals(S)
        expected_error = 'The input matrix is not definite positive (eigenvalues: {})'.format(eig_vals)
        self.assertEqual(str(context.exception), expected_error)

    def test_universal_anisotropy(self):
        C11, C12, C44 = 173, 33, 18
        C = StiffnessTensor.cubic(C11=C11, C12=C12, C44=C44)
        Z = 2 * C44 / (C11 - C12)
        assert 6/5 * (Z**0.5 - Z**(-0.5))**2 == approx(C.universal_anisotropy)

if __name__ == '__main__':
    unittest.main()
