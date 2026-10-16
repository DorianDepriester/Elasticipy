import unittest
import numpy as np

from elasticipy.tensors.finite_strain import DeformationGradient
from elasticipy.hyperelasticity import NeoHooke, MooneyRivlin, Yeoh
from elasticipy.tensors.elasticity import StiffnessTensor
from elasticipy.tensors.stress_strain import StrainTensor, StressTensor

C = 100.
D = 1/500
nh_comp = NeoHooke(C, D=D)
stiff = StiffnessTensor.isotropic(G=2 * C, K=2 / D)
nh_incomp = NeoHooke(C)
yh = Yeoh(C)
yh_3 = Yeoh([39.8, -131.9, 82.8])

C_23 = np.array([[0, 0.087, 0.005],
                   [0.35, 0.012, 0.003]])
mr_23 = MooneyRivlin(C_23)

class TestNeoHooke(unittest.TestCase):
    def test_M_N(self):
        assert nh_comp.M == 1
        assert nh_incomp.M == 1
        assert nh_comp.N == 0

    def test_small_strain_comparison(self):
        eps = np.array([[11, 12, 13], [12, 22, 23], [13, 23, 33]]) * 1e-7
        F = DeformationGradient(eps) + DeformationGradient.eye()
        s_finite = nh_comp.stress_from_F(F)
        s_small = stiff * StrainTensor(eps)
        np.testing.assert_almost_equal(s_finite.matrix, s_small.matrix)

        eps = 1e-9
        F = DeformationGradient.eye() * (1 - eps)
        s_finite = nh_comp.stress_from_F(F)
        p = s_finite.hydrostatic_pressure()
        np.testing.assert_almost_equal(p / eps /3, 2/D, decimal=4)
        np.testing.assert_array_equal(s_finite.matrix, -np.eye(3)*p)

    def test_incompressible(self):
        assert not nh_incomp.is_compressible()
        assert nh_comp.is_compressible()

        F = DeformationGradient.eye() * 1.1
        with self.assertRaises(ValueError) as context:
            _ = nh_incomp.stress_from_F(F)
        expected_error = "For incompressible behaviour, the determinant of the gradient must be 1."
        self.assertEqual(str(context.exception), expected_error)

    def test_stress_free_state(self):
        F = DeformationGradient.eye()
        np.testing.assert_almost_equal(nh_comp.stress_from_F(F).matrix, np.zeros((3, 3)))
        lam = 1.5
        F = DeformationGradient(np.diag([lam, lam**-0.5, lam**-0.5]))
        np.testing.assert_almost_equal(
            nh_comp.stress_from_B(F.B).matrix.trace(), 0)

    def test_isochoric_uniaxial(self):
        lam = 1.4
        F = DeformationGradient.diag([lam, lam ** -0.5, lam ** -0.5])  # J = 1
        sigma = nh_comp.stress_from_F(F)
        B = np.diag([lam ** 2, lam ** -1, lam ** -1])
        sigma_ref = 2 * C * (B - np.trace(B) / 3 * np.eye(3))
        np.testing.assert_allclose(sigma.matrix, sigma_ref, atol=1e-5)

    def test_spherical(self):
        lam = 1.4
        F = DeformationGradient.eye() * lam
        sigma = nh_comp.stress_from_F(F)
        J = F.J
        expected = 2*J*(J-1) / D * np.eye(3) / J
        np.testing.assert_array_almost_equal(sigma.matrix, expected)

    def test_analytical_vs_derivative(self):
        F = DeformationGradient.diag([1.5, 1.6, 1.7])
        stress = nh_comp.stress_from_F(F)
        derivative = nh_comp.stress_from_derivative(F.B, h=1e-7)
        np.testing.assert_allclose(stress.matrix, derivative.matrix, atol=1e-6)

    def test_nh_vs_MooneyRivelin(self):
        mr = MooneyRivlin(C=[[0,0],[C,0]], D=D)
        F = DeformationGradient([[1, 0.1, 0.2],[0.3, 1.4, 0.5], [0.6, 0.7, 1.8]])
        stress_nh = nh_comp.stress_from_derivative(F.B, h=1e-6)
        stress_mr = mr.stress_from_F(F, h=1e-6)
        np.testing.assert_allclose(stress_nh.matrix, stress_mr.matrix, atol=1-3)

    def test_stress_from_F_array(self):
        mag = [1,1.5,2]
        F = DeformationGradient.isochoric_tensile([1,0,0],mag)
        sigma = nh_comp.stress_from_F(F)
        for i, magi in enumerate(mag):
            np.testing.assert_array_almost_equal(sigma[i].matrix, nh_comp.stress_from_F(F[i]).matrix)

    def test_fit(self):
        stretch = np.linspace(0, 1, 500)
        mat = [True, False]
        for i in range(2):
            for j in range(2):
                true_strain = mat[i]
                true_stress = mat[j]
                tensile_stress = nh_incomp.tensile_curve(stretch, true_stress=true_stress, true_strain=true_strain)
                nh_fit = NeoHooke.fit(stretch, tensile_stress, true_stress=true_stress, true_strain=true_strain)
                np.isclose(nh_fit.C, nh_incomp.C)

    def test_C_property(self):
        assert nh_comp.C == C

    def test_repr(self):
        assert nh_incomp.__repr__() == 'NeoHooke(C=100.)'
        assert nh_comp.__repr__() == 'NeoHooke(C=100., D=500.0)'

    def test_str(self):
        assert nh_incomp.__str__() == ('Generalized incompressible Mooney-Rivlin hyper-elastic model\n'
                                'Cij=\n'
                                '[[  0.]\n'
                                ' [100.]]')

        assert nh_comp.__str__() == ('Generalized Mooney-Rivlin hyper-elastic model\n'
                                'Cij=\n'
                                '[[  0.]\n'
                                ' [100.]]\n'
                                'D=[500.]')

    def test_setter(self):
        a = NeoHooke(1)
        a.C = 2
        np.testing.assert_array_equal(a._C, np.array([[0.], [2.]]))

        a.D = 1
        np.testing.assert_array_equal(a._D, np.array([1.]))

class TestYeoh(unittest.TestCase):
    def test_M_N(self):
        assert yh_3.M == 3
        assert yh_3.N == 0

    def test_shear(self):
        eps = 1e-3
        F = DeformationGradient.isochoric_tensile([1,0,0], eps)
        sigma = yh.stress_from_F(F, h=1e-4)
        s = sigma.C[0,0] - sigma.C[1,1]
        E = s/eps
        E_th = 6 * C
        assert np.isclose(E, E_th)

    def test_yeoh_vs_neoHooke(self):
        F= DeformationGradient.isochoric_tensile([1,0,0], 2)
        assert yh.potential_from_F(F) == nh_incomp.potential_from_F(F)

    def test_analytical_vs_derivative(self):
        F = DeformationGradient.isochoric_tensile([1,0,0], 1)
        stress = yh.stress_from_F(F)
        derivative = yh.stress_from_derivative(F.B, h=1e-7)
        np.testing.assert_allclose(stress.matrix, derivative.matrix, atol=1e-6)

    def test_fit(self):
        stretch = np.linspace(0,1, 100)
        mat = [True, False]
        for i in range(2):
            for j in range(2):
                true_strain = mat[i]
                true_stress = mat[j]
                tensile_stress = yh_3.tensile_curve(stretch, true_strain=true_strain, true_stress=true_stress)
                yh_fit = Yeoh.fit(stretch, tensile_stress, true_strain=true_strain, true_stress=true_stress)
                np.testing.assert_allclose(yh_fit.C, yh_3.C, atol=1e-3)

    def test_C_property(self):
        np.testing.assert_array_equal(yh.C, np.array(C))

    def test_setter(self):
        a = Yeoh([1, 2])
        Cnew = np.array([2, 3])
        a.C = Cnew
        np.testing.assert_array_equal(a._C, np.array([[0.], [2.], [3.]]))

        a.D = 1
        np.testing.assert_array_equal(a._D, np.array([1.]))

    def test_repr(self):
        assert yh.__repr__() == 'Yeoh(C=[100.])'

    def test_str(self):
        assert yh.__str__() == ('Generalized incompressible Mooney-Rivlin hyper-elastic model\n'
                                'Cij=\n'
                                '[[  0.]\n'
                                ' [100.]]')

class TestMooneyRivelin(unittest.TestCase):
    def test_fit(self):
        stretch = np.linspace(0, 1, 500)
        tensile_stress = mr_23.tensile_curve(stretch)
        mr_fit = MooneyRivlin.fit(stretch, tensile_stress, M=1, N=2)
        np.testing.assert_array_almost_equal(mr_fit.C, mr_23.C)

    def test_M_N(self):
        assert mr_23.M == 1
        assert mr_23.N == 2

    def test_repr(self):
        assert mr_23.__repr__() == 'MooneyRivlin(C=[[0.   , 0.087, 0.005],\n [0.35 , 0.012, 0.003]])'

    def test_str(self):
        assert mr_23.__str__() == ('Generalized incompressible Mooney-Rivlin hyper-elastic model\n'
                                   'Cij=\n'
                                   '[[0.    0.087 0.005]\n'
                                   ' [0.35  0.012 0.003]]')

    def test_setter(self):
        a = MooneyRivlin(np.zeros(2))
        Cnew = np.array([[0, 1],[1, 1]])
        a.C = Cnew
        np.testing.assert_array_equal(a._C, Cnew)

        a.D = 1
        np.testing.assert_array_equal(a._D, np.array([1.]))

if __name__ == '__main__':
    unittest.main()
