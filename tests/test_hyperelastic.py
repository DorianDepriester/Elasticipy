import unittest
import numpy as np
from elasticipy.tensors.finite_strain import DeformationGradient
from elasticipy.hyperelasticity import NeoHooke, MooneyRivlin, Yeoh
from elasticipy.tensors.elasticity import StiffnessTensor
from elasticipy.tensors.stress_strain import StrainTensor, StressTensor

C = 100.
D = 500.
nh_comp = NeoHooke(C, D=D)
nh_incomp = NeoHooke(C)
yh = Yeoh(C)

def compute_tensile_curve(model):
    stretch = np.linspace(0, 1, 100)
    F = DeformationGradient.isochoric_tensile([1, 0, 0], stretch)
    sigma_dev = model.stress_from_F(F)
    p = sigma_dev.C[1, 1]  # sigma_dev_22
    sigma = sigma_dev + StressTensor.pressure(p)
    return stretch, sigma.C[0, 0]

class TestNeoHooke(unittest.TestCase):
    def test_small_strain_compr(self):
        eps = np.array([[11, 12, 13], [12, 22, 23], [13, 23, 33]]) * 1e-7
        F = DeformationGradient(eps) + DeformationGradient.eye()
        s_finite = nh_comp.stress_from_F(F)
        stiff = StiffnessTensor.isotropic(G=2*C, K=2/D)
        s_small = stiff * StrainTensor(eps)
        np.testing.assert_almost_equal(s_finite.matrix, s_small.matrix)

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
        stretch, tensile_stress = compute_tensile_curve(nh_incomp)
        nh_fit = NeoHooke.fit(stretch, tensile_stress)
        assert nh_fit.C == nh_incomp.C

class TestYeoh(unittest.TestCase):
    def test_yeoh(self):
        F= DeformationGradient.isochoric_tensile([1,0,0], 2)
        assert yh.potential_from_F(F) == nh_incomp.potential_from_F(F)

if __name__ == '__main__':
    unittest.main()
