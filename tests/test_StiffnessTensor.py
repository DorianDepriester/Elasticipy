import unittest
from FourthOrderTensor import ComplianceTensor, tensorFromCrystalSymmetry
import numpy as np
from pytest import approx

Smat = np.array([[8, -3, -2, 0, 14, 0],
                 [-3, 8, -5, 0, -8, 0],
                 [-2, -5, 10, 0, 0, 0],
                 [0, 0, 0, 12, 0, 0],
                 [14, -8, 0, 0, 116, 0],
                 [0, 0, 0, 0, 0, 12]])/1000
S = ComplianceTensor(Smat)


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
        C = tensorFromCrystalSymmetry(C11=lame1 + 2 * lame2,
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


if __name__ == '__main__':
    unittest.main()
