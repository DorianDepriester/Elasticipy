import unittest
from matplotlib import pyplot as plt
import numpy as np

from Elasticipy.FourthOrderTensor import StiffnessTensor
from pytest import approx


C = StiffnessTensor.cubic(C11=186, C12=134, C44=77)
E = C.Young_modulus # SphericalFunction
E_mean = 126.28067650635076
E_std = 31.58751357560234
G = C.shear_modulus # HypersphericalFunction
G_mean = 47.07147379585229
G_std = 14.14600864639266
SEED = 123  # Used for Monte Carlo integrations (e.g. for G.mean())


class TestSphericalFunction(unittest.TestCase):
    def test_plot3D(self):
        fig = plt.figure()
        _, ax = E.plot3D(fig=fig)
        np.testing.assert_allclose(ax.xaxis.v_interval, [-174.66981, 175.43156])
        np.testing.assert_allclose(ax.yaxis.v_interval, [-175.05069, 175.05069])
        np.testing.assert_allclose(ax.zaxis.v_interval, [-131.28802, 131.28802])

    def test_plot_xyz_section(self):
        fig = plt.figure()
        _, axs = E.plot_xyz_sections(fig=fig)
        assert axs[0].title._text == 'X-Y plane'
        assert axs[1].title._text == 'X-Z plane'
        assert axs[2].title._text == 'Y-Z plane'

    def test_plot_as_pole_figure(self):
        _, ax = E.plot_as_pole_figure(show=False)
        np.testing.assert_allclose(ax.dataLim.intervalx, [-0.01578689775673263, 6.298972204936319])
        np.testing.assert_allclose(ax.dataLim.intervaly, [-0.0160285339468867, 1.5868248607417832])

    def test_add_sub_mult(self):
        E_plus = E + E
        E_min = E - E
        E_mult = 2 * E
        E_plus_one = E + 1
        assert E_plus.mean() == approx(2 * E_mean, rel=1e-3)
        assert E_min.mean() == approx(0)
        assert E_mult.mean() == approx(2 * E_mean, rel=1e-3)
        assert E_plus_one.mean() == approx(E_mean + 1, rel=1e-3)

    def test_mean_std(self):
        assert E_mean == approx(E.mean(method='Monte Carlo', n_evals=10000), rel=1e-2)
        assert E_std == approx(E.std(method='Monte Carlo', n_evals=10000), rel=1e-2)


class TestHyperSphericalFunction(unittest.TestCase):
    def test_plot3D(self):
        fig = plt.figure()
        _, ax = G.plot3D(fig=fig)
        np.testing.assert_allclose(ax.xaxis.v_interval, [-86.363067,  87.737754])
        np.testing.assert_allclose(ax.yaxis.v_interval, [-87.435894,  87.435894])
        np.testing.assert_allclose(ax.zaxis.v_interval, [-80.208333,  80.208333])

    def test_plot_xyz_section(self):
        fig = plt.figure()
        _, axs = G.plot_xyz_sections(fig=fig)
        assert axs[0].title._text == 'X-Y plane'
        assert axs[1].title._text == 'X-Z plane'
        assert axs[2].title._text == 'Y-Z plane'

    def test_plot_as_pole_figure(self):
        _, ax = G.plot_as_pole_figure(show=False)
        np.testing.assert_allclose(ax.dataLim.intervalx, [-0.01578689775673263, 6.298972204936319])
        np.testing.assert_allclose(ax.dataLim.intervaly, [-0.0160285339468867, 1.5868248607417832])

    def test_add_sub_mult(self):
        G_plus = G + G
        G_min = G - G
        G_mult = 2 * G
        assert G_plus.mean(seed=SEED, n_evals=10000) == approx(2 * G_mean, rel=5e-3)
        assert G_min.mean(seed=SEED, n_evals=10000) == approx(0)
        assert G_mult.mean(seed=SEED, n_evals=10000) == approx(2 * G_mean, rel=5e-3)

    def test_mean_std(self):
        assert G_mean == approx(G.mean(seed=SEED, n_evals=10000), rel=5e-3)
        assert G_std == approx(G.std(seed=SEED, n_evals=10000), rel=1e-2)


if __name__ == '__main__':
    unittest.main()