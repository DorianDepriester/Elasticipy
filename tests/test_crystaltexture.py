import unittest

from orix.quaternion import Orientation

from elasticipy.crystal_texture import DiscreteTexture, FibreTexture, CompositeTexture, UniformTexture
from orix.vector import Miller, Vector3d
from orix.crystal_map import Phase
import numpy as np
from elasticipy.tensors.elasticity import StiffnessTensor
from pytest import approx
from scipy.stats import kstest

PHASE = Phase(point_group="m-3m")
C = StiffnessTensor.monoclinic(phase_name='TiNi',
                               C11=231, C12=127, C13=104,
                               C22=240, C23=131, C33=175,
                               C44=81, C55=11, C66=85,
                               C15=-18, C25=1, C35=-3, C46=3)

def orientation_checker(texture, hkl, uvw):
    orientation = texture.orientation
    m_hkl = Miller(uvw=hkl, phase=PHASE)
    x = ~orientation * m_hkl.symmetrise(unique=True)
    u_hkl = x.data
    cos_hkl = np.dot(u_hkl, [0, 0, 1]) / np.linalg.norm(u_hkl, axis=1)
    assert np.any(np.isclose(cos_hkl, 1))

    m_uvw = Miller(uvw=uvw, phase=PHASE)
    x = ~orientation * m_uvw.symmetrise(unique=True)
    u_uvw = x.data
    cos_uvw = np.dot(u_uvw, [1, 0, 0]) / np.linalg.norm(u_uvw, axis=1)
    assert np.any(np.isclose(cos_uvw, 1))


class TestUniformTexture(unittest.TestCase):
    def test_uniform(self):
        t = UniformTexture()
        assert C * t == C.Voigt_average()
        S = C.inv()
        assert S * t == S.Reuss_average()

    def test_uniform_random(self):
        t = UniformTexture()
        o = t.sample(10000, seed=132)
        v = Vector3d(~o * Vector3d([0,0,1]))
        ks1 = kstest(v.data[:, 2], 'uniform', args=(-1, 2))
        ks2 = kstest(v.azimuth, 'uniform', args=(0, 2*np.pi))
        assert 0.05 < ks1[1] < 0.95 # Use p-value
        assert 0.05 < ks2[1] < 0.95  # Use p-value

class TestDiscreteTexture(unittest.TestCase):
    def test_cube(self):
        t = DiscreteTexture.cube()
        orientation_checker(t, [0,0,1], [1,0,0])

    def test_Goss(self):
        t = DiscreteTexture.Goss()
        orientation_checker(t, [1,1,0], [1,0,0])

    def test_Brass(self):
        t = DiscreteTexture.brass()
        orientation_checker(t, [1,1,0], [1,1,2])

    def test_GossBrass(self):
        t = DiscreteTexture.GossBrass()
        orientation_checker(t, [1,1,0], [1,1,5])

    def test_Copper(self):
        t = DiscreteTexture.copper()
        orientation_checker(t, [1,1,2], [1,1,1])

    def test_A(self):
        t = DiscreteTexture.A()
        orientation_checker(t, [1,1,0], [1,1,1])

    def test_P(self):
        t = DiscreteTexture.P()
        orientation_checker(t, [0,1,1], [2,1,1])

    def test_CuT(self):
        t = DiscreteTexture.CuT()
        orientation_checker(t, [5,5,2], [1,1,5])

    def test_S(self):
        t = DiscreteTexture.S()
        orientation_checker(t, [1,2,3], [6,3,4])

    def test_repr(self):
        t = DiscreteTexture.Goss()
        assert t.__repr__() == 'Crystallographic texture\nφ1=0.00°, ϕ=45.00°, φ2=0.00°'

    def test_mult_stiffness(self):
        t = DiscreteTexture.S()
        Crot = C * t
        assert Crot == C * t.orientation

    def test_mult_float(self):
        t = DiscreteTexture.Goss()
        tm = t * 0.5
        assert isinstance(tm, DiscreteTexture)
        assert t.orientation == tm.orientation
        assert tm.weight == 0.5

        tm2 = 0.5 * t
        assert isinstance(tm2, DiscreteTexture)
        assert t.orientation == tm2.orientation

    def test_discrete_random(self):
        t = DiscreteTexture.Goss()
        o = t.sample(10)
        for oi in o:
            assert oi == t.orientation

class TestFibreTexture(unittest.TestCase):
    def test_from_Euler(self):
        t = FibreTexture.from_Euler(phi1=0, Phi=10)
        assert t.__repr__() == 'Fibre texture\nφ1= 0°, ϕ= 10°'

        t = FibreTexture.from_Euler(phi1=0, phi2=10)
        assert t.__repr__() == 'Fibre texture\nφ1= 0°, φ2= 10°'

        t = FibreTexture.from_Euler(Phi=0, phi2=10)
        assert t.__repr__() == 'Fibre texture\nϕ= 0°, φ2= 10°'

        t = FibreTexture.from_Euler(Phi=0, phi2=10 * np.pi / 180, degrees=False)
        assert t.__repr__() == 'Fibre texture\nϕ= 0.0°, φ2= 10.0°'

    def test_from_Euler_axis(self):
        t = FibreTexture.from_Euler(Phi=0, phi2=0)
        assert t.axis.dot(Vector3d([0,0,1])) == 1. or t.axis.dot(Vector3d([0,0,1])) == -1.

        t = FibreTexture.from_Euler(phi1=0, phi2=0)
        assert t.axis.dot(Vector3d([0,0,1])) == 1. or t.axis.dot(Vector3d([1,0,0])) == -1.

    def test_from_Miller_axis(self):
        m = Miller(uvw=[1, 0, 0], phase=PHASE)
        texture = FibreTexture.from_Miller_axis(m, [0, 0, 1])
        assert texture.__repr__() == 'Fibre texture\n<1. 0. 0.> || [0, 0, 1]'

    def test_mult_stiffness(self):
        # Check that a fibre texture along [0,0,1] leads to transverse isotropy
        m = Miller(uvw=[1, 0, 0], phase=PHASE)
        texture = FibreTexture.from_Miller_axis(m, [0, 0, 1])
        Crot = C * texture
        Exy = Crot.Young_modulus.eval([[1,0,0],[1,1,0],[0,1,0]])
        assert (approx(Exy[0]) == Exy[1]) and (approx(Exy[0]) == Exy[2])

    def test_mult_float(self):
        m = Miller(uvw=[1, 0, 0], phase=PHASE)
        t = FibreTexture.from_Miller_axis(m, [0, 0, 1])
        tm = t * 0.5
        assert isinstance(tm, FibreTexture)
        assert t.orientation == tm.orientation
        assert tm.weight == 0.5

        tm2 = 0.5 * t
        assert isinstance(tm2, FibreTexture)
        assert t.orientation == tm2.orientation

    def test_fibre_random(self):
        t = FibreTexture.from_Euler(Phi=0, phi2=0)
        o = t.sample(1000, seed=123)
        v = Vector3d(~o * Vector3d([1,0,0]))
        ks = kstest(v.azimuth, 'uniform', args=(0, 2*np.pi))
        assert 0.05 < ks[1] < 0.95  # Use p-value
        v = Vector3d(~o * Vector3d([0,0,1]))
        np.testing.assert_array_almost_equal(v.data[:,0], np.zeros(1000))
        np.testing.assert_array_almost_equal(v.data[:, 1], np.zeros(1000))

        m = Miller(uvw=[1, 1, 1], phase=PHASE)
        t = FibreTexture.from_Miller_axis(m, [0, 0, 1])
        o = t.sample(1000)
        v = Vector3d(~o * m)
        np.testing.assert_array_almost_equal(v.data[:,0], np.zeros(1000))
        np.testing.assert_array_almost_equal(v.data[:, 1], np.zeros(1000))


class TestCompositeTexture(unittest.TestCase):
    def test_add_Textures(self):
        tg = DiscreteTexture.Goss()
        tb = DiscreteTexture.brass()
        tm = tb + tg
        assert isinstance(tm, CompositeTexture)
        wgts = [t.weight for t in tm.texture_list]
        assert wgts == [1., 1.]
        assert tm.texture_list[0] == tb
        assert tm.texture_list[1] == tg

    def test_mult(self):
        tg = DiscreteTexture.Goss()
        tb = DiscreteTexture.brass()
        tm = 0.5 * (tb + tg)
        wgts = [t.weight for t in tm.texture_list]
        assert wgts == [0.5, 0.5]

    def test_add_TexturesMix(self):
        tg = DiscreteTexture.Goss()
        tb = DiscreteTexture.brass()
        tc = DiscreteTexture.cube()
        tm1 = 0.5 * tg + 0.3 * tb
        tm2 = tm1 + 0.4 * tc
        assert isinstance(tm2, CompositeTexture)
        orientations = [t.orientation for t in tm2.texture_list]
        assert orientations == [tg.orientation, tb.orientation, tc.orientation]
        weights = [t.weight for t in tm2.texture_list]
        assert weights == [0.5, 0.3, 0.4]

        tm3 = 0.4 * tc + tm1
        assert isinstance(tm3, CompositeTexture)
        orientations = [t.orientation for t in tm3.texture_list]
        assert orientations == [tc.orientation, tg.orientation, tb.orientation]
        weights = [t.weight for t in tm3.texture_list]
        assert weights == [0.4, 0.5, 0.3]

        tm4 = tm1 + tm2
        assert isinstance(tm4, CompositeTexture)
        orientations = [t.orientation for t in tm4.texture_list]
        assert orientations == [t.orientation for t in tm1.texture_list] + [t.orientation for t in tm2.texture_list]

    def test_repr(self):
        t1 = DiscreteTexture.Goss()
        m = Miller(uvw=[1, 0, 0], phase=PHASE)
        t2 = FibreTexture.from_Miller_axis(m, [0, 0, 1])
        t3 = FibreTexture.from_Euler(phi1=0, Phi=10)
        t4 = UniformTexture()
        tm = t1 + t2 + t3 + 0.5*t4
        assert isinstance(tm, CompositeTexture)
        expected_str = ('Mixture of crystallographic textures\n'
                        ' Wgt.  Type      Component\n'
                        ' ------------------------------------------------------------\n')
        expected_str += ' 1.00  discrete  φ1=0.00°, ϕ=45.00°, φ2=0.00°\n'
        expected_str += ' 1.00  fibre     <1. 0. 0.> || [0, 0, 1]\n'
        expected_str += ' 1.00  fibre     φ1= 0°, ϕ= 10°\n'
        expected_str += ' 0.50  uniform   Uniform distribution over SO(3)'
        assert tm.__repr__() == expected_str

    def test_mean(self):
        t2 = DiscreteTexture.cube()
        t1 = DiscreteTexture.Goss()
        tm = t1 + t2
        Cmean = C * tm
        assert isinstance(Cmean, StiffnessTensor)
        ori = Orientation.from_euler([[0, 0, 0], [0, 45, 0]], degrees=True)
        assert Cmean == (C * ori).Voigt_average()
        assert Cmean == tm * C

        tm_wgt = 1.5 * t1 + 0.5 * t2
        Cmean = tm_wgt.mean_tensor(C)
        assert isinstance(Cmean, StiffnessTensor)
        Cmean_voigt = StiffnessTensor.weighted_average((C * t1, C * t2), [1.5, 0.5], 'Voigt')
        np.testing.assert_almost_equal(Cmean_voigt.matrix(), Cmean.matrix())

    def test_uniform(self):
        t = 0.3 * UniformTexture() + 0.5 * UniformTexture()
        np.testing.assert_array_almost_equal(C.Voigt_average().matrix(), (C * t).matrix())

    def test_random_sampling(self):
        w = [0.2, 0.3, 0.5]
        cube = DiscreteTexture.cube()
        goss = DiscreteTexture.Goss()
        brass = DiscreteTexture.brass()
        n = 100000
        tm = w[0] *cube + w[1] * goss + w[2] * brass
        sample = tm.sample(n, seed=123)
        n_cube = np.count_nonzero(sample.angle==cube.orientation.angle)
        assert approx(n_cube, rel=0.02) == w[0] * n
        n_goss = np.count_nonzero(sample.angle == goss.orientation.angle)
        assert approx(n_goss, rel=0.02) == w[1] * n
        n_brass = np.count_nonzero(sample.angle == brass.orientation.angle)
        assert approx(n_brass, rel=0.02) == w[2] * n


if __name__ == '__main__':
    unittest.main()
