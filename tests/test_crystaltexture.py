import unittest
from elasticipy.crystal_texture import CrystalTexture, FibreTexture, CrystalTextureMix
from orix.vector import Miller, Vector3d
from orix.crystal_map import Phase
import numpy as np
from elasticipy.tensors.elasticity import StiffnessTensor
from pytest import approx

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


class TestCrystalTexture(unittest.TestCase):
    def test_uniform(self):
        t = CrystalTexture.uniform()
        assert t.orientation is None
        assert C * t == C.Voigt_average()
        S = C.inv()
        assert S * t == S.Reuss_average()

    def test_cube(self):
        t = CrystalTexture.Cube()
        orientation_checker(t, [0,0,1], [1,0,0])

    def test_Goss(self):
        t = CrystalTexture.Goss()
        orientation_checker(t, [1,1,0], [1,0,0])

    def test_Brass(self):
        t = CrystalTexture.Brass()
        orientation_checker(t, [1,1,0], [1,1,2])

    def test_GossBrass(self):
        t = CrystalTexture.GossBrass()
        orientation_checker(t, [1,1,0], [1,1,5])

    def test_Copper(self):
        t = CrystalTexture.Copper()
        orientation_checker(t, [1,1,2], [1,1,1])

    def test_A(self):
        t = CrystalTexture.A()
        orientation_checker(t, [1,1,0], [1,1,1])

    def test_P(self):
        t = CrystalTexture.P()
        orientation_checker(t, [0,1,1], [2,1,1])

    def test_CuT(self):
        t = CrystalTexture.CuT()
        orientation_checker(t, [5,5,2], [1,1,5])

    def test_S(self):
        t = CrystalTexture.S()
        orientation_checker(t, [1,2,3], [6,3,4])

    def test_repr(self):
        t = CrystalTexture.Goss()
        assert t.__repr__() == 'Crystallographic texture\nphi1=0.00°, Phi=45.00°, phi2=0.00°'

    def test_mult_stiffness(self):
        t = CrystalTexture.S()
        Crot = C * t
        assert Crot == C * t.orientation

    def test_mult_float(self):
        t = CrystalTexture.Goss()
        tm = t * 0.5
        assert isinstance(tm, CrystalTexture)
        assert t.orientation == tm.orientation
        assert tm.weight == 0.5

        tm2 = 0.5 * t
        assert isinstance(tm2, CrystalTexture)
        assert t.orientation == tm2.orientation

class TestFibreTexture(unittest.TestCase):
    def test_from_Euler(self):
        t = FibreTexture.from_euler(phi1=0, Phi=10)
        assert t.__repr__() == 'Fibre texture\nphi1= 0°, Phi= 10°'

        t = FibreTexture.from_euler(phi1=0, phi2=10)
        assert t.__repr__() == 'Fibre texture\nphi1= 0°, phi2= 10°'

        t = FibreTexture.from_euler(Phi=0, phi2=10)
        assert t.__repr__() == 'Fibre texture\nPhi= 0°, phi2= 10°'

        t = FibreTexture.from_euler(Phi=0, phi2=10 * np.pi / 180, degrees=False)
        assert t.__repr__() == 'Fibre texture\nPhi= 0.0°, phi2= 10.0°'

    def test_from_Euler_axis(self):
        t = FibreTexture.from_euler(Phi=0, phi2=0)
        assert t.axis.dot(Vector3d([0,0,1])) == 1. or t.axis.dot(Vector3d([0,0,1])) == -1.

        t = FibreTexture.from_euler(phi1=0, phi2=0)
        assert t.axis.dot(Vector3d([0,0,1])) == 1. or t.axis.dot(Vector3d([1,0,0])) == -1.

    def test_from_Miller_axis(self):
        m = Miller(uvw=[1, 0, 0], phase=PHASE)
        texture = FibreTexture.from_Miller_axis(m, [0, 0, 1])
        assert texture.__repr__() == 'Fibre texture\n<1. 0. 0.> || [0, 0, 1]\nPoint group: m-3m'

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


class TestCrystalTextureMix(unittest.TestCase):
    def test_add(self):
        tm = CrystalTexture.Goss() + CrystalTexture.Brass()
        assert isinstance(tm, CrystalTextureMix)
        wgts = [t.weight for t in tm.texture_list]
        assert wgts == [1., 1.]

if __name__ == '__main__':
    unittest.main()
