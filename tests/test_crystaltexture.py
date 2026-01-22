import unittest
from elasticipy.crystal_texture import CrystalTexture, FibreTexture
from orix.vector import Miller, Vector3d
from orix.crystal_map import Phase
import numpy as np

phase = Phase(point_group="m-3m")

def orientation_checker(texture, hkl, uvw):
    orientation = texture.orientation
    m_hkl = Miller(uvw=hkl, phase=phase)
    x = ~orientation * m_hkl.symmetrise(unique=True)
    u_hkl = x.data
    cos_hkl = np.dot(u_hkl, [0, 0, 1]) / np.linalg.norm(u_hkl, axis=1)
    assert np.any(np.isclose(cos_hkl, 1))

    m_uvw = Miller(uvw=uvw, phase=phase)
    x = ~orientation * m_uvw.symmetrise(unique=True)
    u_uvw = x.data
    cos_uvw = np.dot(u_uvw, [1, 0, 0]) / np.linalg.norm(u_uvw, axis=1)
    assert np.any(np.isclose(cos_uvw, 1))


class TestCrystalTexture(unittest.TestCase):
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




if __name__ == '__main__':
    unittest.main()
