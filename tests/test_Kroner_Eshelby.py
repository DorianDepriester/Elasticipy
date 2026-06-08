import unittest
from elasticipy.homogenization.kroner_eshelby import polarization_tensor, Kroner_Eshelby
from elasticipy.tensors.elasticity import StiffnessTensor
from elasticipy.tensors.fourth_order import SymmetricFourthOrderTensor
import numpy as np
from pytest import approx

mu_i, mu_m = 100, 200
K_i = 1000
K_m = 3000
Ci = StiffnessTensor.isotropic(K=K_i, G=mu_i)
Cm = StiffnessTensor.isotropic(K=K_m, G=mu_m)

# Hexagonal case (Be) in https://link.springer.com/article/10.1007/s00707-017-1811-x
C11, C12, C13, C33, C44 = 292.3, 26.7, 14, 336.4, 162.5
Chex = StiffnessTensor.hexagonal(C11=C11, C12=C12, C13=C13, C33=C33, C44=C44)

class TestKronerEshelby(unittest.TestCase):
    def test_polarization_tensor_sphere(self):
        E = polarization_tensor(Cm, 1,1,1, 100, 50)
        alpha = 1 / (3*K_m + 4 * mu_m)
        beta = 3 * (K_m + 2 * mu_m) / (5 * mu_m * (3*K_m + 4 * mu_m))
        J = SymmetricFourthOrderTensor.identity_spherical_part()
        K = SymmetricFourthOrderTensor.identity_deviatoric_part()
        Eth = alpha * J + beta * K
        np.testing.assert_array_almost_equal(E.full_tensor, Eth.full_tensor)

    def test_polarization_tensor_disk(self):
        E = polarization_tensor(Cm, 1e4,1e4,1, 100, 50)
        assert approx(E.C33, rel=1e-3) == 1 / (K_m + 4 * mu_m/3)

    def test_hexagonal_sphere(self):
        # DOI 10.1007/s00707-017-1811-x
        # https://link.springer.com/article/10.1007/s00707-017-1811-x
        # Table 1 p. 1981
        E = polarization_tensor(Chex, 1, 1, 1, 100, 500)
        S = E.ddot(Chex).full_tensor
        assert approx(S[0, 0, 0, 0]) == 0.465339
        assert approx(S[0, 0, 1, 1]) == -0.0359279
        assert approx(S[0, 0, 2, 2]) == -0.0618267
        assert approx(S[2, 2, 0, 0]) == -0.0567276
        assert approx(S[2, 0, 2, 0]) == 0.273238
        assert approx(S[0, 1, 0, 1]) == 0.2506334

    def test_hexagonal_disk(self):
        # DOI 10.1007/s00707-017-1811-x
        # https://link.springer.com/article/10.1007/s00707-017-1811-x
        # Eq. (126) and (129)
        E = polarization_tensor(Chex, 10000, 10000, 1, 50, 2000)
        S = E.ddot(Chex).full_tensor
        assert approx(S[2, 2, 2, 2]) == 1.
        assert approx(S[2, 2, 0, 0], rel=1e-3) == C13 / C33
        assert approx(S[2, 2, 1, 1], rel=1e-3) == C13 / C33
        assert approx(S[2, 0, 2, 0], rel=1e-4) == 0.5
        assert approx(S[2, 1, 2, 1], rel=1e-4) == 0.5

    def test_hexagonal_fibre(self):
        # DOI 10.1007/s00707-017-1811-x
        # https://link.springer.com/article/10.1007/s00707-017-1811-x
        # Eqs. (128) and (129)
        E = polarization_tensor(Chex, 1, 1, 10000, 50, 2000)
        S = E.ddot(Chex).full_tensor
        assert approx(S[0, 0, 0, 0]) == (5 * C11 + C12) / (8 * C11)
        assert approx(S[1, 1, 1, 1]) == (5 * C11 + C12) / (8 * C11)
        assert approx(S[0, 0, 1, 1], rel=1e-4) == (3 * C12 - C11) / (8 * C11)
        assert approx(S[1, 1, 0, 0]) == (3 * C12 - C11) / (8 * C11)
        assert approx(S[0, 0, 2, 2], rel=1e-4) == C13 / 2 / C11
        assert approx(S[1, 1, 2, 2], rel=1e-4) == C13 / 2 / C11
        assert approx(S[2, 0, 2, 0]) == 1 / 4
        assert approx(S[1, 2, 1, 2]) == 1 / 4
        assert approx(S[0, 1, 0, 1]) ==  (3 * C11 - C12) / (8 * C11)

    def test_Kroner_Eshelby_isotropic(self):
        Cm, msg = Kroner_Eshelby((Ci, Ci))
        np.testing.assert_array_almost_equal(Cm.matrix(), Ci.matrix())


if __name__ == '__main__':
    unittest.main()
