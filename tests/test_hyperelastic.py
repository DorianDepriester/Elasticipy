import unittest
import numpy as np
from elasticipy.tensors.finite_strain import DeformationGradient
from elasticipy.hyperelasticity import NeoHooke
from elasticipy.tensors.elasticity import StiffnessTensor
from elasticipy.tensors.stress_strain import StrainTensor

class TestNeoHooke(unittest.TestCase):
    def test_small_strain(self):
        C10 = 100
        D = 500
        nh = NeoHooke(C10, D)
        eps = np.array([[11, 12, 13], [12, 22, 23], [13, 23, 33]]) * 1e-7
        F = DeformationGradient(eps) + DeformationGradient.eye()
        s_finite = nh.stress_from_gradient(F)
        C = StiffnessTensor.isotropic(G=2*C10, K=2*D)
        s_small = C * StrainTensor(eps)
        np.testing.assert_almost_equal(s_finite.matrix, s_small.matrix)

if __name__ == '__main__':
    unittest.main()
