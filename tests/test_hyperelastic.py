import unittest
import numpy as np
from elasticipy.tensors.finite_strain import DeformationGradient
from elasticipy.hyperelasticity import NeoHooke
from elasticipy.tensors.elasticity import StiffnessTensor
from elasticipy.tensors.stress_strain import StrainTensor

C = 100
D = 500
nh_comp = NeoHooke(C, D=D)
nh_incomp = NeoHooke(C)


class TestNeoHooke(unittest.TestCase):
    def test_small_strain_compr(self):
        eps = np.array([[11, 12, 13], [12, 22, 23], [13, 23, 33]]) * 1e-7
        F = DeformationGradient(eps) + DeformationGradient.eye()
        s_finite = nh_comp.stress_from_gradient(F)
        stiff = StiffnessTensor.isotropic(G=2*C, K=2/D)
        s_small = stiff * StrainTensor(eps)
        np.testing.assert_almost_equal(s_finite.matrix, s_small.matrix)



if __name__ == '__main__':
    unittest.main()
