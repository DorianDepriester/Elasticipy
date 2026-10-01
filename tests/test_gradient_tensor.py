import unittest
from elasticipy.tensors.finite_strain import GradientTensor
import numpy as np

class TestSecondOrderTensor(unittest.TestCase):
    def test_polar(self):
        F = GradientTensor.rand()
        R, U = F.polar()
        assert U == F.U
        np.testing.assert_array_equal(R, F.R)

        R, V = F.polar(side='left')
        assert V == F.V
        np.testing.assert_array_equal(R, F.R)