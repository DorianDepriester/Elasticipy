import unittest
from elasticipy.yield_criteria import DruckerPrager, MohrCoulomb, TrescaCriterion, VonMisesCriterion
from elasticipy.tensors.stress_strain import StressTensor
import numpy as np

tensile_x = StressTensor.tensile([1, 0, 0], 1)
tensile_y = StressTensor.tensile([0, 1, 0], 1)
shear = StressTensor.shear([1, 0, 0], [0, 1, 0], 1)
biaxial = tensile_x + 2 *tensile_y

class TestDruckerPrager(unittest.TestCase):
    def test_vonMises(self):
        pg_vm = DruckerPrager(k=1, alpha=0)
        for stress in [tensile_x, tensile_y, shear, biaxial]:
            assert pg_vm.yield_function(stress) == stress.vonMises() / np.sqrt(3) - 1

            mises_crit = VonMisesCriterion()
            np.testing.assert_array_almost_equal(pg_vm.normal(stress).matrix, mises_crit.normal(stress).matrix)


class TestMohrCoulomb(unittest.TestCase):
    def test_Tresca(self):
        mv_tr = MohrCoulomb(c=1, phi=0)
        for stress in [tensile_x, tensile_y, shear, biaxial]:
            assert mv_tr.yield_function(stress) == stress.Tresca() - 2

            tresca_crit = TrescaCriterion()
            np.testing.assert_array_almost_equal(mv_tr.normal(stress).matrix, tresca_crit.normal(stress).matrix)


if __name__ == '__main__':
    unittest.main()
