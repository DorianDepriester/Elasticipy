import unittest
from elasticipy.yield_criteria import DruckerPrager, VonMisesCriterion
from elasticipy.tensors.stress_strain import StressTensor
import numpy as np

class TestDruckerPrager(unittest.TestCase):
    def test_vonMises(self):
        pg_vm = DruckerPrager(k=1, alpha=0)
        tensile_x = StressTensor.tensile([1,0,0], 1)
        tensile_y = StressTensor.tensile([1,0,0], 1)
        shear = StressTensor.shear([1,0,0], [0,1,0], 1)
        biaxial = tensile_x +  tensile_y
        for stress in [tensile_x, tensile_y, shear, biaxial]:
            assert pg_vm.yield_function(stress) == stress.vonMises() / np.sqrt(3) - 1

            mises_crit = VonMisesCriterion()
            np.testing.assert_array_almost_equal(pg_vm.normal(stress).matrix, mises_crit.normal(stress).matrix)







if __name__ == '__main__':
    unittest.main()
