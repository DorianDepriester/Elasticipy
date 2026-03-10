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
        pg_vm = DruckerPrager(0, 1)
        for stress in [tensile_x, tensile_y, shear, biaxial]:
            assert pg_vm.yield_function(stress) == stress.vonMises() / np.sqrt(3) - 1

            mises_crit = VonMisesCriterion()
            np.testing.assert_array_almost_equal(pg_vm.normal(stress).matrix, mises_crit.normal(stress).matrix)

    def test_yield_function(self):
        k, alpha = 2, 0.1
        pg = DruckerPrager(alpha, k)
        sy_tensile = k/(alpha + 3**(-0.5))
        sy_compres = k / (alpha - 3 ** (-0.5))
        assert pg.yield_function(tensile_x * sy_tensile) == 0.0
        assert pg.yield_function(tensile_x * sy_compres) == 0.0

    def test_inequality_fit(self):
        c, phi = 2, -10
        dp_1 = DruckerPrager.from_cohesion_friction_angle(c, phi, fit='inside')
        dp_2 = DruckerPrager.from_cohesion_friction_angle(c, phi, fit='middle')
        dp_3 = DruckerPrager.from_cohesion_friction_angle(c, phi, fit='outside')
        rand_stress = StressTensor.rand(shape=1000, seed=123)
        f1 = dp_1.yield_function(rand_stress)
        f2 = dp_2.yield_function(rand_stress)
        f3 = dp_3.yield_function(rand_stress)
        assert np.all(np.logical_and(f1 > f2, f2 > f3))

class TestMohrCoulomb(unittest.TestCase):
    def test_Tresca(self):
        mv_tr = MohrCoulomb(1, 0)
        for stress in [tensile_x, tensile_y, shear, biaxial]:
            assert mv_tr.yield_function(stress) == stress.Tresca() - 2

            tresca_crit = TrescaCriterion()
            np.testing.assert_array_almost_equal(mv_tr.normal(stress).matrix, tresca_crit.normal(stress).matrix)

class TestVonMisesCriterion(unittest.TestCase):
    def test_scale_stress(self):
        vm = VonMisesCriterion(yield_stress=100)
        stress = vm.scale_stress_to_yield_surface(tensile_x)
        np.testing.assert_array_almost_equal(stress.matrix, tensile_x.matrix * 100)

if __name__ == '__main__':
    unittest.main()
