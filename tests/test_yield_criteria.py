import unittest
from elasticipy.yield_criteria import DruckerPrager, MohrCoulomb, TrescaCriterion, VonMisesCriterion
from elasticipy.tensors.stress_strain import StressTensor, StrainTensor
import numpy as np
from pytest import approx

tensile_x = StressTensor.tensile([1, 0, 0], 1)
tensile_y = StressTensor.tensile([0, 1, 0], 1)
shear = StressTensor.shear([1, 0, 0], [0, 1, 0], 1)
biaxial = tensile_x + 2 *tensile_y
K = 3 / 2 * 1 / 3 ** 0.5

def generic_test_from_tensile_compression_stresses(obj, criterion):
    with obj.assertRaises(ValueError) as context:
        criterion.from_tensile_compression_stress(100, 100)
    obj.assertEqual(str(context.exception), 'The compression yield stress must be negative.')
    with obj.assertRaises(ValueError) as context:
        criterion.from_tensile_compression_stress(-100, -150)
    obj.assertEqual(str(context.exception), 'The tensile yield stress must be positive.')

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

    def test_normality(self):
        k, alpha = 2, 0.1
        dp = DruckerPrager(alpha, k)
        stress = StressTensor.eye()
        unit_strain = StrainTensor.eye()
        assert dp.normal(stress) == unit_strain/unit_strain.eq_strain()

    def test_from_tensile_compression_stress(self):
        s_t = 100
        s_c = -150
        mc = DruckerPrager.from_tensile_compression_stress(s_t, s_c)
        assert mc.yield_function(tensile_x * s_t) == approx(0.)
        assert mc.yield_function(tensile_x * s_c) == approx(0.)

    def test_from_tensile_compression_stress_valid_args(self):
        generic_test_from_tensile_compression_stresses(self, DruckerPrager)


class TestMohrCoulomb(unittest.TestCase):
    def test_Tresca(self):
        mv_tr = MohrCoulomb(1, 0)
        for stress in [tensile_x, tensile_y, shear, biaxial]:
            assert mv_tr.yield_function(stress) == stress.Tresca() - 2

            tresca_crit = TrescaCriterion()
            np.testing.assert_array_almost_equal(mv_tr.normal(stress).matrix, tresca_crit.normal(stress).matrix)

    def test_normality(self):
        mv_tr = MohrCoulomb(1, 0)
        stress = StressTensor.eye()
        unit_strain = StrainTensor.eye()
        assert mv_tr.normal(stress) == unit_strain/unit_strain.eq_strain()

    def test_from_tensile_compression_stress(self):
        s_t = 100
        s_c = -150
        mc = MohrCoulomb.from_tensile_compression_stress(s_t, s_c)
        assert mc.yield_function(tensile_x * s_t) == 0.
        assert mc.yield_function(tensile_x * s_c) == 0.

    def test_from_tensile_compression_stress_valid_args(self):
        generic_test_from_tensile_compression_stresses(self, MohrCoulomb)

class TestTrescaCriterion(unittest.TestCase):
    def test_normality_Tresca(self):
        biaxial = (StressTensor.tensile([1,0,0],[0, 1, 1, 1, 1, 1, 0]) +
                   StressTensor.tensile([0,1,0],[-1, -1, -0.5, 0, 0.5, 1, 1]))
        n = TrescaCriterion.normal(biaxial)
        assert n[0] == VonMisesCriterion.normal(biaxial[0])
        assert n[2] == K * np.diag([1, -1, 0])
        assert n[2] == K * np.diag([1, -1, 0])
        assert n[3] == VonMisesCriterion.normal(biaxial[3])
        assert n[4] == K * np.diag([1, 0, -1])
        assert n[5] == VonMisesCriterion.normal(biaxial[5])
        assert n[6] == VonMisesCriterion.normal(biaxial[6])

        # Check that the magnitude of the normal is 1
        np.testing.assert_array_equal(n.eq_strain(), np.ones(biaxial.shape))
        triaxial = StressTensor(np.diag([1,2,4]))
        n = TrescaCriterion.normal(triaxial)
        assert n == K * np.diag([-1, 0, 1])
        assert n.eq_strain() == 1.0

class TestVonMisesCriterion(unittest.TestCase):
    def test_scale_stress(self):
        vm = VonMisesCriterion(yield_stress=100)
        stress = vm.scale_stress_to_yield_surface(tensile_x)
        np.testing.assert_array_almost_equal(stress.matrix, tensile_x.matrix * 100)

    def test_normality_J2(self):
        tensile_stress = StressTensor.tensile([1, 0, 0], 1)
        normal = VonMisesCriterion.normal(tensile_stress)
        assert normal == np.diag([1., -0.5, -0.5])

        shear_stress = StressTensor.shear([1, 0, 0], [0, 1, 0], 1)
        normal = VonMisesCriterion.normal(shear_stress)
        normal_th = K * np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 0]])
        np.testing.assert_array_almost_equal(normal.matrix, normal_th)

if __name__ == '__main__':
    unittest.main()
