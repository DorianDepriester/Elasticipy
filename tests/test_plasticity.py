import unittest
import numpy as np
from Elasticipy.Plasticity import JohnsonCook
from Elasticipy.Plasticity import normality_rule
from pytest import approx

from Elasticipy.StressStrainTensors import StressTensor, StrainTensor

A, B, C = 792, 510, 0.014
m, n = 1.03, 0.26
eps_dot_ref = 1
T0, Tm = 25, 1500
JC    = JohnsonCook(A=A, B=B, n=n)
JC_rd = JohnsonCook(A=A, B=B, n=n, C=C, eps_dot_ref=eps_dot_ref)
JC_td = JohnsonCook(A=A, B=B, n=n, m=1.03, T0=T0, Tm=Tm)
JC_rtd= JohnsonCook(A=A, B=B, n=n, C=C, eps_dot_ref=eps_dot_ref, m=m, T0=T0, Tm=Tm)
k = 3 / 2 * 1 / 3 ** 0.5


class TestJohnsonCook(unittest.TestCase):
    def test_yield_stress(self):
        assert JC.flow_stress(0) == A
        assert JC_rd.flow_stress(0, eps_dot=eps_dot_ref) == A
        assert JC_td.flow_stress(0, T=T0) == A

    def test_rate_dependence(self):
        assert JC_rd.flow_stress(0.1, eps_dot=2) == JC_rd.flow_stress(0.1, eps_dot=1) * (1 + C*np.log(2))
        for model in (JC, JC_td):   # Check that an error is thrown if the model is not rate-dependent
            with self.assertRaises(ValueError) as context:
                _ = model.flow_stress(0.1, eps_dot=2)
            self.assertEqual(str(context.exception), 'C and eps_dot_ref must be defined for using a rate-dependent model')

    def test_temperature_dependence(self):
        assert JC_td.flow_stress(0.1, T=T0) == JC.flow_stress(0.1)
        assert JC_td.flow_stress(0.1, T=Tm) == 0.0
        for model in (JC, JC_rd):   # Check that an error is thrown if the model is not temperature-dependent
            with self.assertRaises(ValueError) as context:
                _ = model.flow_stress(0.1, T=T0)
            self.assertEqual(str(context.exception), 'T0, Tm and m must be defined for using a temperature-dependent model')

    def test_compute_strain_increment(self):
        JC2 = JohnsonCook(A=A, B=B, n=n)
        strain0 = 0.1

        # Test temperature-independent model
        stress = JC2.flow_stress(strain0)
        strain1 = JC2.compute_strain_increment(stress)
        assert strain1 == approx(strain0)

        # Now try with a full tensor
        JC.reset_strain()
        assert JC.plastic_strain == 0.0
        stress = StressTensor.tensile([1,0,0], A+10)
        strain = JC.compute_strain_increment(stress)
        normalized_strain = strain / strain.eq_strain()
        np.testing.assert_array_almost_equal(normalized_strain.matrix, np.diag([1,-0.5,-0.5]))

        # Test temperature-dependent model
        stress = JC_td.flow_stress(strain0, T=500)
        strain2 = JC_td.compute_strain_increment(stress, T=500)
        assert strain2 == approx(strain0)

        # What if we use try to use the temperature on temperature-independent model
        with self.assertRaises(ValueError) as context:
            _ = JC.flow_stress(0.1, T=T0)
        self.assertEqual(str(context.exception), 'T0, Tm and m must be defined for using a temperature-dependent model')

        # Check that if stress < A, the strain is zero
        assert JC.compute_strain_increment(A) == 0.0

        # Check that if the temperature is larger than Tm, the strain is infinite
        assert JC_td.compute_strain_increment(0, T=Tm) == np.inf


    def test_normality_J2(self):
        tensile_stress = StressTensor.tensile([1,0,0], 1)
        normal = normality_rule(tensile_stress)
        assert normal == np.diag([1., -0.5, -0.5])

        shear_stress = StressTensor.shear([1, 0, 0], [0, 1, 0], 1)
        normal = normality_rule(shear_stress)
        normal_th = k * np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 0]])
        np.testing.assert_array_almost_equal(normal.matrix, normal_th)

    def test_normality_Tresca(self):
        biaxial = (StressTensor.tensile([1,0,0],[0, 1, 1, 1, 1, 1, 0]) +
                   StressTensor.tensile([0,1,0],[-1, -1, -0.5, 0, 0.5, 1, 1]))
        n = normality_rule(biaxial, criterion='Tresca')
        assert n[0] == normality_rule(biaxial[0])
        assert n[2] == k * np.diag([1, -1, 0])
        assert n[2] == k * np.diag([1, -1, 0])
        assert n[3] == normality_rule(biaxial[3])
        assert n[4] == k * np.diag([1, 0, -1])
        assert n[5] == normality_rule(biaxial[5])
        assert n[6] == normality_rule(biaxial[6])

        # Check that the magnitude of the normal is 1
        np.testing.assert_array_equal(n.eq_strain(), np.ones(biaxial.shape))

        triaxial = StressTensor(np.diag([1,2,4]))
        n = normality_rule(triaxial, criterion='Tresca')
        assert n == k*np.diag([-1,0,1])
        assert n.eq_strain() == 1.0


    def test_apply_strain(self):
        strain = StrainTensor.tensile([1,0,0], 1) + StrainTensor.tensile([0, 1, 0], -0.5) + StrainTensor.tensile([0, 0, 1], -0.5)
        JC.apply_strain(0.0)    # Try with float
        assert JC.plastic_strain == 0.0
        JC.apply_strain(strain) # Try with StrainTensor
        assert JC.plastic_strain == 1.0
        JC.apply_strain(-1.0)
        assert JC.plastic_strain == 2






if __name__ == '__main__':
    unittest.main()
