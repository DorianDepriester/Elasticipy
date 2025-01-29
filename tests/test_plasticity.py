import unittest
import numpy as np
from Elasticipy.Plasticity import JohnsonCook

A, B, C = 792, 510, 0.014
m, n = 1.03, 0.26
eps_dot_ref = 1
T0, Tm = 25, 1500
JC    = JohnsonCook(A=A, B=B, n=n)
JC_rd = JohnsonCook(A=A, B=B, n=n, C=C, eps_dot_ref=eps_dot_ref)
JC_td = JohnsonCook(A=A, B=B, n=n, m=1.03, T0=T0, Tm=Tm)
JC_rtd= JohnsonCook(A=A, B=B, n=n, C=C, eps_dot_ref=eps_dot_ref, m=m, T0=T0, Tm=Tm)


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

    def test_compute_strain(self):
        strain0 = np.linspace(0,1)

        # Test temperature-independent model
        stress = JC.flow_stress(strain0)
        strain1 = JC.compute_strain(stress)
        np.testing.assert_array_almost_equal(strain0, strain1)

        # Test temperature-dependent model
        stress = JC_td.flow_stress(strain0, T=500)
        strain1 = JC_td.compute_strain(stress, T=500)
        np.testing.assert_array_almost_equal(strain0, strain1)

        # What if we use try to use the temperature on temperature-independent model
        with self.assertRaises(ValueError) as context:
            _ = JC.flow_stress(0.1, T=T0)
        self.assertEqual(str(context.exception), 'T0, Tm and m must be defined for using a temperature-dependent model')



if __name__ == '__main__':
    unittest.main()
