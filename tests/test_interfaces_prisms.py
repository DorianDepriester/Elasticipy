import unittest
from Elasticipy.interfaces.prisms_plasticity import from_quadrature_file, from_stressstrain_file
from Elasticipy.tensors.second_order import SecondOrderTensor
from Elasticipy.tensors.stress_strain import StressTensor
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
quadrature_file = os.path.join(current_dir, 'QuadratureOutputs.csv')

class TestPRISMInterfaces(unittest.TestCase):
    def test_from_quadrature(self):
        stress = from_quadrature_file(quadrature_file)
        assert isinstance(stress, StressTensor)
        assert stress[0].C[0, 0] == 1.59893e+01
        assert stress[0].C[1, 1] == 7.76097
        assert stress[0].C[2, 2] == 6.51261

    def test_from_stressstrain_with_fields(self):
        fields = ('grain ID', 'phase ID', 'det(J)', 'twin', 'coordinates', 'orientation', 'elastic gradient',
                  'plastic gradient', 'stress')
        a = from_quadrature_file(quadrature_file, returns=fields)
        assert len(a) == len(fields)
        assert isinstance(a[0], np.ndarray)
        assert isinstance(a[1], np.ndarray)
        assert isinstance(a[2], np.ndarray)
        assert isinstance(a[3], np.ndarray)
        assert isinstance(a[4], np.ndarray)
        assert isinstance(a[5], np.ndarray)
        assert isinstance(a[6], SecondOrderTensor)
        assert isinstance(a[7], SecondOrderTensor)
        assert isinstance(a[8], StressTensor)





if __name__ == '__main__':
    unittest.main()
