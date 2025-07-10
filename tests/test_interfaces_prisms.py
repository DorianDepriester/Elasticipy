import unittest
from Elasticipy.interfaces.prisms_plasticity import from_quadrature_file, from_stressstrain_file
from Elasticipy.tensors.second_order import SecondOrderTensor
from Elasticipy.tensors.stress_strain import StressTensor
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
quadrature_file = os.path.join(current_dir, 'QuadratureOutputs.csv')

quadrature_data = pd.read_csv(quadrature_file, header=None, usecols=range(0,37))

class TestPRISMSInterfaces(unittest.TestCase):
    def test_from_quadrature(self):
        stress = from_quadrature_file(quadrature_file)
        assert isinstance(stress, StressTensor)
        for i in range(0,len(quadrature_data)):
            assert stress[i].C[0, 0] == quadrature_data.iloc[i,28]
            assert stress[i].C[1, 1] == quadrature_data.iloc[i,29]
            assert stress[i].C[2, 2] == quadrature_data.iloc[i,30]
            assert stress[i].C[0, 1] == quadrature_data.iloc[i,31]
            assert stress[i].C[0, 2] == quadrature_data.iloc[i,32]
            assert stress[i].C[1, 2] == quadrature_data.iloc[i,34]

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
