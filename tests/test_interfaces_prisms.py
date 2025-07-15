import unittest
from Elasticipy.interfaces.PRISMS import from_quadrature_file, from_stressstrain_file
from Elasticipy.tensors.second_order import SecondOrderTensor, SymmetricSecondOrderTensor
from Elasticipy.tensors.stress_strain import StressTensor
import numpy as np
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
quadrature_file = os.path.join(current_dir, 'interfaces/PRISMS/QuadratureOutputs.csv')
stressstrain_file = os.path.join(current_dir, 'interfaces/PRISMS/stressstrain.txt')

quadrature_data = pd.read_csv(quadrature_file, header=None, usecols=range(0,37))
stressstrain_data = pd.read_csv(stressstrain_file, sep='\t')

class TestPRISMSInterfaces(unittest.TestCase):
    def test_from_quadrature(self):
        stress = from_quadrature_file(quadrature_file)
        assert stress.shape == (len(quadrature_data),)
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

    def test_from_stressstrain(self):
        E, stress = from_stressstrain_file(stressstrain_file)
        assert E.shape == stress.shape == (len(stressstrain_data),)
        assert isinstance(E, SymmetricSecondOrderTensor)
        assert isinstance(stress, StressTensor)
        for i in range(0,len(stressstrain_data)):
            assert E[i].C[0, 0] == stressstrain_data['Exx'][i]
            assert E[i].C[1, 1] == stressstrain_data['Eyy'][i]
            assert E[i].C[2, 2] == stressstrain_data['Ezz'][i]
            assert E[i].C[1, 2] == stressstrain_data['Eyz'][i]
            assert E[i].C[0, 2] == stressstrain_data['Exz'][i]
            assert E[i].C[0, 1] == stressstrain_data['Exy'][i]
            assert stress[i].C[0, 0] == stressstrain_data['Txx'][i]
            assert stress[i].C[1, 1] == stressstrain_data['Tyy'][i]
            assert stress[i].C[2, 2] == stressstrain_data['Tzz'][i]
            assert stress[i].C[1, 2] == stressstrain_data['Tyz'][i]
            assert stress[i].C[0, 2] == stressstrain_data['Txz'][i]
            assert stress[i].C[0, 1] == stressstrain_data['Txy'][i]



if __name__ == '__main__':
    unittest.main()
