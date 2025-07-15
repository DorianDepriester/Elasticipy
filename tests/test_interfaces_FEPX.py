import unittest
from Elasticipy.interfaces.FEPX import from_step_file, from_results_folder
from Elasticipy.tensors.second_order import SecondOrderTensor, SymmetricSecondOrderTensor, \
    SkewSymmetricSecondOrderTensor
from Elasticipy.tensors.stress_strain import StrainTensor, StressTensor
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
FEPX_DATA = os.path.join(current_dir,'interfaces/FEPX/simulation.sim/results/elts/')
SIZE_FEPX_DATA = 2453
NSTEP_FEPX_DATA = 3

class TestFEPX(unittest.TestCase):
    def test_stress_from_file(self):
        a = from_step_file(FEPX_DATA + 'strain/strain.step0')
        assert isinstance(a, StrainTensor)
        assert a.shape == (SIZE_FEPX_DATA,)

    def test_strain_from_file(self):
        a = from_step_file(FEPX_DATA + 'stress/stress.step0')
        assert isinstance(a, StressTensor)
        assert a.shape == (SIZE_FEPX_DATA,)

    def test_stress_from_folder(self):
        a = from_results_folder(FEPX_DATA + 'strain')
        assert isinstance(a, StrainTensor)
        assert a.shape == (NSTEP_FEPX_DATA, SIZE_FEPX_DATA)

    def test_defrate_from_folder(self):
        a = from_results_folder(FEPX_DATA + 'defrate')
        assert isinstance(a, SymmetricSecondOrderTensor)
        assert a.shape == (NSTEP_FEPX_DATA, SIZE_FEPX_DATA)

    def test_velgrad_from_folder(self):
        a = from_results_folder(FEPX_DATA + 'velgrad')
        assert isinstance(a, SecondOrderTensor) and not isinstance(a, SymmetricSecondOrderTensor)
        assert a.shape == (NSTEP_FEPX_DATA, SIZE_FEPX_DATA)

    def test_spinrate_from_folder(self):
        a = from_results_folder(FEPX_DATA + 'spinrate')
        assert isinstance(a, SkewSymmetricSecondOrderTensor)
        assert a.shape == (NSTEP_FEPX_DATA, SIZE_FEPX_DATA)

    def test_orientation_from_folder(self):
        a = from_results_folder(FEPX_DATA + 'ori')
        assert isinstance(a, np.ndarray)
        assert a.shape == (NSTEP_FEPX_DATA, SIZE_FEPX_DATA, 3)

if __name__ == '__main__':
    unittest.main()
