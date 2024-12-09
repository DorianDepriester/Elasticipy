import unittest
from pytest import approx
import pandas as pd
import numpy as np

from Elasticipy.FourthOrderTensor import StiffnessTensor
from scipy.spatial.transform import Rotation
from Elasticipy.FourthOrderTensor import _indices2str
from Elasticipy.CrystalSymmetries import SYMMETRIES

data_base = pd.read_json('MaterialsProject.json')
rotations = Rotation.random(10000)

def variant_selection(symmetry, point_group):
    for variant_group in symmetry.keys():
        elements = [elem.strip() for elem in variant_group.split(",")]
        if point_group in elements:
            return symmetry[variant_group]
    return None


def crystal_symmetry_tester(symmetry_name, point_group=None):
    symmetry = SYMMETRIES[symmetry_name]
    if point_group is None:
        materials_of_interest = data_base[data_base.symmetry == symmetry_name]
        required_fields = symmetry.required
    else:
        materials_of_interest = data_base[data_base.point_group == point_group]
        variant = variant_selection(symmetry, point_group)
        required_fields = variant.required
    for index, row in materials_of_interest.iterrows():
        matrix = np.array(row['C'])
        kwargs = dict()
        for indices in required_fields:
            component_name = 'C' + _indices2str(indices)
            kwargs[component_name] = matrix[*indices]
        constructor = getattr(StiffnessTensor, symmetry_name.lower())
        C = constructor(**kwargs)
        assert np.all(C.matrix == approx(matrix, abs=0.5))

class test_StiffnessConstructor(unittest.TestCase):
    def test_averages(self):
        rel = 5e-2
        for index, row in data_base.iterrows():
            matrix = row['C']
            symmetry = row['symmetry']
            C = StiffnessTensor(matrix, symmetry=symmetry)
            Gvoigt = C.Voigt_average().shear_modulus.mean()
            Greuss = C.Reuss_average().shear_modulus.mean()
            Gvrh = C.Hill_average().shear_modulus.mean()
            assert row['Gvoigt'] == approx(Gvoigt, rel=rel)
            assert row['Greuss'] == approx(Greuss, rel=rel)
            assert row['Gvrh'] == approx(Gvrh, rel=rel)

            C_rotated = C * rotations
            Gvoigt = C_rotated.Voigt_average().shear_modulus.mean()
            Greuss = C_rotated.Reuss_average().shear_modulus.mean()
            Gvrh = C_rotated.Hill_average().shear_modulus.mean()
            assert row['Gvoigt'] == approx(Gvoigt, rel=rel)
            assert row['Greuss'] == approx(Greuss, rel=rel)
            assert row['Gvrh'] == approx(Gvrh, rel=rel)

    def test_stiffness_cubic(self):
        crystal_symmetry_tester('Cubic')

    def test_stiffness_hexagonal(self):
        crystal_symmetry_tester('Hexagonal')

    def test_stiffness_trigonal(self):
        crystal_symmetry_tester('Trigonal', point_group='32')
        crystal_symmetry_tester('Trigonal', point_group='-3')

    def test_stiffness_tetragonal(self):
        crystal_symmetry_tester('Tetragonal', point_group='-42m')
        crystal_symmetry_tester('Tetragonal', point_group='-4')

    def test_stiffness_orthorhombic(self):
        crystal_symmetry_tester('Orthorhombic')

    def test_stiffness_monoclinic(self):
        crystal_symmetry_tester('Monoclinic', 'Diad || y')



if __name__ == '__main__':
    unittest.main()
