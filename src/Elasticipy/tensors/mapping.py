import numpy as np

a = np.sqrt(2)
KELVIN_MAPPING_MATRIX = np.array([[1, 1, 1, a, a, a],
                                  [1, 1, 1, a, a, a],
                                  [1, 1, 1, a, a, a],
                                  [a, a, a, 2, 2, 2],
                                  [a, a, a, 2, 2, 2],
                                  [a, a, a, 2, 2, 2], ])

VOIGT_MAPPING_MATRIX_COMPLIANCE = [[1, 1, 1, 2, 2, 2],
                                  [1, 1, 1, 2, 2, 2],
                                  [1, 1, 1, 2, 2, 2],
                                  [2, 2, 2, 4, 4, 4],
                                  [2, 2, 2, 4, 4, 4],
                                  [2, 2, 2, 4, 4, 4]]

class MappingConvention:
    matrix = np.array(KELVIN_MAPPING_MATRIX)

    @property
    def mapping_inverse(self):
        return self

class KelvinMapping(MappingConvention):
    name = 'Kelvin'

class VoigtMapping(MappingConvention):
    name = 'Voigt'

    def __init__(self, tensor='Stiffness'):
        if tensor == 'Stiffness':
            self.matrix = np.ones((6,6))
            self.tensor_type = 'Stiffness'
        else:
            self.matrix = np.array(VOIGT_MAPPING_MATRIX_COMPLIANCE)
            self.tensor_type = 'Compliance'

    @property
    def mapping_inverse(self):
        if self.tensor_type == 'Stiffness':
            return VoigtMapping(tensor='Compliance')
        else:
            return VoigtMapping(tensor='Stiffness')