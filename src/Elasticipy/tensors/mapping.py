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
    """
    Generic class for defining the mapping convention to build a 4th-order tensor from a (6,6) matrix, and the mapping
    convention to use for reciprocal tensor.

    Attributes
    ----------
    matrix : numpy.ndarray
        (6,6) matrix evidencing the coefficient between the 4-index and the 2-index notations
    mapping_inverse : MappingConvention
        Mapping convention to use for the reciprocal tensor
    """
    matrix = np.array(KELVIN_MAPPING_MATRIX)

    @property
    def mapping_inverse(self):
        return self

class KelvinMapping(MappingConvention):
    name = 'Kelvin'

class VoigtMapping(MappingConvention):
    name = 'Voigt'

    def __init__(self, tensor='Stiffness'):
        """
        Create a Voigt mapping convention.

        Parameters
        ----------
        tensor : str
            It can be 'stiffness' or 'compliance'
            Type of tensor we define. Depending on this, the mapping convention will change (see notes).

        Notes
        -----
        For stiffness-like tensors (if ``tensor=stiffness``), the mapping matrix will be:

        .. math::

            \\begin{bmatrix}
                1 & 1 & 1 & 1 & 1 & 1\\\\
                1 & 1 & 1 & 1 & 1 & 1\\\\
                1 & 1 & 1 & 1 & 1 & 1\\\\
                1 & 1 & 1 & 1 & 1 & 1\\\\
                1 & 1 & 1 & 1 & 1 & 1\\\\
                1 & 1 & 1 & 1 & 1 & 1\\\\
            \end{bmatrix}

        Conversely, for compliance-like tensors (if ``tensor=compliance``), the mapping matrix will be:

        .. math::

            \\begin{bmatrix}
                1 & 1 & 1 & \\sqrt{2} & \\sqrt{2} & \\sqrt{2}\\\\
                1 & 1 & 1 & \\sqrt{2} & \\sqrt{2} & \\sqrt{2}\\\\
                1 & 1 & 1 & \\sqrt{2} & \\sqrt{2} & \\sqrt{2}\\\\
                \\sqrt{2} & \\sqrt{2} & \\sqrt{2} & 2 & 2 & 2\\\\
                \\sqrt{2} & \\sqrt{2} & \\sqrt{2} & 2 & 2 & 2\\\\
                \\sqrt{2} & \\sqrt{2} & \\sqrt{2} & 2 & 2 & 2\\\\
            \end{bmatrix}
        """
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