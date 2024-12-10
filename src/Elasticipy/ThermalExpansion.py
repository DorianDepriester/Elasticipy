from Elasticipy.SecondOrderTensor import SymmetricSecondOrderTensor
from StressStrainTensors import StrainTensor
import numpy as np

class ThermalExpansionTensor(SymmetricSecondOrderTensor):
    name = 'Thermal expansion tensor'

    def __init__(self, matrix, symmetry='triclinic'):
        matrix = np.asarray(matrix)
        if matrix.shape == (3, 3):
            super().__init__(matrix)
            self.symmetry = symmetry
        else:
            raise ValueError('Matrix must have shape (3, 3).')

    def apply_temperature(self, T):
        if isinstance(T, (float, int)):
            eps_mat = self.matrix * T
        else:
            T = np.asarray(T)
            eps_mat = np.einsum('ij,...->...ij', self.matrix, T)
        return StrainTensor(eps_mat)

    @classmethod
    def isotropic(cls, alpha):
        """
        Create an isotropic thermal expansion tensor.

        Parameters
        ----------
        alpha : float
            Thermal expansion coefficient.
        Returns
        -------
        ThermalExpansionTensor
        """
        return cls(np.eye(3)*alpha, symmetry='isotropic')

    @classmethod
    def orthotropic(cls, alpha_1, alpha_2, alpha_3):
        """
        Create an orthotropic thermal expansion tensor.

        Parameters
        ----------
        alpha_1, alpha_2, alpha_3 : float
            Thermal expansion coefficient along the first, second and third axes, respectively.

        Returns
        -------
        ThermalExpansionTensor
        """
        return cls(np.diag([alpha_1, alpha_2, alpha_3]), symmetry='orthotropic')

    @classmethod
    def orthorhombic(cls, *args):
        """
        Create a thermal expansion tensor corresponding to an orthotropic thermal expansion coefficient.

        This function is an alias for orthotropic().

        Parameters
        ----------
        args : list
            Orthotropic thermal expansion coefficient.

        Returns
        -------
        ThermalExpansionTensor

        See Also
        --------
        orthotropic
        """
        return cls.orthotropic(*args)

    @classmethod
    def monoclinic(cls, alpha_11, alpha_22, alpha_33, alpha_13=None, alpha_12=None):
        """
        Create a thermal expansion tensor for monoclinic symmetry.

        If alpha_13, the Diad || z is assumed. If alpha_12, the Diad || z is assumed. Therefore, these two parameters
        are exclusive.

        Parameters
        ----------
        alpha_11, alpha_22, alpha_33 : float
            Thermal expansion coefficient along the first, second and third axes, respectively.
        alpha_13 : float, optional
            Thermal expansion coefficient corresponding to XZ shear (for Diad || y)
        alpha_12: float, optional
            Thermal expansion coefficient corresponding to XY shear (for Diad || z)

        Returns
        -------
        ThermalExpansionTensor
        """
        matrix = np.diag([alpha_11, alpha_22, alpha_33])
        if (alpha_13 is not None) and (alpha_12 is None):
            matrix[0, 2] = matrix[2, 0]= alpha_13
        elif (alpha_12 is not None) and (alpha_13 is None):
            matrix[0, 1] = matrix[1, 0]= alpha_12
        elif (alpha_13 is not None) and (alpha_12 is not None):
            raise ValueError('alpha_13 and alpha_12 cannot be used together.')
        else:
            raise ValueError('Either alpha_13 or alpha_12 must be provided.')
        return cls(matrix, symmetry='monoclinic')


    @classmethod
    def triclinic(cls, alpha_11=0., alpha_12=0., alpha_13=0., alpha_22=0., alpha_23=0., alpha_33=0.):
        """
        Create a thermal expansion tensor for triclinic symmetry.

        Parameters
        ----------
        alpha_11, alpha_12, alpha_13, alpha_22, alpha_23, alpha_33 : float
            Values of the thermal expansion coefficients

        Returns
        -------
        ThermalExpansionTensor
        """
        mat = [[alpha_11, alpha_12, alpha_13],
               [alpha_12, alpha_22, alpha_23],
               [alpha_13, alpha_23, alpha_33]]
        return cls(np.diag(mat), symmetry='triclinic')

    @classmethod
    def transverse_isotropic(cls, alpha_1, alpha_3):
        """
        Create a thermal expansion tensor for transverse isotropic symmetry.

        Parameters
        ----------
        alpha_1 : float
            Thermal expansion coefficient along the first and second axes
        alpha_3 : float
            Thermal expansion coefficient along the third axis

        Returns
        -------
        ThermalExpansionTensor
        """
        return cls(np.diag([alpha_1, alpha_1, alpha_3]), symmetry='transverse isotropic')
