from Elasticipy.SecondOrderTensor import SymmetricSecondOrderTensor
from Elasticipy.StressStrainTensors import StrainTensor
import numpy as np
from scipy.spatial.transform import Rotation

class ThermalExpansionTensor(SymmetricSecondOrderTensor):
    name = 'Thermal expansion tensor'

    def __mul__(self, other):
        if isinstance(other, Rotation):
            return super().__mul__(other)
        else:
            other = np.asarray(other)
            other_expanded = other[..., None, None]
            other_with_eye = other_expanded * np.ones(3)
            new_mat = self.matrix * other_with_eye
            return StrainTensor(new_mat)

    def apply_temperature(self, temperature, mode='pair'):
        """
        Apply temperature increase to the thermal expansion tensor, or to the array.

        Application can be made pair-wise, or considering all cross-combinations (see below).

        Parameters
        ----------
        temperature : float or numpy.ndarray
        mode : str, optional
            If "pair" (default), the temperatures are applied pair-wise on the tensor array. Broadcasting rule applies
            If "cross", all cross combinations are considered. Therefore, if ``C=A.apply_temperature(T, mode="cross")``,
            then ``C.shape=A.shape + T.shape``.

        Returns
        -------
        StrainTensor
            Strain corresponding to the applied temperature increase(s).
        """
        temperature = np.asarray(temperature)
        if mode == 'pair':
            matrix = self.matrix*temperature[...,np.newaxis,np.newaxis]
        elif mode == 'cross':
            indices_self = ALPHABET[:self.ndim]
            indices_temp = ALPHABET[:len(temperature.shape)].upper()
            ein_str = indices_self + 'ij,' + indices_temp + '->' + indices_self + indices_temp + 'ij'
            matrix = np.einsum(ein_str, self.matrix, temperature)
        else:
            raise ValueError('Invalid mode. It could be either "pair" or "cross".')
        return StrainTensor(matrix)


    def matmul(self, other):
        """
        Matrix like product with array of Rotations, resulting either in rotated ThermalExpansionTensor or StrainTensor.

        Compute the product between the tensor and an array of Rotations or a numpy array in a "matrix-product" way,*
        that is by computing each of the products. If T.shape=(m,n,o,...) and other.shape=(p,q,r,...), then::

            T.matmul(other).shape = (m,n,o,...,p,q,r,...)

        Parameters
        ----------
        other : np.ndarray or Rotation
            Value to multiply by.
        Returns
        -------
        ThermalExpansionTensor or StrainTensor
            If other is a Rotation, the tensor is just rotated, thus returning a ThermalExpansionTensor.
            If other is an array, it is assumed that it corresponds to the temperature increase, thus returning a
            StrainTensor.
        """
        if isinstance(other, Rotation):
            return super().matmul(other)
        else:
            return self.apply_temperature(other, mode='cross')


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
        return cls(np.eye(3)*alpha)

    @classmethod
    def orthotropic(cls, alpha_11, alpha_22, alpha_33):
        """
        Create an orthotropic thermal expansion tensor.

        Parameters
        ----------
        alpha_11, alpha_22, alpha_33 : float
            Thermal expansion coefficients along the first, second and third axes, respectively.

        Returns
        -------
        ThermalExpansionTensor
        """
        return cls(np.diag([alpha_11, alpha_22, alpha_33]))

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
        return cls(matrix)


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
        return cls(mat)

    @classmethod
    def transverse_isotropic(cls, alpha_11, alpha_33):
        """
        Create a thermal expansion tensor for transverse isotropic symmetry.

        Parameters
        ----------
        alpha_11 : float
            Thermal expansion coefficient along the first and second axes
        alpha_33 : float
            Thermal expansion coefficient along the third axis

        Returns
        -------
        ThermalExpansionTensor
        """
        return cls(np.diag([alpha_11, alpha_11, alpha_33]))

    @property
    def volumetric_coefficient(self):
        """ Returns the volumetric thermal expansion coefficient."""
        return self.I1