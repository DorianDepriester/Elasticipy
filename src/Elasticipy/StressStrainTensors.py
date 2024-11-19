import numpy as np
from Elasticipy.SecondOrderTensor import SecondOrderTensor


class StrainTensor(SecondOrderTensor):
    """
    Class for manipulating symmetric strain tensors or arrays of symmetric strain tensors.

    """
    name = 'Strain tensor'
    voigt_map = [1, 1, 1, 2, 2, 2]

    def principalStrains(self):
        """
        Values of the principals strains.

        If the tensor array is of shape [m,n,...], the results will be of shape [m,n,...,3].

        Returns
        -------
        np.ndarray
            Principal strain values
        """
        return self.eig()[0]

    def volumetricStrain(self):
        """
        Volumetric change (1st invariant of the strain tensor)

        Returns
        -------
        np.array or float
            Volumetric change
        """
        return self.I1


class StressTensor(SecondOrderTensor):
    """
    Class for manipulating stress tensors or arrays of stress tensors.
    """
    name = 'Stress tensor'

    def principalStresses(self):
        """
        Values of the principals stresses.

        If the tensor array is of shape [m,n,...], the results will be of shape [m,n,...,3].

        Returns
        -------
        np.ndarray
            Principal stresses
        """
        return np.real(self.eig()[0])

    @property
    def J2(self):
        return self.I1**2 / 3 - self.I2

    def vonMises(self):
        """
        von Mises equivalent stress.

        Returns
        -------
        np.ndarray or float
            von Mises equivalent stress

        See Also
        --------
        Tresca : Tresca equivalent stress
        """
        return np.sqrt(3 * self.J2)

    def Tresca(self):
        """
         Tresca(-Guest) equivalent stress.

         Returns
         -------
         np.ndarray or float
             Tresca equivalent stress

        See Also
        --------
        vonMises : von Mises equivalent stress
        """
        ps = self.principalStresses()
        return np.max(ps, axis=-1) - np.min(ps, axis=-1)

    def hydrostaticPressure(self):
        """
        Hydrostatic pressure

        Returns
        -------
        np.ndarray or float

        See Also
        --------
        sphericalPart : spherical part of the stress
        """
        return -self.I1/3
