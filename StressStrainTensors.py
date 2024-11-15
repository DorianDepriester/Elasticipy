import numpy as np

from SecondOrderTensor import SecondOrderTensor


class StrainTensor(SecondOrderTensor):
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
        return self.firstInvariant()


class StressTensor(SecondOrderTensor):
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
        p = (self.C[0, 0] - self.C[1, 1])**2 + (self.C[0, 0] - self.C[2, 2])**2 + (self.C[1, 1] - self.C[2, 2])**2 + \
            6*self.C[0, 1]**2 + 6*self.C[0, 2]**2 + 6*self.C[1, 2]**2
        return np.sqrt(0.5*p)

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
        return -self.firstInvariant()/3
