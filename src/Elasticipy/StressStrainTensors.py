import numpy as np
from Elasticipy.SecondOrderTensor import SymmetricSecondOrderTensor


class StrainTensor(SymmetricSecondOrderTensor):
    """
    Class for manipulating symmetric strain tensors or arrays of symmetric strain tensors.

    """
    name = 'Strain tensor'
    voigt_map = [1, 1, 1, 2, 2, 2]

    def principal_strains(self):
        """
        Values of the principals strains.

        If the tensor array is of shape [m,n,...], the results will be of shape [m,n,...,3].

        Returns
        -------
        np.ndarray
            Principal strain values
        """
        return self.eigvals()

    def volumetric_strain(self):
        """
        Volumetric change (1st invariant of the strain tensor)

        Returns
        -------
        np.array or float
            Volumetric change
        """
        return self.I1

    def eq_strain(self):
        return np.sqrt(3/2 * self.J2)

    def elastic_energy(self, stress):
        """
        Compute the elastic energy.

        Parameters
        ----------
        stress : StressTensor
            Corresponding stress tensor

        Returns
        -------
        Volumetric elastic energy
        """
        return 0.5 * self.ddot(stress)


class StressTensor(SymmetricSecondOrderTensor):
    """
    Class for manipulating stress tensors or arrays of stress tensors.
    """
    name = 'Stress tensor'

    def principal_stresses(self):
        """
        Values of the principals stresses.

        If the tensor array is of shape [m,n,...], the results will be of shape [m,n,...,3].

        Returns
        -------
        np.ndarray
            Principal stresses
        """
        return self.eigvals()

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
        ps = self.principal_stresses()
        return ps[...,-1] - ps[...,0]

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

    def elastic_energy(self, strain):
        """
        Compute the elastic energy.

        Parameters
        ----------
        strain : StrainTensor
            Corresponding strain tensor

        Returns
        -------
        Volumetric elastic energy
        """
        return 0.5 * self.ddot(strain)