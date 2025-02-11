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
        numpy.ndarray or float
            Volumetric change
        """
        return self.I1

    def eq_strain(self):
        """von Mises equivalent strain"""
        return np.sqrt(2/3 * self.ddot(self))

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
        return ps[...,0] - ps[...,-1]

    def hydrostatic_pressure(self):
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

    def elastic_energy(self, strain, mode='pair'):
        """
        Compute the elastic energy.

        Parameters
        ----------
        strain : StrainTensor
            Corresponding elastic strain tensor
        mode : str, optional
            If 'pair' (default), the elastic energies are computed element-wise. Broadcasting rule applies.
            If 'cross', each cross-combination of stress and strain are considered.

        Returns
        -------
        numpy.ndarray
            Volumetric elastic energy
        """
        return 0.5 * self.ddot(strain, mode=mode)