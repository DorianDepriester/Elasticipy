import numpy as np
from Elasticipy.tensors.second_order import SymmetricSecondOrderTensor


class StrainTensor(SymmetricSecondOrderTensor):
    """
    Class for manipulating symmetric strain tensors or arrays of symmetric strain tensors.

    """
    name = 'Strain tensor'
    _voigt_map = [1, 1, 1, 2, 2, 2]

    def principal_strains(self):
        """
        Values of the principals strains.

        If the tensor array is of shape [m,n,...], the results will be of shape [m,n,...,3].

        Returns
        -------
        np.ndarray
            Principal strain values

        Examples
        --------
        For a single strain value, the principal strain values are composed of 3 float. E.g.:

        >>> from Elasticipy.tensors.stress_strain import StrainTensor
        >>> eps = StrainTensor.shear([1,0,0],[0,1,0],1e-3)
        >>> eps.principal_strains()
        array([ 0.001,  0.   , -0.001])

        For strain tensor array, the shape of the returned array will depend on that of the strain array. E.g.:

        >>> tau = [1,2,3,4] # increasing magnitude
        >>> eps_2d = StrainTensor.shear([1,0,0],[0,1,0],tau)/1000
        >>> eps_2d.principal_strains()
        array([[ 0.001,  0.   , -0.001],
               [ 0.002,  0.   , -0.002],
               [ 0.003,  0.   , -0.003],
               [ 0.004,  0.   , -0.004]])
        """
        return self.eigvals()

    def volumetric_strain(self):
        """
        Volumetric change (1st invariant of the strain tensor)

        Returns
        -------
        numpy.ndarray or float
            Volumetric change

        Examples
        --------
        At first, try with pure shear:

        >>> from Elasticipy.tensors.stress_strain import StrainTensor
        >>> eps = StrainTensor.shear([1,0,0],[0,1,0],1e-3)
        >>> eps.volumetric_strain()
        0.0

        Now try with hydrastatic straining:

        >>> import numpy as np
        >>> eps_hydro = StrainTensor(-np.eye(3)) / 1000
        >>> eps_hydro
        Strain tensor
        [[-0.001 -0.    -0.   ]
         [-0.    -0.001 -0.   ]
         [-0.    -0.    -0.001]]
        >>> eps_hydro.volumetric_strain()
        -0.003
        """
        return self.I1

    def eq_strain(self):
        """von Mises equivalent strain

        Returns
        -------
        numpy.ndarray or float
            If the input tensor is single, the result will be float. Instead, an Numpy array will be returned.

        Notes
        -----
        The von Mises equivalent strain is defined as:

        .. math::

            \\sqrt{\\frac23 \\varepsilon_{ij}\\varepsilon_{ij}}

        Examples
        --------
        >>> from Elasticipy.tensors.stress_strain import StrainTensor
        >>> StrainTensor.tensile([1,0,0], 1e-3).eq_strain()
        0.000816496580927726
        >>> StrainTensor.shear([1,0,0],[0,1,0], 1e-3).eq_strain()
        0.0011547005383792514
        """
        return np.sqrt(2/3 * self.ddot(self))

    def elastic_energy(self, stress, mode='pair'):
        """
        Compute the elastic energy.

        Parameters
        ----------
        stress : StressTensor
            Corresponding stress tensor
        mode : str, optional
            If 'pair' (default), the elastic energies are computed element-wise. Broadcasting rule applies.
            If 'cross', each cross-combination of stress and strain are considered.

        Returns
        -------
        float or numpy.ndarray
            Volumetric elastic energy

        Examples
        --------
        Let consider an isotropic material (e.g. steel), undergoing tensile strain. In order to compute the volumetric
        elastic energy, one can do the following:

        >>> from Elasticipy.tensors.stress_strain import StrainTensor
        >>> from Elasticipy.tensors.elasticity import StiffnessTensor
        >>> C = StiffnessTensor.isotropic(E=210e3, nu=0.28)
        >>> eps = StrainTensor.tensile([1,0,0],1e-3) # Define the strain
        >>> sigma = C * eps # Compute the stress
        >>> sigma
        Stress tensor
        [[268.46590909   0.           0.        ]
         [  0.         104.40340909   0.        ]
         [  0.           0.         104.40340909]]

        Then, the volumetric elastic energy is:

        >>> eps.elastic_energy(sigma)
        0.13423295454545456
        """
        return 0.5 * self.ddot(stress, mode=mode)

    def draw_Mohr_circles(self):
        fig, ax = super().draw_Mohr_circles()
        ax.set_xlabel(ax.get_xlabel() + ' strain')
        ax.set_ylabel(ax.get_ylabel() + ' strain')
        return fig, ax


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

        Examples
        --------
        For (single-valued) tensile stress:

        >>> from Elasticipy.tensors.stress_strain import StressTensor
        >>> sigma = StressTensor.tensile([1,0,0],1)
        >>> sigma.vonMises()
        1.0

        For (single-valued) shear stress:

        >>> sigma = StressTensor.shear([1,0,0],[0,1,0],1)
        >>> sigma.vonMises()
        1.7320508075688772

        For arrays of stresses :

        >>> import numpy as np
        >>> sigma_xx = np.linspace(0,1,5)
        >>> sigma_xy = np.linspace(0,1,5)
        >>> sigma = StressTensor.tensile([1,0,0],sigma_xx) + StressTensor.shear([1,0,0],[0,1,0],sigma_xy)
        >>> sigma.vonMises()
        array([-0. ,  0.5,  1. ,  1.5,  2. ])
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

        Examples
        --------
        For (single-valued) tensile stress:

        >>> from Elasticipy.tensors.stress_strain import StressTensor
        >>> sigma = StressTensor.tensile([1,0,0],1)
        >>> sigma.Tresca()
        1.0

        For (single-valued) shear stress:

        >>> sigma = StressTensor.shear([1,0,0],[0,1,0],1)
        >>> sigma.Tresca()
        2.0

        For arrays of stresses :

        >>> import numpy as np
        >>> sigma_xx = np.linspace(0,1,5)
        >>> sigma_xy = np.linspace(0,1,5)
        >>> sigma = StressTensor.tensile([1,0,0],sigma_xx) + StressTensor.shear([1,0,0],[0,1,0],sigma_xy)
        >>> sigma.Tresca()
        array([0.        , 0.55901699, 1.11803399, 1.67705098, 2.23606798])
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

    def draw_Mohr_circles(self):
        """
        Draw the Mohr circles of the stress tensor.

        This function only works for single-valued tensors.

        Returns
        -------
        fig : matplotlib.figure.Figure
            handle to Matplotlib figure
        ax : matplotlib.axes.Axes
            handle to Matplotlib axes

        Examples
        --------
        In order to illustrate this function, we consider a triaxial tensile stress:

        >>> from Elasticipy.tensors.stress_strain import StressTensor
        >>> sigma = StressTensor.tensile([1,0,0],1) + StressTensor.tensile([0,1,0],3)

        The princiapl stresses are obviously 0, 1 and 2:

        >>> sigma.principal_stresses()
        array([3., 1., 0.])

        These principal stresses can be directly plotted with:

        >>> fig, ax = sigma.draw_Mohr_circles()
        >>> fig.show()
        """
        fig, ax = super().draw_Mohr_circles()
        ax.set_xlabel(ax.get_xlabel() + ' stress')
        ax.set_ylabel(ax.get_ylabel() + ' stress')
        return fig, ax

    def triaxiality(self):
        """
        Compute the stress triaxiality.

        It is defined as the hydrostatic stress to the von Mises equivalent stress ratio (see Notes).

        Returns
        -------
        float or np.ndarray
            Stress triaxiality. A float is returned if the tensor is single-valued, otherwise, an array of the same
            shape of the tensor is returned.

        See Also
        --------
        hydrostatic_pressure : compute the hydrostatic pressure
        vonMises : compute the von Mises equivalent stress
        Lode_angle : compute the Lode angle

        Notes
        -----
        The stress triaxiality is defined as follows:

        .. math::

            \\eta = \\frac{-p}{\\sigma_{vM}}

        where :math:`p` and :math:`\\sigma_{vM}` are the hydrostatic pressure and the von Mises equivalent stress,
        respectively.

        Examples
        --------
        One can check that the stress triaxiality is for simple tensile is 1/3:

        >>> from Elasticipy.tensors.stress_strain import StressTensor
        >>> s1 = StressTensor.tensile([1,0,0],1.)
        >>> s1.triaxiality()
        0.3333333333333333

        For a stress array (e.g. for biaxial tensile stress):

        >>> s2 = s1 + StressTensor.tensile([0,1,0],[0,0.5,1])
        >>> s2.triaxiality()
        array([0.33333333, 0.57735027, 0.66666667])
        """
        return self.I1 / self.vonMises() / 3