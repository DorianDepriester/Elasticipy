import numpy as np
from elasticipy.tensors.second_order import SecondOrderTensor, SymmetricSecondOrderTensor
from elasticipy.tensors.stress_strain import StrainTensor

class DeformationGradient(SecondOrderTensor):
    name = "Deformation Gradient tensor"

    def __init__(self, mat):
        J = np.linalg.det(mat)
        if np.any(np.asarray(J) <= 0):
            raise ValueError("The determinant of the deformation tensor must be positive.")
        super().__init__(mat)

    @classmethod
    def rand(cls, **kwargs):
        a = SymmetricSecondOrderTensor.rand(**kwargs)
        a = a * a.I3
        return cls(a.matrix)

    @property
    def U(self):
        """
        Right stretch tensor

        Returns the symmetric Second-order tensor `U` such that:

        .. math::

            \\mathbf{F} = \\mathbf{R}\\cdot\\mathbf{U}

        where :math:`\\mathbf{U}` is the positive definite and :math:`\\mathbf{F}` is an orthogonal matrix.

        Returns
        -------
        SymmetricSecondOrderTensor

        See Also
        --------
        V : Left stretch tensor
        """
        _, U = self.polar()
        return U

    @property
    def V(self):
        """
        Left stretch tensor

        Returns the symmetric Second-order tensor `U` such that:

        .. math::

            \\mathbf{F} = \\mathbf{V}\\cdot\\mathbf{R}

        where :math:`\\mathbf{V}` is the positive semi definite and :math:`\\mathbf{F}` is an orthogonal matrix.

        Returns
        -------
        SymmetricSecondOrderTensor

        See Also
        --------
        U : Right stretch tensor
        """
        _, V = self.polar(side='left')
        return V

    @property
    def R(self):
        """
        Rotational part of the polar decomposition

        Returns the orthogonal matrix `R` such that:

        .. math::

            \\mathbf{F} = \\mathbf{R}\\cdot\\mathbf{U}

        Returns
        -------
        numpy.ndarray
            Orthogonal matrix corresponding to the rotation part of the polar decomposition

        See Also
        --------
        U : Right stretch tensor
        V : Left stretch tensor
        """
        R, _ = self.polar()
        return R

    @property
    def J(self):
        """
        Compute the determinant of the tensor (Jacobian).

        It is actually an alias for ``F.I3``

        Returns
        -------
        float or numpy.ndarray
            Determinant of the gradient tensor, or array of determinants
        """
        return self.I3

    @property
    def C(self):
        """
        Compute the right Cauchy-Green tensor.

        The right Cauchy-Green tensor is defined as:

        .. math::

            \\mathbf{C} = \\mathbf{F}^\\top\\cdot\\mathbf{F}

        Returns
        -------
        CauchyGreenTensor

        See Also
        --------
        E : Green-Lagrangian tensor
        B : Left Cauchy-Green tensor
        """
        return CauchyGreenTensor(self.T.dot(self))

    @property
    def B(self):
        """
        Compute the left Cauchy-Green tensor.

        The left Cauchy-Green tensor is defined as:

        .. math::

            \\mathbf{C} = \\mathbf{F}\\cdot\\mathbf{F}^\\top

        Returns
        -------
        CauchyGreenTensor

        See Also
        --------
        E : Green-Lagrangian tensor
        C : Right Cauchy-Green tensor
        """
        return CauchyGreenTensor(self.dot(self.T))

    def Green_Lagrangian(self):
        """
        Compute the Green-Lagrangian tensor from the deformation tensor F.

        The Green-Lagrangian tensor is defined as:

        .. math::

            \\mathbf{E} = \\frac12\\left(\\mathbf{C} - \\mathbf{I}\\right)

        where :math:`\\mathbf{C}` is the right Cauchy-Green tensor.

        Returns
        -------
        SymmetricSecondOrderTensor
            Green-Lagrangian tensor

        See Also
        --------
        C : Right Cauchy-Green tensor
        B : Left Cauchy-Green tensor
        """
        I = SymmetricSecondOrderTensor.eye(shape=self.shape)
        return 0.5 * (self.C - I)

    @property
    def E(self):
        """
        Compute the Green-Lagrangian tensor.

        It is actually an alias for ``F.Green_Lagrangian()``.

        Returns
        -------
        SymmetricSecondOrderTensor
            Green-Lagrangian tensor

        See Also
        --------
        C : Right Cauchy-Green tensor
        B : Left Cauchy-Green tensor
        small_strain : Compute the associated small strain tensor
        """
        return self.Green_Lagrangian()

    def small_strain(self):
        """
        Under the small strain assumption (SSA), make first order approximation to compute the small strain tensor.

        Under the SSA, we have:

        .. math::

            \\mathbf{\\epsilon} = \\frac12 (\\nabla\\mathbf{U} + \\nabla^\\top\\mathbf{U})

        with

        .. math::

            \\nabla\\mathbf{U} = \\mathbf{F} - \\mathbf{I}

        Returns
        -------
        StrainTensor
            Small strain tensor

        See Also
        --------
        E : Green-Lagrangian tensor
        """
        gradU = self - SymmetricSecondOrderTensor.eye(shape=self.shape)
        return StrainTensor(gradU, force_symmetry=True)

    @classmethod
    def tensile(cls, u, magnitude):
        t = SecondOrderTensor.tensile(u, magnitude)
        I = SecondOrderTensor.eye(shape=t.shape)
        return cls(t.matrix + I.matrix)

    @classmethod
    def shear(cls, u, v, magnitude):
        t = SecondOrderTensor.shear(u, v, magnitude)
        I = SecondOrderTensor.eye(shape=t.shape)
        return cls(t.matrix + I.matrix)

    @classmethod
    def isochoric_tensile(cls, u, stretch):
        """
        Create a tensor, or a tensor array, corresponding to the isochoric tensile state (i.e. det(F)=1).

        Parameters
        ----------
        u : list or tuple or numpy.ndarray
            direction of principal stretch
        stretch : float or list of floats or numpy.ndarray
            Value(s) of the stretch (rel. elongation) along the tensile direction

        Returns
        -------
        DeformationGradient

        See Also
        --------
        tensile : create a gradient tensor corresponding to the pure tensile state
        shear : create a gradient tensor corresponding to the pure shear state

        Examples
        --------
        Create a gradient tensor corresponding to stretch along the first direction ([1,0,0]), whereas the stretch along
        all orthogonal directions are such that det(F)=1:

        >>> from elasticipy.tensors.finite_strain import DeformationGradient
        >>> F = DeformationGradient.isochoric_tensile([1,0,0], stretch=1)
        >>> print(F)
        Second-order tensor
        [[2.         0.         0.        ]
         [0.         0.70710678 0.        ]
         [0.         0.         0.70710678]]

        The value for the stretch can be a list (or a numpy array), e.g.:

        >>> F = DeformationGradient.isochoric_tensile([1,0,0], stretch=[1,2,3,4])
        >>> print(F)
        Second-order tensor
        Shape=(4,)

        >>> print(F[-1])
        Second-order tensor
        [[4.  0.  0. ]
         [0.  0.5 0. ]
         [0.  0.  0.5]]

        One can check that the determinant is always one:

        >>> print(F.I3)
        [1. 1. 1. 1.]
        """
        u = u / np.linalg.norm(u)
        p = np.einsum('i,j->ij', u, u)
        q = np.eye(3) - p
        stretch = np.asarray(stretch)
        Fuu = stretch + 1
        einsum = 'ij,...->...ij'
        a = np.einsum(einsum, p, Fuu)
        b = np.einsum(einsum, q, Fuu ** (-0.5))
        return cls(a + b)

    def elongation(self, u):
        """
        Compute the relative elongation of the material along a given direction.

        Parameters
        ----------
        u : list or tuple or numpy.ndarray
            direction along which one wants to compute the elongation

        Returns
        -------
        float or numpy.ndarray
            relative elongation
        """
        C = self.C
        u = u / np.linalg.norm(u)
        a = np.einsum('i,...ij,j->...', u, C.matrix, u)
        return np.maximum(a, 0.)**0.5 -1

    def volumetric_strain(self):
        """
        Compute the volumetric strain.

        It is defined as:

        .. math::

            \\frac{\\Delta v}{v} = det(\\mathbf{F}) - 1

        Returns
        -------
        float or numpy.ndarray
            Volumetric relative change
        """
        return self.I3 - 1

class CauchyGreenTensor(SymmetricSecondOrderTensor):
    name = "Cauchy-Green tensor"

    @property
    def J(self):
        """
        Determinant of the gradient (Jacobian).

        It is computed as follows:

        .. math::

            J = \\sqrt{det(\\mathbf{C})}

        Returns
        -------

        """
        return np.sqrt(self.I3)

    @property
    def I1_bar(self):
        """
        First invariant of the Cauchy-Green tensor, accounting for the spherical part of deformation

        It is defined as:

        .. math::

            \\bar{I}_1 = I_1J^{-2/3}

        Returns
        -------
        float or numpy.ndarray
        """
        return self.I1 * self.J**(-2 / 3)

    @property
    def I2_bar(self):
        """
        Second invariant of the Cauchy-Green tensor, accounting for the spherical part of deformation

        It is defined as:

        .. math::

            \\bar{I}_2 = I_2J^{-4/3}

        Returns
        -------
        float or numpy.ndarray
        """
        return self.I2 * self.J ** (-4 / 3)