import numpy as np
from elasticipy.tensors.second_order import SecondOrderTensor, SymmetricSecondOrderTensor
from elasticipy.tensors.stress_strain import StrainTensor

class DeformationGradient(SecondOrderTensor):
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
        """
        return self.Green_Lagrangian()

    def small_strain(self):
        """
        Under the small strain assumption, make first order approximation to compute the small strain tensor.

        Under the small strain assumption, we have:

        .. math::

            \\mathbf{\\epsilon} = \\frac12 (\\nabla\\mathbf{U} + \\nabla^\\top\\mathbf{U})

        with

        .. math::

            \\nabla\\mathbf{U} = \\mathbf{F} - \\mathbf{I}

        Returns
        -------
        StrainTensor
            Small strain tensor
        """
        gradU = self - SymmetricSecondOrderTensor.eye(shape=self.shape)
        return StrainTensor(gradU, force_symmetry=True)

class CauchyGreenTensor(SymmetricSecondOrderTensor):
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

            \\bar{I}_1 = I_1J**{-2/3}

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

            \\bar{I}_2 = I_2J**{-4/3}

        Returns
        -------
        float or numpy.ndarray
        """
        return self.I2 * self.J ** (-4 / 3)