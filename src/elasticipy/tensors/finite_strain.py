from elasticipy.tensors.second_order import SecondOrderTensor, SymmetricSecondOrderTensor

class DeformationGradient(SecondOrderTensor):
    @property
    def U(self):
        """
        Right stretch tensor

        Returns the symmetric Second-order tensor `U` such that:

        .. math::

            \\mathbf{F} = \\mathbf{R}\\cdot\\mathbf{U}

        where :math:`\\mathbf{R}` is the positive semi definite

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

        where :math:`\\mathbf{R}` is the positive semi definite

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
        SymmetricSecondOrderTensor

        See Also
        --------
        E : Green-Lagrangian tensor
        B : Left Cauchy-Green tensor
        """
        return SymmetricSecondOrderTensor(self.T.dot(self))

    @property
    def B(self):
        """
        Compute the left Cauchy-Green tensor.

        The left Cauchy-Green tensor is defined as:

        .. math::

            \\mathbf{C} = \\mathbf{F}\\cdot\\mathbf{F}^\\top

        Returns
        -------
        SymmetricSecondOrderTensor

        See Also
        --------
        E : Green-Lagrangian tensor
        C : Right Cauchy-Green tensor
        """
        return SymmetricSecondOrderTensor(self.dot(self.T))

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
            Green-Lagrangian tensor.
        """
        return self.Green_Lagrangian()