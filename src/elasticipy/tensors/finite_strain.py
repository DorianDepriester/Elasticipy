from elasticipy.tensors.second_order import SecondOrderTensor, SymmetricSecondOrderTensor

class DeformationGradient(SecondOrderTensor):
    @property
    def U(self):
        """
        Right stretch tensor

        Returns
        -------
        SymmetricSecondOrderTensor
        """
        _, U = self.polar()
        return U

    @property
    def V(self):
        """
        Left stretch tensor

        Returns
        -------
        SymmetricSecondOrderTensor
        """
        _, V = self.polar(side='left')
        return V

    @property
    def R(self):
        """
        Rotational part of the polar decomposition

        Returns
        -------
        numpy.ndarray
            Orthogonal matrix corresponding to the rotation part of the polar decomposition
        """
        R, _ = self.polar()
        return R

    @property
    def C(self):
        """
        Compute the right Cauchy-Green tensor.

        The right Cauchy-Green tensor is defined as:

        ..math::

            \\mathbf{C} = \\mathbf{F}^\top\\cdot\\mathbf{F}

        Returns
        -------
        SymmetricSecondOrderTensor
        """
        return SymmetricSecondOrderTensor(self.T.dot(self))

    @property
    def B(self):
        """
        Compute the left Cauchy-Green tensor.

        The left Cauchy-Green tensor is defined as:

        ..math::

            \\mathbf{C} = \\mathbf{F}\\cdot\\mathbf{F}^\\top

        Returns
        -------
        SymmetricSecondOrderTensor
        """
        return SymmetricSecondOrderTensor(self.dot(self.T))

    def Green_Lagrangian(self):
        """
        Compute the Green-Lagrangian tensor from the deformation tensor F.

        The Green-Lagrangian tensor is defined as:

        ..math::

            E = \\frac12\\left(\\mathbf{C} - \\mathbf{I}\\right)

        Returns
        -------
        SymmetricSecondOrderTensor
            Green-Lagrangian tensor
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