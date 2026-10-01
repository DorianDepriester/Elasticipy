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