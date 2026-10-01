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
        R, _ = self.polar()
        return R