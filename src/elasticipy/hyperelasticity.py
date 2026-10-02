from elasticipy.tensors.stress_strain import StressTensor
from elasticipy.tensors.second_order import SymmetricSecondOrderTensor
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import approx_fprime

class HyperElastic(ABC):

    def potential_from_F(self, F):
        """
        Compute the potential function from the deformation gradient F.

        Parameters
        ----------
        F : DeformationGradient
            tensor or tensor array of defomation gradients

        Returns
        -------
        float
            Potential function

        See Also
        --------
        potential_from_B : compute the potential function from the left Cauchy-Green tensor
        """
        return self.potential_from_B(F.B)

    @abstractmethod
    def potential_from_B(self, B):
        """
        Compute the potential function from the left Cauchy-Green tensor

        Parameters
        ----------
        B : SymmetricSecondOrderTensor
            Left Cauchy-Green tensor

        Returns
        -------
        float
            Potential function
        """
        pass

    def _potential_from_B_voigt(self, b_flat):
        B = SymmetricSecondOrderTensor.from_Voigt(b_flat)
        return self.potential_from_B(B)

    def stress_from_B(self, B, h=1e-6):
        """
        Compute the Cauchy stress tensor from the left Cauchy-Green tensor.

        If not implemented, the stress tensor is computed by finite difference from the derivative of the potential
        function (see Notes).

        Parameters
        ----------
        B :  SymmetricSecondOrderTensor
            Left Cauchy-Green tensor
        h : float, optional
            step size to perform finite difference calculation.

        Returns
        -------
        StressTensor

        Notes
        -----
        The Cauchy stress tensor derives from the elastic potential function according to the following formula:

        .. math::

            \\mathbf{\\sigma} = \\frac2J \\frac{\\partial W}{\\partial \\mathbf{B}}\\cdot\\mathbf{B}
        """
        b = B.to_Voigt()
        dWdB = approx_fprime(b, self._potential_from_B_voigt, h)
        dWdB_full = SymmetricSecondOrderTensor.from_Voigt(dWdB)
        J = np.sqrt(B.I3)
        return 2 / J * StressTensor(dWdB_full.dot(B))

    def stress_from_F(self, F, h=1e-6):
        """
        Compute the stress tensor from the deformation gradient tensor

        Parameters
        ----------
        F : DeformationGradient
            Deformation gradient tensor
        h : float, optional
            step size to perform finite difference calculation.

        Returns
        -------
        StressTensor
        """
        return self.stress_from_B(F.B, h=h)

class MooneyRivlin(HyperElastic):
    def __init__(self, C, D=None):
        """
        Create a Mooney-Rivlin hyper-elastic model

        Parameters
        ----------
        C : list of list or numpy.ndarray
            Material constants relative to deviatoric parts.
        D : list of float or numpy.ndarray, optional
            Material constants relative to volumetric part. If not provided, the material is supposed to be
            incompressible.

        Notes
        -----
        The compressible Mooney-Rivlin potential is defined as:

        .. math::

            W = \\sum_{i,j=0}^N C_{ij}(I_1-3)î(I_2-3)^j + \\sum_{k=1}\\frac{(J-1)^{2k}}{D_k}
        """
        self.C = np.asarray(C)
        self.D = D

    def potential_from_B(self, B):
        I1 = B.I1
        I2 = B.I2
        J = np.sqrt(B.I3)
        I1_bar = J ** (-2 / 3) * I1
        I2_bar = J ** (-4 / 3) * I2
        W = np.zeros_like(I1)
        C = self.C
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                W += C[i, j] * (I1_bar-3)**i * (I2_bar - 3)**j
        D = self.D
        if D is not None:
            for k in range(len(D)):
                W += (J-1)**(2*k + 2) / D[k]
        return W

class NeoHooke(HyperElastic):
    def __init__(self, C10, D1):
        self.C10 = C10
        self.D1 = D1

    def stress_from_gradient(self, gradient):
        J = gradient.J
        p = 2 * self.D1 * J * (J-1)
        I = SymmetricSecondOrderTensor.eye(shape=gradient.shape)
        tau = p * I + 2 * self.C10 / J**(2/3) * gradient.B.deviatoric_part()
        return StressTensor(tau / J)