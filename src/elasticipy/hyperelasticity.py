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
        B : CauchyGreenTensor
            Left Cauchy-Green tensor

        Returns
        -------
        float
            Potential function
        """
        pass

    def is_compressible(self):
        """
        Check whether the model corresponds to compressible behaviour or not.

        Returns
        -------
        bool
            True if the behaviour is compressible, False otherwise.
        """
        return True

    def _potential_from_B_voigt(self, b_flat):
        B = SymmetricSecondOrderTensor.from_Voigt(b_flat)
        return self.potential_from_B(B)

    def _stress_from_derivative(self, B, h):
        b = B.to_Voigt()
        dWdB = approx_fprime(b, self._potential_from_B_voigt, h)
        dWdB_full = SymmetricSecondOrderTensor.from_Voigt(dWdB)
        J = np.sqrt(B.I3)
        return 2 / J * StressTensor(dWdB_full.dot(B))

    def stress_from_B(self, B, h=1e-6):
        """
        Compute the Cauchy stress tensor from the left Cauchy-Green tensor.

        If not hard-coded, the stress tensor is computed by finite difference from the derivative of the potential
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
        return self._stress_from_derivative(B, h)

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
        J = F.J
        if not self.is_compressible() and np.any(np.abs(J-1)>1e-6):
            raise ValueError("For incompressible behaviour, the determinant of the gradient must be 1.")
        return self.stress_from_B(F.B, h=h)

class MooneyRivlin(HyperElastic):
    def __init__(self, C, D=None):
        """
        Create a Mooney-Rivlin hyper-elastic model

        Parameters
        ----------
        C : float or list of list or numpy.ndarray
            Material constants relative to deviatoric parts, provided as a NxN matrix (see notes)
        D : list of float or numpy.ndarray, optional
            Material constants relative to volumetric part. If not provided, the material is supposed to be
            incompressible.

        Notes
        -----
        The compressible Mooney-Rivlin potential is defined as:

        .. math::

            W = \\sum_{i,j=0}^N C_{ij}(\\bar{I}_1-3)^i(\\bar{I}_2-3)^j + \\sum_{k=1}^M\\frac{(J-1)^{2k}}{D_k}

        with

        .. math::

            \\bar{I}_1 = I_1J^{-2/3}
            \\bar{I}_2 = I_2J^{-4/3}
        """
        self.C = np.asarray(C)
        self.D = D

    def is_compressible(self):
        return self.D is not None

    def potential_from_B(self, B):
        J = B.J
        W = np.zeros_like(J)
        C = self.C
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                W += C[i, j] * (B.I1_bar-3)**i * (B.I2_bar - 3)**j
        D = self.D
        if D is not None:
            for k, Dk in enumerate(self.D, start=1):
                W += (J-1)**(2*k) / Dk
        return W

class NeoHooke(MooneyRivlin):
    def __init__(self, C, D=None):
        """
        Create a Neo-Hooke hyperelastic model

        The compressible Neo-Hooke model defines the potential function as:

        .. math::

             W = C(\bar{I}_1 - 3) + \\frac{(J-1)^2}{D}

        where :math:`C` and :math:`D` are the material constants and :math:`\bar{I}_1=I_1J^{-2/3}`. If D is None
        (default), the material is assumed to be incompressible and the corresponding part in the equation above is
        omitted.

        Parameters
        ----------
        C : float
            Material constant relative to deviatoric part
        D : float, optional
            Material compressibility
        """
        super().__init__(C, D=D)

    def potential_from_B(self, B):
        if self.D is None:
            return self.C * (B.I1 - 3)
        else:
            return self.C * (B.I1_bar - 3) + (B.J - 1)**2 / self.D

    def stress_from_B(self, B, **kwargs):
        J = B.J
        tau = 2 * self.C / J ** (2/3) * B.deviatoric_part()
        if self.D is not None:
            p = 2 * J * (J-1) / self.D
            tau = tau + p * SymmetricSecondOrderTensor.eye(shape=gradient.shape)
        return StressTensor(tau / J)