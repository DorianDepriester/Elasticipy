from elasticipy.tensors.stress_strain import StressTensor
from elasticipy.tensors.second_order import SymmetricSecondOrderTensor
from abc import ABC
import numpy as np

class HyperElastic(ABC):
    def potential(self, F):
        pass

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

    def potential(self, F):
        I1 = F.B.I1
        I2 = F.B.I2
        W = np.zeros_like(I1)
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                W += C[i, j] * (I1-3)**i * (I2 - 3)**j
        if D is not None:
            J = F.J
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