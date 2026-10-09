from elasticipy.tensors.finite_strain import CauchyGreenTensor, DeformationGradient
from elasticipy.tensors.stress_strain import StressTensor
from elasticipy.tensors.second_order import SymmetricSecondOrderTensor
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
import numpy as np

class HyperElastic(ABC):
    def potential_from_F(self, F):
        """
        Compute the potential function from the deformation gradient F.

        Parameters
        ----------
        F : DeformationGradient
            tensor or tensor array of deformation gradients

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
        B = CauchyGreenTensor.from_Voigt(b_flat)
        return self.potential_from_B(B)

    def stress_from_B_analytical(self, B):
        raise NotImplementedError()

    def stress_from_derivative(self, B, h=1e-6):
        """
        Compute the stress tensor from the derivative of the potential function wrt. the left Cauchy-Green tensor B.

        Parameters
        ----------
        B : CauchyGreenTensor
            Left Cauchy-Green tensor
        h : step size to use for finite differences

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
        h_mat = np.eye(6) * h
        shape = B.shape + (6,)
        dWdB_voigt = np.zeros(shape)
        for i in range(6):
            DWp = self._potential_from_B_voigt(b + h_mat[i])
            DWm = self._potential_from_B_voigt(b - h_mat[i])
            dWdB_voigt[..., i] = (DWp - DWm) / 2 / h
        dWdB_full = SymmetricSecondOrderTensor.from_Voigt(dWdB_voigt)
        J = np.sqrt(B.I3)
        s = 2 * StressTensor(dWdB_full.dot(B), force_symmetry=True) / J
        if self.is_compressible():
            return s
        else:
            return s.deviatoric_part()

    def stress_from_B(self, B, h=1e-6):
        """
        Compute the Cauchy stress tensor from the left Cauchy-Green tensor.

        If the method `stress_from_B_analytical` is not implemented, the stress tensor is computed by finite difference
        from the derivative of the potential function (`stress_from_derivative`).

        Parameters
        ----------
        B :  CauchyGreenTensor
            Left Cauchy-Green tensor
        h : float, optional
            step size to perform finite difference calculation.

        Returns
        -------
        StressTensor

        Notes
        -----
        If the material is incompressible, only the deviatoric part of the stress tensor is returned.

        See Also
        --------
        stress_from_B_analytical : analytical expression of the stress tensor as a function of B
        stress_from_derivative : numerical evaluation of the stress tensor for the derivative of the potential function
        """
        try:
            return self.stress_from_B_analytical(B)
        except NotImplementedError:
            return self.stress_from_derivative(B, h=h)

    def stress_from_F(self, F, h=1e-6):
        """
        Compute the stress tensor from the deformation gradient tensor.

        If the model is incompressible, the returned stress is the deviatoric stress, as the hydrostatic pressure cannot
        be estimated.

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

    @classmethod
    def fit(cls, stretch, tensile_stress, **kwargs):
        """
        Fit an hyperelastic model from stress/strain values given by a tensile test.

        Parameters
        ----------
        stretch : list or numpy.ndarray
            relative elongation (engineering strain)
        tensile_stress : list or numpy.ndarray
            True stress
        kwargs
            keyword arguments

        Returns
        -------
        cls
            Fitted hyperelastic model
        """
        pass

    def tensile_curve(self, stretch):
        """
        Compute the tensile curve of the hyperelastic model.

        Parameters
        ----------
        stretch : float or list or numpy.ndarray
            Engineering strain

        Returns
        -------
        float or numpy.ndarray
            True stress
        """
        if self.is_compressible():
            raise NotImplementedError("Tensile curve is not implemented for compressible materials")
        else:
            F = DeformationGradient.isochoric_tensile([1, 0, 0], stretch)
            sigma_dev = self.stress_from_F(F)
            return sigma_dev.C[0, 0] - sigma_dev.C[1, 1]

class MooneyRivlin(HyperElastic):
    def __init__(self, C, D=None):
        """
        Create a Mooney-Rivlin hyper-elastic model

        Parameters
        ----------
        C : list or numpy.ndarray
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
        self.C = np.atleast_2d(C)
        if self.C[0, 0] != 0.:
            raise ValueError('C[0,0] must be zero.')
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
        if self.D is not None:
            if isinstance(self.D, float):
                D = (self.D,)
            else:
                D = self.D
            for k, Dk in enumerate(D, start=1):
                W += (J-1)**(2*k) / Dk
        return W

    @classmethod
    def _fit(cls, stretch, tensile_stress, M, N, **kwargs):
        M = M + 1
        N = N + 1
        def fun(x, *C_flat):
            C_flat_full = np.concatenate(([0.], C_flat))
            C = np.asarray(C_flat_full).reshape(M, N)
            model = MooneyRivlin(C)
            return model.tensile_curve(stretch)
        C0 = np.zeros((M, N))
        E = tensile_stress/stretch
        Emean = E[np.isfinite(E)].mean()
        C0[1,0] = Emean / 6
        C0_flat = C0.flatten()
        C_flat_opt, _ = curve_fit(fun, stretch, tensile_stress, p0=C0_flat[1:], *kwargs)
        return C_flat_opt

    @classmethod
    def fit(cls, stretch, tensile_stress, M=3, N=3, **kwargs):
        """
        Fit an incompressible Mooney-Rivlin hyper-elastic model on tensile curve data.

        Parameters
        ----------
        stretch : list or numpy.ndarray
            relative elongation (engineering strain)
        tensile_stress : list or numpy.ndarray
            True stress
        M : int, optional
            Maximum degree to consider for I1 dependence
        N : int, optional
            Maximum degree to consider for I2 dependence
        kwargs
            keyword arguments passed to scipy.optimize.curve_fit

        Returns
        -------
        cls
            Fitted hyperelastic model
        """
        C_flat = cls._fit(stretch, tensile_stress, M, N)
        C_full = np.concatenate(([0.], C_flat))
        return cls(C_full.reshape(M+1, N+1))

class NeoHooke(MooneyRivlin):
    def __init__(self, C, D=None):
        """
        Create a Neo-Hooke hyperelastic model

        The compressible Neo-Hooke model defines the potential function as:

        .. math::

             W = C(\\bar{I}_1 - 3) + \\frac{(J-1)^2}{D}

        where :math:`C` and :math:`D` are the material constants and :math:`\\bar{I}_1=I_1J^{-2/3}`. If D is None
        (default), the material is assumed to be incompressible and the corresponding part in the equation above is
        omitted.

        Parameters
        ----------
        C : float
            Material constant relative to deviatoric part
        D : float, optional
            Material compressibility
        """
        super().__init__([[0.], [C]], D=D)

    def stress_from_B_analytical(self, B, **kwargs):
        J = B.J
        C = self.C[1,0]
        tau = B.deviatoric_part() * 2 * C / J ** (2/3)
        if self.D is not None:
            p = 2 * J * (J-1) / self.D
            tau = tau + SymmetricSecondOrderTensor.eye(shape=B.shape) * p
        return StressTensor(tau / J)

    @classmethod
    def fit(cls, stretch, tensile_stress):
        def fun(x, C):
            nh_test = cls(C)
            F = DeformationGradient.isochoric_tensile([1, 0, 0], x)
            sigma_dev = nh_test.stress_from_F(F)
            return sigma_dev.C[0, 0] - sigma_dev.C[1, 1]

        C_opt, _ = curve_fit(fun, stretch, tensile_stress)
        return cls(C_opt[0])

class Yeoh(MooneyRivlin):
    def __init__(self, C, D=None):
        """
        Create a Yeoh hyper-elastic model

        The compressible Yeoh potential is a particular case of the (generalized)
        Mooney-Rivlin model, in which the deviatoric terms depend on the first
        invariant only:

        .. math::

            W = \\sum_{i=1}^N C_{i0}(\\bar{I}_1-3)^i + \\sum_{k=1}^M\\frac{(J-1)^{2k}}{D_k}

        Parameters
        ----------
        C : float or list of float
            Material constants relative to deviatoric part, ordered as [C10, C20, ...]
        D : list of float, optional
            Material constants relative to volumetric part. If not provided,
            the material is supposed to be incompressible.
        """
        C = np.asarray(C)
        if C.ndim > 1:
            raise ValueError('C must be a 1-D list of floats, e.g. [C10, C20, C30]')
        C = np.atleast_1d(C)
        C = np.concatenate(([0.], C)).reshape(-1, 1)
        super().__init__(C, D)

    def stress_from_B_analytical(self, B, **kwargs):
        if self.is_compressible():
            raise NotImplementedError
        else:
            dWdI1 = np.zeros(B.shape)
            C = self.C[:,0]
            for i, Ci in enumerate(C[1:], start=1):
                dWdI1 += i * Ci * (B.I1 - 3)**(i-1)
            a =  2 * B.deviatoric_part() * dWdI1
            return StressTensor(a.matrix)

    @classmethod
    def fit(cls, stretch, tensile_stress, M=3, **kwargs):
        C_flat = super()._fit(stretch, tensile_stress, M=M, N=0, **kwargs)
        return cls(C_flat)