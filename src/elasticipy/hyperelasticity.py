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
        B = F.CauchyGreen(side='left')
        return self.stress_from_B(B, h=h)

    @classmethod
    def fit(cls, tensile_strain, tensile_stress, C0=None, **kwargs):
        """
        Fit an hyperelastic model from stress/strain values given by a tensile test.

        Parameters
        ----------
        tensile_strain : list or numpy.ndarray
            relative elongation (engineering strain)
        tensile_stress : list or numpy.ndarray
            True stress
        C0 : list or numpy.ndarray, optional
            Initial guess for the fitted parameters
        kwargs
            keyword arguments

        Returns
        -------
        cls
            Fitted hyperelastic model
        """
        pass

    def tensile_curve(self, longitudinal_strain, true_strain=False, true_stress=False):
        """
        Compute the tensile curve of the hyperelastic model.

        Parameters
        ----------
        longitudinal_strain : float or list or numpy.ndarray
            Longitudinal strain. It can relate to conventional (engineering) strain or a true (i.e. logarithmic) strain;
            see below.
        true_strain: bool, optional
            Specify if the provided strain correspond to the true strain or the engineering strain.
        true_stress: bool, optional
            Sets if the returned tensile curve corresponds to the true stress (accounting for cross-section changes) or
            the engineering stress.

        Returns
        -------
        float or numpy.ndarray
            Tensile stress
        """
        if self.is_compressible():
            raise NotImplementedError("Tensile curve is not implemented for compressible materials")
        else:
            if true_strain:
                longitudinal_strain = np.exp(longitudinal_strain) - 1
            F = DeformationGradient.isochoric_tensile([1, 0, 0], longitudinal_strain)
            sigma_dev = self.stress_from_F(F)
            sigma = sigma_dev.C[0, 0] - sigma_dev.C[1, 1]
            if true_stress:
                return sigma
            else:
                return sigma * (F.C[1,1] * F.C[2,2])

class MooneyRivlin(HyperElastic):
    def __init__(self, C, D=None):
        """
        Create a generalized Mooney-Rivlin hyper-elastic model

        Parameters
        ----------
        C : list or numpy.ndarray
            Material constants relative to deviatoric parts, provided as a MxN matrix (see notes)
        D : list of float or numpy.ndarray, optional
            Material constants relative to volumetric part. If not provided, the material is supposed to be
            incompressible.

        Notes
        -----
        The generalized compressible Mooney-Rivlin potential is defined as:

        .. math::

            W = \\sum_{i=0}^M\\sum_{j=0}^N C_{ij}(\\bar{I}_1-3)^i(\\bar{I}_2-3)^j + \\sum_{k=1}^P\\frac{(J-1)^{2k}}{D_k}

        with

        .. math::

            \\bar{I}_1 = I_1J^{-2/3}
            \\bar{I}_2 = I_2J^{-4/3}
        """
        self._C = np.atleast_2d(C)
        if self._C[0, 0] != 0.:
            raise ValueError('C[0,0] must be zero.')
        if D is not None:
            self._D = np.atleast_1d(D)
        else:
            self._D = None

    def __repr__(self):
        C = np.array2string(self.C, separator=', ')
        msg = f'{self.__class__.__name__}(C={C}'
        if self.is_compressible():
            msg += f', D={self.D}'
        msg += ')'
        return msg

    def __str__(self):
        C = str(self._C)
        head = 'Generalized incompressible Mooney-Rivlin hyper-elastic model'
        if self.is_compressible():
            head = 'Generalized Mooney-Rivlin hyper-elastic model'
            return f'{head}\nCij=\n{C}\nD={self._D}'
        return f'{head}\nCij=\n{C}'

    @property
    def C(self):
        """
        Parameters of the hyperelastic model

        Returns
        -------
        float or numpy.ndarray
            Parameters of the hyperelastic model, returned in a human-readable way
        """
        return self._C

    @property
    def D(self):
        """
        Coefficients of the J dependence

        Returns
        -------
        float or numpy.ndarray
        """
        return self._D

    def is_compressible(self):
        return self._D is not None

    def potential_from_B(self, B):
        J = B.J
        W = np.zeros_like(J)
        C = self._C
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                W += C[i, j] * (B.I1_bar-3)**i * (B.I2_bar - 3)**j
        if self._D is not None:
            if isinstance(self._D, float):
                D = (self._D,)
            else:
                D = self._D
            for k, Dk in enumerate(D, start=1):
                W += (J-1)**(2*k) / Dk
        return W

    @staticmethod
    def _initial_guess_fit(tensile_strain, tensile_stress, M, N):
        strain_thres = 1e-2
        mask = tensile_strain < strain_thres
        eps = tensile_strain[mask]
        sig = tensile_stress[mask]
        C = np.zeros((M, N))
        if np.count_nonzero(mask) > 2:
            E, _ = np.polyfit(eps, sig, 1)
        else:
            E = (tensile_stress[-1] - tensile_stress[0]) / (tensile_strain[-1] - tensile_strain[0])
        C[1, 0] = max(E / 6, 1e-6)
        return C.flatten()[1:]

    @classmethod
    def _fit(cls, strain, stress, M, N, true_stress, true_strain, C0=None, **kwargs):
        M = M + 1
        N = N + 1
        shape = (M, N)
        def fun(x, *C_flat):
            C_flat_full = np.concatenate(([0.], C_flat))
            C = np.asarray(C_flat_full).reshape(shape)
            model = MooneyRivlin(C)
            return model.tensile_curve(x, true_strain=true_strain, true_stress=true_stress)
        lb = -np.ones(shape)*np.inf
        lb[1,0] = 0.0
        ub = np.ones(shape)*np.inf
        bounds = [lb.flatten()[1:], ub.flatten()[1:]]
        if C0 is None:
            C0 = cls._initial_guess_fit(strain, stress, M, N)
        C_flat_opt, _ = curve_fit(fun, strain, stress, p0=C0, bounds=bounds, **kwargs)
        return C_flat_opt

    @classmethod
    def fit(cls, tensile_strain, tensile_stress, M=3, N=3, true_strain=False, true_stress=False, C0=None, **kwargs):
        """
        Fit an incompressible Mooney-Rivlin hyper-elastic model on tensile curve data.

        Parameters
        ----------
        tensile_strain : list or numpy.ndarray
            relative elongation (engineering strain)
        tensile_stress : list or numpy.ndarray
            True stress
        M : int, optional
            Maximum degree to consider for I1 dependence
        N : int, optional
            Maximum degree to consider for I2 dependence
        true_strain: bool, optional
            Sets if the provided strain correspond to true (i.e. logarithmic) or engineering definition
        true_stress: bool, optional
            Sets if the provided stress correspond to true (accounting for cross-section changes) or engineering
            definition of the tensile stress
        C0 : list or np.ndarray, optional
            Initial guess for C
        kwargs
            keyword arguments passed to scipy.optimize.curve_fit

        Returns
        -------
        cls
            Fitted hyperelastic model
        """
        C_flat = cls._fit(tensile_strain, tensile_stress, M, N, true_stress, true_strain, C0=C0, **kwargs)
        C_full = np.concatenate(([0.], C_flat))
        return cls(C_full.reshape(M+1, N+1))

    @property
    def M(self):
        """
        Returns the maximum degree to consider for I1 dependence.

        Returns
        -------
        int
            Maximum degree to consider for I1 dependence
        """
        return self._C.shape[0] - 1

    @property
    def N(self):
        """
        Returns the maximum degree to consider for I2 dependence.

        Returns
        -------
        int
            Maximum degree to consider for I2 dependence
        """
        return self._C.shape[1] - 1

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

    @property
    def C(self):
        return self._C[1:,0]

    def stress_from_B_analytical(self, B, **kwargs):
        J = B.J
        # dW/dI1_bar
        dWdI1 = np.zeros(B.shape)
        C = np.atleast_1d(self._C[1:, 0])
        for i, Ci in enumerate(C, start=1):
            dWdI1 += i * Ci * (B.I1_bar - 3) ** (i - 1)
        # deviatoric stress (unloaded: pressure term) + volumetric contribution
        tau = 2 * B.deviatoric_part() * dWdI1 / J ** (2 / 3)  # careful: see note below
        if self._D is not None:
            D = np.atleast_1d(self._D)
            p = np.zeros(B.shape)
            for k, Dk in enumerate(D, start=1):
                p += 2 * k * J * (J - 1) ** (2 * k - 1) / Dk
            tau = tau + SymmetricSecondOrderTensor.eye(shape=B.shape) * p
        return StressTensor(tau / J)

    @classmethod
    def fit(cls, tensile_strain, tensile_stress, M=3, true_stress=False, true_strain=False, C0=None, **kwargs):
        C_flat = super()._fit(tensile_strain, tensile_stress,
                              M=M, N=0,
                              true_stress=true_stress, true_strain=true_strain,
                              C0=C0, **kwargs)
        return cls(C_flat)

class NeoHooke(Yeoh):
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
        super().__init__(C, D=D)

    @property
    def C(self):
        return self._C[1,0]

    @property
    def D(self):
        if self._D is not None:
            return float(self._D[0])
        else:
            return None