import numpy as np
from Elasticipy.StressStrainTensors import StrainTensor, StressTensor


class IsotropicHardening:
    """
    Template class for isotropic hardening plasticity models
    """
    def __init__(self, criterion='von Mises'):
        """
        Create an instance of a plastic model, assuming isotropic hardening

        Parameters
        ----------
        criterion : str or PlasticityCriterion
            Plasticity criterion to use. Can be 'von Mises', 'Tresca' or 'J2'. J2 is the same as von Mises.
        """
        if isinstance(criterion, str):
            criterion = criterion.lower()
            if criterion in ('von mises', 'mises', 'vonmises', 'j2'):
                self.criterion = VonMisesPlasticity
            elif criterion == 'tresca':
                self.criterion = TrescaPlasticity
            else:
                raise ValueError('The criterion can be "Tresca", "von Mises" or "J2".')
        else:
            self.criterion = criterion
        self.plastic_strain = 0.0

    def flow_stress(self, strain, **kwargs):
        pass

    def apply_strain(self, strain, **kwargs):
        """
        Apply strain to the current JC model.

        This function updates the internal variable to store hardening state.

        Parameters
        ----------
        strain : float or StrainTensor
        kwargs : dict
            Keyword arguments passed to flow_stress()

        Returns
        -------
        float
            Associated flow stress (positive)

        See Also
        --------
        flow_stress : compute the flow stress, given a cumulative equivalent strain
        """
        if isinstance(strain, float):
            self.plastic_strain += np.abs(strain)
        elif isinstance(strain, StrainTensor):
            self.plastic_strain += strain.eq_strain()
        else:
            raise ValueError('The applied strain must be float of StrainTensor')
        return self.flow_stress(self.plastic_strain, **kwargs)

    def compute_strain_increment(self, stress, **kwargs):
        pass

    def reset_strain(self):
        self.plastic_strain = 0.0


class JohnsonCook(IsotropicHardening):
    def __init__(self, A, B, n, C=None, eps_dot_ref=1.0, m=None, T0=25, Tm=None, criterion='von Mises'):
        """
        Constructor for a Jonhson-Cook (JC) model.

        The JC model is an exponential-law strain hardening model, which can take into account strain-rate sensibility
        and temperature-dependence (although they are not mandatory). See notes for details.

        Parameters
        ----------
        A : float
            Yield stress
        B : float
            Work hardening coefficient
        n : float
            Work hardening exponent
        C : float, optional
            Strain-rate sensitivity coefficient
        eps_dot_ref : float, optional
            Reference strain-rate
        m : float, optional
            Temperature sensitivity exponent
        T0 : float, optional
            Reference temperature
        Tm : float, optional
            Melting temperature (at which the flow stress is zero)
        criterion : str or PlasticityCriterion, optional
            Plasticity criterion to use. It can be 'von Mises' or 'Tresca'.

        Notes
        -----
        The flow stress (:math:`\\sigma`) depends on the strain (:math:`\\varepsilon`),
        the strain rate :math:`\\dot{\\varepsilon}` and
        the temperature (:math:`T`) so that:

        .. math::

                \\sigma = \\left(A + B\\varepsilon^n\\right)
                        \\left(1 + C\\log\\left(\\frac{\\varepsilon}{\\dot{\\varepsilon}_0}\\right)\\right)
                        \\left(1-\\theta^m\\right)

        with

        .. math::

                \\theta = \\begin{cases}
                            \\frac{T-T_0}{T_m-T_0} & \\text{if } T<T_m\\\\
                            1                      & \\text{otherwise}
                            \\end{cases}
        """
        super().__init__(criterion=criterion)
        self.A = A
        self.B = B
        self.C = C
        self.n = n
        self.m = m
        self.eps_dot_ref = eps_dot_ref
        self.T0 = T0
        self.Tm = Tm

    def flow_stress(self, eps_p, eps_dot=None, T=None):
        """
        Compute the flow stress from the Johnson-Cook model

        Parameters
        ----------
        eps_p : float or list or tuple or numpy.ndarray
            Equivalent plastic strain
        eps_dot : float or list or tuple or numpy.ndarray, optional
            Equivalent plastic strain rate. If float, the strain-rate is supposed to be homogeneous for every value of
            eps_p.
        T : float or list or tuple or np.ndarray
            Temperature. If float, the temperature is supposed to be homogeneous for every value of eps_p.
        Returns
        -------
        float or numpy.ndarray
            Flow stress
        """
        eps_p = np.asarray(eps_p)
        stress = (self.A + self.B * eps_p**self.n)

        if eps_dot is not None:
            eps_dot = np.asarray(eps_dot)
            if (self.C is None) or (self.eps_dot_ref is None):
                raise ValueError('C and eps_dot_ref must be defined for using a rate-dependent model')
            stress *= (1 + self.C * np.log(eps_dot / self.eps_dot_ref))

        if T is not None:
            T = np.asarray(T)
            if self.T0 is None or self.Tm is None or self.m is None:
                raise ValueError('T0, Tm and m must be defined for using a temperature-dependent model')
            theta = (T - self.T0) / (self.Tm - self.T0)
            theta = np.clip(theta, None, 1.0)
            stress *= (1 - theta**self.m)

        return stress



    def compute_strain_increment(self, stress, T=None, apply_strain=True, criterion='von Mises'):
        """
        Given the equivalent stress, compute the strain increment with respect to the normality rule.

        Parameters
        ----------
        stress : float or StressTensor
            Equivalent stress to compute the stress from, or full stress tensor.
        T : float
            Temperature
        apply_strain : bool, optional
            If true, the JC model will be updated to account for the applied strain (hardening)
        criterion : str, optional
            Plasticity criterion to consider to compute the equivalent stress and apply the normality rule.
            It can be 'von Mises', 'Tresca' or 'J2'. 'J2' is equivalent to 'von Mises'.

        Returns
        -------
        StrainTensor or float
            Increment of plastic strain. If the input stress is float, only the magnitude of the increment will be
            returned (float value). If the stress is of type StressTensor, the returned value will be a full
            StrainTensor.

        See Also
        --------
        apply_strain : apply strain to the JC model and updates its hardening value
        """
        if isinstance(stress, StressTensor):
            eq_stress = self.criterion.eq_stress(stress)
        else:
            eq_stress = stress
        if T is None:
            if eq_stress > self.A:
                k = eq_stress  - self.A
                total_strain = (1 / self.B * k) ** (1 / self.n)
                strain_increment = np.max((total_strain - self.plastic_strain, 0))
            else:
                strain_increment = 0.0
        else:
            if self.T0 is None or self.Tm is None or self.m is None:
                raise ValueError('T0, Tm and m must be defined for using a temperature-dependent model')
            else:
                if T >= self.Tm:
                    strain_increment = np.inf
                else:
                    theta = (T - self.T0) / (self.Tm - self.T0)
                    theta_m = theta**self.m
                    k = (eq_stress / (1 - theta_m) - self.A)
                    if k<0:
                        strain_increment = 0.0
                    else:
                        total_strain = (1/self.B * k)**(1/self.n)
                        strain_increment = np.max((total_strain - self.plastic_strain, 0))
        if apply_strain:
            self.apply_strain(strain_increment)

        if isinstance(stress, StressTensor):
            n = self.criterion.normal(stress)
            return n * strain_increment
        else:
            return strain_increment

    def reset_strain(self):
        """
        Reinitialize the plastic strain to 0
        """
        self.plastic_strain = 0.0


class PlasticityCriterion:
    @staticmethod
    def eq_stress(stress, **kwargs):
        """
        Return the equivalent stress, with respect to the plasticity criterion.

        Parameters
        ----------
        stress : StressTensor
            Stress to compute the equivalent stress from
        kwargs : dict
            keyword arguments passed to the function
        Returns
        -------
        float or numpy.ndarray
        """
        pass

    def normal(self, stress, **kwargs):
        """
        Apply the normality rule

        Parameters
        ----------
        stress : StressTensor
            Stress tensor to apply the normality rule
        kwargs : dict
            Keyword arguments passed to the function

        Returns
        -------
        StrainTensor
            Normalized direction of plastic flow
        """
        pass

class VonMisesPlasticity(PlasticityCriterion):
    @staticmethod
    def eq_stress(stress, **kwargs):
        return stress.vonMises()

    @staticmethod
    def normal(stress, **kwargs):
        eq_stress = stress.vonMises()
        dev_stress = stress.deviatoric_part()
        gradient_tensor = dev_stress / eq_stress
        return StrainTensor(3 / 2 * gradient_tensor.matrix)

class TrescaPlasticity(PlasticityCriterion):
    @staticmethod
    def eq_stress(stress, **kwargs):
        return stress.Tresca()

    @staticmethod
    def normal(stress, **kwargs):
        vals, dirs = stress.eig()
        u1 = dirs[..., 0]
        u3 = dirs[..., 2]
        s1 = vals[..., 0]
        s2 = vals[..., 1]
        s3 = vals[..., 2]
        a = np.einsum('...i,...j->...ij', u1, u1)
        b = np.einsum('...i,...j->...ij', u3, u3)
        normal = a - b
        singular_points = np.logical_or(s2 == s1, s2 == s3)
        normal[singular_points] = VonMisesPlasticity().normal(stress[singular_points]).matrix
        normal[np.logical_and(s2 == s1, s2 == s3)] = 0.0
        strain = StrainTensor(normal)
        return strain / strain.eq_strain()

class DruckerPrager(PlasticityCriterion):
    def __init__(self, alpha):
        """
        Create a Drucker-Prager (DG) plasticity criterion.

        Parameters
        ----------
        alpha : float
            Pressure dependence parameters (see notes for details)

        Notes
        -----
        The pressure-dependent DG plasticity criterion assumes that the equivalent stress is defined as:


        """
        self.alpha = alpha

    def eq_stress(self, stress, **kwargs):
        return (stress.J2**0.5 + self.alpha * stress.I1) / (1/3**0.5 + self.alpha)

    def normal(self, stress, **kwargs):
        J2 = stress.J2
        gradient = stress.deviatoric_part() / (2 * J2**0.5) + self.alpha * StressTensor.eye(stress.shape)
        strain = StrainTensor(gradient.matrix)
        return strain / strain.eq_strain()