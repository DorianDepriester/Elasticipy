import numpy as np



class JohnsonCook:
    def __init__(self, A, B, n, C=None, eps_dot_ref=1.0, m=None, T0=25, Tm=None):
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
        numpy.ndarray
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


    def compute_strain(self, stress, T=None):
        """
        Given the equivalent stress, compute the strain

        Parameters
        ----------
        stress : float or numpy.ndarray
            Equivalent stress tom compute the stress from
        T : float or list or tuple or numpy.ndarray
            Temperature

        Returns
        -------
        numpy.ndarray
            Equivalent strain
        """
        stress = np.asarray(stress)
        if T is None:
            theta_m=0.0
        else:
            if self.T0 is None or self.Tm is None or self.m is None:
                raise ValueError('T0, Tm and m must be defined for using a temperature-dependent model')
            else:
                theta = (T - self.T0) / (self.Tm - self.T0)
                theta_m = np.clip(theta**self.m, None, 1.0)
        return (1/self.B * ( stress / (1 - theta_m) - self.A))**(1/self.n)
