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
        criterion : str, optional
            Plasticity criterion to use. Can be 'von Mises', 'Tresca' or 'J2'. J2 is the same as von Mises.
        """
        self.criterion = criterion
        self.plastic_strain = 0.0

    def flow_strain(self):
        pass

    def apply_strain(self, strain, **kwargs):
        pass

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
            criterion = criterion.lower()
            if criterion in ('von mises', 'j2'):
                eq_stress = stress.vonMises()
            elif criterion =='tresca':
                eq_stress = stress.Tresca()
            else:
                raise NotImplementedError('Only von Mises and Tresca criteria have been implement yet.')
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
            n = normality_rule(stress, criterion=criterion)
            return n * strain_increment
        else:
            return strain_increment

    def reset_strain(self):
        """
        Reinitialize the plastic strain to 0
        """
        self.plastic_strain = 0.0


def normality_rule(stress, criterion='von Mises'):
    """
    Apply the normality rule for plastic flow, given a yield criterion.

    The stress can be a single tensor, or an array of tensors.

    Parameters
    ----------
    stress : StressTensor
        Stress tensor to apply the normality rule from
    criterion : str, optional
        Name of the criterion to use. Can be either 'von Mises' or 'Tresca'

    Returns
    -------
    StrainTensor
        If a single stress tensor is passed, the returned array will be of shape

    Notes
    -----
    The singular points for the Tresca criterion are treated as the von Mises criterion, which is equivalent to the
    average of the two adjacent normals of the domain.
    """
    if criterion.lower()=='von mises':
        eq_stress = stress.vonMises()
        dev_stress= stress.deviatoric_part()
        gradient_tensor = dev_stress / eq_stress
        return StrainTensor(3/2 * gradient_tensor.matrix)
    elif criterion.lower()=='tresca':
        vals, dirs = stress.eig()
        u1 = dirs[...,0]
        u3 = dirs[...,2]
        s1 = vals[...,0]
        s2 = vals[..., 1]
        s3 = vals[...,2]
        A = np.einsum('...i,...j->...ij',u1, u1)
        B = np.einsum('...i,...j->...ij',u3, u3)
        normal = A - B
        singular_points = np.logical_or(s2==s1, s2==s3)
        normal[singular_points] = normality_rule(stress[singular_points], criterion='von Mises').matrix
        normal[np.logical_and(s2==s1, s2==s3)] = 0.0
        strain = StrainTensor(normal)
        return strain / strain.eq_strain()
    else:
        raise NotImplementedError('The normality rule is only implemented for von Mises (J2) and Tresca criteria.')