import numpy as np
from scipy.integrate import trapezoid

from elasticipy.tensors.elasticity import StiffnessTensor
from elasticipy.tensors.fourth_order import FourthOrderTensor, SymmetricFourthOrderTensor
from scipy.optimize import fixed_point

from elasticipy.tensors.mapping import KelvinMapping

I = FourthOrderTensor.identity()


def gamma(C_macro_local, phi, theta, a1, a2, a3):
    s1 = np.sin(theta)*np.cos(phi) / a1
    s2 = np.sin(theta)*np.sin(phi) / a2
    s3 = np.cos(theta) / a3
    s = np.stack((s1,s2,s3), axis=-1)
    Dinv = C_macro_local.Christoffel_tensor(s).inv()
    a1 = np.einsum('mnik,mnj,mnl->mnijkl', Dinv.matrix, s, s)
    return SymmetricFourthOrderTensor(a1, force_symmetries=True)


def polarization_tensor(C, a1, a2, a3, n_phi=100, n_theta=50):
    """
    Numerically compute the polarization tensor of an elliptical inclusion

    Parameters
    ----------
    C : StiffnessTensor
        macroscopic stiffness tensor
    a1 : float
        first half-principal axis
    a2 : float
        second half-principal axis
    a3 : float
        third half-principal axis
    n_phi : int, optional
        number of integration point along azimutal plane (default 100)
    n_theta : int, optional
        number of integration point along theta plane (default 50)

    Returns
    -------
    FourthOrderTensor
        Polarization tensor
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='xy')
    g = gamma(C, phi_grid, theta_grid, a1, a2, a3)
    integrand = g * np.sin(theta_grid)
    a = trapezoid(integrand.full_tensor, theta, axis=0)
    b = trapezoid(a, phi, axis=0)/(4*np.pi)
    return SymmetricFourthOrderTensor(b)


def localization_tensor(C_macro, C_incl, a1, a2, a3, n_phi=100, n_theta=50):
    """
    Numerically compute the localization tensor of an elliptical inclusion

    Parameters
    ----------
    C_macro : StiffnessTensor
        Macroscopic stiffness tensor
    C_incl : StiffnessTensor
        Stiffness tensor of inclusion
    a1 : float
        first half-principal axis
    a2 : float
        second half-principal axis
    a3 : float
        third half-principal axis
    n_phi : int, optional
        number of integration point along azimutal plane (default 100)
    n_theta : int, optional
        number of integration point along theta plane (default 50)

    Returns
    -------
    FourthOrderTensor
        Localization tensor
    """
    E = polarization_tensor(C_macro, a1, a2, a3, n_phi=n_phi, n_theta=n_theta)
    if C_incl==np.inf:
        return I - E.ddot(C_macro)
    else:
        Ainv = E.ddot(C_incl - C_macro) + I
        return Ainv.inv()


def Kroner_Eshelby(Cs, particle_sizes=None, orientations=None,
                   volume_fractions=None, n_phi=50, n_theta=100, **kwargs):
    if isinstance(Cs, (tuple, list)):
        Cs = StiffnessTensor.stack(Cs)
    if orientations is not None:
        Cs_local = (Cs * orientations.inv())        # Stiffness tensors written in the particules' frames
    else:
        Cs_local = Cs

    # Initial guess
    if np.logical_not(np.any(np.logical_or(Cs == 0., Cs == np.inf))):
        method = 'Hill'
    elif np.logical_not(np.any(Cs == np.inf)):
        method = 'Voigt'
    elif np.logical_not(np.any(Cs == 0.)):
        method = 'Reuss'
    else:
        raise NotImplemented
    C_macro_0 = StiffnessTensor.weighted_average(Cs, volume_fractions=volume_fractions, method=method).to_Kelvin()

    if particle_sizes is None:
        a1 = a2 = a3 = np.ones(Cs.shape[0])
    else:
        particle_sizes = np.asarray(particle_sizes)
        if particle_sizes.ndim == 1:
            a1, a2, a3 = particle_sizes
        elif particle_sizes.ndim == 2:
            a1, a2, a3 = np.asarray(particle_sizes).T

    def fun(C_macro):
        C_macro = StiffnessTensor(C_macro, mapping=KelvinMapping, force_symmetries=True)
        m = Cs.shape[0]
        A_local = FourthOrderTensor.zeros(m)
        if orientations is not None:
            C_macro_local = C_macro * (orientations.inv())
        else:
            C_macro_local = C_macro
        for i in range(m):
            if C_macro_local.shape:
                C_macro_local_i = C_macro_local[i]  # Macroscopic stiffness written in the i-th particule's frame
            else:
                C_macro_local_i = C_macro_local
            A_local[i] = localization_tensor(C_macro_local_i, Cs_local[i], a1[i], a2[i], a3[i], n_phi=n_phi, n_theta=n_theta)
        if orientations is None:
            A = A_local
        else:
            A = A_local * orientations
        Q = Cs.ddot(A)
        CiAi_mean = Q.weighted_average(weights=volume_fractions)
        return CiAi_mean.matrix()

    sol = fixed_point(fun, C_macro_0, **kwargs)
    return StiffnessTensor.from_Kelvin(sol)
