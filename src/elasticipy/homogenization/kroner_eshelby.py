import numpy as np
from elasticipy.tensors.elasticity import StiffnessTensor
from elasticipy.tensors.fourth_order import FourthOrderTensor, SymmetricFourthOrderTensor
from scipy.optimize import fixed_point
from elasticipy.tensors.mapping import VoigtMapping

I = FourthOrderTensor.identity()

def gamma(C_macro_local, phi, theta, a1, a2, a3):
    """
    Compute the gamma tensor (integrand for the polarization tensor)

    Parameters
    ----------
    C_macro_local : StiffnessTensor
        Macroscopic stiffness tensor, written in the coordinate system align with the principal axes of the ellipsoid.
    phi : float or ndarray
        phi angle in the spherical coordinate system
    theta : float or ndarray
        theta angle in the spherical coordinate system
    a1, a2, a3  : float
        half-lengths of the ellipsoid

    Returns
    -------
    SymmetricFourthOrderTensor
        gamma tensor, expressed in the coordinate system align with the principal axes of the ellipsoid.

    See Also
    --------
    localization_tensor : compute the localization tensor
    elasticipy.tensors.elasticity.StiffnessTensor.Christoffel_tensor : compute the 2nd-order Christoffel tensor

    Notes
    -----
    Given the macroscopic stiffness tensor **C**, the gamma tensor is defined as:

    .. math::

        \\gamma_{ijkl}\\left(\phi,\\theta\\right) = \\frac{1}{4}\\left(
        D^{-1}_{ik}u_ju_l + D^{-1}_{jk}u_iu_l + D^{-1}_{il}u_ju_k + D^{-1}_{jl}u_iu_k
        \\right)

    where **D** denotes the Christoffel tensor, such that:

    .. math::

        D_{ik}=C_{ijkl}u_ju_l

    with

    .. math::

		u_1 =\\frac{\\sin\\theta\\cos\\phi}{a_1}\\text{, }
		u_2 =\\frac{\\sin\\theta\\sin\\phi}{a_2}\\text{, }
		u_3 =\\frac{\\cos\\theta}{a_3}
    """
    s1 = np.sin(theta)*np.cos(phi) / a1
    s2 = np.sin(theta)*np.sin(phi) / a2
    s3 = np.cos(theta) / a3
    s = np.stack((s1,s2,s3), axis=-1)
    Dinv = C_macro_local.Christoffel_tensor(s).inv()
    a1 = np.einsum('mnik,mnj,mnl->mnijkl', Dinv.matrix, s, s)
    return SymmetricFourthOrderTensor(a1, force_symmetries=True)

def _vec_components(C):
    Cxx = ("C11", "C12", "C13", "C14", "C15", "C16",
           "C22", "C23", "C24", "C25", "C26",
           "C33", "C34", "C35", "C36",
           "C44", "C45", "C46",
           "C55", "C56",
           "C66")
    return np.array([getattr(C, Ci) for Ci in Cxx])


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

    See Also
    --------
    gamma : compute gamma tensor
    localization_tensor : compute the localization tensor

    Notes
    -----
    The polarization tensor is defined as:

    .. math::

	    P_{ijkl}=\\frac{1}{4\\pi}\\int_{0}^{\\pi}\\int_{0}^{2\\pi}
	    \\gamma_{ijkl}\\left(\\phi,\\theta\\right)\\sin\\theta\\mathrm{d}\\phi\\mathrm{d}\\theta

    where :math:`\\mathbf{\\gamma}` denotes the gamma tensor, such that:

    .. math::

        \\gamma_{ijkl}\\left(\phi,\\theta\\right) = \\frac{1}{4}\\left(
        D^{-1}_{ik}u_ju_l + D^{-1}_{jk}u_iu_l + D^{-1}_{il}u_ju_k + D^{-1}_{jl}u_iu_k
        \\right)
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='xy')
    g = gamma(C, phi_grid, theta_grid, a1, a2, a3)
    integrand = g * np.sin(theta_grid)
    a = integrand.integrate(theta, axis=0)
    return a.integrate(phi, axis=0)/(4*np.pi)


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

    See Also
    --------
    polarization_tensor : compute polarization tensor

    Notes
    -----
    The localization tensor is defined as:

    .. math::

        \\mathbf{A}= \\left[\\mathbf{P}:\left( \\mathbf{C}-\\mathbf{C}^{macro}\\right) + \\mathbf{I}\\right]^{-1}

    where :math:`\\mathbf{P}`, :math:`\\mathbf{C}` and :math:`\\mathbf{C}^{macro}` denote the polarization tensor, the
    particule stiffness tensor and the macroscopic stiffness tensor, respectively.
    """
    E = polarization_tensor(C_macro, a1, a2, a3, n_phi=n_phi, n_theta=n_theta)
    Ainv = E.ddot(C_incl - C_macro) + I
    return Ainv.inv()


def kroner_eshelby(Cs, particle_sizes=None, orientations=None,
                   volume_fractions=None, n_phi=100, n_theta=50, method='del2', maxiter=1000):
    """
    Compute the homogenized stiffness tensor of an ellipsoidal-inclusion
    composite using the (self-consistent) Kröner-Eshelby scheme.

    Each phase is modeled as an ellipsoidal inclusion embedded in the
    (unknown) macroscopic medium. The effective stiffness is defined as

    .. math::

        \\mathbf{C}_{eff} = <\\mathbf{C}_i : \\mathbf{A}_i>

    where :math:`A_i` is the localization tensor of the i-th inclusion,

    .. math::

        A_i = (\\mathbf{I} + \\mathbf{P}(\\mathbf{C}_{eff}) : (\\mathbf{C}_i - \\mathbf{C}_{eff}))^{-1},

    and :math:`\\mathbf{P}(\\mathbf{C}_{eff})` is the Eshelby polarization tensor of the ellipsoid,
    computed by angular quadrature over the unit sphere. The resulting
    implicit equation on :math:`\\mathbf{P}(\\mathbf{C}_{eff})` is solved by fixed-point iteration, with
    the initial guess chosen as the Hill (or Voigt/Reuss) average of the phase stiffnesses.

    Notes
    -----
    * With spherical inclusions, the scheme reduces to the classic
      Kröner self-consistent estimate for polycrystals.
    * `Cs` are interpreted as given in the global (sample) frame, and
      `orientations` map each particle frame onto the global frame.
    * A RuntimeError is raised if the fixed-point iteration does not
      converge within `maxiter` iterations.

    Parameters
    ----------
    Cs : StiffnessTensor or tuple/list of StiffnessTensor
        Stiffness tensors of the particles, expressed in the global frame.
    particle_sizes : array_like, optional
        Half-lengths (a1, a2, a3) of the ellipsoid principal axes, given
        as an array of shape (m, 3) (one row of half-axes per particle).
        If None, spherical inclusions are assumed (default).
    orientations : scipy.spatial.transform.Rotation or orix Orientation, optional
        Rotations mapping each particle frame onto the global frame.
        If None, all particles are assumed aligned with the global
        frame (default).
    volume_fractions : array_like, optional
        Volume fraction of each particle. If None, uniform fractions are
        assumed (default).
    n_phi : int, optional
        Number of azimuthal integration points (default 100).
    n_theta : int, optional
        Number of polar integration points (default 50).
    method : str, optional
        Fixed-point iteration method, forwarded to
        scipy.optimize.fixed_point (default 'del2').
    maxiter : int, optional
        Maximum number of fixed-point iterations (default 1000).

    Returns
    -------
    StiffnessTensor
        Homogenized (macroscopic) stiffness tensor.

    See Also
    --------
    polarization_tensor : Eshelby polarization tensor of an ellipsoid.
    localization_tensor : Localization tensor of an ellipsoidal inclusion.
    """
    if isinstance(Cs, (tuple, list)):
        Cs = StiffnessTensor.stack(Cs)
    elif not Cs.shape:
        Cs = StiffnessTensor.stack([Cs])
    if orientations is not None:
        Cs_local = (Cs * orientations.inv())        # Stiffness tensors written in the particules' frames
    else:
        Cs_local = Cs

    # Initial guess
    if np.logical_not(np.any(np.logical_or(Cs == 0., Cs == np.inf))):
        avg_method = 'Hill'
    elif np.logical_not(np.any(Cs == np.inf)):
        avg_method = 'Voigt'
    elif np.logical_not(np.any(Cs == 0.)):
        avg_method = 'Reuss'
    else:
        raise NotImplementedError
    C_macro_0 = StiffnessTensor.weighted_average(Cs, volume_fractions=volume_fractions, method=avg_method)

    if particle_sizes is None:
        a1 = a2 = a3 = np.ones(Cs.shape[0])
    else:
        particle_sizes = np.asarray(particle_sizes)
        if particle_sizes.ndim == 1:
            a1, a2, a3 = particle_sizes
        elif particle_sizes.ndim == 2:
            a1, a2, a3 = np.asarray(particle_sizes).T

    def fun(Cxx):
        C_macro = StiffnessTensor.triclinic(*Cxx, check_positive_definite=False)
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
        CiAi_mean.mapping = VoigtMapping()
        vec = _vec_components(CiAi_mean)
        return vec

    vec_0 = _vec_components(C_macro_0)
    sol = fixed_point(fun, vec_0, method=method, maxiter=maxiter)
    return StiffnessTensor.triclinic(*sol)