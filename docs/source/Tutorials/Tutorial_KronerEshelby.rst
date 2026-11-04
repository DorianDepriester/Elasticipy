Kröner-Eshelby homogenization
-----------------------------

This tutorial explains how to compute the homogenized (macroscopic) stiffness tensor of a composite or a
polycrystal using the ``kroner_eshelby()`` function.

In the Kröner-Eshelby (self-consistent) scheme, each phase is modeled as an ellipsoidal inclusion embedded in
the (unknown) macroscopic medium. The effective stiffness is defined as

.. math::

    \mathbf{C}^{eff} = \left\langle \mathbf{C}_i : \mathbf{A}_i \right\rangle

where :math:`\mathbf{A}_i` is the localization tensor of the i-th inclusion:

.. math::

    \mathbf{A}_i = \left[ \mathbb{I} + \mathbb{P}(\mathbf{C}^{eff}) : (\mathbf{C}_i - \mathbf{C}^{eff}) \right]^{-1}

in which :math:`\mathbb{P}` is the Eshelby polarization tensor of the ellipsoid. Since
:math:`\mathbf{C}^{eff}` appears on both sides, the equation is solved by fixed-point iteration, starting
from the Hill average (or the Voigt/Reuss average for degenerate phases).

.. note::
    The stiffness tensors passed to ``kroner_eshelby()`` are interpreted as expressed in the global
    (sample) frame, while the ``orientations`` argument maps each particle frame onto the global frame.

Two isotropic phases
====================

Let's start with a composite made of two isotropic phases: a stiff matrix reinforced by softer inclusions,
in equal volume fractions.

.. doctest::

    >>> from elasticipy.tensors.elasticity import StiffnessTensor
    >>> from elasticipy.homogenization.kroner_eshelby import kroner_eshelby
    >>> C_inclusion = StiffnessTensor.isotropic(K=1000, G=100)
    >>> C_matrix = StiffnessTensor.isotropic(K=3000, G=200)
    >>> C_eff = kroner_eshelby((C_inclusion, C_matrix), volume_fractions=[0.5, 0.5])

Since the phases are isotropic and spherical inclusions are assumed (the default), the resulting macroscopic
behaviour is isotropic as well. This can be verified through the universal anisotropy factor, which is equal to
zero for a purely isotropic material:

    >>> print(C_eff.universal_anisotropy)
    2.5713262630233658e-09
    >>> print(C_eff.bulk_modulus)
    1543.5773653350213
    >>> print(C_eff.C44)
    142.7715520149519

Note that the bulk and shear moduli are always located between the Reuss and Voigt bounds:

    >>> K_voigt = 0.5 * 1000 + 0.5 * 3000
    >>> K_reuss = 1 / (0.5 / 1000 + 0.5 / 3000)
    >>> print(K_reuss <= C_eff.bulk_modulus <= K_voigt)
    True

Polycrystal with random orientations
====================================

The Kröner-Eshelby scheme was originally introduced for polycrystals. Let's consider a Be-like hexagonal
single crystal, whose stiffness tensor is randomly oriented 100 times to build an untextured polycrystal.
The rotations can be described either with ``orix`` or with ``scipy``. Here, we use ``orix``:

.. doctest::

    >>> import numpy as np
    >>> from orix.quaternion import Orientation
    >>> np.random.seed(0)
    >>> C_be = StiffnessTensor.hexagonal(C11=292.3, C12=26.7, C13=14, C33=336.4, C44=162.5)
    >>> orientations = Orientation.random(100)
    >>> Cs = [C_be * o for o in orientations]
    >>> C_poly = kroner_eshelby(Cs, orientations=orientations)
    >>> print(C_poly)
    Stiffness tensor (in Voigt mapping):
    [[312.14  14.47  16.1    0.36  -0.58  -2.65]
     [ 14.47 314.77  15.08  -0.27  -0.05  -1.78]
     [ 16.1   15.08 311.11  -0.67   0.     1.7 ]
     [  0.36  -0.27  -0.67 149.04  -2.18  -0.97]
     [ -0.58  -0.05   0.   -2.18 147.76  -0.49]
     [ -2.65  -1.78   1.7   -0.97  -0.49 148.96]]

As expected, the resulting macroscopic stiffness is almost isotropic (the residual anisotropy is due to the
finite number of orientations). The corresponding mean elastic moduli are:

    >>> print(C_poly.Young_modulus.mean())
    311.0985092753945
    >>> print(C_poly.shear_modulus.mean())
    148.61481207772414

Ellipsoidal inclusions
======================

By default, the inclusions are assumed to be spherical. Non-spherical (ellipsoidal) inclusions can be
considered through the ``particle_sizes`` argument, which expects an array of shape ``(m, 3)`` with the
half-lengths (a1, a2, a3) of the ellipsoid principal axes of each phase.

Let's now consider a fibre-reinforced composite: both phases are modeled as long fibres (a3 >> a1 = a2),
which promotes a transversely isotropic macroscopic behaviour:

.. doctest::

    >>> C_fibre = kroner_eshelby((C_be, C_matrix), volume_fractions=[0.3, 0.7],
    ...                          particle_sizes=np.array([[1., 1., 50.],
    ...                                                    [1., 1., 50.]]))
    >>> print(C_fibre)
    Stiffness tensor (in Voigt mapping):
    [[902.15 549.45 569.63   0.    -0.    -0.  ]
     [549.45 902.15 569.63   0.    -0.    -0.  ]
     [569.63 569.63 973.3    0.    -0.    -0.  ]
     [  0.     0.     0.   187.94   0.     0.  ]
     [ -0.    -0.    -0.     0.   187.94   0.  ]
     [ -0.    -0.    -0.     0.     0.   176.35]]

The resulting stiffness tensor is indeed transversely isotropic, with a stiffer response along the fibre
axis (direction 3) than in the transverse plane, in agreement with the Hill bounds.

Going further
=============

- The accuracy of the angular quadrature can be tuned with ``n_phi`` and ``n_theta`` (number of azimuthal
  and polar integration points, respectively).
- The fixed-point iteration is performed with ``scipy.optimize.fixed_point``: a ``RuntimeError`` is raised
  if no convergence is reached within ``maxiter`` iterations. For strongly contrasted phases, consider
  increasing ``maxiter`` or changing the iteration ``method``.
- See the API documentation of :func:`elasticipy.homogenization.kroner_eshelby.kroner_eshelby`,
  ``polarization_tensor`` and ``localization_tensor`` for details.
