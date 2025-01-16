Averaging methods
-----------------

This tutorial explains how to compute the Voigt, Reuss and Hill averages from a given stiffness tensor and a finite or
an infinite set of rotations.

The corresponding elastic moduli yet depends on the model, which actually depends on the underlying
assumption, namely:

- Voigt (i.e. homogeneous strain assumption);
- Reuss (i.e. homogeneous stress assumption);
- Voigt-Reuss-Hill (or Hill, for short).

The later is just the arithmetic mean between the Voigt and the Reuss stiffnesses.

As an example, let's consider the stiffness tensor for monoclinic TiNi:

.. doctest::

    >>> from Elasticipy.FourthOrderTensor import StiffnessTensor
    >>> C = StiffnessTensor.monoclinic(phase_name='TiNi',
    ...                                C11=231, C12=127, C13=104,
    ...                                C22=240, C23=131, C33=175,
    ...                                C44=81, C55=11, C66=85,
    ...                                C15=-18, C25=1, C35=-3, C46=3)

Infinite number of orientations (the non-textured case)
=======================================================
If we consider an infinite number of rotations, uniformly distributed on SO3, this necessarily leads to an isotropic
behaviour. In this case, the aforementioned averages can be estimated as follows:

    >>> C_Voigt = C.Voigt_average()
    >>> C_Reuss = C.Reuss_average()
    >>> C_Hill = C.Hill_average()

    Let's see how ``C_Hill`` looks like:

    >>> print(C_Hill)
    Stiffness tensor (in Voigt notation):
    [[200.92610744 119.59583533 119.59583533   0.           0.
        0.        ]
     [119.59583533 200.92610744 119.59583533   0.           0.
        0.        ]
     [119.59583533 119.59583533 200.92610744   0.           0.
        0.        ]
     [  0.           0.           0.          40.66513606   0.
        0.        ]
     [  0.           0.           0.           0.          40.66513606
        0.        ]
     [  0.           0.           0.           0.           0.
       40.66513606]]
    Symmetry: Triclinic

    As a comparison, let's see how the underlying assumption impair the Young moduli:

    >>> E_Voigt = C_Voigt.Young_modulus
    >>> E_Reuss = C_Reuss.Young_modulus
    >>> E_Hill = C_Hill.Young_modulus
    >>> print(E_Voigt.mean(), E_Reuss.mean(), E_Hill.mean())
    145.66862361382908 76.13802022165397 111.67690532486962


Finite number of orientations
=============================
If one wants to consider a finite set of orientations, or consider that these orientations are not uniformly distributed
over the SO3 (i.e. if we have a crystallographic texture), the first steps consists in estimating the mean of all the
rotated stiffnesses/compliances (depending on the model).

As an example, we assume that the material displays a perfect fiber texture along **z**. In terms of Bunge-Euler angles,
it means that we have Phi=0 for each orientation:

    >>> from scipy.spatial.transform import Rotation
    >>> import numpy as np
    >>> phi1 = np.random.random(10000)*2*np.pi # Random sampling from 0 to 2pi
    >>> Phi = phi2 = np.zeros(10000)
    >>> Euler_angles = np.array([phi1, Phi, phi2]).T
    >>> rotations = Rotation.from_euler('ZXZ', Euler_angles)    # Bunge-Euler angles

Now generate the rotated stiffness tensors:

    >>> C_rotated = C * rotations

And compute the Hill averages :

    >>> C_Hill = C_rotated.Hill_average()

Let's see how the Young modulus is distributed over space:

    >>> C_Hill.Young_modulus.plot3D() # doctest: +SKIP

.. image:: images/E_hill_fiber.png
    :width: 400

It is thus clear that the fiber along the *z* axis results in a transverse-isotropic behavior in the X-Y plane.

Plotting Voigt, Reuss and Hill averages at once
===============================================

Above, we have only used the Hill average for estimating the macroscopic elastic response. In order to evidence the
influence of the method (namely Voigt, Reuss or Hill), we can plot the directional Young moduli on orthogonal
sections (see :ref:`here<plotting>` for details) for each of the aforementioned methods as follows:

    >>> for method in ['Reuss', 'Hill', 'Voigt']:
    ...     C_avg = C_rotated.average(method)
    ...     if method == 'Reuss':
    ...         fig, axs = C_avg.Young_modulus.plot_xyz_sections(label='Reuss') # Create fig and axes (sections)
    ...     else:
    ...         fig, axs = C_avg.Young_modulus.plot_xyz_sections(fig=fig, axs=axs, label=method) # Use existing axes
    >>> axs[-1].legend() # doctest: +SKIP

which gives:


.. image:: images/E_VRH_sections.png