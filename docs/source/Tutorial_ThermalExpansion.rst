Thermal expansion
-----------------
In this tutorial, we will see how we can model thermal expansion using Elasticipy.

Isotropic case
==============
At first, let's consider steel as an example, assuming an isotropic behaviour:

.. doctest::

    >>> from Elasticipy.ThermalExpansion import ThermalExpansionTensor as ThEx
    >>> alpha = ThEx.isotropic(11e-6)
    >>> print(alpha)
    Thermal expansion tensor
    [[1.1e-05 0.0e+00 0.0e+00]
     [0.0e+00 1.1e-05 0.0e+00]
     [0.0e+00 0.0e+00 1.1e-05]]

To compute the strain due to an increase of temperature, just multiply ``alpha`` by this amount:

    >>> dT = 10
    >>> alpha * dT
    Strain tensor
    [[0.00011 0.      0.     ]
     [0.      0.00011 0.     ]
     [0.      0.      0.00011]]

We can also compute the strain values from a series (i.e. an array) of temperature increases, and get an array of strain
with the corresponding shape:

    >>> eps = alpha * [0,1,2]
    >>> print(eps)
    Strain tensor
    Shape=(3,)
    >>> eps[0]
    Strain tensor
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    >>> eps[-1]
    Strain tensor
    [[2.2e-05 0.0e+00 0.0e+00]
     [0.0e+00 2.2e-05 0.0e+00]
     [0.0e+00 0.0e+00 2.2e-05]]

See :ref:`here<multidimensional-arrays>` for details about multidimensional arrays of tensors.

Anisotropic case
================
The anisotropic case just works as before. For instance, if we consider carbon fibers:

    >>> alpha = ThEx.transverse_isotropic(alpha_11=5.6e-6, alpha_33=-0.4e-6)
    >>> print(alpha)
    Thermal expansion tensor
    [[ 5.6e-06  0.0e+00  0.0e+00]
     [ 0.0e+00  5.6e-06  0.0e+00]
     [ 0.0e+00  0.0e+00 -4.0e-07]]

As is, we consider that the fiber is parallel to the Z axis. We can obviously rotate it. For instance, if we want to
align the fiber with the X axis:

    >>> from scipy.spatial.transform import Rotation
    >>> rotation = Rotation.from_euler('Y', 90, degrees=True) # Rotate of 90° around Y axis
    >>> alpha * rotation
    Thermal expansion tensor
    [[-4.00000000e-07  0.00000000e+00  1.33226763e-21]
     [ 0.00000000e+00  5.60000000e-06  0.00000000e+00]
     [ 1.33226763e-21  0.00000000e+00  5.60000000e-06]]

If we want to consider multiple orientations at once, we can create an array of thermal expansion tensors,
:ref:`just as we do with strain/stress tensors<strain_rotations>`:

    >>> rotations = Rotation.random(10000, random_state=123) # 10k random orientations. random_state ensure reproducibility
    >>> alpha_rotated = alpha * rotations
    >>> print(alpha_rotated)
    Thermal expansion tensor
    Shape=(10000,)

If we want to compute the strain for each combination of orientations and temperatures in ``[0,1,2]`` (as done above),
we can use the ``matmul`` operator:

    >>> eps = alpha_rotated.matmul([0,1,2])
    >>> print(eps)
    Strain tensor
    Shape=(10000, 3)

For instance we can check out the maximum value for initial (0°) and final (2°) temperatures:

    >>> eps[:,0].max()    # 0 because it corresponds to 0°
    Strain tensor
    [[ 0. -0. -0.]
     [-0.  0. -0.]
     [-0. -0.  0.]]
    >>> eps[:,-1].max()
    Strain tensor
    [[1.12000000e-05 5.99947076e-06 5.99905095e-06]
     [5.99947076e-06 1.11999999e-05 5.99621436e-06]
     [5.99905095e-06 5.99621436e-06 1.11999992e-05]]

We see that the maximum value for the shear strain is consistent with the
`Mohr circle <https://en.wikipedia.org/wiki/Mohr%27s_circle>`_, as we have:

.. math::

    \max \varepsilon_{xy} = \frac{\max(\alpha_{11}, \alpha_{22}, \alpha_{33})
    - \min(\alpha_{11}, \alpha_{22}, \alpha_{33}) }{2}\times 2° C=6\times 10^{-6}



