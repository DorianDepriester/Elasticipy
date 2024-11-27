Tutorial: working with stress and strain tensors
========================================================

Introduction
------------
This tutorial illustrates how we work on strain and stress tensors, and how Elasticipy handles arrays of tensors.

Single tensors
--------------
Let's start with basic operations with the stress tensor. For instance, we can compute the von Mises and Tresca equivalent stresses:


.. doctest::

    >>> import numpy as np
    >>> from Elasticipy.StressStrainTensors import StressTensor, StrainTensor
    >>> stress = StressTensor([[0, 1, 0],
    ...                       [1, 0, 0],
    ...                       [0, 0, 0]])
    >>> print(stress.vonMises(), stress.Tresca())
    1.7320508075688772 2.0

So now, let's have a look on the the strain tensor, and compute the principal strains and the volumetric change:

    >>> strain = StrainTensor([[0, 1e-3, 0],
    ...                   [1e-3, 0, 0],
    ...                   [0, 0, 0]])
    >>> print(strain.principalStrains())
    [ 0.001 -0.001  0.   ]
    >>> print(strain.volumetricStrain())
    0.0

Linear elasticity
--------------------------------
This section is dedicated to linear elasticity, hence introducing the fourth-order stiffness tensor.
As an example, create a stiffness tensor corresponding to ferrite:

    >>> from Elasticipy.FourthOrderTensor import StiffnessTensor
    >>> C = StiffnessTensor.fromCrystalSymmetry(symmetry='cubic', phase_name='ferrite',
    ...                                         C11=274, C12=175, C44=89)
    >>> print(C)
    Stiffness tensor (in Voigt notation) for ferrite:
    [[274. 175. 175.   0.   0.   0.]
     [175. 274. 175.   0.   0.   0.]
     [175. 175. 274.   0.   0.   0.]
     [  0.   0.   0.  89.   0.   0.]
     [  0.   0.   0.   0.  89.   0.]
     [  0.   0.   0.   0.   0.  89.]]
    Symmetry: cubic


Considering the previous strain, evaluate the corresponding stress:

    >>> sigma = C * strain
    >>> print(sigma)
    Stress tensor
    [[0.    0.178 0.   ]
     [0.178 0.    0.   ]
     [0.    0.    0.   ]]

Conversely, one can compute the compliance tensor:

    >>> S = C.inv()
    >>> print(S)
    Compliance tensor (in Voigt notation) for ferrite:
    [[ 0.00726819 -0.00283282 -0.00283282  0.          0.          0.        ]
     [-0.00283282  0.00726819 -0.00283282  0.          0.          0.        ]
     [-0.00283282 -0.00283282  0.00726819  0.          0.          0.        ]
     [ 0.          0.          0.          0.01123596  0.          0.        ]
     [ 0.          0.          0.          0.          0.01123596  0.        ]
     [ 0.          0.          0.          0.          0.          0.01123596]]
    Symmetry: cubic

and check that we retrieve the correct (initial) strain:

    >>> print(S * sigma)
    Strain tensor
    [[0.    0.001 0.   ]
     [0.001 0.    0.   ]
     [0.    0.    0.   ]]

.. _multidimensional-arrays:

Multidimensional tensor arrays
------------------------------
Elasticipy allows to process thousands of tensors at one, with the aid of tensor arrays.
For instance, we start by creating an array of 10 stresses:

    >>> n_array = 10
    >>> sigma = StressTensor.zeros(n_array)  # Initialize the array to zero-stresses
    >>> sigma.C[0, 1] = sigma.C[1, 0] = np.linspace(0, 100, n_array)    # The shear stress is linearly increasing
    >>> print(sigma[0])     # Check the initial value of the stress...
    Stress tensor
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    >>> print(sigma[-1])    # ...and the final value.
    Stress tensor
    [[  0. 100.   0.]
     [100.   0.   0.]
     [  0.   0.   0.]]

The corresponding strain array is evaluated with the same syntax as before:

    >>> eps = S * sigma
    >>> print(eps[0])     # Now check the initial value of strain...
    Strain tensor
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    >>> print(eps[-1])    # ...and the final value.
    Strain tensor
    [[0.         0.56179775 0.        ]
     [0.56179775 0.         0.        ]
     [0.         0.         0.        ]]

We can compute the corresponding elastic energies:

    >>> energy = 0.5*sigma.ddot(eps)
    >>> print(energy)     # print the elastic energy
    [ 0.          0.69357747  2.77430989  6.24219725 11.09723956 17.33943682
     24.96878901 33.98529616 44.38895825 56.17977528]

Apply rotations
---------------
Rotations can be applied on the tensors. If multiple rotations are applied at once, this results in tensor arrays.
Rotations are defined by ``scipy.transform.Rotation``.

    >>> from scipy.spatial.transform import Rotation

For example, let's consider a random set of 1000 rotations:

    >>> n_ori = 1000
    >>> random_state = 1234 # This is just to ensure reproducibility
    >>> rotations = Rotation.random(n_ori, random_state=random_state)

These rotations can be applied on the strain tensor

    >>> eps_rotated = eps.matmul(rotations)


The ``matmul()`` just works like the matrix product, thus increasing the dimensionality of the array.
Here, we thus get an array of shape (10, 1000).

    >>> print(eps_rotated.shape)
    (10, 1000)

Therefore, we can compute the corresponding rotated stress array:

    >>> sigma_rotated = C * eps_rotated
    >>> print(sigma_rotated.shape)    # Check out the shape of the stresses
    (10, 1000)

And get the stress back to the initial coordinate system:

    >>> sigma = sigma_rotated * rotations.inv()   # Go back to initial frame

Finally, we can estimate the mean stresses among all the orientations:

    >>> sigma_mean = sigma.mean(axis=1)     # Compute the mean over all orientations
    >>> print(sigma_mean[-1]) # random
    Stress tensor
    [[ 5.35134832e-01  8.22419895e+01  2.02619662e-01]
     [ 8.22419895e+01 -4.88440590e-01 -1.52733598e-01]
     [ 2.02619662e-01 -1.52733598e-01 -4.66942413e-02]]

Actually, a more straightforward method is to define a set of rotated stiffness tensors, and compute their Reuss average:

    >>> C_rotated = C * rotations
    >>> C_Voigt = C_rotated.Voigt_average()

Which yields the same results in terms of stress:

    >>> sigma_Voigt = C_Voigt * eps
    >>> print(sigma_Voigt[-1])
    Stress tensor
    [[ 5.35134832e-01  8.22419895e+01  2.02619662e-01]
     [ 8.22419895e+01 -4.88440590e-01 -1.52733598e-01]
     [ 2.02619662e-01 -1.52733598e-01 -4.66942413e-02]]

See :ref:`here<Averaging methods>` for further details about the averaging methods.