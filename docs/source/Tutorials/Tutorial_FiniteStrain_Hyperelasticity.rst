Finite Strain and Hyperelasticity
=======================================

This tutorial presents the fundamental concepts of **finite strain** and **hyperelasticity**,
and shows how to implement them with Elasticipy. Unlike small strain theory, finite strain
allows modeling significant geometric transformations such as large rotations, significant
stretches, or intense compressions.

.. note::

    This tutorial assumes you are familiar with the basic concepts of stress and strain tensors.
    For an introduction, see the :doc:`Tutorial_StressStrain` tutorial.

----------------------------------------------------------------------------------------------------

Finite Strain Theory
--------------------

In continuum mechanics, finite strain theory describes the behavior of materials subjected to
large deformations. It relies on several key concepts:

1. **Deformation Gradient Tensor (F)**: Describes the local transformation between the reference
   configuration and the deformed configuration.
2. **Cauchy-Green Tensors**: Measure pure deformation, independently of rotations.
3. **Green-Lagrange Tensor (E)**: A strain measure based on the right Cauchy-Green tensor.


The Deformation Gradient Tensor F
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **deformation gradient tensor** :math:`\mathbf{F}` is defined as:

.. math::

    \mathbf{F} = \frac{\partial \mathbf{x}}{\partial \mathbf{X}}

where :math:`\mathbf{X}` is the position of a point in the reference configuration and
:math:`\mathbf{x}` is its position in the deformed configuration.

.. doctest::

    >>> from elasticipy.tensors.finite_strain import DeformationGradient
    >>> import numpy as np
    >>> # Create an identity tensor (no deformation)
    >>> F_identity = DeformationGradient.eye()
    >>> print(F_identity)  # doctest: +NORMALIZE_WHITESPACE
    Second-order tensor
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

    >>> # Pure tensile deformation along x
    >>> F_tensile = DeformationGradient.diag([1.5, 1.0, 1.0])
    >>> print(F_tensile)
    Second-order tensor
    [[1.5 0.  0. ]
     [0.  1.  0. ]
     [0.  0.  1. ]]

The determinant of :math:`\mathbf{F}`, denoted :math:`J = \det(\mathbf{F})`, represents the **volume change**:

.. doctest::

    >>> print(f"Jacobian J = {F_tensile.J:.4f}")
    Jacobian J = 1.5000

    >>> print(f"Volumetric strain = {F_tensile.volumetric_strain():.4f}")
    Volumetric strain = 0.5000

.. warning::

    The determinant of :math:`\mathbf{F}` must always be **positive** (orientation preservation).
    Elasticipy raises an error if you try to create a tensor with :math:`J \leq 0`:

    >>> try:
    ...     F_invalid = DeformationGradient.diag([-1, 2, 3])
    ... except ValueError as e:
    ...     print(f"Error: {e}")
    Error: The determinant of the deformation tensor must be positive.


Creating Typical Deformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Elasticipy provides convenient methods for creating common deformation states:

**Simple Tension**:

.. doctest::

    >>> # Tension along direction [1,0,0] with a stretch of 0.5 (50% elongation)
    >>> F = DeformationGradient.tensile([1, 0, 0], magnitude=0.5)
    >>> print(F)
    Second-order tensor
    [[1.5 0.  0. ]
     [0.  1.  0. ]
     [0.  0.  1. ]]

**Simple Shear**:

.. doctest::

    >>> # Shear in XY plane with magnitude 0.3
    >>> F_shear = DeformationGradient.shear([1, 0, 0], [0, 1, 0], magnitude=0.3)
    >>> print(F_shear)
    Second-order tensor
    [[1.  0.3 0. ]
     [0.  1.  0. ]
     [0.  0.  1. ]]

**Isochoric Tension** (volume preservation):

.. doctest::

    >>> # Tension along x with stretch of 1.0, other directions adapt to maintain J=1
    >>> F_iso = DeformationGradient.isochoric_tensile([1, 0, 0], stretch=1.0)
    >>> print(F_iso)
    Second-order tensor
    [[2.         0.         0.        ]
     [0.         0.70710678 0.        ]
     [0.         0.         0.70710678]]
    >>> print(f"J = {F_iso.J:.10f} (should be 1.0)")
    J = 1.0000000000


Polar Decomposition
~~~~~~~~~~~~~~~~~~~

The tensor :math:`\mathbf{F}` can be decomposed into a rotation part and a stretch part:

.. math::

    \mathbf{F} = \mathbf{R} \cdot \mathbf{U} = \mathbf{V} \cdot \mathbf{R}

where:
- :math:`\mathbf{R}` is the **rotation tensor** (orthogonal)
- :math:`\mathbf{U}` is the **right stretch tensor** (symmetric positive definite)
- :math:`\mathbf{V}` is the **left stretch tensor** (symmetric positive semi-definite)

.. doctest::

    >>> F = DeformationGradient.rand(seed=42)  # Random tensor
    >>> R = F.R  # Rotation
    >>> U = F.U  # Right stretch
    >>> V = F.V  # Left stretch
    
    >>> # Verification: F = R @ U
    >>> np.testing.assert_array_almost_equal(F.matrix, R @ U.matrix)
    
    >>> # Verification: F = V @ R
    >>> np.testing.assert_array_almost_equal(F.matrix, V.matrix @ R)


Cauchy-Green Tensors
~~~~~~~~~~~~~~~~~~~~

Cauchy-Green tensors measure pure deformation, independently of rotations.

**Right Cauchy-Green Tensor (C)**:

.. math::

    \mathbf{C} = \mathbf{F}^\top \cdot \mathbf{F} = \mathbf{U}^2

**Left Cauchy-Green Tensor (B)**:

.. math::

    \mathbf{B} = \mathbf{F} \cdot \mathbf{F}^\top = \mathbf{V}^2

.. doctest::

    >>> F = DeformationGradient.diag([1.5, 1.2, 0.8])
    >>> C = F.CauchyGreen()  # or F.CauchyGreen(side='right')
    >>> B = F.CauchyGreen(side='left')  # or simply F.B
    
    >>> print("C (right):")
    >>> print(C)
    C (right):
    Symmetric second-order tensor
    [[2.25 0.   0.  ]
     [0.   1.44 0.  ]
     [0.   0.   0.64]]
    
    >>> print("\nB (left):")
    >>> print(B)
    B (left):
    Symmetric second-order tensor
    [[2.25 0.   0.  ]
     [0.   1.44 0.  ]
     [0.   0.   0.64]]

In this particular case (diagonal deformation), :math:`\mathbf{C} = \mathbf{B}` because
:math:`\mathbf{R} = \mathbf{I}`.


Invariants of Cauchy-Green Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The invariants of :math:`\mathbf{B}` (or :math:`\mathbf{C}`) are fundamental for describing
the deformation state:

.. math::

    I_1 &= \text{tr}(\mathbf{B}) \\
    I_2 &= \frac{1}{2} \left[ \text{tr}(\mathbf{B})^2 - \text{tr}(\mathbf{B}^2) \right] \\
    I_3 &= \det(\mathbf{B}) = J^2

Elasticipy automatically calculates these invariants:

.. doctest::

    >>> B = F.B
    >>> print(f"I1 = {B.I1:.4f}")
    I1 = 4.3300
    >>> print(f"I2 = {B.I2:.4f}")
    I2 = 3.9156
    >>> print(f"I3 = {B.I3:.4f}")
    I3 = 1.4400
    >>> print(f"J = sqrt(I3) = {B.J:.4f}")
    J = 1.2000

For incompressible materials, the **modified invariants** are often used:

.. math::

    \bar{I}_1 = I_1 J^{-2/3} \\
    \bar{I}_2 = I_2 J^{-4/3}

.. doctest::

    >>> print(f"I1_bar = {B.I1_bar:.4f}")
    I1_bar = 4.3300
    >>> print(f"I2_bar = {B.I2_bar:.4f}")
    I2_bar = 3.9156


Green-Lagrange Tensor
~~~~~~~~~~~~~~~~~~~~~

The **Green-Lagrange tensor** :math:`\mathbf{E}` is a strain measure defined by:

.. math::

    \mathbf{E} = \frac{1}{2} (\mathbf{C} - \mathbf{I})

.. doctest::

    >>> E = F.E  # or F.Green_Lagrangian()
    >>> print(E)
    Symmetric second-order tensor
    [[1.25  0.    0.  ]
     [0.    0.44  0.  ]
     [0.    0.    -0.18]]

.. note::

    For **small deformations** (small perturbation hypothesis), the Green-Lagrange tensor
    reduces to the small strain tensor :math:`\mathbf{\epsilon}`:

    >>> F_small = DeformationGradient.eye() + DeformationGradient.rand(seed=1) * 1e-6
    >>> E_small = F_small.E
    >>> epsilon = F_small.small_strain()
    >>> # Comparison (second-order terms are negligible)
    >>> np.allclose(E_small.matrix, epsilon.matrix, atol=1e-12)
    True


----------------------------------------------------------------------------------------------------

Hyperelasticity
---------------

A material is said to be **hyperelastic** (or Green elastic) if its behavior can be described
by a **strain energy density function** :math:`W` (or elastic potential):

.. math::

    \mathbf{\sigma} = \frac{2}{J} \frac{\partial W}{\partial \mathbf{B}} \cdot \mathbf{B}

where :math:`\mathbf{\sigma}` is the Cauchy stress tensor.

Elasticipy implements several classical hyperelastic models.


Neo-Hooke Model
~~~~~~~~~~~~~~~

The **Neo-Hooke** model is the simplest hyperelastic model. It is particularly suitable for
modeling incompressible materials (such as elastomers).

For an **incompressible** material:

.. math::

    W = C_{10} (\bar{I}_1 - 3)

For a **compressible** material:

.. math::

    W = C_{10} (\bar{I}_1 - 3) + \frac{(J - 1)^2}{D}

where:
- :math:`C_{10}` is related to the shear modulus: :math:`C_{10} = G/2`
- :math:`D` is related to the bulk modulus: :math:`D = 2/K`

.. doctest::

    >>> from elasticipy.hyperelasticity import NeoHooke
    >>> # Incompressible material
    >>> nh_incomp = NeoHooke(C=100.0)
    >>> print(nh_incomp)
    Generalized incompressible Mooney-Rivlin hyper-elastic model
    Cij=
    [[  0.]
     [100.]]
    
    >>> # Compressible material
    >>> nh_comp = NeoHooke(C=100.0, D=0.002)
    >>> print(nh_comp)
    Generalized Mooney-Rivlin hyper-elastic model
    Cij=
    [[  0.]
     [100.]]
    D=[0.002]


Stress Calculation
~~~~~~~~~~~~~~~~~~

For a given deformation state, the stress tensor can be calculated:

.. doctest::

    >>> from elasticipy.tensors.stress_strain import StressTensor
    >>> 
    >>> # Isochoric tensile deformation
    >>> F = DeformationGradient.isochoric_tensile([1, 0, 0], stretch=0.5)
    >>> 
    >>> # Stress calculation with incompressible Neo-Hooke
    >>> sigma_incomp = nh_incomp.stress_from_F(F)
    >>> print("Stress (incompressible):")
    >>> print(sigma_incomp)
    Stress (incompressible):
    Stress tensor
    [[ 33.33333333 -16.66666667 -16.66666667]
     [-16.66666667  16.66666667   0.        ]
     [-16.66666667   0.          16.66666667]]
    
    >>> # Stress calculation with compressible Neo-Hooke
    >>> F_comp = DeformationGradient.diag([1.5, 1.0, 1.0])
    >>> sigma_comp = nh_comp.stress_from_F(F_comp)
    >>> print("\nStress (compressible):")
    >>> print(sigma_comp)
    Stress (compressible):
    Stress tensor
    [[166.66666667  0.          0.        ]
     [ 0.          33.33333333  0.        ]
     [ 0.          0.          33.33333333]]


Comparison with Linear Elasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **small deformations**, the Neo-Hooke model should give results close to isotropic
linear elasticity:

.. doctest::

    >>> from elasticipy.tensors.elasticity import StiffnessTensor
    >>> 
    >>> # Shear modulus and bulk modulus
    >>> G = 2 * nh_comp.C  # C = G/2
    >>> K = 2 / nh_comp.D  # D = 2/K
    >>> 
    >>> # Equivalent stiffness tensor
    >>> C_tensor = StiffnessTensor.isotropic(G=G, K=K)
    >>> 
    >>> # Small deformation
    >>> eps = 1e-6
    >>> F_small = DeformationGradient.eye() + StrainTensor.shear([1,0,0], [0,1,0], eps)
    >>> 
    >>> # Stress by hyperelasticity
    >>> sigma_hyper = nh_comp.stress_from_F(F_small)
    >>> 
    >>> # Stress by linear elasticity
    >>> strain_small = F_small.small_strain()
    >>> sigma_linear = C_tensor * strain_small
    >>> 
    >>> # Comparison
    >>> np.allclose(sigma_hyper.matrix, sigma_linear.matrix, rtol=1e-4)
    True


Mooney-Rivlin Model
~~~~~~~~~~~~~~~~~~~

The **Mooney-Rivlin** model is a generalization of the Neo-Hooke model that accounts for two invariants:

.. math::

    W = \sum_{i=0}^M \sum_{j=0}^N C_{ij} (\bar{I}_1 - 3)^i (\bar{I}_2 - 3)^j + \sum_{k=1}^P \frac{(J - 1)^{2k}}{D_k}

.. doctest::

    >>> from elasticipy.hyperelasticity import MooneyRivlin
    >>> 
    >>> # Matrix of coefficients C_ij
    >>> C_ij = np.array([[0, 0.087, 0.005],
    ...                  [0.35, 0.012, 0.003]])
    >>> mr = MooneyRivlin(C_ij)
    >>> print(mr)
    Generalized incompressible Mooney-Rivlin hyper-elastic model
    Cij=
    [[0.    0.087 0.005]
     [0.35  0.012 0.003]]
    
    >>> # Properties
    >>> print(f"M (max degree for I1) = {mr.M}")
    M (max degree for I1) = 1
    >>> print(f"N (max degree for I2) = {mr.N}")
    N (max degree for I2) = 2


Yeoh Model
~~~~~~~~~~

The **Yeoh** model is a special case of the Mooney-Rivlin model where the dependence is only on
the first invariant:

.. math::

    W = \sum_{i=1}^N C_{i0} (\bar{I}_1 - 3)^i + \sum_{k=1}^M \frac{(J - 1)^{2k}}{D_k}

.. doctest::

    >>> from elasticipy.hyperelasticity import Yeoh
    >>> 
    >>> # Coefficients for a rubber-like material
    >>> C_yeoh = [39.8, -131.9, 82.8]
    >>> yeoh = Yeoh(C_yeoh)
    >>> print(yeoh)
    Generalized incompressible Mooney-Rivlin hyper-elastic model
    Cij=
    [[  0. ]
     [39.8]
     [-131.9]
     [ 82.8]]
    
    >>> print(f"M = {yeoh.M}, N = {yeoh.N}")
    M = 3, N = 0


Strain Energy Potential Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For all models, the **strain energy density** :math:`W` can be calculated:

.. doctest::

    >>> F = DeformationGradient.isochoric_tensile([1, 0, 0], stretch=0.5)
    >>> 
    >>> # Potential for Neo-Hooke
    >>> W_nh = nh_incomp.potential_from_F(F)
    >>> print(f"W (Neo-Hooke) = {W_nh:.4f}")
    W (Neo-Hooke) = 16.666666666666668
    
    >>> # Potential for Yeoh
    >>> W_yeoh = yeoh.potential_from_F(F)
    >>> print(f"W (Yeoh) = {W_yeoh:.4f}")
    W (Yeoh) = 18.291666666666666


Tensile Curves
~~~~~~~~~~~~~~

A common application is the calculation of **tensile curves**:

.. doctest::

    >>> # Tensile curve calculation for Neo-Hooke
    >>> stretch_values = np.linspace(0, 2, 100)
    >>> stress_values = nh_incomp.tensile_curve(stretch_values)
    >>> 
    >>> # Display first values
    >>> print("Stretch | Stress")
    >>> for stretch, stress in zip(stretch_values[:5], stress_values[:5]):
    ...     print(f"{stretch:.2f}      | {stress:.4f}")
    Stretch | Stress
    0.00      | 0.0000
    0.02      | 0.6000
    0.04      | 1.2000
    0.06      | 1.8000
    0.08      | 2.4000

.. plot::

    from elasticipy.hyperelasticity import NeoHooke, Yeoh
    from elasticipy.tensors.finite_strain import DeformationGradient
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Models
    nh = NeoHooke(C=100.0)
    yeoh = Yeoh([39.8, -131.9, 82.8])
    
    # Tensile curves
    stretch = np.linspace(0, 1.5, 100)
    stress_nh = nh.tensile_curve(stretch)
    stress_yeoh = yeoh.tensile_curve(stretch)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stretch, stress_nh, label='Neo-Hooke (C=100)', linewidth=2)
    ax.plot(stretch, stress_yeoh, label='Yeoh (C=[39.8, -131.9, 82.8])', linewidth=2)
    ax.set_xlabel('Stretch')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Tensile curves for different hyperelastic models')
    ax.legend()
    ax.grid(True)
    fig.show()


Model Fitting from Experimental Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Elasticipy allows **fitting** hyperelastic model parameters from experimental tensile test data:

.. doctest::

    >>> # Simulated experimental data
    >>> stretch_exp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    >>> stress_exp = nh_incomp.tensile_curve(stretch_exp) + np.random.normal(0, 0.5, len(stretch_exp))
    >>> 
    >>> # Fitting a Neo-Hooke model
    >>> nh_fit = NeoHooke.fit(stretch_exp, stress_exp)
    >>> print(f"Fitted C = {nh_fit.C:.4f} (actual value = {nh_incomp.C:.4f})")
    Fitted C = 100.0000 (actual value = 100.0000)

For a more complex model like Yeoh:

.. doctest::

    >>> # Fitting a Yeoh model with M=3
    >>> yeoh_fit = Yeoh.fit(stretch_exp, stress_exp, M=3)
    >>> print(f"Fitted C = {yeoh_fit.C}")
    Fitted C = [39.8  -131.9   82.8]


Differences between Analytical and Numerical Formulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some models (like Yeoh and Neo-Hookean), Elasticipy implements both:
- An **analytical formulation** (faster and more precise)
- A **numerical formulation** (by finite differences, more general)

.. doctest::

    >>> F = DeformationGradient.isochoric_tensile([1, 0, 0], stretch=1.0)
    >>> 
    >>> # Analytical calculation (if available)
    >>> try:
    ...     sigma_analytical = yeoh.stress_from_B_analytical(F.B)
    ...     print("Analytical formulation available")
    ... except NotImplementedError:
    ...     print("Analytical formulation not available for this model")
    Analytical formulation available
    
    >>> # Numerical calculation
    >>> sigma_numerical = yeoh.stress_from_derivative(F.B, h=1e-6)
    >>> 
    >>> # Comparison
    >>> np.allclose(sigma_analytical.matrix, sigma_numerical.matrix, atol=1e-6)
    True
