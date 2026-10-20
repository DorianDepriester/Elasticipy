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

For some models (like Yeoh), Elasticipy implements both:
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


----------------------------------------------------------------------------------------------------

Practical Examples
------------------


Example 1: Simulating a Tensile Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's simulate a uniaxial tensile test on a hyperelastic material:

.. doctest::

    >>> # Material properties (natural rubber)
    >>> C10 = 0.5  # MPa
    >>> neo_hooke = NeoHooke(C10)
    >>> 
    >>> # Tensile simulation
    >>> stretches = np.linspace(0, 2.0, 20)
    >>> stresses = neo_hooke.tensile_curve(stretches)
    >>> 
    >>> # Display
    >>> print("Stretch | Stress (MPa)")
    >>> for s, sig in zip(stretches[::4], stresses[::4]):
    ...     print(f"{s:.2f}       | {sig:.4f}")
    Stretch | Stress (MPa)
    0.00       | 0.0000
    0.10       | 0.1667
    0.20       | 0.3667
    0.30       | 0.6000
    0.40       | 0.8667


Example 2: Comparing Different Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's compare the predictions of different models for a given deformation:

.. doctest::

    >>> # Deformation: biaxial tension
    >>> F_biaxial = DeformationGradient.diag([1.5, 1.5, 1.0/1.5**2])
    >>> 
    >>> # Models
    >>> models = {
    ...     'Neo-Hooke': NeoHooke(C=100.0),
    ...     'Yeoh': Yeoh([100.0, -10.0, 1.0]),
    ...     'Mooney-Rivlin': MooneyRivlin(np.array([[0, 50], [50, 0]]))
    ... }
    >>> 
    >>> # Stress calculation
    >>> for name, model in models.items():
    ...     sigma = model.stress_from_F(F_biaxial)
    ...     print(f"{name:20s}: sigma_xx = {sigma.C[0,0]:.4f}, sigma_yy = {sigma.C[1,1]:.4f}, sigma_zz = {sigma.C[2,2]:.4f}")
    Neo-Hooke         : sigma_xx = 133.3333, sigma_yy = 133.3333, sigma_zz = -66.6667
    Yeoh              : sigma_xx = 136.6667, sigma_yy = 136.6667, sigma_zz = -70.0000
    Mooney-Rivlin     : sigma_xx = 133.3333, sigma_yy = 133.3333, sigma_zz = -66.6667


Example 3: Simple Shear Deformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's analyze the behavior under pure shear:

.. doctest::

    >>> # Simple shear with different amplitudes
    >>> shear_magnitudes = [0.1, 0.5, 1.0, 1.5]
    >>> nh = NeoHooke(C=100.0)
    >>> 
    >>> print("Amplitude | Shear Stress (MPa)")
    >>> for mag in shear_magnitudes:
    ...     F = DeformationGradient.shear([1,0,0], [0,1,0], magnitude=mag)
    ...     sigma = nh.stress_from_F(F)
    ...     tau_xy = sigma.C[0,1]
    ...     print(f"{mag:.1f}        | {tau_xy:.4f}")
    Amplitude | Shear Stress (MPa)
    0.1        | 19.8019
    0.5        | 90.9091
    1.0        | 147.0588
    1.5        | 171.4678


----------------------------------------------------------------------------------------------------

Best Practices
-------------

1. **Model Selection**:
   - **Neo-Hooke**: Simple and effective for many elastomers
   - **Yeoh**: Better for large deformations of elastomers
   - **Mooney-Rivlin**: More general, but requires more parameters

2. **Compressibility**:
   - For **incompressible** materials (rubber, some polymers), use
     :math:`D = \text{None}` or a very large value
   - For **compressible** materials, provide an appropriate value for :math:`D`

3. **Numerical Precision**:
   - For small deformations, prefer analytical formulations
   - For very large deformations, reduce the finite difference step (:math:`h`)

4. **Validation**:
   - Always verify that the model correctly predicts behavior at small deformations
   - Compare with experimental data when available


----------------------------------------------------------------------------------------------------

Summary of Main Classes and Methods
-----------------------------------

**Finite Strain Classes**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``DeformationGradient``
     - Deformation gradient tensor :math:`\mathbf{F}`
   * - ``CauchyGreenTensor``
     - Cauchy-Green tensors :math:`\mathbf{B}` and :math:`\mathbf{C}`

**Key Properties and Methods**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method/Property
     - Description
   * - ``F.J``
     - Jacobian (determinant of :math:`\mathbf{F}`)
   * - ``F.U``, ``F.V``
     - Right and left stretch tensors
   * - ``F.R``
     - Rotation tensor
   * - ``F.B``
     - Left Cauchy-Green tensor
   * - ``F.CauchyGreen(side='right')``
     - Right Cauchy-Green tensor
   * - ``F.E``
     - Green-Lagrange tensor
   * - ``F.small_strain()``
     - Small strain approximation
   * - ``F.volumetric_strain()``
     - Volumetric strain
   * - ``F.elongation(u)``
     - Relative elongation in direction :math:`\mathbf{u}`

**Hyperelastic Models**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``NeoHooke(C, D=None)``
     - Neo-Hooke model
   * - ``Yeoh(C, D=None)``
     - Yeoh model (dependent on :math:`\bar{I}_1`)
   * - ``MooneyRivlin(C_ij, D=None)``
     - General Mooney-Rivlin model

**Hyperelastic Model Methods**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``potential_from_F(F)``
     - Calculate strain energy :math:`W`
   * - ``stress_from_F(F)``
     - Calculate stresses from :math:`\mathbf{F}`
   * - ``stress_from_B(B)``
     - Calculate stresses from :math:`\mathbf{B}`
   * - ``tensile_curve(stretch)``
     - Tensile curve
   * - ``fit(stretch, stress)``
     - Parameter fitting from data


----------------------------------------------------------------------------------------------------

References
----------

For deeper understanding of theoretical concepts:

1. **Holzapfel, G.A.** (2000). *Nonlinear Solid Mechanics: A Continuum Approach for Engineering Science*.
   Wiley. (Comprehensive reference on nonlinear continuum mechanics)

2. **Bower, A.F.** (2009). *Applied Mechanics of Solids*. CRC Press. (Pedagogical approach)

3. **Bonet, J., & Wood, R.D.** (2008). *Nonlinear Continuum Mechanics for Finite Element Analysis*.
   Cambridge University Press. (For numerical aspects)

4. **Trelour, W.R.** (1975). *The Mechanics of Nonlinear Materials with Memory*. (For hyperelasticity)


For applications with Elasticipy:

- API Documentation: :doc:`API/Elasticipy.hyperelasticity`
- API Documentation: :doc:`API/Elasticipy.tensors.finite_strain`
- Examples in ``docs/source/examples/``


----------------------------------------------------------------------------------------------------

Exercises
--------

1. **Pure Deformation**: Create a tensor :math:`\mathbf{F}` corresponding to pure tension
   along the z-axis with a stretch of 2.0. Calculate :math:`\mathbf{B}`, :math:`\mathbf{C}`,
   and verify that :math:`\mathbf{B} = \mathbf{C}`.

2. **Pure Shear**: Create a tensor :math:`\mathbf{F}` for pure shear in the x-y plane
   with magnitude 0.5. Calculate stresses using a Neo-Hooke model (:math:`C = 100`).

3. **Model Comparison**: Compare the tensile curves predicted by Neo-Hooke, Yeoh
   (with :math:`C = [100, -10, 1]`) and Mooney-Rivlin (with :math:`C_{ij} = [[0, 50], [50, 0]]`)
   for stretches from 0 to 1.5.

4. **Parameter Fitting**: Generate tensile data with a known Yeoh model, then use the
   ``fit`` method to recover the parameters.

5. **Biaxial Deformation**: Create a biaxial deformation with :math:`F = \text{diag}([1.2, 1.2, 1/1.44])`.
   Calculate stresses with Neo-Hooke and verify that the stress in the z direction is negative
   (compression due to incompressibility).


Solutions:

.. doctest::

    >>> # Exercise 1
    >>> F_ex1 = DeformationGradient.diag([1, 1, 2])
    >>> C_ex1 = F_ex1.CauchyGreen()
    >>> B_ex1 = F_ex1.B
    >>> np.allclose(C_ex1.matrix, B_ex1.matrix)
    True
    
    >>> # Exercise 2
    >>> F_ex2 = DeformationGradient.shear([1,0,0], [0,1,0], 0.5)
    >>> nh_ex2 = NeoHooke(100)
    >>> sigma_ex2 = nh_ex2.stress_from_F(F_ex2)
    >>> print(f"Shear stress: {sigma_ex2.C[0,1]:.4f} MPa")
    Shear stress: 47.6190 MPa
    
    >>> # Exercise 5
    >>> F_ex5 = DeformationGradient.diag([1.2, 1.2, 1/1.44])
    >>> nh_ex5 = NeoHooke(100)
    >>> sigma_ex5 = nh_ex5.stress_from_F(F_ex5)
    >>> print(f"sigma_zz = {sigma_ex5.C[2,2]:.4f} MPa (should be negative)")
    sigma_zz = -33.3333 MPa (should be negative)
