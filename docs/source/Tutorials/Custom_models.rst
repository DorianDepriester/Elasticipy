Extending Elasticipy: implementing custom models
-----------------------------------------------
Elasticipy does not claim to implement every plasticity or hyperelasticity model that exists. Instead, it provides
two abstract base classes that cover the *generic* machinery (normality rule, plotting, scaling to the yield
surface, stress computation from the energy potential, tensile curves...) so that implementing a new model boils
down to writing only what is specific to it: the model equations.

This tutorial shows how to do so through two worked examples:

- the `Hosford yield criterion <https://en.wikipedia.org/wiki/Yield_surface#Hosford_yield_function>`_ for plasticity,
- the `Gent hyperelastic model <https://en.wikipedia.org/wiki/Gent_(hyperelastic_model)>`_ for hyperelasticity.

Anatomy of the ``YieldCriterion`` class
=======================================
All the plasticity criteria available in Elasticipy (von Mises, Tresca, Drucker-Prager, Mohr-Coulomb, Hill...)
derive from the ``YieldCriterion`` abstract class, available in the ``elasticipy.yield_criteria`` module. This base
class already implements, for every subclass:

- ``normal``: the normality rule, that is the direction of plastic flow,
- ``normal_from_derivative``: numerical evaluation of the normality rule by finite differences,
- ``plot2D`` / ``plot3D``: plotting the yield surface in the biaxial / principal stress space,
- ``scale_stress_to_yield_surface``: automatic scaling of a stress tensor so that it lies on the yield surface,
- ``draw_surface_normal``: drawing the normal to the yield surface at a given location.

To create a new yield criterion, one only needs to subclass ``YieldCriterion`` and define the ``yield_function``
method, which is the only abstract method. It takes a ``StressTensor`` (or an array of stress tensors) and returns
:math:`f(\sigma)`, negative if and only if the stress state is elastic.

The Hosford yield criterion
===========================
The Hosford criterion is a generalization of von Mises and Tresca which depends on a single exponent :math:`a`.
It is defined by the equivalent stress:

.. math::

    \sigma_{eq} = \left[\frac{1}{2}\left(|\sigma_1-\sigma_2|^a + |\sigma_2-\sigma_3|^a + |\sigma_3-\sigma_1|^a\right)\right]^{1/a}

where :math:`\sigma_1, \sigma_2, \sigma_3` are the principal stresses. The yield function then reads
:math:`f(\sigma)=\sigma_{eq}-\sigma_y`, where :math:`\sigma_y` is the tensile yield stress. For :math:`a=2` and
:math:`a\to\infty`, one recovers the von Mises and Tresca criteria, respectively.

A minimal implementation is:

.. code-block:: python

    import numpy as np
    from elasticipy.yield_criteria import YieldCriterion

    class HosfordCriterion(YieldCriterion):
        name = 'Hosford'

        def __init__(self, a, yield_stress=1.0):
            self.a = a
            self.yield_stress = yield_stress

        def eq_stress(self, stress, **kwargs):
            s = stress.principal_stresses()
            return (0.5 * (np.abs(s[..., 0] - s[..., 1]) ** self.a
                          + np.abs(s[..., 1] - s[..., 2]) ** self.a
                          + np.abs(s[..., 2] - s[..., 0]) ** self.a)) ** (1 / self.a)

        def yield_function(self, stress):
            return self.eq_stress(stress) - self.yield_stress

        @property
        def _plot_bounds(self):
            s = self.yield_stress
            return (-s, s), (-s, s)

That is all that is required: the ``name`` class attribute is used as a default label in the plots, and the
``_plot_bounds`` property is optional (it only helps ``plot2D`` select a relevant plotting range). Note that the
implementation is fully vectorized: ``stress`` can be a single stress tensor or an arbitrarily-shaped array of
stress tensors, exactly like the criteria shipped with Elasticipy.

Let us instantiate the criterion and check the yield function on a uniaxial stress state:

.. doctest::

    >>> from elasticipy.tensors.stress_strain import StressTensor
    >>> hosford = HosfordCriterion(a=8, yield_stress=100.)
    >>> stress = StressTensor.tensile([1, 0, 0], 1)
    >>> float(hosford.yield_function(stress))
    -99.0

The ``scale_stress_to_yield_surface`` method is inherited as is:

.. doctest::

    >>> hosford.scale_stress_to_yield_surface(stress)
    Stress tensor
    [[99.99999999  0.          0.        ]
     [ 0.          0.          0.        ]
     [ 0.          0.          0.        ]]

The normality rule is also available: since we did not define the ``normal_analytical`` method, the base class
automatically falls back to a finite-difference evaluation of the gradient of the yield function:

.. doctest::

    >>> hosford.normal(stress)
    Strain tensor
    [[ 1.   0.   0. ]
     [ 0.  -0.5  0. ]
     [ 0.   0.  -0.5]]

Finally, the yield surface can be plotted exactly like for any other criterion:

.. code-block:: python

    fig, ax = hosford.plot2D()

For efficiency (or for singular points where finite differences are inaccurate), one can also implement the
optional ``normal_analytical`` method and return the analytical gradient of the yield function as a
``StrainTensor``; ``normal`` will then pick it up automatically. The Tresca and Mohr-Coulomb implementations in
``elasticipy/yield_criteria.py`` are good examples of such gradient computations, including the treatment of
singular points.

Anatomy of the ``HyperElastic`` class
======================================
Hyperelastic models all derive from the ``HyperElastic`` abstract class of the ``elasticipy.hyperelasticity``
module. Again, most of the work is already done by the base class:

- ``potential_from_F``: energy potential from the deformation gradient,
- ``stress_from_B`` / ``stress_from_F``: Cauchy stress, either analytically if the subclass defines
  ``stress_from_B_analytical``, or by finite differences on the potential otherwise,
- ``tensile_curve``: simulation of a tensile test (including the lateral contraction for compressible models),
- ``fit``: least-square fitting on tensile data.

The only abstract method is ``potential_from_B``, which returns the energy potential :math:`W` from the left
Cauchy-Green tensor :math:`\mathbf{B}`. Subclasses can also override ``is_compressible`` (which returns ``True``
by default) to declare incompressible behaviour; in this case, ``stress_from_B`` returns the deviatoric stress
and ``tensile_curve`` enforces isochoric deformation.

The Gent hyperelastic model
===========================
The Gent model was designed to account for the finite extensibility of polymer chains. Its potential reads:

.. math::

    W = -\frac{C_1 J_m}{2}\ln\left(1-\frac{\bar{I}_1-3}{J_m}\right)

where :math:`\bar{I}_1` is the first invariant of the isochoric left Cauchy-Green tensor, :math:`C_1` is the
small-strain shear modulus and :math:`J_m` is the maximum allowable value of :math:`\bar{I}_1-3`. The model is
incompressible in its standard form. The full implementation is:

.. code-block:: python

    import numpy as np
    from elasticipy.hyperelasticity import HyperElastic

    class Gent(HyperElastic):

        def __init__(self, C1, Jm):
            self.C1 = C1
            self.Jm = Jm

        def is_compressible(self):
            return False

        def potential_from_B(self, B):
            return -0.5 * self.C1 * self.Jm * np.log(1 - (B.I1_bar - 3) / self.Jm)

That is all: the stress tensor is then obtained by finite differences on the potential, so no analytical
derivation is needed. Let us simulate a tensile test with :math:`C_1=1` and :math:`J_m=10`. As expected for an
incompressible model, the returned stress is purely deviatoric (the hydrostatic pressure cannot be estimated):

.. doctest::

    >>> from elasticipy.tensors.finite_strain import DeformationGradient
    >>> gent = Gent(C1=1.0, Jm=10.)
    >>> F = DeformationGradient.isochoric_tensile([1, 0, 0], 0.5)
    >>> gent.stress_from_F(F)
    Stress tensor
    [[ 1.12094395  0.          0.        ]
     [ 0.         -0.56047198  0.        ]
     [ 0.          0.         -0.56047198]]

The ``tensile_curve`` method is also inherited, so the model can directly be compared to experimental data:

.. doctest::

    >>> print(gent.tensile_curve(0.5))
    1.1209439529677767

For better performance, one can implement the ``stress_from_B_analytical`` method to return the analytical Cauchy
stress; ``stress_from_B`` will use it automatically instead of the finite-difference route. The ``MooneyRivlin``
and ``Yeoh`` classes in ``elasticipy/hyperelasticity.py`` illustrate this pattern, as well as the definition of
custom ``fit`` classmethods.

Going further
=============
A few final remarks:

- **Multidimensional support**: all the stress/strain tensor classes in Elasticipy seamlessly work on arrays of
  tensors. Whenever possible, write the equations so that they remain vectorized (using NumPy operations and the
  ``...`` ellipsis in indices), as done above for the Hosford criterion.
- **Docstrings**: if you plan to contribute your model to Elasticipy (which is more than welcome!), please
  document it with NumPy-style docstrings, as done for the classes shipped with the package.
- **Testing**: the test suite (``tests/``) contains one test module per feature; adding a few tests for your
  custom model is the best way to secure it against future refactoring of the base classes.
