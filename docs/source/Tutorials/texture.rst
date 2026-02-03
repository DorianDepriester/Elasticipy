Working with crystallographic textures
---------------------------------------

With the help of `orix <https://orix.readthedocs.io/en/stable/index.html>`_, Elasticipy allows to compute averages based
on crystallographic texture (in addition to single orientations; see :ref:`rotations`).

Define and compose textures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discrete textures
=================
A series of "usual" texture components (e.g. Goss, cube, Brass etc.) are already implemented in Elasticipy:

.. doctest::

    >>> from elasticipy.crystal_texture import DiscreteTexture
    >>> goss = DiscreteTexture.Goss()
    >>> goss
    Crystallographic texture
    φ1=0.00°, ϕ=45.00°, φ2=0.00°

This texture actually consists in a single orientation (as opposed to fibre textures, see below). It can be used to
rotate a stiffness tensor as follows:

    >>> from elasticipy.tensors.elasticity import StiffnessTensor
    >>> C = StiffnessTensor.cubic(C11=186, C12=134, C44=77) # Copper, mp-30
    >>> C * goss
    Stiffness tensor (in Voigt mapping):
    [[ 1.86000000e+02  1.34000000e+02  1.34000000e+02  0.00000000e+00
       0.00000000e+00  0.00000000e+00]
     [ 1.34000000e+02  2.37000000e+02  8.30000000e+01  1.42108547e-14
       0.00000000e+00  0.00000000e+00]
     [ 1.34000000e+02  8.30000000e+01  2.37000000e+02 -7.10542736e-15
       0.00000000e+00  0.00000000e+00]
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  2.60000000e+01
       0.00000000e+00  0.00000000e+00]
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       7.70000000e+01  0.00000000e+00]
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
       0.00000000e+00  7.70000000e+01]]

Fibre textures
==============
Fibre textures are defined as uniformly distributed orientations around a given axis, leading to a single line when
plotting an Orientation Distribution Function (ODF), hence the name. Therefore, there are two ways to define such
textures.

The first one consists in defining the *uvw* direction to align with a given axis (related to the sample coordinate
system). For instance, let's consider the texture such that the <111> direction is aligned with the Z axis:

    >>> from elasticipy.crystal_texture import FibreTexture
    >>> from orix.crystal_map import Phase
    >>> from orix.vector.miller import Miller
    >>>
    >>> phase = Phase(point_group='m-3m')   # Cubic symmetry
    >>> m = Miller(uvw=[1,1,1], phase=phase)
    >>> fibre_111 = FibreTexture.from_Miller_axis(m, [0,0,1])
    >>> fibre_111
    Fibre texture
    <1. 1. 1.> || [0, 0, 1]

The other way is to use two (out the the three) Bunge-Euler angles to define the possible orientations (assuming that
orientations are uniformly distributed over the remaining angle). E.g.:

    >>> fibre_phi2 = FibreTexture.from_Euler(phi1=0., Phi=0.)
    >>> fibre_phi2
    Fibre texture
    φ1= 0.0°, ϕ= 0.0°

Then, the average stiffness resulting from such orientation distribution can be estimated from Voigt, Reuss or Hill
method by integration over the all possible rotations. E.g.:

    >>> Chill = C.Hill_average(orientations=fibre_phi2)

One can check that this results in a transversely isotropic behaviour by plotting the values of the Young modulus in
planar sections. The full code is:

.. plot::

    >>> from elasticipy.tensors.elasticity import StiffnessTensor
    >>> from elasticipy.crystal_texture import FibreTexture
    >>>
    >>> C = StiffnessTensor.cubic(C11=186, C12=134, C44=77)
    >>> fibre_phi2 = FibreTexture.from_Euler(phi1=0., Phi=0.)
    >>> Chill = C.Hill_average(orientations=fibre_phi2)
    >>> E = Chill.Young_modulus
    >>> fig, ax = E.plot_xyz_sections()
    >>> fig.show()

Composite textures
==================
Different textures can be composed together to create a ``CompositeTexture`` object. For instance, if we consider a
a cubic material which exhibits 40% of uniform texture, 30% of <111>||[0,0,1] and balanced copper texture, we can do the
following:

    >>> from elasticipy.crystal_texture import UniformTexture
    >>> t = 0.4 * UniformTexture() + 0.3 * fibre_111 +  0.3 * DiscreteTexture.copper()
    >>> t
    Mixture of crystallographic textures
     Wgt.  Type      Component
     ------------------------------------------------------------
     0.40  uniform   Uniform distribution over SO(3)
     0.30  fibre     <1. 1. 1.> || [0, 0, 1]
     0.30  discrete  φ1=90.00°, ϕ=35.26°, φ2=45.00°

Again, the Hill average can be computed as follows:

    >>> C.Hill_average(orientations=t)
    Stiffness tensor (in Voigt mapping):
    [[ 2.18433638e+02  1.19058181e+02  1.16508181e+02 -1.16263805e-15
       4.90396075e-10  1.30236624e-16]
     [ 1.19058181e+02  2.15883638e+02  1.19058181e+02 -3.45188669e-15
       3.60624458e+00  1.30996473e-15]
     [ 1.16508181e+02  1.19058181e+02  2.18433638e+02 -2.66453526e-15
      -3.60624458e+00  2.66930323e-17]
     [-1.59872116e-15 -3.28225710e-15 -3.73034936e-15  4.13627287e+01
       1.41602644e-16  3.60624458e+00]
     [ 4.90392835e-10  3.60624458e+00 -3.60624458e+00 -6.74210370e-16
       3.88127287e+01 -1.40788787e-15]
     [ 2.19967723e-15  3.77034682e-15  1.59277697e-15  3.60624458e+00
      -2.72082736e-17  4.00877287e+01]]

Random sampling
~~~~~~~~~~~~~~
Sample of orientations can be drawn from all kind of textures. While a "random" sample from a single discrete texture
does not make any sense, it can be of great interest for fibres or composite textures.

For instance, a sample of 10 orientations can be drawn from the composite texture defined above as follows:

    >>> sample = t.sample(num=10, seed=123) # Seed here is used to ensure reproducibility
    >>> sample
    Orientation (10,) 1
    [[ 0.1157  0.5901  0.2194 -0.7683]
     [ 0.3954 -0.6714 -0.421   0.4644]
     [ 0.9571 -0.1986 -0.2022  0.0609]
     [ 0.0802 -0.7012  0.3947 -0.5883]
     [ 0.1952 -0.8854  0.3814  0.1802]
     [-0.4814  0.4494  0.097   0.7463]
     [ 0.3647 -0.2798 -0.1159 -0.8805]
     [ 0.3647 -0.2798 -0.1159 -0.8805]
     [ 0.3647 -0.2798 -0.1159 -0.8805]
     [ 0.3647 -0.2798 -0.1159 -0.8805]]

One may note that here, the 4 last orientations are the same. This is because it relates to the copper texture:

    >>> DiscreteTexture.copper().orientation
    Orientation (1,) 1
    [[ 0.3647 -0.2798 -0.1159 -0.8805]]

Plotting
~~~~~~~~
Pole figures can be drawn to evidence how each texture works. For instance, the (pure) Goss texture results in the
following pole figure:

.. plot::

    >>> from elasticipy.crystal_texture import DiscreteTexture
    >>> from orix.crystal_map import Phase
    >>> from orix.vector.miller import Miller
    >>>
    >>> phase = Phase(point_group='m-3m')   # Cubic symmetry
    >>> m = Miller(uvw=[1,0,0], phase=phase)
    >>> Goss = DiscreteTexture.Goss()
    >>> fig, ax = Goss.plot_as_pole_figure(m.symmetrise(unique=True))
    >>> fig.show()

Fibre texture can be drawn in a simular way. E.g.:

.. plot::

    >>> from elasticipy.crystal_texture import FibreTexture
    >>> from orix.crystal_map import Phase
    >>> from orix.vector.miller import Miller
    >>>
    >>> phase = Phase(point_group='m-3m')   # Cubic symmetry
    >>> m = Miller(uvw=[1,1,1], phase=phase)
    >>> fibre_111 = FibreTexture.from_Miller_axis(m, [1,0,0])
    >>> fig, ax = fibre_111.plot_as_pole_figure(m.symmetrise(unique=True))
    >>> fig.show()
