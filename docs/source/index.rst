Welcome to Elasticipy's Documentation!
======================================

.. image:: https://img.shields.io/pypi/v/Elasticipy
   :alt: PyPI - Version
   :target: https://pypi.org/project/elasticipy/

.. image:: https://img.shields.io/pypi/dm/Elasticipy
   :alt: PyPI - Downloads
   :target: https://pypistats.org/packages/elasticipy

.. image:: https://img.shields.io/badge/Hosted%20on%20-%20GitHub%20-%20blue
   :alt: Static Badge
   :target: https://github.com/DorianDepriester/Elasticipy

.. image:: https://img.shields.io/pypi/l/Elasticipy
   :alt: PyPI - License
   :target: https://github.com/DorianDepriester/Elasticipy/blob/main/LICENSE

.. image:: https://readthedocs.org/projects/elasticipy/badge/?version=latest
   :alt: ReadTheDoc
   :target: https://elasticipy.readthedocs.io/

.. image:: https://zenodo.org/badge/876162900.svg
  :alt: doi:10.5281/zenodo.14501849
  :target: https://doi.org/10.5281/zenodo.14501849

.. image:: https://codecov.io/gh/DorianDepriester/Elasticipy/graph/badge.svg?token=VUZPEUPBH1
   :alt: coverage
   :target: https://codecov.io/gh/DorianDepriester/Elasticipy

.. image:: https://img.shields.io/pypi/pyversions/Elasticipy
   :alt: PyPI - Python Version



Purpose of this package
-----------------------
This Python's package is dedicated to work on mechanical elasticity-related tensors; namely: stress, strain and
stiffness/compliance tensors.

It provides a couple of ways to perform basic operations on these tensors in a user-friendly way. In addition, it
handles arrays of tensors, allowing to perform thousands of data at once (see an example :ref:`here<multidimensional-arrays>`).

It also comes with plotting features (e.g. spatial dependence of Young modulus).

Fundamentals of continuum mechanics
-----------------------------------
Linear elasticity
~~~~~~~~~~~~~~~~~

In continuum mechanics, the stress is described by the second-order Cauchy's stress tensor :math:`\boldsymbol{\sigma}`
whereas the strain tensor :math:`\boldsymbol{\varepsilon}`, under the small strain assumption is defined as:

.. math::

    \boldsymbol{\varepsilon}=\frac12 \left(\boldsymbol{\nabla}\boldsymbol{u} + \boldsymbol{\nabla}\boldsymbol{u}^T\right)

where :math:`\boldsymbol{\nabla}\boldsymbol{u}` denotes the displacement gradient.

In linear elasticity, the stress-strain relationship is governed by the generalized Hooke's law:

.. math::

    \sigma_{ij} = C_{ijkl} \varepsilon_{kl}


where :math:`\boldsymbol{C}` is the fourth-order **stiffness tensor**, which depends on the material's properties.
Due to symmetries in stress, strain, and material response, :math:`\boldsymbol{C}` exhibits various symmetries, reducing
the number of independent constants: from 2 (for isotropic materials), to 21 (for triclinic materials).

In isotropic materials, the elasticity tensor can be expressed in terms of two constants: the **Lamé parameters**
:math:`\lambda` :math:`\mu` (the shear modulus):

.. math::

    \sigma_{ij} = \lambda \delta_{ij} \varepsilon_{kk} + 2 \mu \varepsilon_{ij}.

For anisotropic materials, the structure of :math:`\boldsymbol{C}` depends on the material's symmetry (e.g., transverse-
isotropic, cubic, hexagonal,...). Elasticipy automates the computation of effective properties, symmetry-adapted
reductions, and analysis for these complex cases.

Voigt Notation
~~~~~~~~~~~~~~

To simplify the representation of the fourth-order elasticity tensor :math:`\boldsymbol{C}`, **Voigt notation** is
commonly used.
This notation reduces :math:`\boldsymbol{C}` to a symmetric :math:`6\times 6` matrix :math:`C_{\alpha\beta}`, leveraging
the symmetries of the stress and strain tensors. The mapping of indices follows the convention:

.. math::

    \begin{aligned}
    (11) & \rightarrow 1, & (22) & \rightarrow 2, & (33) & \rightarrow 3, \\
    (23), (32) & \rightarrow 4, & (13), (31) & \rightarrow 5, & (12), (21) & \rightarrow 6.
    \end{aligned}

This transformation allows the stress-strain relationship to be expressed in a more compact form:

.. math::

    \begin{bmatrix}
    \sigma_{11} \\
    \sigma_{22} \\
    \sigma_{33} \\
    \sigma_{23} \\
    \sigma_{13} \\
    \sigma_{12}
    \end{bmatrix}
    =
    \begin{bmatrix}
    C_{11} & C_{12} & C_{13} & C_{14} & C_{15} & C_{16} \\
    C_{12} & C_{22} & C_{23} & C_{24} & C_{25} & C_{26} \\
    C_{13} & C_{23} & C_{33} & C_{34} & C_{35} & C_{36} \\
    C_{14} & C_{24} & C_{34} & C_{44} & C_{45} & C_{46} \\
    C_{15} & C_{25} & C_{35} & C_{45} & C_{55} & C_{56} \\
    C_{16} & C_{26} & C_{36} & C_{46} & C_{56} & C_{66}
    \end{bmatrix}
    \begin{bmatrix}
    \varepsilon_{11} \\
    \varepsilon_{22} \\
    \varepsilon_{33} \\
    2\varepsilon_{23} \\
    2\varepsilon_{13} \\
    2\varepsilon_{12}
    \end{bmatrix}.

The symmetric :math:`6\times 6` stiffness matrix :math:`C_{\alpha\beta}` is much easier to work with compared to the full
:math:`3\times 3\times 3\times 3` tensor, while retaining all the essential information about a material's elastic
properties. Elasticipy fully supports operations in Voigt notation, making it a practical tool for exploring elastic
properties in isotropic and anisotropic materials.

Features of this package
------------------------

Features of Elasticipy include:

- Computation of elasticity tensors,
- Analysis of elastic anisotropy and wave propagation,
- Multidimensional arrays of strain and stress tensors,
- Rotation of tensors,
- Integration with crystal symmetry groups,
- Visualization and tutorials for ease of use,
- A graphical user interface to plot the spatial dependence of engineering constants.

Elasticipy streamlines the exploration of linear elasticity, making it accessible for applications in materials science,
geophysics, and mechanical engineering.


Installation
------------
To install this package, simply run::

    pip install Elasticipy


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   ./Tutorials.rst
   modules