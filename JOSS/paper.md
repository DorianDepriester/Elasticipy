---
title: 'Elasticipy: A Python package for linear elasticity and tensor analysis'
tags:
  - Python
  - Continuum Mechanics
  - Linear elasticity
  - Thermal expansion
  - Anisotropy
  - Crystals
  - Polycrystals
  - Materials science
authors:
  - name: Dorian Depriester
    orcid: 0000-0002-2881-8942
    equal-contrib: false
    affiliation: '1'
    corresponding: true
  - name: Régis Kubler
    orcid: 0000-0001-7781-5855
    affiliation: '1'
affiliations:
 - index: 1
   name: Arts et Métiers Institute of Technology, MSMP, Aix-en-Provence, F-13617, France
   ror: 04yzbzc51
date: 20 January 2025
bibliography: paper.bib
---

# Summary

Elasticipy is a Python library designed to streamline computation and manipulation of elasticity tensors for materials and 
crystalline materials, taking their specific symmetries into account. It provides tools to manipulate, visualize, and 
analyze tensors --such as stress, strain and stiffness tensors-- simplifying workflows for materials scientists and 
engineers.

# Statement of Need

In continuum mechanics, the deformation of a material is described by the second-order strain tensor (usually denoted 
$\boldsymbol{\varepsilon}$) whereas the stress is described by the second-order Cauchy's stress tensor 
($\boldsymbol{\sigma}$). Under the linear elasticity assumption, the relationship between the elastic strain $\boldsymbol{\varepsilon}$
and $\boldsymbol{\sigma}$, known as the generalized Hooke's law, is given through the fourth-order stiffness tensor $\boldsymbol{C}$ with:

$$\boldsymbol{\sigma}=\boldsymbol{C}:\boldsymbol{\varepsilon}$$

where "$:$" denotes the tensor product contrated twice, so that:

$$\sigma_{ij}=C_{ijk\ell}\varepsilon_{k\ell}$$

In order to simplify the above equations, one usually uses the so-called Voigt notation, 
which reads:
$$\begin{bmatrix}
\sigma_{11}\\
\sigma_{22}\\
\sigma_{33}\\
\sigma_{23}\\
\sigma_{13}\\
\sigma_{12}
\end{bmatrix}
=
\begin{bmatrix}
C_{1111}    & C_{1122}      & C_{1133}  & C_{1123} & C_{1113}  & C_{1112}\\
            & C_{2222}      & C_{2233}  & C_{2223} & C_{2213}  & C_{2212}\\
            &               & C_{3333}  & C_{3323} & C_{3313}  & C_{3312}\\
            &               &           & C_{2323} & C_{2313}  & C_{2312}\\
            & \mathrm{sym.} &           &          & C_{1313}  & C_{1312}\\
            &           &               &          &           & C_{1212}\\
\end{bmatrix}
\begin{bmatrix}
\varepsilon_{11}\\
\varepsilon_{22}\\
\varepsilon_{33}\\
2\varepsilon_{23}\\
2\varepsilon_{13}\\
2\varepsilon_{12}
\end{bmatrix}
$$

The values of $\boldsymbol{C}$ depend on the material, whereas its pattern (set of zero-components, or linear 
relationships between them) depends on the material's symmetry [@nye], as outlined in \autoref{fig:Nye}. 

Pymatgen [@pymatgen] provides some built-in functions to work on strain, stress and elasticity but lacks some 
functionalities about the tensor analysis. Conversely, Elate [@elate] is a project dedicated to analysis of stiffness 
and compliance tensors (e.g. plotting directional engineering constants, such as Young modulus). It is implemented in 
[the Materials Project](https://next-gen.materialsproject.org/) [@MaterialsProject]. AnisoVis [@AnisoVis] is similar to 
Elate, but works on MATLAB\textsuperscript{\textregistered}.

![Patterns of stiffness and compliance tensors of crystals, depending on their symmetry [@nye]. 
With courtesy of Pr. Pamela Burnley.\label{fig:Nye}](Nye.png)


Therefore, the purpose of Elasticipy is to combine the functionalities of Pymatgen and Elate into a consistent 
Python project dedicated to continuum mechanics. Its aim is to propose an easy-to-use and efficient tool with the following features:

  - intuitive Python-based APIs for defining and manipulating second- and fourth-order tensors, such as strain, stress
and stiffness;

  - support for standard crystal symmetry groups [@nye] to facilitate the definition of stiffness/compliance components; 

  - visualization tools for understanding directional elastic behavior (Young modulus, shear modulus and Poisson ratio);

  - a collection of built-in methods to easily and efficiently perform fundamental operations on tensors (rotations 
[@meanElastic], products, invariants, statistical analysis etc.);

  - averaging techniques, such as Voigt--Reuss--Hill [@hill], for textured and non-textured polycrystalline 
multiphased aggregates.

In order to evidence some of these features, \autoref{fig:Young}.a) illustrates the directional Young modulus of 
copper (Cu) single crystal as a 3D surface, whereas \autoref{fig:Young}.b) shows the same values as a pole figure (Lambert 
projection). In \autoref{fig:Young}.c), the Young modulus of a polycrystalline Cu displaying a perfect $[001]$ fiber 
texture has been estimated with different averaging methods (namely Voigt, Reuss and Hill [@hill]), then plotted as 
orthogonal sections.

![Young modulus (GPa) of Cu single crystal as a 3D surface (a) or a pole figure (b); 
Young modulus of Cu polycrystal with $[001]$ fiber texture, plotted in three orthogonal sections, depending on the
averaging method. \label{fig:Young}](YoungModulus.png)

Elasticipy also introduces the concept of *tensor arrays*, in a similar way as in MTEX [@MTEX], allowing to 
process several tensors at once with simple and highly efficient commands. In order to highlight the performances 
of Elasticipy, \autoref{fig:pymatgen} shows the wall-time required to perform two basic operations on tensors (namely, 
apply the generalized Hooke's law and compute the von Mises equivalent stress) as 
functions of the number of considered tensors. This demonstrates that, when processing large datasets of tensors 
($n>10^3$), basic tensor operations are 1 to 2 orders of magnitude faster in Elasticipy compared to Pymatgen. 
These performance gains are achieved by leveraging NumPy's array broadcasting capabilities [@NumPy].
However, as tensor algebra is not the primary focus of Pymatgen, Elasticipy is designed to complement rather than 
replace it. Elasticipy supports seamless conversion between its own data structures and those of Pymatgen, allowing 
users to integrate both tools and benefit from Pymatgen's extensive features beyond tensor analysis. Elasticipy is also
compatible with Orix [@orix], a Python library for analysing orientations and crystal symmetry.

![Performance comparison between Elasticipy and pymatgen.\label{fig:pymatgen}](ElasticipyVSpymatgen.png){ width=75% }


# Possible extensions

It is worth mentioning that Elasticipy provides a full framework for working on tensors, allowing to extend the analyses
to other averaging methods (e.g. self-consistent models), possibly beyond linear elasticity problems (e.g. plasticity) 
with ease. It already implements thermal expansion.

# Usage

## Plot directional engineering constants

\autoref{fig:Young}.a) and b) were rendered with the following syntax:

````python
from Elasticipy.FourthOrderTensor import StiffnessTensor
C = StiffnessTensor.cubic(C11=186, C12=134, C44=77)
E = C.Young_modulus
E.plot3D(n_phi=500, n_theta=500)
E.plot_as_pole_figure()
````

## Create an array of rotated stiffness tensors and compute average

When considering a finite set of orientations, an array of stiffness tensors can be built to account for the rotations:

````python
from scipy.spatial.transform import Rotation
import numpy as np
n = 10000
phi1 = np.random.random(n)*2*np.pi  # Random sampling from 0 to 2pi
Euler_angles = np.array([phi1,  np.zeros(n),  np.zeros(n)]).T # Fibre texture
rotations = Rotation.from_euler('ZXZ', Euler_angles) # Bunge-Euler angles
C_rotated = C * rotations # n-lenght tensor array
````

Then, the Voigt--Reuss--Hill average [@hill] can be computed from the tensor array:

````python
C_VRH = C_rotated.Hill_average()
````

Finally, the corresponding Young moduli can be plotted in orthogonal sections, as shown in \autoref{fig:Young}.c), with:

````python
C_VRH.Young_modulus.plot_xyz_sections()
````

## Arrays of stress/strain tensor

Efforts have been made to provide out-of-the-box simple syntaxes for common operations. For example, the following
will create a tensor array corresponding to evenly-spaced strain along $[1,0,0]$ axis:

````python
from Elasticipy.StressStrainTensors import StrainTensor
m = 1000                     # length of tensor array
mag = np.linspace(0, 0.1, m) # Strain magnitude
strain = StrainTensor.tensile([1,0,0], mag)
````

Given the stiffness tensor ``C`` (see above), one can compute the corresponding stress array with:
````python
stress = C * strain
````
Finally, ``stress.von_Mises()`` returns an array of length ``n`` and data type ``float64``, providing all the 
corresponding von Mises equivalent stresses.

# References