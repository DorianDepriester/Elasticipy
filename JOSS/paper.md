---
title: 'Elasticipy: A Python package for elasticity and tensor analysis'
tags:
  - Python
  - Continuum Mechanics
  - Linear elasticity
  - Thermal expansion
  - Anisotropy
  - Crystals
  - Polycrystals
authors:
  - name: Dorian Depriester
    orcid: 0000-0002-2881-8942
    equal-contrib: false
    affiliation: '1'
  - name: Régis Kubler
    orcid: 0000-0001-7781-5855
    affiliation: '1'
affiliations:
 - index: 1
   name: Arts et Métiers Institute of Technology, MSMP, Aix-en-Provence, F-13617, France
   ror: 04yzbzc51
date: 15 January 2025
bibliography: paper.bib
---

# Summary

Elasticipy is a Python library designed to streamline the computation of elasticity tensors for materials and 
crystalline materials, taking their specific symmetries into account. It provides tools to manipulate, visualize, and 
analyze these tensors, simplifying workflows for materials scientists an engineers.

# Statement of Need

Strain analysis is crucial in fields such as materials science and engineering. Elasticity 
tensors, which govern the stress-strain relationships in materials, are complex to compute and analyze, especially when 
accounting for crystal or material symmetries. Existing software solutions often lack accessibility or do not fully 
support complex symmetry operations, making them challenging for non-specialist users or those seeking rapid prototyping
and analysis.

Elasticipy addresses this gap by providing:

  - Intuitive Python-based APIs for defining and manipulating second- and fourth-order tensors, such as strain, stress
and stiffness;

  - Support for standard crystal symmetry groups [@nye] to facilitate the definition of stiffness/compliance components; 

  - Visualization tools for understanding directional elastic behavior.

Unlike other software such as pymatgen [@pymatgen] or Elate [@elate], Elasticipy emphasizes ease of use, flexibility, 
and integration with existing Python workflows. In addition, it introduces the concept of *tensor arrays*, in a similar 
way as in MTEX [@MTEX], allowing to process thousands of tensors at once (e.g. rotation of tensors) with simple and 
highly efficient commands. In order to highlight the performances of Elasticipy, \autoref{fig:compa} shows the wall-time 
required to perform two basic operations on tensors, as functions of the number of considered tensors. This evidences 
that, when processing large number of tensors ($>10^3$), basic operations on tensors are 1 to 2 orders of magnitude 
faster when using Elasticipy than pymatgen.

![Performance comparison between Elasticipy and pymatgen.\label{fig:compa}](ElasticipyVSpymatgen.png){ width=75% }

Nevertheless, as tensor algebra is not the core of pymatgen, Elasticipy supports conversion to pymatgen, and vice versa. 
It also allows direct imports of elastic data from 
[the Materials Project](https://next-gen.materialsproject.org/) [@MaterialsProject].

# References