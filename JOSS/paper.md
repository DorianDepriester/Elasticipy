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
    affiliation: '1'
  - name: Régis Kubler
    orcid: 0000-0001-7781-5855
    affiliation: '1'
affiliations:
 - index: 1
   name: Arts et Métiers Institute of Technology, MSMP, Aix-en-Provence, F-13617, France
date: 2025-01-15
bibliography: paper.bib

# Summary

Elasticipy is a Python package designed to simplify the analysis of material elasticity properties. It enables the 
computation, visualization, and analysis of elasticity tensors while incorporating crystal symmetries. By targeting a 
diverse audience, Elasticipy is suitable for researchers, engineers, and educators looking to explore elastic behavior 
in materials. It is easy to install via PyPI and offers robust documentation, practical tutorials, and a user-friendly 
interface. 

# Statement of Need

Crystal elasticity analysis is crucial in fields such as materials science, physics, and engineering. Elasticity 
tensors, which govern the stress-strain relationships in materials, are complex to compute and analyze, especially when 
accounting for crystal symmetries. Existing software solutions often lack accessibility or do not fully support complex 
symmetry operations, making them challenging for non-specialist users or those seeking rapid prototyping and analysis.

Elasticipy addresses this gap by providing:

  - Intuitive Python-based APIs for defining and manipulating elasticity tensors.

  - Support for standard crystal symmetry groups, ensuring physically accurate tensor operations.

  - Visualization tools for understanding directional elastic behavior.

Unlike other software such as pymatgen [@pymatgen] or [@elate], Elasticipy emphasizes ease of use, flexibility, and 
integration with existing Python workflows. Its modular design and comprehensive documentation make it accessible for 
both experts and non-specialists. In addition, it introduces the concept of *tensor arrays*, allowing to process thousands of tensors
at once (e.g. rotation of tensors) in a user-friendly and highly efficient way.

# References