Welcome to Elasticipy's Documentation!
======================================

.. image:: https://img.shields.io/pypi/v/Elasticpy?link=https%3A%2F%2Fpypi.org%2Fproject%2FElasticipy%2F
   :alt: PyPI - Version

.. image:: https://img.shields.io/pypi/dm/Elasticpy
   :alt: PyPI - Downloads

.. image:: https://img.shields.io/pypi/l/Elasticipy
   :alt: PyPI - License

.. image:: https://readthedocs.org/projects/elasticipy/badge/?version=latest
   :alt: ReadTheDoc


Purpose of this package
-----------------------
This Python's package is dedicated to work on mechanical elasticity-related tensors; namely: stress, strain and
stiffness/compliance tensors.

It provides a couple of ways to perform basic operations on these tensors in a user-friendly way. In addition, it
handles arrays of tensors, allowing to perform thousands of data at once (see an example :ref:`here<multidimensional-arrays>`).

It also comes with plotting features (e.g. spatial dependence of Young modulus).

Installation
------------
To install this package, simply run::

    pip install Elasticipy


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   ./example_StressStrain.rst
   ./example_StiffnessTensor.rst
   ./Tutorial_AveragingMethods.rst
   modules