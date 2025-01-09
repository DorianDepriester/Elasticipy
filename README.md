# Elasticipy 
[![PyPI - Version](https://img.shields.io/pypi/v/Elasticipy?link=https%3A%2F%2Fpypi.org%2Fproject%2FElasticipy%2F)](https://pypi.org/project/elasticipy/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/Elasticipy?link=https%3A%2F%2Fpypi.org%2Fproject%2FElasticipy%2F)](https://pypistats.org/packages/elasticipy)
[![PyPI - License](https://img.shields.io/pypi/l/Elasticipy)](https://github.com/DorianDepriester/Elasticipy/blob/main/LICENSE)
[![ReadTheDoc](https://readthedocs.org/projects/elasticipy/badge/?version=latest)](https://elasticipy.readthedocs.io/)
[![DOI](https://zenodo.org/badge/876162900.svg)](https://doi.org/10.5281/zenodo.14501849)
[![codecov](https://codecov.io/gh/DorianDepriester/Elasticipy/graph/badge.svg?token=VUZPEUPBH1)](https://codecov.io/gh/DorianDepriester/Elasticipy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Elasticipy)



A python toolkit to manipulate strain and strain tensors, and other linear elasticity-related tensors (e.g. stiffness). 
This package also provides a collection of easy-to-use and very fast tools to work on stress and strain tensors.

## Main features
Among other features, this package implements:

- Computation of elasticity tensors,
- Analysis of elastic anisotropy and wave propagation,
- Working with multidimensional arrays of strain and stress tensors,
- Thermal expansion tensors,
- Rotation of tensors,
- Integration with crystal symmetry groups,
- Visualization and tutorials for ease of use,
- A graphical user interface to plot the spatial dependence of engineering constants,
- Compatibility with the [Materials Project](https://next-gen.materialsproject.org/) API.

## Installation
Elasticipy can be installed with PIP:
````
pip install Elasticipy
````

## Documentation
Tutorials and full documentation are available on [ReadTheDoc](https://elasticipy.readthedocs.io/).

## Sources
The source code is available on [GitHub](https://github.com/DorianDepriester/Elasticipy) under the [MIT licence](https://github.com/DorianDepriester/Elasticipy/blob/c6c3d441a2d290ab8f4939992d5d753a1ad3bdb0/LICENSE).

## Tests and Code Coverage

The project uses unit tests with `pytest` and coverage reports generated using `coverage`. These reports are hosted on 
[codecov](https://app.codecov.io/gh/DorianDepriester/Elasticipy).

### **Coverage Exclusions**
Certain parts of the code, particularly those related to graphical user interfaces (GUIs) or visual plotting, are 
**excluded from code coverage analysis**. This includes the following file:

- **`src/Elasticipy/gui.py`**