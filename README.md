[![PyPI - Version](https://img.shields.io/pypi/v/Elasticipy?link=https%3A%2F%2Fpypi.org%2Fproject%2FElasticipy%2F)](https://pypi.org/project/elasticipy/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/Elasticipy?link=https%3A%2F%2Fpypi.org%2Fproject%2FElasticipy%2F)](https://pypistats.org/packages/elasticipy)
[![Conda Version](https://img.shields.io/conda/v/conda-forge/Elasticipy)](https://anaconda.org/channels/conda-forge/packages/elasticipy/overview)
![Conda Downloads](https://img.shields.io/conda/d/conda-forge/Elasticipy)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/DorianDepriester/Elasticipy)
![GitHub Repo stars](https://img.shields.io/github/stars/DorianDepriester/Elasticipy)
[![PyPI - License](https://img.shields.io/pypi/l/Elasticipy)](https://github.com/DorianDepriester/Elasticipy/blob/main/LICENSE)
[![ReadTheDoc](https://readthedocs.org/projects/elasticipy/badge/?version=latest)](https://elasticipy.readthedocs.io/)
[![codecov](https://codecov.io/gh/DorianDepriester/Elasticipy/graph/badge.svg?token=VUZPEUPBH1)](https://codecov.io/gh/DorianDepriester/Elasticipy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Elasticipy)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07940/status.svg)](https://doi.org/10.21105/joss.07940)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DorianDepriester/Elasticipy/HEAD?urlpath=%2Fdoc%2Ftree%2FElasticipy_for_the_Impatient.ipynb)


# ![Elasticipy](docs/source/logo/logo_text.png)

A python toolkit to manipulate stress and strain tensors, and other linear elasticity-related tensors (e.g. stiffness). 
This package also provides a collection of easy-to-use and very fast tools to work on stress and strain tensors.

## :rocket: Main features
Among other features, this package implements:

- Computation of elasticity tensors,
- Analysis of elastic anisotropy and wave propagation,
- Working with multidimensional arrays of tensors,
- Thermal expansion tensors,
- Rotation of tensors,
- Integration with crystal symmetry groups,
- Visualization and tutorials for ease of use,
- A graphical user interface to plot the spatial dependence of engineering constants,
- Compatibility with the [Materials Project](https://next-gen.materialsproject.org/) API, [pymatgen](https://pymatgen.org/) and 
[orix](https://orix.readthedocs.io/).

## :snake: Installation
Elasticipy can be installed with PIP:
````
pip install Elasticipy
````

On anaconda, one can also use:
````
conda install conda-forge::elasticipy
````

## :books: Documentation
Tutorials and full documentation are available on [ReadTheDoc](https://elasticipy.readthedocs.io/).

## :hourglass_flowing_sand: Elasticipy in a nutshell
Take a 5-minute tour through Elasticipy's main features by running the online Jupyter Notebook, hosted on 
[Binder](https://mybinder.org/v2/gh/DorianDepriester/Elasticipy/HEAD?urlpath=%2Fdoc%2Ftree%2FElasticipy_for_the_Impatient.ipynb).
 

## :mag: Sources
The source code is available on [GitHub](https://github.com/DorianDepriester/Elasticipy) under the [MIT licence](https://github.com/DorianDepriester/Elasticipy/blob/c6c3d441a2d290ab8f4939992d5d753a1ad3bdb0/LICENSE).

## :umbrella: Tests and Code Coverage

The project uses unit tests with `pytest` and coverage reports generated using `coverage`. These reports are hosted on 
[codecov](https://app.codecov.io/gh/DorianDepriester/Elasticipy).

### **Coverage Exclusions**
Certain parts of the code, particularly those related to graphical user interfaces (GUIs) or visual plotting, are 
**excluded from code coverage analysis**. This includes the following file:

- **`src/Elasticipy/gui.py`**

## :mortar_board: Cite this package
If you use Elasticipy, please cite [![DOI](https://joss.theoj.org/papers/10.21105/joss.07940/status.svg)](https://doi.org/10.21105/joss.07940)

You can use the following BibTeX entry:
````bibtex
@article{Elasticipy, 
    doi = {10.21105/joss.07940}, 
    url = {https://doi.org/10.21105/joss.07940}, 
    year = {2025}, 
    publisher = {The Open Journal}, 
    volume = {10}, 
    number = {115}, 
    pages = {7940}, 
    author = {Depriester, Dorian and Kubler, Régis}, 
    title = {Elasticipy: A Python package for linear elasticity and tensor analysis}, 
    journal = {Journal of Open Source Software}
}
````
