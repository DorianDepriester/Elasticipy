Installation
============

Elasticipy is compatible with Python ≥ 3.9 and can be installed either via
`pip` or `conda` (through the *conda-forge* channel).

Installation with pip
---------------------

The recommended way to install Elasticipy is from PyPI:

.. code-block:: bash

   pip install elasticipy

This command installs the latest stable release along with its required
dependencies.

Editable installation (development mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development purposes or to modify the source code locally, Elasticipy can
be installed in **editable mode**.

First, clone the GitHub repository:

.. code-block:: bash

   git clone https://github.com/DorianDepriester/Elasticipy.git
   cd Elasticipy

Then install the package in editable mode:

.. code-block:: bash

   pip install -e .

Any change made to the source code will be immediately reflected without
reinstalling the package.

Installation with conda
-----------------------

Elasticipy is also available on **conda-forge**.

Using conda (or mamba), install Elasticipy with:

.. code-block:: bash

   conda install conda-forge::elasticipy

or, for faster dependency resolution (recommended):

.. code-block:: bash

   mamba install -c conda-forge elasticipy

Using a dedicated conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to create a dedicated conda environment before installation:

.. code-block:: bash

   conda create -n elasticipy python=3.12
   conda activate elasticipy
   conda install -c conda-forge elasticipy

