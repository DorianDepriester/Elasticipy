Save, read and download stiffness values
----------------------------------------
In this tutorial, we will see how we can write and read stiffness data to/from text files, and take advantage of the
Materials Project to automatically fetch data from its online database.

Read & write files
==================

Write stiffness to file
~~~~~~~~~~~~~~~~~~~~~~~
To save a given stiffness tensor to a plain text file, just use the ``save_to_txt()`` command. For instance:

.. doctest::

    >>> from Elasticipy.FourthOrderTensor import StiffnessTensor
    >>> C = StiffnessTensor.cubic(C11=186, C12=134, C44=77, phase_name='Cu')
    >>> C.save_to_txt('Stiffness_Cu.txt')

By default, the file will contain the stiffness matrix, and additional data as well (symmetry and phase name). To save
the matrix only, just use the option ``matrix_only=True``.

Load stiffness from file
~~~~~~~~~~~~~~~~~~~~~~~~
To read entries from an existing text file, just run ``StiffnessTensor.from_txt_file(<file_path>)``.
Using the example above:

    >>> StiffnessTensor.from_txt_file('Stiffness_Cu.txt')
    Stiffness tensor (in Voigt notation) for Cu:
    [[186. 134. 134.   0.   0.   0.]
     [134. 186. 134.   0.   0.   0.]
     [134. 134. 186.   0.   0.   0.]
     [  0.   0.   0.  77.   0.   0.]
     [  0.   0.   0.   0.  77.   0.]
     [  0.   0.   0.   0.   0.  77.]]
    Symmetry: cubic


Download data from the Materials Project
========================================

The `Materials Project (MP) <https://materialsproject.org/>`_ contains various data about plenty materials, and stiffness
values for some of them. They can be directly reached in Python thanks to the MP API, which is already implemented in
Elasticipy.

Requirements
~~~~~~~~~~~~
To be able to fetch data from MP, you need to register first, then get an API key.
This may be accessible on `your dashboard <https://materialsproject.org/dashboard>`_, and may look like this:

    ``Ag5Kljg5iGcZ45mJ7bC7T9q4L56KgBcZ``

In Python, you will need to install ``mp_api``, for instance with PIP:

    ``pip install mp_api``

Fetching data from materials names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
On MP, each material is given a unique name, like *mp-xxx* (where *xxx* is a number). Given this name, one can import the stiffness tensor (if
available). For instance, to fetch data for `mp-1048 <https://next-gen.materialsproject.org/materials/mp-1048>`_
(monoclinic TiNi):

.. doctest::

    >>> api_key = '<Your API key>'
    >>> StiffnessTensor.from_MP("mp-1048", api_key=api_key) # doctest: +SKIP
    Stiffness tensor (in Voigt notation) for TiNi:
    [[231. 127. 104.  -0. -18.   0.]
     [127. 240. 131.  -0.   1.   0.]
     [104. 131. 175.  -0.  -3.  -0.]
     [ -0.  -0.  -0.  81.  -0.   3.]
     [-18.   1.  -3.  -0.  11.   0.]
     [  0.   0.  -0.   3.   0.  85.]]
    Symmetry: Monoclinic

.. note::

    In order to avoid passing the ``api_key`` as an argument, you can also define an environment variable, named
    ``MP_API_KEY`` and containing the proper string value.


Instead of a single material name, a list of material IDs can be passed to ``from_MP()``; in this case, the
function will return a list of stiffness tensors. If a material is not found (or if no data about the elastic properties
are available), the corresponding stiffness tensor will be ``None``.


