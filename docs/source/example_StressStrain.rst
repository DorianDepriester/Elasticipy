Tutorial: working with stress and strain tensors
========================================================

Introduction
------------
This tutorial illustrates how we work on strain and stress tensors, and how Elasticipy handles arrays of tensors.

Single tensors
--------------
Let's start with basic operations with the stress tensor. For instance, we can compute the von Mises and Tresca equivalent stresses:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :linenos:
   :lines: 4-12

So now, let's have a look on the the strain tensor, and compute the principal strains and the volumetric change:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :linenos:
   :lines: 17-21

Linear elasticity
--------------------------------
This section is dedicated to linear elasticity, hence introducing the fourth-order stiffness tensor.
As an example, create a stiffness tensor corresponding to ferrite:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 26-30

Considering the previous strain, evaluate the corresponding stress:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 32-33

Conversely, one can compute the compliance tensor:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 35-36

and check that we retrieve the correct (initial) strain:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 38

.. _multidimensional-arrays:

Multidimensional tensor arrays
------------------------------
Elasticipy allows to process thousands of tensors at one, with the aid of tensor arrays.
For instance, we start by creating an array of 10 stresses:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 43-47

The corresponding strain array is evaluated with the same syntax as before:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 49-51

We can compute the corresponding elastic energies:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 53-54

Apply rotations
---------------
Rotations can be applied on the tensors. If multiple rotations are applied at once, this results in tensor arrays.
Rotations are defined by ``scipy.transform.Rotation``.

For example, let's consider a random set of 1000 rotations:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 59-62

These rotations can be applied on the strain tensor

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 64

The ``matmul()`` just works like the matrix product, thus increasing the dimensionality of the array.
In our case, we get an array of shape (10, 1000).

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 65

Therefore, we can compute the corresponding rotated strain array:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 67-68

And get the stress back to the initial coordinate system:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 70

Finally, we can estimate the mean stresses among all the orientations:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 71-72

Actually, a more straightforward method is to define a set of rotated stiffness tensors, and compute their Reuss average:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 74-75

Which yields the same results in terms of stress:

.. literalinclude:: ../../Examples/Example_StressStrain_arrays.py
   :language: python
   :lines: 76-77