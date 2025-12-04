Manipulation of 4th-order tensors
---------------------------------

This tutorial illustrates how you can take advantage of the ``FourthOrderTensor`` class (and its subclasses
``StiffnessTensor`` and ``ComplianceTensor``) to manipulate 4th-order tensors and perform common algebra on them.


.. note::

    As all common 4th-order tensors met in mechanics display at least minor symmetry
    (:math:`T_{ijkl}=T_{jikl}=T_{ijlk}`), Elasticipy only handles such type of tensors with ``FourthOrderTensor`` (and
    it subclasses).

Mapping conventions
========================

First, it is important to recall that the matrix representation of 4-th order tensors is usually done by two different
means: the Voigt mapping or the Kelvin mapping.

Voigt mapping
~~~~~~~~~~~~~~

In Voigt mapping, the components of a 4th-order tensor :math:`\boldsymbol{C}` are given as follows:

.. math::

    \boldsymbol{C}^{Voigt}=
    \begin{bmatrix}
    C_{1111}    & C_{1122} & C_{1133}   & C_{1123} & C_{1113} & C_{1112} \\
    C_{2211}    & C_{2222} & C_{2233}   & C_{2223} & C_{2213} & C_{2212} \\
    C_{3311}    & C_{3322} & C_{3333}   & C_{3323} & C_{3313} & C_{3312} \\
    C_{2311}    & C_{2322} & C_{2333}   & C_{2323} & C_{2313} & C_{2312} \\
    C_{1311}    & C_{1322} & C_{1333}   & C_{1323} & C_{1313} & C_{1312} \\
    C_{1211}    & C_{1222} & C_{1233}   & C_{1223} & C_{1213} & C_{1212}
    \end{bmatrix}

For instance, the matrix representation of the identity tensor, w.r.t to mapping convention above, is:

.. math::

    \boldsymbol{I}^{Voigt}=
    \begin{bmatrix}
    1 & 0 & 0 & 0       & 0         & 0\\
    0 & 1 & 0 & 0       & 0         & 0\\
    0 & 0 & 1 & 0       & 0         & 0\\
    0 & 0 & 0 & \frac12 & 0         & 0\\
    0 & 0 & 0 & 0       & \frac12   & 0\\
    0 & 0 & 0 & 0       & 0         & \frac12\\
    \end{bmatrix}

.. note::

    The Fourth-order identity tensor is defined as:

    .. math::

        I_{ijkl}=\frac12\left(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}\right)

Let :math:`\boldsymbol{S}` be the inverse tensor of :math:`\boldsymbol{C}`
(i.e :math:`\boldsymbol{S}=\boldsymbol{C}^{-1}`). When using the Voigt convention, the matrix representation of
:math:`\boldsymbol{S}` is:

.. math::

    \boldsymbol{S}^{Voigt^{-1}}=
    \begin{bmatrix}
    S_{1111}    & S_{1122}  & S_{1133}   & 2S_{1123} & 2S_{1113} & 2S_{1112} \\
    S_{2211}    & S_{2222}  & S_{2233}   & 2S_{2223} & 2S_{2213} & 2S_{2212} \\
    S_{3311}    & S_{3322}  & S_{3333}   & 2S_{3323} & 2S_{3313} & 2S_{3312} \\
    2S_{2311}   & 2S_{2322} & 2S_{2333}  & 4S_{2323} & 4S_{2313} & 4S_{2312} \\
    2S_{1311}   & 2S_{1322} & 2S_{1333}  & 4S_{1323} & 4S_{2323} & 4S_{1312} \\
    2S_{1211}   & 2S_{1222} & 2S_{1233}  & 4S_{1223} & 4S_{2312} & 4S_{1212}
    \end{bmatrix}

In this case, the matrix representation of the identity tensor, w.r.t to the mapping convention above, is:

.. math::

    \boldsymbol{I}^{Voigt^{-1}}=
    \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0\\
    0 & 1 & 0 & 0 & 0 & 0\\
    0 & 0 & 1 & 0 & 0 & 0\\
    0 & 0 & 0 & 2 & 0 & 0\\
    0 & 0 & 0 & 0 & 2 & 0\\
    0 & 0 & 0 & 0 & 0 & 2\\
    \end{bmatrix}

The two conventions above allows:

.. math::

    \boldsymbol{C}^{Voigt}=\left(\boldsymbol{S}^{Voigt^{-1}}\right)^{-1}


Kelvin mapping
~~~~~~~~~~~~~~

Conversely, the Kelvin(-Mandel) mapping conventions gives:

.. math::

    \boldsymbol{C}^{Kelvin}=
    \begin{bmatrix}
    C_{1111}            & C_{1122}         & C_{1133}           & \sqrt{2}C_{1123} & \sqrt{2}C_{1113} & \sqrt{2}C_{1112} \\
    C_{2211}            & C_{2222}         & C_{2233}           & \sqrt{2}C_{2223} & \sqrt{2}C_{2213} & \sqrt{2}C_{2212} \\
    C_{3311}            & C_{3322}         & C_{3333}           & \sqrt{2}C_{3323} & \sqrt{2}C_{3313} & \sqrt{2}C_{3312} \\
    \sqrt{2}C_{2311}    & \sqrt{2}C_{2322} & \sqrt{2}C_{2333}   & 2C_{2323} & 2C_{2313} & 2C_{2312} \\
    \sqrt{2}C_{1311}    & \sqrt{2}C_{1322} & \sqrt{2}C_{1333}   & 2C_{1323} & 2C_{1313} & 2C_{1312} \\
    \sqrt{2}C_{1211}    & \sqrt{2}C_{1222} & \sqrt{2}C_{1233}   & 2C_{1223} & 2C_{1213} & 2C_{1212}
    \end{bmatrix}

and

.. math::

    \boldsymbol{S}^{Kelvin}=
    \begin{bmatrix}
    S_{1111}            & S_{1122}         & S_{1133}           & \sqrt{2}S_{1123} & \sqrt{2}S_{1113} & \sqrt{2}S_{1112} \\
    S_{2211}            & S_{2222}         & S_{2233}           & \sqrt{2}S_{2223} & \sqrt{2}S_{2213} & \sqrt{2}S_{2212} \\
    S_{3311}            & S_{3322}         & S_{3333}           & \sqrt{2}S_{3323} & \sqrt{2}S_{3313} & \sqrt{2}S_{3312} \\
    \sqrt{2}S_{2311}    & \sqrt{2}S_{2322} & \sqrt{2}S_{2333}   & 2S_{2323} & 2S_{2313} & 2S_{2312} \\
    \sqrt{2}S_{1311}    & \sqrt{2}S_{1322} & \sqrt{2}S_{1333}   & 2S_{1323} & 2S_{1313} & 2S_{1312} \\
    \sqrt{2}S_{1211}    & \sqrt{2}S_{1222} & \sqrt{2}S_{1233}   & 2S_{1223} & 2S_{1213} & 2S_{1212}
    \end{bmatrix}

which obviously allows:

.. math::

    \boldsymbol{C}^{Kelvin}=\left(\boldsymbol{S}^{Kelvin}\right)^{-1}

and

.. math::

    \boldsymbol{I}^{Kelvin}=
    \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0\\
    0 & 1 & 0 & 0 & 0 & 0\\
    0 & 0 & 1 & 0 & 0 & 0\\
    0 & 0 & 0 & 1 & 0 & 0\\
    0 & 0 & 0 & 0 & 1 & 0\\
    0 & 0 & 0 & 0 & 0 & 1\\
    \end{bmatrix}

In Elasticpy
~~~~~~~~~~~~

The Kelvin mapping preserves tensor norms and inner products between second-order tensors. This makes it particularly suitable
for tensor algebra, projections and eigendecompositions. Therefore, in Elasticipy, the Kelvin mapping is used for all
underlying operations, although is transparent from the end-user point of view. As a consequence, the two aforementioned conventions
can be used (and mixed) independently, as they are only used a representation mean.

For ``FourthOrderTensor`` and ``SymmetricFourthOrderTensor``, the default mapping convention is Kelvin's [#f1]_:

.. doctest::

    >>> from Elasticipy.tensors.fourth_order import FourthOrderTensor
    >>> I_kelvin = FourthOrderTensor.eye()
    >>> print(I_kelvin)
    4th-order tensor (in Kelvin mapping):
    [[1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 1.]]

Still, the Voigt mapping can be used instead:

    >>> from Elasticipy.tensors.mapping import VoigtMapping
    >>> I_voigt = FourthOrderTensor.eye(mapping=VoigtMapping())
    >>> print(I_voigt)
    4th-order tensor (in Voigt mapping):
    [[1.  0.  0.  0.  0.  0. ]
     [0.  1.  0.  0.  0.  0. ]
     [0.  0.  1.  0.  0.  0. ]
     [0.  0.  0.  0.5 0.  0. ]
     [0.  0.  0.  0.  0.5 0. ]
     [0.  0.  0.  0.  0.  0.5]]

Actually, both ``I_kelvin`` and ``I_voigt`` defined above are *exactly the same tensors* (identity tensor here). The only difference is their
representation. Indeed, one can check that their components as full representations (as (3,3,3,3) arrays) are equal:

    >>> import numpy as np
    >>> np.array_equal(I_kelvin.full_tensor, I_voigt.full_tensor)
    True

Actually, a more direct way to check this equality is:

    >>> I_kelvin == I_voigt
    True

As said above, the two tensors above can `mixed`, although they are not represented the same way. E.g.:

    >>> I_kelvin + I_voigt
    4th-order tensor (in Kelvin mapping):
    [[2. 0. 0. 0. 0. 0.]
     [0. 2. 0. 0. 0. 0.]
     [0. 0. 2. 0. 0. 0.]
     [0. 0. 0. 2. 0. 0.]
     [0. 0. 0. 0. 2. 0.]
     [0. 0. 0. 0. 0. 2.]]


Tensor algebra
==============

    This section illustrates how basic algebra can be done on 4th-order tensors with Elasticpy.

Tensor products
~~~~~~~~~~~~~~~~

Between two 4th-order tensors
++++++++++++++++++++++++++++++

Let consider two random tensors:

    >>> T1 = FourthOrderTensor.rand()
    >>> T2 = FourthOrderTensor.rand()

The tensor product, contracted twice, between ``T1`` and ``T2`` is just:

    >>> prod = T1.ddot(T2)
    >>> print(prod) # doctest: +SKIP
    4th-order tensor (in Kelvin mapping):
    [[3.38803756 2.42773891 3.19611497 4.19371704 4.86279192 5.84923528]
     [2.0285819  1.75427124 1.82935727 3.06920803 3.72681217 3.47202109]
     [2.31206735 1.64688191 2.08154427 3.55532315 3.40044051 4.51753924]
     [3.40513449 2.85863369 2.84788701 5.65277162 5.49084006 5.78541462]
     [4.51742123 3.13127224 3.84471626 6.16850941 6.67736519 6.80033276]
     [4.50532375 2.79409697 4.54388471 6.28615133 7.08068387 8.52605556]]

.. note::

    The tensor product contracted twice between two 4th-order tensors :math:`\boldsymbol{A}` and :math:`\boldsymbol{B}`
    is defined as:

        .. math::

            \left[\boldsymbol{A}:\boldsymbol{B}\right]_{ijkl} = A_{ijmn}B_{mnkl}

Note that this also works for *arrays* of tensors. E.g.:

    >>> T3 = FourthOrderTensor.rand(shape=(3,4))
    >>> prod = T1.ddot(T3)

The magic here is that we have applied the ":" operator between ``T1`` and every tensor in the tensor array ``T3``.
Therefore, the shape of the result is a tensor array as well:

    >>> prod.shape
    (3, 4)

This syntax also works when working with two tensor arrays, as long as they can be broadcast. E.g.:

    >>> T4 = FourthOrderTensor.rand(shape=(4,))
    >>> prod = T3.ddot(T4)
    >>> prod.shape
    (3, 4)

If one wants to evaluate *every* cross combinations of tensor products:

    >>> T3.ddot(T4, mode='cross')
    4th-order tensor array of shape (3, 4, 4)

Between 4th and a 2nd -order tensors
++++++++++++++++++++++++++++++++++++++

The same applies when working with 2nd-order tensors (provided by the ``SecondOrderTensor`` class):

    >>> from Elasticipy.tensors.second_order import SecondOrderTensor
    >>> t = SecondOrderTensor.rand(shape=(4,))
    >>> T1.ddot(t)
    Second-order tensor
    Shape=(4,)
    >>> T3.ddot(t)
    Second-order tensor
    Shape=(3, 4)

Rotations
~~~~~~~~~~
Elasticipy supports scipy's ``Rotation`` class to "rotate" tensors.
As an example, we consider the "one" 4th-order tensor (i.e. tensor populated with 1s):


    >>> from scipy.spatial.transform import Rotation
    >>> ones = FourthOrderTensor.ones()
    >>> print(ones)
    4th-order tensor (in Kelvin mapping):
    [[1.         1.         1.         1.41421356 1.41421356 1.41421356]
     [1.         1.         1.         1.41421356 1.41421356 1.41421356]
     [1.         1.         1.         1.41421356 1.41421356 1.41421356]
     [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
     [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
     [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]]

Now we define a single rotation (say of 90 degrees around the X axis):

    >>> rot = Rotation.from_euler('X', 90, degrees=True)

Then, we can apply the rotation to the ``ones`` tensor as follows:

    >>> rotated_tensor = ones.rotate(rot)

Let's have a look on the rotated tensor:

    >>> rotated_tensor
    4th-order tensor (in Kelvin mapping):
    [[ 1.          1.          1.         -1.41421356  1.41421356 -1.41421356]
     [ 1.          1.          1.         -1.41421356  1.41421356 -1.41421356]
     [ 1.          1.          1.         -1.41421356  1.41421356 -1.41421356]
     [-1.41421356 -1.41421356 -1.41421356  2.         -2.          2.        ]
     [ 1.41421356  1.41421356  1.41421356 -2.          2.         -2.        ]
     [-1.41421356 -1.41421356 -1.41421356  2.         -2.          2.        ]]

Alternatively, one can also use the following syntax:

    >>> rotated_tensor = ones * rot

This also works with series of rotations. For instance:

    >>> rand_rots = Rotation.random(1000) # 1000 random rotations over SO(3)
    >>> rotated_tensor = ones * rand_rots
    >>> print(rotated_tensor)
    4th-order tensor array of shape (1000,)

A tensor array can be subscripted to access each tensor value:

    >>> rotated_tensor[0]   # doctest: +SKIP
    4th-order tensor (in Kelvin mapping):
    [[ 0.79057598  0.48313729  1.39371692  1.16047974 -1.48447911 -0.87402144]
     [ 0.48313729  0.29525517  0.85172916  0.70919312 -0.90719581 -0.53413254]
     [ 1.39371692  0.85172916  2.4570021   2.04582519 -2.61700799 -1.54082403]
     [ 1.16047974  0.70919312  2.04582519  1.70345833 -2.17905424 -1.28296862]
     [-1.48447911 -0.90719581 -2.61700799 -2.17905424  2.78743383  1.64116619]
     [-0.87402144 -0.53413254 -1.54082403 -1.28296862  1.64116619  0.96627459]]

or it can be rotated once again. For instance, to get back to original tensors:

    >>> rot_inv = rand_rots.inv()
    >>> rotated_back_tensor = rotated_tensor * rot_inv
    >>> rotated_back_tensor
    4th-order tensor array of shape (1000,)
    >>> rotated_back_tensor[0]
    4th-order tensor (in Kelvin mapping):
    [[1.         1.         1.         1.41421356 1.41421356 1.41421356]
     [1.         1.         1.         1.41421356 1.41421356 1.41421356]
     [1.         1.         1.         1.41421356 1.41421356 1.41421356]
     [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
     [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]
     [1.41421356 1.41421356 1.41421356 2.         2.         2.        ]]


Arithmetic mean
~~~~~~~~~~~~~~~~

Working with tensor array allows statistical operations, such as computation of arithmetic means, E.g.:

    >>> rotated_tensor.mean() # doctest: +SKIP
    4th-order tensor (in Kelvin mapping):
    [[ 1.90633472  0.60401454  0.60381129 -0.0087537  -0.00545366  0.01853883]
     [ 0.60401454  1.6874382   0.57078235  0.0350303   0.02462755  0.03668225]
     [ 0.60381129  0.57078235  1.84901073  0.02031739 -0.00815819 -0.00252591]
     [-0.0087537   0.0350303   0.02031739  1.1415647  -0.00357218  0.03482861]
     [-0.00545366  0.02462755 -0.00815819 -0.00357218  1.20762258 -0.0123796 ]
     [ 0.01853883  0.03668225 -0.00252591  0.03482861 -0.0123796   1.20802908]]


.. rubric:: Footnotes

.. [#f1] For ``StiffnessTensor`` and ``ComplianceTensor`` classes, the default mapping conventions are Voigt's.