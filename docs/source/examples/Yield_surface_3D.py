"""
===========================================================
Plot in 3D the Drucker-Prager yield surface
===========================================================

We plot the Drucker-Prager yield surface in the 3D principal stresse space

.. note::

    This replicates the figure shown on wikipedia to illustrate the DP yield criterion  (see
    `here <https://en.wikipedia.org/wiki/Drucker%E2%80%93Prager_yield_criterion>`_).
"""
from elasticipy.plasticity import DruckerPrager
mises = DruckerPrager(c=2, phi=-20)
mises.plot_3D(xmin=-6, xmax=6, ymin=-6, ymax=6, zmin=-6, zmax=6)


