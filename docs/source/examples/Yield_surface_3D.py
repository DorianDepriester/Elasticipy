"""
============================================================
3D plot the Drucker-Prager and Mohr-Coulomb yield surface
============================================================

We plot the Drucker-Prager and Mohr-coulomb yield surfaces in the 3D principal stresses space.

.. note::

    This example is inspired by the figure shown on wikipedia to illustrate the DP yield criterion (see
    `here <https://en.wikipedia.org/wiki/Drucker%E2%80%93Prager_yield_criterion>`_).
"""
from elasticipy.yield_criteria import DruckerPrager, MohrCoulomb

pg = DruckerPrager.from_cohesion_friction_angle(2, -20)
fig=pg.plot_3D(xrange=(-6, 6), yrange=(-6, 6), zrange=(-6, 6))

mc = MohrCoulomb(2, -20)
mc.plot_3D(fig=fig, color='blue')