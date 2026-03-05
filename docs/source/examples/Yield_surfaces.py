"""
===========================================================
Plot the Tresca hexagon and the von Mises ellipse
===========================================================

We plot the elastic domain, with respect to the von Mises and Tresca criteria, on one single
Matplotlib axis.

.. note::

    This somehow replicates the figure shown on wikipedia to illustrate the difference between those
    criteria (see `here <https://en.wikipedia.org/wiki/Yield_surface#von_Mises_yield_surface>`_).
"""
from elasticipy.plasticity import VonMisesPlasticity, TrescaPlasticity
sigma_y = 100 # yield stress
mises = VonMisesPlasticity(yield_stress=sigma_y)
tresca = TrescaPlasticity(yield_stress=sigma_y)
fig, ax = mises.plot_2D()
fig, ax = tresca.plot_2D(fig=fig, ax=ax, color='blue')
ax.legend()
fig.show()


