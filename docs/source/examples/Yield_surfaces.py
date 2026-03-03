"""
===========================================================
Plot the Tresca hexagon and the von Mises ellipse
===========================================================
This example shows how to plot the elastic domain, with respect to the von Mises and Tresca criteria, on one single
Matplotlib axis.
"""
from elasticipy.plasticity import VonMisesPlasticity, TrescaPlasticity
sigma_y = 100 # yield stress
fig, ax = VonMisesPlasticity().plot_2D(yield_stress=sigma_y)
fig, ax = TrescaPlasticity().plot_2D(yield_stress=sigma_y, fig=fig, ax=ax, color='blue')
ax.legend()
fig.show()





