"""
=================================================================
Plot the elastic domain in the deviatoric plane (pi-plane)
=================================================================

We plot the elastic domain, with respect to the von Mises and Tresca criteria, in the deviatoric
plane, that is the plane where the three principal stresses sum up to zero. The projections of the
three principal stress axes are shown as three axes 120° apart from each other.

.. note::

    The Tresca hexagon is inscribed in the von Mises circle: both criteria coincide for uniaxial
    tension and compression, but Tresca is more conservative for multiaxial stress states.
"""
from elasticipy.yield_criteria import VonMisesCriterion, TrescaCriterion

sigma_y = 100  # yield stress
mises = VonMisesCriterion(yield_stress=sigma_y)
tresca = TrescaCriterion(yield_stress=sigma_y)
fig, ax = mises.plot_deviatoric()
fig, ax = tresca.plot_deviatoric(fig=fig, ax=ax, color='blue')
ax.legend()
fig.show()
