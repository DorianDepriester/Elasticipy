"""
=============================================================
Plot in 2D the Mohr-Coulomb and Drucker-Prager yield surface
=============================================================

The example illustrates in 2D the differences between the Mohr-Coulomb and the Drucker-Prager yield criteria.

"""
from elasticipy.yield_criteria import DruckerPrager, MohrCoulomb

sigma_y = 100 # yield stress
mc = MohrCoulomb(c=2, phi=-20)

#####################################################################################
# When expressed in terms of cohesion (c) and friction angle (phi), there are three ways to 'fit' the Drucker-Prager
# (DP) yield criterion with that of Mohr-Coulomb
# (see `here <https://en.wikipedia.org/wiki/Drucker%E2%80%93Prager_yield_criterion#Expressions_in_terms_of_cohesion_and_friction_angle>`_:

pg1 = DruckerPrager(c=2, phi=-20, fit='inside')
pg2 = DruckerPrager(c=2, phi=-20, fit='middle')
pg3 = DruckerPrager(c=2, phi=-20, fit='outside')

fig, ax = mc.plot_2D()
fig, ax = pg1.plot_2D(fig=fig, ax=ax, color='blue', alpha=0., label='DP (inside)')
fig, ax = pg2.plot_2D(fig=fig, ax=ax, color='green', alpha=0., label='DP (middle)')
fig, ax = pg3.plot_2D(fig=fig, ax=ax, color='pink', alpha=0., label='DP (outside)')
ax.legend()
fig.show()


