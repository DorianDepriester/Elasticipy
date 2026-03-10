"""
=============================================================
Plot the
=============================================================

The example illustrates in 2D the differences between the Mohr-Coulomb and the Drucker-Prager yield criteria.

"""


###################################################
# Plot the yield surface
#----------------------------------
from elasticipy.yield_criteria import MohrCoulomb
mc = MohrCoulomb(c, -20)
fig=mc.plot_3D(xrange=(-6, 6), yrange=(-6, 6), zrange=(-6, 6))

###################################################
# Plot the surface normals
#----------------------------------
from elasticipy.tensors.stress_strain import StressTensor

tensile_stress_x = StressTensor.tensile([1,0,0], 1)
tensile_stress_y = StressTensor.tensile([0,1,0], 1)
biaxial_tension  = tensile_stress_x + tensile_stress_y
tension_compress = tensile_stress_x - tensile_stress_y

fig = mc.plot_surface_normal(fig, tensile_stress_x, color='black', auto_scale=True, label='Tensile x')
fig = mc.plot_surface_normal(fig, tensile_stress_y, color='gray', auto_scale=True, label='Tensile y')
fig = mc.plot_surface_normal(fig, biaxial_tension, color='blue', auto_scale=True, label='Biaxial tension')
fig = mc.plot_surface_normal(fig, tension_compress, color='green', auto_scale=True, label='Tensile/compression')

########################
# .. Notes::
#
# The `auto_scale=True` option automatically scales the provided stress so that the yield surface is reached.


