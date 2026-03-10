"""
==============================================================
Illustrate the normality rule in the principle stresses space
==============================================================
The example shows how to illustrate the normality rule based on the 3D representation the yield surface
"""


###################################################
# Plot the yield surface
#----------------------------------
from elasticipy.yield_criteria import MohrCoulomb
mc = MohrCoulomb(2, -20)
fig=mc.plot_3D(xrange=(-6, 6), yrange=(-6, 6), zrange=(-6, 6))

###################################################
# Plot the surface normals
#----------------------------------
from elasticipy.tensors.stress_strain import StressTensor

tensile_stress_x = StressTensor.tensile([1,0,0], 1)
tensile_stress_y = StressTensor.tensile([0,1,0], 1)
biaxial_tension  = tensile_stress_x + tensile_stress_y
tension_compress = tensile_stress_x - tensile_stress_y

fig = mc.draw_surface_normal(fig, tensile_stress_x, color='black', auto_scale=True, label='Tensile x')
fig = mc.draw_surface_normal(fig, tensile_stress_y, color='gray', auto_scale=True, label='Tensile y')
fig = mc.draw_surface_normal(fig, biaxial_tension, color='blue', auto_scale=True, label='Biaxial tension')
mc.draw_surface_normal(fig, tension_compress, color='green', auto_scale=True, label='Tensile/compression')

########################
# .. Notes::
#
#     The `auto_scale=True` option automatically scales the provided stress so that the yield surface is reached.


