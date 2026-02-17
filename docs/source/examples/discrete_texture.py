"""
=====================================
Average stiffness of discrete texture
=====================================

This example shows how to compute the Voigt average of crystals with a mixture of discrete textures.
"""
###############################################################################
# Define fibre texture
# -------------------------------------
#
# We define the stiffness tensor for BCC austenite using its elastic constants. They are taken from the
# `Materials Project (mp-13) <https://next-gen.materialsproject.org/materials/mp-13>`_.
from elasticipy.tensors.elasticity import StiffnessTensor
C = StiffnessTensor.cubic(C11=274, C12=175, C44=89)

###############################################################################
# Create two discrete textures
# -------------------------------------------------
# The Brass texture corresponds to {110}<112>, whereas the copper texture corresponds to {112}<111>:
from elasticipy.crystal_texture import DiscreteTexture
brass = DiscreteTexture.brass()
copper = DiscreteTexture.copper()
print(brass)
print(copper)

######################################
# Create a composite texture
# ------------------------------------
# We assume that the overall texture consists in 60% brass and 40% copper:
texture = 0.6 * brass + 0.4 * copper
print(texture)

###############################################################################
# Plot the pole figures
# --------------------------
fig, ax = texture.plot_as_pole_figure([1,0,0], symmetrise=True)

###############################################################################
# Compute the Voigt average
# --------------------------
# The Voigt average of the polycrystal is:
Cvoigt = C.Voigt_average(orientations=texture)
print(Cvoigt)

###############################################################################
# Plot the Young modulus
# -------------------------------
Cvoigt.Young_modulus.plot3D()


