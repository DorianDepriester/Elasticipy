"""
===================================
Average stiffness of fibre texture
===================================

This example shows how to compute the Voigt-Reuss-Hill average of crystals with perfect fibre texture.
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
# Create a gamma fibre texture
# -------------------------------------------------
# The gamma texture correspond to alignment of  <1 1 1> with normal direction.
from elasticipy.crystal_texture import FibreTexture
gamma = FibreTexture.gamma()
print(gamma)

######################################
# Plot the corresponding pole figure
# ------------------------------------
# The pole figure of uvw directions, defined with the help of
# `orix <https://orix.readthedocs.io/en/stable/index.html>`_, as pole figures. Eg.:
from orix.crystal_map import Phase
from orix.vector.miller import Miller
phase = Phase(point_group='m-3m')   # BCC
miller = Miller(uvw=[1,0,0], phase=phase)
fig, ax = gamma.plot_as_pole_figure(miller.symmetrise(unique=True))

###############################################################################
# Compute the Hill average
# --------------------------
# The Voigt-Reuss-Hill average of the polycrystal is:
Chill = C.Voigt_average(orientations=gamma)
print(Chill)

###############################################################################
# Check transversely isotropy
# ---------------------------
# The transversely isotropy, resulting from the fibre texture along ND, can be evidenced by looking at the Young
# modulus, which is constant along any direction in XY plane:
Chill.Young_modulus.plot3D()

