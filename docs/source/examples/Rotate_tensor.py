"""
===========================================================
Apply a rotation to stiffness tensors
===========================================================

This example shows how to apply a rotation to a stiffness tensor.
"""
###############################################################################
# Define the stiffness tensor for copper
# ---------------------------------------
#
# We define the stiffness tensor for a cubic copper using its elastic constants. They are taken from the
# `Materials Project (mp-30) <https://next-gen.materialsproject.org/materials/mp-30?formula=Cu>`_.
from elasticipy.tensors.elasticity import StiffnessTensor
C = StiffnessTensor.cubic(C11=186, C12=134, C44=77)
print(C)

###############################################################################
# Plot the the Young modulus of the crystal
# ------------------------------------------------------------
# The illustrates the cubic symmetries of the elastic moduli, one can look at
# the Young modulus:
E = C.Young_modulus
E.plot3D()

###############################################################################
# Rotate the tensor
# --------------------
# Assume a rotation of 30° around the X axis:
from scipy.spatial.transform import Rotation
r = Rotation.from_euler('X', 30, degrees=True)

###############################################################################
# Now apply rotation, and check out the "shape" of such tensor:
Crot = C * r
print(Crot)

###############################################################################
# Finally, plot the corresponding Young modulus:
Crot.Young_modulus.plot3D()


