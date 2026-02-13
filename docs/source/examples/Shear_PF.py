"""
===========================================================
Plot shear modulus as a pole figure
===========================================================

This example shows how to plot the directional dependence of the shear modulus as a pole figure.
"""
###############################################################################
# Define the stiffness tensor for NiTi
# -------------------------------------
#
# We define the stiffness tensor for a monoclinic NiTi material using its elastic constants. They are taken from the
# `Materials Project (mp-1048) <https://github.com/DorianDepriester/Elasticipy>`_.
from elasticipy.tensors.elasticity import StiffnessTensor
C = StiffnessTensor.monoclinic(
    phase_name='TiNi',
    C11=231, C12=127, C13=104,
    C22=240, C23=131, C33=175,
    C44=81, C55=11, C66=85,
    C15=-18, C25=1, C35=-3, C46=3
)
print("Stiffness tensor for NiTi:\n", C)

###############################################################################
# Get the shear modulus from the stiffness tensor
# ------------------------------------------------------
G = C.shear_modulus
print("Shear modulus:")
print(G)

###############################################################################
# Plot it as a pole figure in Lamber (equal area) projection
# ---------------------------------------------------------------
fig, ax = G.plot_as_pole_figure()


