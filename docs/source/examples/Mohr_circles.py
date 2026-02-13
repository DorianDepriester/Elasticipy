"""
===========================================================
Draw Mohr circles of stress tensor
===========================================================

This example shows how to automatically draw the Mohr circles from a given stress tensor.
"""
###############################################################################
# Create a random stress tensor
# -------------------------------------
from elasticipy.tensors.stress_strain import StressTensor
s = StressTensor.rand(seed=123) # Use seed to ensure reproducibility
print(s)

###############################################################################
# Draw the corresponding Mohr circles
# --------------------
fig, ax = s.draw_Mohr_circles()


