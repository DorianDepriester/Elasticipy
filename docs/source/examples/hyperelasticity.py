"""
=====================================
Hyperelastic behaviour (Neo-Hookean)
=====================================

This example shows how to model hyperelastic behaviour.
"""
###############################################################################
# Create the hyper-elastic model
# -------------------------------------
# We consider the Neo-Hookean model
from elasticipy.hyperelasticity import NeoHooke
nh = NeoHooke(37.9) # Incompressible case

###############################################################################
# Create a deformation gradient tensor corresponding to tensile test
# -------------------------------------
# We consider that the material is incompressible. Hence, the corresponding gradient tensor array can be defined as
# follows:
from elasticipy.tensors.finite_strain import DeformationGradient
import numpy as np
stretch = np.linspace(1,2)
F = DeformationGradient.isochoric_tensile([1,0,0], stretch)

###############################################################################
# One can check that all values of this tensor array correspond to isochoric case:

print(F.volumetric_strain())

###############################################################################
# Compute the corresponding stress
# -------------------------------------
# As the material is incompressible, the hydrostatic pressure cannot be estimated from the deformation gradient (it also
# depends on the applied load). Therefore, the command below returns the deviatoric part of the stress:

sigma_dev = nh.stress_from_F(F)

###############################################################################
# Actually, the tensile stress is supposed to be zero on transverse directions (load-free surface of the sample). Thus,
# the hydrostatic pressure can be inferred from the tensile deviatoric stresses along transverse directions:
from elasticipy.tensors.stress_strain import StressTensor
p = sigma_dev.C[1,1]    # sigma_22
sigma = sigma_dev + StressTensor.pressure(p)

###############################################################################
# One can check that the longitudinal transverse stress is always zeros. E.g.:
print(sigma[-1])

###############################################################################
# Plot tensile stress vs true stress
# -------------------------------------
# First, compute the longitudinal elongation from `F`

elong = F.elongation([1,0,0])   # Extract relative elongation along x

###############################################################################
# As we here talk about large strain, it's better to use true (logarithmic) strain:
true_strain = np.log(1 + elong)

from matplotlib import pyplot as plt
plt.plot(true_strain, sigma.C[0,0])
plt.xlabel('True strain')
plt.ylabel('Stress (MPa)')
plt.show()