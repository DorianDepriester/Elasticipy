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
C = 37.9 # MPa
nh = NeoHooke(C) # Incompressible case

###############################################################################
# Create a deformation gradient tensor corresponding to tensile test
# -------------------------------------
# We consider that the material is incompressible. Hence, the corresponding gradient tensor array can be defined as
# follows:
from elasticipy.tensors.finite_strain import DeformationGradient
import numpy as np
stretch = np.linspace(0,1) # 0 to 100% elongation
F = DeformationGradient.isochoric_tensile([1,0,0], stretch)

###############################################################################
# One can check that all values of this tensor array correspond to isochoric case:
print(F.volumetric_strain())

###############################################################################
# Compute the corresponding stress
# -------------------------------------
# As the material is incompressible, the hydrostatic pressure cannot be estimated from the deformation gradient (it also
# depends on the applied load). Therefore, the command below returns the deviatoric part of the stress only:
sigma_dev = nh.stress_from_F(F)

###############################################################################
# Actually, on tensile tests, the tensile stress is supposed to be zero on transverse directions (load-free surface of
# the sample). Thus, the hydrostatic pressure can be inferred from the tensile deviatoric stresses along transverse
# directions:
from elasticipy.tensors.stress_strain import StressTensor
p = sigma_dev.C[1,1]    # sigma_dev_22
sigma = sigma_dev + StressTensor.pressure(p) # sigma = sigma_dev - pI

###############################################################################
# One can check that the longitudinal transverse stress is always zero. E.g.:
print(sigma[-1])

###############################################################################
# Plot tensile stress vs true strain
# -------------------------------------
# As we here talk about large strain, it's better to use true (logarithmic) strain:
true_strain = np.log(1 + stretch)

from matplotlib import pyplot as plt
plt.plot(true_strain, sigma.C[0,0])
plt.xlabel('True strain')
plt.ylabel('Stress (MPa)')
plt.title('Tensile curve of incompressible Neo-Hookean material')
plt.show()