import numpy as np
from Elasticipy.tensors.stress_strain import StressTensor, StrainTensor
from Elasticipy.tensors.elasticity import StiffnessTensor
from scipy.spatial.transform import Rotation

# ======================================================
# Simple example of stress
# ======================================================
stress = StressTensor.shear([1,0,0],[0,1,0],1)
print(stress.vonMises())
print(stress.Tresca())

# ======================================================
# Simple example of strain
# ======================================================
strain = StrainTensor.shear([1,0,0],[0,1,0],1e-3)
print(strain.principal_strains())
print(strain.volumetric_strain())

# ======================================================
# Linear elasticity
# ======================================================
C = StiffnessTensor.cubic(phase_name='ferrite',C11=274, C12=175, C44=89)
print(C)

sigma = C * strain
print(sigma)

S = C.inv()
print(S)
print(S * sigma)

# ======================================================
# Multidimensional tensor arrays
# ======================================================
n_array = 10
sigma_xy = np.linspace(0, 100, n_array)
sigma = StressTensor.shear([1,0,0], [0,1,0],sigma_xy)
print(sigma[0])     # Check the initial value of the stress...
print(sigma[-1])    # ...and the final value.

eps = S * sigma
print(eps[0])     # Now check the initial value of strain...
print(eps[-1])    # ...and the final value.

energy = 0.5*sigma.ddot(eps)
print(energy)     # print the elastic energy

# ======================================================
# Apply random rotations
# ======================================================
n_ori = 1000
rotations = Rotation.random(n_ori)

eps_rotated = eps.rotate(rotations, mode='cross')
print(eps_rotated.shape)    # Just to check how it looks like

sigma_rotated = C * eps_rotated
print(sigma_rotated.shape)    # Check out the shape of the stresses

sigma = sigma_rotated * rotations.inv()         # Go back to initial frame
sigma_mean = sigma.mean(axis=1)     # Compute the mean over all orientations
print(sigma_mean[-1])

C_rotated = C * rotations
C_Voigt = C_rotated.Voigt_average() # Now check that the previous result is consistent with Voigt average
sigma_Voigt = C_Voigt * eps
print(sigma_Voigt[-1])

fig, ax = sigma_Voigt[-1].draw_Mohr_circles()
fig.show()

