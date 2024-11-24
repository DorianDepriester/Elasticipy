# ======================================================
# Imports and simple example of stress
# ======================================================
import numpy as np
from Elasticipy.StressStrainTensors import StressTensor, StrainTensor


stress = StressTensor([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
print(stress.vonMises())
print(stress.Tresca())

# ======================================================
# Simple example of strain
# ======================================================
strain = StrainTensor([[0, 1e-3, 0],
                       [1e-3, 0, 0],
                       [0, 0, 0]])
print(strain.principalStrains())
print(strain.volumetricStrain())

# ======================================================
# Linear elasticity
# ======================================================
from Elasticipy.FourthOrderTensor import StiffnessTensor

C = StiffnessTensor.fromCrystalSymmetry(symmetry='cubic', phase_name='ferrite',
                                        C11=274, C12=175, C44=89)
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
sigma = StressTensor.zeros(n_array)  # Initialize the array to zero-stresses
sigma.C[0, 1] = sigma.C[1, 0] = np.linspace(0, 100, n_array)    # The shear stress is linearly increasing
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
from scipy.spatial.transform import Rotation

n_ori = 1000
rotations = Rotation.random(n_ori)

eps_rotated = eps.matmul(rotations)
print(eps_rotated.shape)    # Just to check how it looks like

sigma_rotated = C * eps_rotated
print(sigma_rotated.shape)    # Check out the shape of the stresses

sigma = sigma_rotated * rotations.inv()         # Go back to initial frame
sigma_mean = sigma.mean(axis=1)     # Compute the mean over all orientations
print(sigma_mean[-1])

C_rotated = C * rotations
C_Voigt = C_rotated.Voigt_average()
sigma_Voigt = C_Voigt * eps
print(sigma_Voigt[-1])

