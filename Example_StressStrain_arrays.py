import numpy as np

from Elasticipy.FourthOrderTensor import tensorFromCrystalSymmetry
from Elasticipy.StressStrainTensors import StressTensor
from scipy.spatial.transform import Rotation

# We start with a linear evolution of strain
n_slices = 10
sigma = np.zeros((n_slices, 3, 3))
sigma[:, 1, 1] = np.linspace(0, 1e-3, n_slices)
sigma = StressTensor(sigma)     # Convert it to strain
print(sigma)

# Now we consider TiNi material:
C = tensorFromCrystalSymmetry(symmetry='monoclinic', diad='y', phase_name='TiNi',
                              C11=231, C12=127, C13=104,
                              C22=240, C23=131, C33=175,
                              C44=81, C55=11, C66=85,
                              C15=-18, C25=1, C35=-3, C46=3)

# Estimate the compliance tensor
S = C.inv()

# Compute strain
eps = S*sigma

# Check eps for end values
print('Strain (at 0) :')
print(eps[0])
print('Strain (at end) :')
print(eps[-1])

# Compute elastic energy
energy = 0.5*sigma.ddot(eps)
print(energy)

# Now let consider a set of n rotations
n = 100
rotations = Rotation.random(n)
sigma_rotated = sigma.matmul(rotations)
print(sigma_rotated)    # Just to check how it looks like
eps_rotated = S*sigma_rotated
# Check mean strain value at the beginning and the end
print('Strain (at 0) :')
print(eps_rotated.mean(axis=1)[0])
print('Strain (at end) :')
print(eps_rotated.mean(axis=1)[-1])
