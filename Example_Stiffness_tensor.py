from FourthOrderTensor import tensorFromCrystalSymmetry
from scipy.spatial.transform import Rotation
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot

# First, let consider the NiTi material:
C = tensorFromCrystalSymmetry(symmetry='monoclinic', diad='y', phase_name='TiNi',
                              C11=231, C12=127, C13=104,
                              C22=240, C23=131, C33=175,
                              C44=81, C55=11, C66=85,
                              C15=-18, C25=1, C35=-3, C46=3)
print(C)

# Let's have a look on its Young modulus
E = C.Young_modulus
# See min/max values
print(E)
# Now illustrate the spatial dependence
E.plot_xyz_sections()   # As 2D sections...
E.plot()                # ...or in 3D
print(E.max())

# Apply a random rotation on stiffness tensor
rotation = Rotation.from_euler('zxz', [0, 45, 0], degrees=True)
Crot = C*rotation
# Check that the Young modulus has changed as well
Crot.Young_modulus.plot()

# Now let's consider the shear modulus
G = C.shear_modulus
G.plot_xyz_sections()   # Plot sections with min, max and mean
G.plot(which='min')     # And plot it in 3D
print(G.min())
print(G.max())

# Finally, let's have a look on the Poisson ratio
nu = C.Poisson_ratio
nu.plot_xyz_sections()
nu.plot(which='max')
print(nu.min())
print(nu.max())

# Now let consider a finite set of orientations
oris = Rotation.random(1000)
Cvoigt = C.Voigt_average(oris)  # Compute the Voigt average
print(Cvoigt.Young_modulus) # Look at the corresponding Young modulis
print(C.Voigt_average().Young_modulus) # Compare with infinite number of orientations


