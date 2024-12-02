from Elasticipy.FourthOrderTensor import StiffnessTensor
from scipy.spatial.transform import Rotation
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot

# First, let consider the NiTi material:
C = StiffnessTensor.fromCrystalSymmetry(symmetry='monoclinic', diad='y', phase_name='TiNi',
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
E.plot3D()                # ...or in 3D
E.plot_as_pole_figure()     # or even with PF
print(E.max())

# Apply a random rotation on stiffness tensor
rotation = Rotation.from_euler('zxz', [0, 45, 0], degrees=True)
Crot = C*rotation
# Check that the Young modulus has changed as well
Crot.Young_modulus.plot3D()

# Now let's consider the shear modulus
G = C.shear_modulus
G.plot_xyz_sections()   # Plot sections with min, max and mean
G.plot3D()     # And plot it in 3D
print(G.min())
print(G.max())

# Finally, let's have a look on the Poisson ratio
nu = C.Poisson_ratio
nu.plot_xyz_sections()
nu.plot3D(which='max')
print(nu.min())
print(nu.max())

# Now let consider a finite set of orientations
oris = Rotation.random(1000)
Crotated = C*oris  # Compute the Voigt average
Cvoigt = Crotated.Voigt_average()
print(Cvoigt.Young_modulus) # Look at the corresponding Young modulis
print(C.Voigt_average().Young_modulus) # Compare with infinite number of orientations


