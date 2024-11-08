from FourthOrderTensor import tensorFromCrystalSymmetry
from scipy.spatial.transform import Rotation
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot

C = tensorFromCrystalSymmetry(symmetry='monoclinic', diad='y', phase_name='TiNi',
                              C11=231, C12=127, C13=104,
                              C22=240, C23=131, C33=175,
                              C44=81, C55=11, C66=85,
                              C15=-18, C25=1, C35=-3, C46=3)
print(C)

# Let consider the Young's modulus first
E = C.Young_modulus
# See min/max values
print(E)
# Now illustrate the spatial dependence
fig, ax = E.plot()

# Apply a random rotation on stiffness tensor
rotation = Rotation.from_euler('zxz', [0, 45, 0], degrees=True)
Crot = C*rotation
Crot.Young_modulus.plot()

# Show spatial dependence of shear modulus
G = C.shear_modulus
G.plot(which='min')

# Show spatial dependence of Poisson ratio
nu = C.Poisson_ratio
nu.plot(which='max')




