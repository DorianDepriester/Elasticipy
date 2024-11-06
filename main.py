import numpy as np

from StiffnessTensor import tensorFromCrystalSymmetry
from Orientation import EulerAngles
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot

C = tensorFromCrystalSymmetry(symmetry='monoclinic', diad='y', phase_name='TiNi',
                              C11=231, C12=127, C13=104,
                              C22=240, C23=131, C33=175,
                              C44=81, C55=11, C66=85,
                              C15=-18, C25=1, C35=-3, C46=3)
print(C)
E = C.Young_modulus
fig, ax=E.plot(n_theta=50, n_phi=50)

epsilon = np.zeros((10, 3, 3))
epsilon[:, 0, 0] = np.linspace(0, 1e-3, 10)
print(C*epsilon)

sigma = np.zeros((3, 3))
sigma[0, 0] = 1
print(C.inv()*epsilon)
print(E.eval([1, 0, 0]))

#C.shear_modulus.plot(n_theta=100)

print(C.Young_modulus.min())
print(C.Young_modulus)

G = C.shear_modulus
G.plot(which='std')


