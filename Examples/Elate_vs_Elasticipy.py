import numpy as np
from elastic import Elastic
from Elasticipy.FourthOrderTensor import StiffnessTensor
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot
from Elasticipy.SphericalFunction import sph2cart, _plot3D
import time
import matplotlib.pyplot as plt


C = StiffnessTensor.monoclinic(phase_name='TiNi',
                               C11=231, C12=127, C13=104,
                               C22=240, C23=131, C33=175,
                               C44=81, C55=11, C66=85,
                               C15=-18, C25=1, C35=-3, C46=3)
Celate = Elastic(list(C.matrix))

# Plotting with Elate
start_time=time.perf_counter()
n_phi, n_theta = 500,500
phi = np.linspace(0, 2*np.pi, n_phi)
theta = np.linspace(0, np.pi, n_theta)

u= np.zeros((n_phi, n_theta, 3))
Eelate = np.zeros((n_phi, n_theta))
for i in range(n_phi):
    for j in range(n_theta):
        Eelate[i,j] = Celate.Young([theta[j], phi[i]])
        u[i,j,:] = sph2cart(phi[i], theta[j])
fig = plt.figure()
ax = _plot3D(fig, u.reshape(n_phi, n_theta, 3), Eelate)
t = time.perf_counter() - start_time
ax.set_title("Elate (t={:.3f} s)".format(t))

# Plotting with Elasticpy
start_time=time.perf_counter()
fig, ax = C.Young_modulus.plot3D(n_phi=n_phi, n_theta=n_theta)
t = time.perf_counter() - start_time
#ax.set_title("Elasticipy (t={:.3f} s)".format(t))
fig.savefig('../JOSS/Plot_E.png', dpi=300)
