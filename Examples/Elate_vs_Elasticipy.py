import numpy as np
from elastic import Elastic
from Elasticipy.FourthOrderTensor import StiffnessTensor
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot
from Elasticipy.SphericalFunction import sph2cart
import time
import matplotlib.pyplot as plt


C = StiffnessTensor.monoclinic(phase_name='TiNi',
                               C11=231, C12=127, C13=104,
                               C22=240, C23=131, C33=175,
                               C44=81, C55=11, C66=85,
                               C15=-18, C25=1, C35=-3, C46=3)
Celate = Elastic(list(C.matrix))
t_Young_elate=[]
t_Young_elast=[]
t_shear_elate=[]
t_shear_elast=[]
n = [1,2,10,20,100,200,1000,2000,10000,20000]

for ni in n:
    n_angle = int(np.sqrt(ni))
    phi = np.linspace(0, 2 * np.pi, n_angle)
    theta = np.linspace(0, np.pi, n_angle)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    phi = phi_grid.flatten()
    theta = theta_grid.flatten()
    u = sph2cart(phi, theta)

    start_time = time.perf_counter()
    E=C.Young_modulus.eval(u)
    t_Young_elast.append(time.perf_counter() - start_time)

    start_time = time.perf_counter()
    Eelate = np.zeros(ni)
    k=0
    for i in range(n_angle):
        for j in range(n_angle):
            Eelate[k] = Celate.Young([theta[k], phi[k]])
            k=k+1
    t_Young_elate.append(time.perf_counter() - start_time)

    # Shear modulus
    n_angle = int(ni**(1/3))
    phi = np.linspace(0, 2 * np.pi, n_angle)
    theta = np.linspace(0, np.pi, n_angle)
    psi = np.linspace(0, np.pi, n_angle)
    phi_grid, theta_grid, psi_grid = np.meshgrid(phi, theta, psi)
    phi = phi_grid.flatten()
    theta = theta_grid.flatten()
    psi = psi_grid.flatten()
    u, v = sph2cart(phi, theta, psi)

    start_time = time.perf_counter()
    G = C.shear_modulus.eval(u, v)
    t_shear_elast.append(time.perf_counter() - start_time)

    start_time = time.perf_counter()
    Gelate = np.zeros(ni)
    p=0
    for i in range(n_angle):
        for j in range(n_angle):
            for k in range(n_angle):
                Gelate[p] = Celate.shear([theta[k], phi[k], psi[k]])
                p=p+1
    t_shear_elate.append(time.perf_counter() - start_time)

fig, ax = plt.subplots()
ax.plot(n, t_Young_elate, label='Young modulus (Elate)', marker="s")
ax.plot(n, t_Young_elast, label='Young modulus (Elasticipy)', marker="o")
ax.plot(n, t_shear_elate, label='Shear modulus (Elate)', linestyle='dotted', marker="s")
ax.plot(n, t_shear_elast, label='Shear modulus (Elasticipy)', linestyle='dotted', marker="o")
plt.legend()
plt.xscale('log')
plt.yscale('log')
ax.set_xlabel('Number of directions')
ax.set_ylabel('CPU time (s)')
ax.set_xlim((1, max(n)))
fig.tight_layout()
plt.show()
fig.savefig('../JOSS/ElasticipyVSelate.png', dpi=300)