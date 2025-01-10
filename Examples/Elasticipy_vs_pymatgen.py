from docutils.utils.math.latex2mathml import layout_styles

from Elasticipy.StressStrainTensors import StrainTensor
from Elasticipy.FourthOrderTensor import StiffnessTensor
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot
import numpy as np
import time

C = StiffnessTensor.transverse_isotropic(Ex=200, Ez=300, nu_yx=0.2, nu_zx=0.3, Gxz=80)
## Elasticipy's stuff
exp = np.arange(0, 8)
n = 10**exp
t_stress_elast=[]
t_stress_mg = []
t_vm_elast=[]
t_vm_mg = []

for ni in n:
    strain = StrainTensor.rand(shape=(ni,))
    start_time = time.perf_counter()
    stress = C*strain
    t_stress_elast.append(time.perf_counter() - start_time)
    start_time = time.perf_counter()
    vm_elast = stress.vonMises()
    t_vm_elast.append(time.perf_counter() - start_time)

    ## Pymatgen's stuff
    Cmg = C.to_pymatgen()
    strain_mg = strain.to_pymatgen()
    start_time = time.perf_counter()
    stress = [Cmg.calculate_stress(strain_mg[i]) for i in range(ni)]
    t_stress_mg.append(time.perf_counter() - start_time)
    start_time = time.perf_counter()
    vm_mg = [stress[i].von_mises for i in range(ni)]
    t_vm_mg.append(time.perf_counter() - start_time)


fig, ax = plt.subplots()
ax.plot(n, t_stress_mg, label='Stiffness-strain product (pymatgen)', marker="s")
ax.plot(n, t_stress_elast, label='Stiffness-strain product (Elasticipy)', marker="o")
ax.plot(n, t_vm_mg, label='von Mises eq. stress (pymatgen)', linestyle='dotted', marker="s")
ax.plot(n, t_vm_elast, label='von Mises eq. stress  (Elasticipy)', linestyle='dotted', marker="o")
plt.legend()
plt.xscale('log')
plt.yscale('log')
ax.set_xlabel('Number of tensors')
ax.set_ylabel('CPU time (s)')
ax.set_xlim((1, max(n)))
fig.tight_layout()
plt.show()
fig.savefig('../JOSS/ElasticipyVSpymatgen.png', dpi=300)