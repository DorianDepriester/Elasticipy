from Elasticipy.Plasticity import JohnsonCook, normality_rule
from Elasticipy.StressStrainTensors import StressTensor, StrainTensor
from Elasticipy.FourthOrderTensor import StiffnessTensor
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot

JC = JohnsonCook(A=363, B=792.7122, n=0.5756, m=1.645, T0=298, Tm=1798, eps_dot_ref=0.6)
C = StiffnessTensor.isotropic(E=210000, nu=0.27)

n = 100
sigma_max = 750
load_path = [np.linspace(0, 400, n),
             np.linspace(400, 0, n),
             np.linspace(0, 450, n),
             np.linspace(450, 0, n),
             np.linspace(0, 500, n),
             np.linspace(500, 0, n),
             np.linspace(0, 550, n)]
stress_mag = np.concatenate(load_path)
stress = StressTensor.tensile([1,0,0], stress_mag)
n_step = len(stress_mag)

elastic_strain = C.inv() * stress
plastic_strain = StrainTensor.zeros(n_step)
eq_stress = stress.vonMises()
eq_strain_bak = 0.0
for i in range(2, n_step):
    strain_increment = JC.compute_strain_increment(stress[i])
    normal = normality_rule(stress[i])
    delta_plastic_strain = strain_increment * normal
    plastic_strain[i] = plastic_strain[i-1] + delta_plastic_strain


eps_xx = elastic_strain.C[0,0]+plastic_strain.C[0,0]
fig, ax = plt.subplots()
ax.plot(eps_xx, stress_mag)
ax.plot(eps_xx, JC.flow_stress(eps_xx))