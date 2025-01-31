from Elasticipy.Plasticity import JohnsonCook
from Elasticipy.StressStrainTensors import StressTensor, StrainTensor
from Elasticipy.FourthOrderTensor import StiffnessTensor
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot


JC = JohnsonCook(A=363, B=792.7122, n=0.5756) # https://doi.org/10.1016/j.matpr.2020.05.213
C = StiffnessTensor.isotropic(E=210000, nu=0.27)

n_step = 100
sigma_max = 725
stress_mag = np.linspace(0, sigma_max, n_step)
stress = StressTensor.tensile([1,0,0], stress_mag)

elastic_strain = C.inv() * stress
plastic_strain = StrainTensor.zeros(n_step)
for i in range(2, n_step):
    strain_increment = JC.compute_strain_increment(stress[i])
    plastic_strain[i] = plastic_strain[i-1] + strain_increment


eps_xx = elastic_strain.C[0,0]+plastic_strain.C[0,0]
fig, ax = plt.subplots()
ax.plot(eps_xx, stress_mag, label='Stress-controlled')


##
from scipy.optimize import minimize_scalar
stress = StressTensor.zeros(n_step)
plastic_strain = StrainTensor.zeros(n_step)
JC.reset_strain()
for i in range(2, n_step):
    def fun(tensile_stress):
        trial_stress = StressTensor.tensile([1,0,0], tensile_stress)
        trial_elastic_strain = C.inv() * trial_stress
        trial_strain_increment = JC.compute_strain_increment(trial_stress, apply_strain=False)
        trial_plastic_strain = plastic_strain[i - 1] + trial_strain_increment
        trial_elongation =  trial_plastic_strain.C[0,0] +  trial_elastic_strain.C[0,0]
        return (trial_elongation - eps_xx[i])**2
    s = minimize_scalar(fun)
    s0 = s.x
    stress.C[0,0][i] = s0
    strain_increment = JC.compute_strain_increment(stress[i])
    plastic_strain[i] = plastic_strain[i-1] + strain_increment

ax.plot(eps_xx, stress.C[0,0], label='Strain-controlled', linestyle='dotted')
ax.legend()
ax.set_xlabel(r'$\varepsilon_{xx}$')
ax.set_ylabel('Tensile stress (MPa)')