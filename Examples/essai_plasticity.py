from Elasticipy.plasticity import JohnsonCook
from Elasticipy.tensors.stress_strain import StressTensor, StrainTensor
from Elasticipy.tensors.elasticity import StiffnessTensor
import numpy as np
from matplotlib import pyplot as plt


JC = JohnsonCook(A=363, B=792.7122, n=0.5756) # https://doi.org/10.1016/j.matpr.2020.05.213
C = StiffnessTensor.isotropic(E=210000, nu=0.27)

n_step = 100
sigma_max = 725
stress_mag = np.linspace(0, sigma_max, n_step)
stress = StressTensor.shear([1,0,0], [0,1,0],stress_mag)

elastic_strain = C.inv() * stress
plastic_strain = StrainTensor.zeros(n_step)
for i in range(2, n_step):
    strain_increment = JC.compute_strain_increment(stress[i])
    plastic_strain[i] = plastic_strain[i-1] + strain_increment


eps_xx = elastic_strain.C[0,1]+plastic_strain.C[0,1]
fig, ax = plt.subplots()
ax.plot(eps_xx, stress_mag, label='Stress-controlled')
ax.set_xlabel(r'$\varepsilon_{xy}$')
ax.set_ylabel('Shear stress (MPa)')

##
from scipy.optimize import minimize_scalar
stress = StressTensor.zeros(n_step)
plastic_strain = StrainTensor.zeros(n_step)
JC.reset_strain()
for i in range(2, n_step):
    def fun(tensile_stress):
        trial_stress = StressTensor.shear([1,0,0], [0,1,0], tensile_stress)
        trial_elastic_strain = C.inv() * trial_stress
        trial_strain_increment = JC.compute_strain_increment(trial_stress, apply_strain=False)
        trial_plastic_strain = plastic_strain[i - 1] + trial_strain_increment
        trial_elongation =  trial_plastic_strain.C[0,1] +  trial_elastic_strain.C[0,1]
        return (trial_elongation - eps_xx[i])**2
    s = minimize_scalar(fun)
    s0 = s.x
    stress.C[0,1][i] = s0
    strain_increment = JC.compute_strain_increment(stress[i])
    plastic_strain[i] = plastic_strain[i-1] + strain_increment

ax.plot(eps_xx, stress.C[0,1], label='Strain-controlled', linestyle='dotted')
ax.legend()


#
JC.reset_strain()
JC_tresca = JohnsonCook(A=363, B=792.7122, n=0.5756, criterion='Tresca')
stress_mag = np.linspace(0, 500, n_step)
stress = StressTensor.shear([1,0,0], [0,1,0],stress_mag)
models = (JC, JC_tresca)
labels = ('von Mises', 'Tresca')

elastic_strain = C.inv() * stress
fig, ax = plt.subplots()
for j, model in enumerate(models):
    plastic_strain = StrainTensor.zeros(n_step)
    for i in range(2, n_step):
        strain_increment = model.compute_strain_increment(stress[i])
        plastic_strain[i] = plastic_strain[i-1] + strain_increment
    eps_xy = elastic_strain.C[0,1]+plastic_strain.C[0,1]
    ax.plot(eps_xy, stress_mag, label=labels[j])
ax.set_xlabel(r'$\varepsilon_{xy}$')
ax.set_ylabel('Shear stress (MPa)')
ax.legend()
fig.show()