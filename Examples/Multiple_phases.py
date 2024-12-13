from Elasticipy.FourthOrderTensor import StiffnessTensor
import numpy as np
from matplotlib import pyplot as plt


C_ferrite = StiffnessTensor.cubic(C11=242, C12=146, C44=116)
C_austenite = StiffnessTensor.cubic(C11=204, C12=137, C44=126)

n_values = 100
fraction_austenite = np.linspace(0, 1, n_values)

fig, ax = plt.subplots()
colors = plt.cm.tab10.colors    # Simple hack to ensure that both E and G will have the same color for a given method
color_index = 0
for method in ('Voigt', 'Hill', 'Reuss'):
    C_aust_avg = C_austenite.average(method)    # Equivalent to C_austenite.{method}_average()
    C_ferr_avg = C_ferrite.average(method)
    E_mean = np.zeros(n_values)
    G_mean = np.zeros(n_values)
    for i in range(n_values):
        f1 = fraction_austenite[i]
        f2 = 1 - f1
        C_tot_mean = StiffnessTensor.weighted_average((C_aust_avg, C_ferr_avg), (f1, f2), method)
        E_mean[i] = C_tot_mean.Young_modulus.mean()
        G_mean[i] = C_tot_mean.shear_modulus.mean()
    ax.plot(fraction_austenite, E_mean, label='E ({})'.format(method), color=colors[color_index])
    ax.plot(fraction_austenite, G_mean, label='G ({})'.format(method), color=colors[color_index], linestyle='--')
    color_index = color_index + 1

ax.legend()
ax.set_xlim([0, 1])
ax.set_xlabel('Fraction of Austenite')
ax.set_ylabel('Young/Shear Modulus (GPa)')
fig.show()



