from elasticipy.tensors.fourth_order import SymmetricFourthOrderTensor
from elasticipy.tensors.second_order import SymmetricSecondOrderTensor
from elasticipy.tensors.second_order import _inv_3x3
import time
import numpy as np
from elasticipy.homogenization.kroner_eshelby import polarization_tensor, gamma
from elasticipy.tensors.elasticity import StiffnessTensor
from scipy.integrate import trapezoid

a = SymmetricSecondOrderTensor.rand(shape=1000)

start_time = time.time()
ainv = np.linalg.inv(a.matrix)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ainv2 = _inv_3x3(a.matrix)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ainv3 = _inv_3x3(a.matrix, sym=True)
print("--- %s seconds ---" % (time.time() - start_time))

print(np.all(ainv == ainv2))


C=StiffnessTensor.cubic(C11=110, C12=60, C44=50)
n_theta, n_phi = 100, 200
a1, a2, a3 = 1, 2, 3

P=polarization_tensor(C, a1,a2,a3, n_theta, n_phi)

theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2 * np.pi, n_phi)
phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='xy')
start_time = time.time()
g = gamma(C, phi_grid, theta_grid, 1, a2, a3)
integrand = (g.full_tensor.T * np.sin(theta_grid).T).T
a = trapezoid(integrand, theta, axis=0)
t = trapezoid(a, phi, axis=0) / (4 * np.pi)
t2 = SymmetricFourthOrderTensor(t)
print("--- %s seconds ---" % (time.time() - start_time))
