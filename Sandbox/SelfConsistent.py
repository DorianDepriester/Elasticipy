from elasticipy.homogenization.kroner_eshelby import Kroner_Eshelby
from elasticipy.tensors.elasticity import StiffnessTensor
from scipy.spatial.transform import Rotation
import time


align_fibre = Rotation.from_euler('Y', 90, degrees=True)
C_fibres = StiffnessTensor.transverse_isotropic(Ex=15, Ez=230, nu_zx=0.2, nu_xy=0.07, Gxz=3.8)
C_fibres = C_fibres * align_fibre
C_matrix = StiffnessTensor.isotropic(E=4.5, G=1.61)

start_time = time.time()
Cmacro = Kroner_Eshelby((C_fibres, C_matrix), particle_sizes=[(1, 1, 1), (1,1,1)], xtol=1, n_phi=200, n_theta=100, maxiter=1000)
print("--- %s seconds ---" % (time.time() - start_time))
print(Cmacro)

start_time = time.time()
Cmacro = Kroner_Eshelby((C_fibres, C_matrix), particle_sizes=[(1, 1, 1), (1,1,1)], xtol=1, n_phi=200, n_theta=100, method='iteration', maxiter=10000)
print("--- %s seconds ---" % (time.time() - start_time))
print(Cmacro)

