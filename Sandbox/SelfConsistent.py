from elasticipy.homogenization.kroner_eshelby import Kroner_Eshelby
from elasticipy.tensors.elasticity import StiffnessTensor

C1 = StiffnessTensor.isotropic(E=210, nu=0.25)
C2 = StiffnessTensor.isotropic(E=70, nu=0.3)
Cmacro, msg = Kroner_Eshelby((C1, C2), display=True, particle_size=(1000, 1000, 1))
