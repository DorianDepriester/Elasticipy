from Elasticipy.tensors.elasticity import StiffnessTensor
from matplotlib import pyplot as plt

C = StiffnessTensor.orthorhombic(phase_name='forsterite',
                                C11=320, C12=68.2, C13=71.6,
                                C22=196.5, C23=76.8, C33=233.5, C44=64, C55=77, C66=78.7)
rho = 3.355

cp, cs_fast, cs_slow = C.wave_velocity(rho)
fig = plt.figure(figsize=(12,4))
cp.plot_as_pole_figure(subplot_args=(131,), title='p wave', fig=fig)
cs_fast.plot_as_pole_figure(subplot_args=(132,), title='s wave (fast)', fig=fig)
cs_slow.plot_as_pole_figure(subplot_args=(133,), title='s wave (slow)', fig=fig)
fig.show()