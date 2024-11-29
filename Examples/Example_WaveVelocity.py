from Elasticipy.FourthOrderTensor import StiffnessTensor
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot
from matplotlib import pyplot as plt

C = StiffnessTensor.fromCrystalSymmetry(symmetry='orthorombic', phase_name='forsterite',
                                        C11=320, C12=68.2, C13=71.6,
                                        C22=196.5, C23=76.8, C33=233.5, C44=64, C55=77, C66=78.7)
rho = 3.355

cp, cs_fast, cs_slow = C.wave_velocity(rho)
print(cp)

fig, _ =cp.plot_as_pole_figure(subplot_args=(131,), title='p wave', show=False)
cs_fast.plot_as_pole_figure(subplot_args=(132,), title='s wave (fast)', fig=fig, show=False)
cs_slow.plot_as_pole_figure(subplot_args=(133,), title='s wave (slow)', fig=fig, show=True)
