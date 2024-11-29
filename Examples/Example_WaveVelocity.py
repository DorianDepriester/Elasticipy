from Elasticipy.FourthOrderTensor import StiffnessTensor
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot

C = StiffnessTensor.fromCrystalSymmetry(symmetry='orthorombic', C11=320, C12=68.2, C13=71.6,
                                        C22=196.5, C23=76.8, C33=233.5, C44=64, C55=77, C66=78.7)
rho = 3.355

cp, cs_fast, cs_slow = C.wave_velocity(rho)
print(cp)
(cs_fast-cs_slow).plot_as_pole_figure(plot_type='contourf')