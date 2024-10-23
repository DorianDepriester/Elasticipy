from StiffnessTensor import tensorFromCrystalSymmetry
import matplotlib as mpl
mpl.use('Qt5Agg')   # Ensure interactive plot

C = tensorFromCrystalSymmetry(symmetry='cubic', C11=387, C12=25, C44=111)
E = C.Young_modulus
print(E)
E.plot()
