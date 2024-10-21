# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from StiffnessTensor import tensorFromCrystalSymmetry
import matplotlib.pyplot as plt

Ct=tensorFromCrystalSymmetry(symmetry='hexagonal', C11=48, C12=15, C13=10, C33=55, C44=16,unit='MPa')
