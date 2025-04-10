import warnings
from Elasticipy.tensors.stress_strain import StressTensor as NewStressTensor
from Elasticipy.tensors.stress_strain import StrainTensor as NewStrainTensor

warnings.warn(
    "The module 'Elasticipy.StressStrainTensors' is deprecated and will be removed in a future release. "
    "Please use 'Elasticipy.tensors.stress_strain' instead.",
    DeprecationWarning,
    stacklevel=2
)

class StressTensor(NewStressTensor):
    pass

class StrainTensor(NewStrainTensor):
    pass
