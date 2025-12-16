import warnings
from elasticipy.tensors.thermal_expansion import ThermalExpansionTensor as NewThermalExpansionTensor

warnings.warn(
    "The module 'elasticipy.ThermalExpansion' is deprecated and will be removed in a future release. "
    "Please use 'elasticipy.tensors.thermal_expansion' instead.",
    DeprecationWarning,
    stacklevel=2
)

class ThermalExpansionTensor(NewThermalExpansionTensor):
    pass