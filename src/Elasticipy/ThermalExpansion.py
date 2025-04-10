import warnings
from Elasticipy.tensors.thermal_expansion import ThermalExpansionTensor as NewThermalExpansionTensor

warnings.warn(
    "The module 'Elasticipy.ThermalExpansion' is deprecated and will be removed in a future release. "
    "Please use 'Elasticipy.tensors.thermal_expansion' instead.",
    DeprecationWarning,
    stacklevel=2
)

class ThermalExpansionTensor(NewThermalExpansionTensor):
    pass