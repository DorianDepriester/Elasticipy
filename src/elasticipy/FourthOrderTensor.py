import warnings
from elasticipy.tensors.elasticity import StiffnessTensor as NewStiffnessTensor
from elasticipy.tensors.elasticity import ComplianceTensor as NewComplianceTensor

warnings.warn(
    "The module 'elasticipy.FourthOrderTensor' is deprecated and will be removed in a future release. "
    "Please use 'elasticipy.tensors.elasticity' instead.",
    DeprecationWarning,
    stacklevel=2
)

class StiffnessTensor(NewStiffnessTensor):
    pass

class ComplianceTensor(NewComplianceTensor):
    pass
