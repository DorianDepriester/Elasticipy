import warnings
from Elasticipy.tensors.elasticity import StiffnessTensor as NewStiffnessTensor
from Elasticipy.tensors.elasticity import ComplianceTensor as NewComplianceTensor

warnings.warn(
    "The module 'Elasticipy.FourthOrderTensor' is deprecated and will be removed in a future release. "
    "Please use 'Elasticipy.tensors.elasticity' instead.",
    DeprecationWarning,
    stacklevel=2
)

class StiffnessTensor(NewStiffnessTensor):
    pass

class ComplianceTensor(NewComplianceTensor):
    pass
