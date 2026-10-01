from elasticipy.tensors.stress_strain import StressTensor
from elasticipy.tensors.second_order import SymmetricSecondOrderTensor

class NeoHooke:
    def __init__(self, C10, D1):
        self.C10 = C10
        self.D1 = D1

    def stress_from_gradient(self, gradient):
        J = gradient.J
        p = 2 * self.D1 * J * (J-1)
        I = SymmetricSecondOrderTensor.eye(shape=gradient.shape)
        tau = p * I + 2 * self.C10 / J**(2/3) * gradient.B.deviatoric_part()
        return StressTensor(tau / J)