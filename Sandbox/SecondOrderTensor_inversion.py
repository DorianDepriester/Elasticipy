from elasticipy.tensors.second_order import SymmetricSecondOrderTensor
from elasticipy.tensors.second_order import _inv_3x3
import time
import numpy as np

a = SymmetricSecondOrderTensor.rand(shape=1000000)

start_time = time.time()
ainv = np.linalg.inv(a.matrix)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ainv2 = _inv_3x3(a.matrix)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ainv3 = _inv_3x3(a.matrix, sym=True)
print("--- %s seconds ---" % (time.time() - start_time))

print(np.all(ainv == ainv2))


