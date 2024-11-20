# Elasticipy
A python toolkit to manipulate elastic constants of materials. 
This package also provides a collection of easy-to-use and very fast tools to work on stress and strain tensors.

## Basic examples
### Working with stiffness tensor
First, create a stiffness tensor, taking into account the material symmetry:
````python
 from Elasticipy.FourthOrderTensor import tensorFromCrystalSymmetry
 
 C = tensorFromCrystalSymmetry(symmetry='cubic', phase_name='ferrite'
                              C11=274, C12=175, C44=89)
````

Plot the directional Young moduli:
````python
C.Young_modulus.plot()
````
Evaluate its mean value:
````python
C.Young_modulus.mean()
````

Evaluate the shear modulus with respect to x and y:
````python
C.shear_modulus.eval([1,0,0], [0,1,0])
````

### Working with stress/strain tensors
````python
import numpy as np
from Elasticipy.StressStrainTensors import StressTensor
````
First, let's create an array of stresses:
````python
n_slices = 10
sigma = np.zeros((n_slices, 3, 3))
sigma[:, 1, 1] = np.linspace(0, 1, n_slices)
sigma = StressTensor(sigma)     # Convert it to stress
````
Now compute the strain:
````python
eps = C.inv()*sigma
````
Check that the strain is of the same shape as the stress:
````python
print(eps.shape)
````
Check out the von Mises equivalent stresses:
````python
print(sigma.vonMises())
````
Let's apply a set of 1000 random rotations to the stiffness:
````python
from scipy.spatial.transform import Rotation

orientations = Rotation.random(1000)
C_rotated = C*orientations
````
Now check the corresponding strains:
````python
eps_rotated = C_rotated.inv()*sigma
````
See the shape of the results:
````python
print(eps_rotated.shape)
````
Just to be sure, look at a particular value of ``eps_rotated``:
````python
print(eps_rotated[0,-1])
````
Evaluate the mean strain values over all orientations 
````python
eps_mean = eps_rotated.mean(axis=0)
````
Actually, a more direct method is to compute the Reuss average of stiffness tensor:
````python
Creuss = C.Reuss_average(orientations=orientations)
````
Then compute the corresponding strain:
````python
eps_reuss = Creuss.inv()*sigma
````
You can check that both the approaches are consistent:
````python
np.all(np.isclose(eps_mean.matrix,eps_reuss.matrix))
````