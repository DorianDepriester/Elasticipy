from scipy.spatial.transform import Rotation
from orix.quaternion import Rotation
import numpy as np
from scipy.integrate import quad_vec

class CrystalTexture:
    pass

class FibreTexture(CrystalTexture):
    def __init__(self, miller, axis):
        self.miller = miller
        self.axis = axis

    def __repr__(self):
        if self.miller.coordinate_format == 'uvw' or self.miller.coordinate_format == 'UVTW':
            miller = s = str(self.miller.uvw[0])
            miller = miller.replace('[', '<').replace(']', '>')
        else:
            miller = s = str(self.miller.hkl[0])
        row_0 =  "Fiber texture with {miller} || {axis}".format(miller=miller, axis=self.axis)
        point_group = self.miller.phase.point_group.name
        row_1 = 'Point group: ' + str(point_group)
        return row_0 + '\n' + row_1

    def mean_tensor(self, tensor):
        ref_orient = Rotation.from_align_vectors(self.miller, self.axis)
        tensor_ref_orient = tensor * ref_orient
        def fun(theta):
            rotation = Rotation.from_axes_angles(self.axis, theta)
            tensor_rotated = tensor_ref_orient * rotation
            return tensor_rotated.to_Kelvin()
        res, *_ = quad_vec(fun, 0, 2 * np.pi)
        return tensor.__class__.from_Kelvin(res / (2*np.pi))