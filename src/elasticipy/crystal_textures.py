from orix.quaternion import Orientation
import numpy as np
from scipy.integrate import quad_vec

class CrystalTexture:
    def __init__(self, orientation):
        self.orientation = orientation

    def __repr__(self):
        return str(self.orientation.to_euler(degrees=True))

    @classmethod
    def Goss(cls):
        o = Orientation.from_euler([0, 45, 0], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def Brass(cls):
        o = Orientation.from_euler([35.26438968, 45, 0], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def Copper(cls):
        o = Orientation.from_euler([90, 35.26438968, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def S(cls):
        o = Orientation.from_euler([58.97991646, 36.6992252, 63.43494882], degrees=True)
        return CrystalTexture(o)


class FibreTexture(CrystalTexture):
    def __init__(self, miller, axis):
        super().__init__(None)
        self.miller = miller
        self.axis = axis

    def __repr__(self):
        if self.miller.coordinate_format == 'uvw' or self.miller.coordinate_format == 'UVTW':
            miller = s = str(self.miller.uvw[0])
            miller = miller.replace('[', '<').replace(']', '>')
        else:
            miller = s = str(self.miller.hkl[0])
        row_0 =  "Fibre texture with {miller} || {axis}".format(miller=miller, axis=self.axis)
        point_group = self.miller.phase.point_group.name
        row_1 = 'Point group: ' + str(point_group)
        return row_0 + '\n' + row_1

    def mean_tensor(self, tensor):
        ref_orient = Orientation.from_align_vectors(self.miller, self.axis)
        tensor_ref_orient = tensor * ref_orient
        def fun(theta):
            rotation = Orientation.from_axes_angles(self.axis, theta)
            tensor_rotated = tensor_ref_orient * rotation
            return tensor_rotated.to_Kelvin()
        res, *_ = quad_vec(fun, 0, 2 * np.pi)
        return tensor.__class__.from_Kelvin(res / (2*np.pi))