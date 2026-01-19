from orix.quaternion import Orientation
from scipy.integrate import quad_vec
import numpy as np

ANGLE_35 = 35.26438968
ANGLE_37 = 36.6992252
ANGLE_59 = 58.97991646
ANGLE_63 = 63.43494882
ANGLE_74 = 74.20683095


class CrystalTexture:
    """
    Class to handle classical crystallographic texture.

    Notes
    -----
    This class implements the crystallographic textures listed by [Lohmuller et al.]_

    References
    ----------
    .. [Lohmuller et al.] Lohmuller, P.; Peltier, L.; Hazotte, A.; Zollinger, J.; Laheurte, P.; Fleury, E. Variations of
    the Elastic Properties of the CoCrFeMnNi High Entropy Alloy Deformed by Groove Cold Rolling.
    Materials 2018, 11, 1337. https://doi.org/10.3390/ma11081337
    """

    def __init__(self, orientation):
        self.orientation = orientation

    def __repr__(self):
        return str(self.orientation.to_euler(degrees=True))

    def mean_tensor(self, tensor):
        return tensor * self.orientation

    @classmethod
    def Goss(cls):
        o = Orientation.from_euler([0, 45, 0], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def Brass(cls):
        o = Orientation.from_euler([ANGLE_35, 45, 0], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def GossBrass(cls):
        o = Orientation.from_euler([ANGLE_74, 90, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def Copper(cls):
        o = Orientation.from_euler([90, ANGLE_35, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def A(cls):
        o = Orientation.from_euler([ANGLE_35, 90, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def P(cls):
        o = Orientation.from_euler([30, 90, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def CuT(cls):
        o = Orientation.from_euler([90, ANGLE_74, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def S(cls):
        o = Orientation.from_euler([ANGLE_59, ANGLE_37, ANGLE_63], degrees=True)
        return CrystalTexture(o)


class FibreTexture(CrystalTexture):
    def __init__(self, miller, axis):
        ref_orient = Orientation.from_align_vectors(miller, axis)
        super().__init__(ref_orient)
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
        tensor_ref_orient = tensor * self.orientation
        def fun(theta):
            rotation = Orientation.from_axes_angles(self.axis, theta)
            tensor_rotated = tensor_ref_orient * rotation
            return tensor_rotated.to_Kelvin()
        res, *_ = quad_vec(fun, 0, 2 * np.pi)
        return tensor.__class__.from_Kelvin(res / (2*np.pi))