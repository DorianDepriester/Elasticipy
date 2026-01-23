from copy import deepcopy

from matplotlib import pyplot as plt
from orix.quaternion import Orientation
from orix.vector import Vector3d
from scipy.integrate import quad_vec
import numpy as np
from elasticipy.polefigure import add_polefigure

ANGLE_35 = 35.26438968
ANGLE_37 = 36.6992252
ANGLE_54 = 54.73561032
ANGLE_59 = 58.97991646
ANGLE_63 = 63.43494882
ANGLE_74 = 74.20683095

def _plot_as_pf(orientations, miller, fig, projection, plot_type='plot'):
    if fig is None:
        fig = plt.figure()
    ax = add_polefigure(fig, projection=projection)
    for m in miller:
        t = Vector3d(~orientations * m)
        xyz = t.data
        r = np.linalg.norm(xyz, axis=1)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi[phi < 0] += 2 * np.pi
        theta = np.arccos(xyz[:, 2] / r)
        if plot_type == 'scatter':
            ax.scatter(phi, theta)
        else:
            ax.plot(phi, theta)
    ax.set_ylim([0, np.pi / 2])
    return fig, ax

class _CrystalTextureBase:
    def __init__(self, orientation):
        """
        Create a single-orientation crystallographic texture

        Parameters
        ----------
        orientation : orix.quaternion.orientation.Orientation or None
            Orientation of the crystallographic texture
        """
        self.orientation = orientation
        self.weight = 0.

    def mean_tensor(self, tensor):
        """
        Perform the texture-weighted mean of a 4th-order tensor.

        Parameters
        ----------
        tensor : FourthOrderTensor
            Reference tensor (unrotated)
        Returns
        -------
        FourthOrderTensor
            mean value of the rotated tensor
        """
        if self.orientation is None:
            return tensor.infinite_random_average()
        else:
            return tensor * self.orientation

    def __mul__(self, other):
        t = deepcopy(self)
        t.weight = other
        return t

    def __rmul__(self, other):
        return self * other

class CrystalTexture(_CrystalTextureBase):
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

    def __repr__(self):
        title = "Crystallographic texture"
        return title + '\nphi1={:.2f}°, Phi={:.2f}°, phi2={:.2f}°'.format(*self.orientation.to_euler(degrees=True)[0])

    @classmethod
    def uniform(cls):
        """
        Create a uniform crystallographic texture

        Returns
        -------
        CrystalTexture
        """
        return cls(None)

    @classmethod
    def Cube(cls):
        """
        Create a Cube crystallographic texture: {100}<100>

        Returns
        -------
        CrystalTexture
        """
        o = Orientation.from_euler([0, 0, 0], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def Goss(cls):
        """
        Create a Goss crystallographic texture: {110}<100>

        Returns
        -------
        CrystalTexture
        """
        o = Orientation.from_euler([0, 45, 0], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def Brass(cls):
        """
        Create a Brass crystallographic texture: {110}<112>

        Returns
        -------
        CrystalTexture
        """
        o = Orientation.from_euler([ANGLE_35, 45, 0], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def GossBrass(cls):
        """
        Create a Goss/Brass crystallographic texture: {110}<115>

        Returns
        -------
        CrystalTexture
        """
        o = Orientation.from_euler([ANGLE_74, 90, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def Copper(cls):
        """
        Create a copper crystallographic texture: {112}<111>

        Returns
        -------
        CrystalTexture
        """
        o = Orientation.from_euler([90, ANGLE_35, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def A(cls):
        """
        Create an "A" crystallographic texture: {110}<111>

        Returns
        -------
        CrystalTexture
        """
        o = Orientation.from_euler([ANGLE_35, 90, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def P(cls):
        """
        Create a "P"" crystallographic texture: {011}<211>

        Returns
        -------
        CrystalTexture
        """
        o = Orientation.from_euler([ANGLE_54, 90, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def CuT(cls):
        """
        Create a CuT crystallographic texture: {552}<115>

        Returns
        -------
        CrystalTexture
        """
        o = Orientation.from_euler([90, ANGLE_74, 45], degrees=True)
        return CrystalTexture(o)

    @classmethod
    def S(cls):
        """
        Create an "S" crystallographic texture: {123}<634>

        Returns
        -------
        CrystalTexture
        """
        o = Orientation.from_euler([ANGLE_59, ANGLE_37, ANGLE_63], degrees=True)
        return CrystalTexture(o)

    def plot_as_pole_figure(self, miller, projection='lambert', fig=None):
        """
        Plot the pole figure of the crystallographic texture

        Parameters
        ----------
        miller : orix.vector.miller.Miller
            Miller indices of directions/planes to plot
        projection : str, optional
            Type of projection to use, it can be either stereographic or Lambert
        fig : matplotlib.figure.Figure, optional
            Handle to existing figure, if needed

        Returns
        -------
        matplotlib.figure.Figure
            Handle to figure

        """
        return _plot_as_pf(self.orientation, miller, fig, projection, plot_type='scatter')

class FibreTexture(_CrystalTextureBase):
    def __init__(self, o, axis):
        super().__init__(o)
        self.axis = Vector3d(axis)
        self._repr = 'Fibre texture'

    @classmethod
    def from_euler(cls, phi1=None, Phi=None, phi2=None, degrees=True):
        if phi1 is None:
            orient1 = Orientation.from_euler([0., Phi, phi2] , degrees=degrees)
            orient2 = Orientation.from_euler([1., Phi, phi2] , degrees=degrees)
            angle_list = {'Phi':Phi, 'phi2':phi2}
        elif Phi is None:
            orient1 = Orientation.from_euler([phi1, 0., phi2], degrees=degrees)
            orient2 = Orientation.from_euler([phi1, 1., phi2], degrees=degrees)
            angle_list = {'phi1':phi1, 'phi2':phi2}
        elif phi2 is None:
            orient1 = Orientation.from_euler([phi1, Phi, 0.] , degrees=degrees)
            orient2 = Orientation.from_euler([phi1, Phi, 1.] , degrees=degrees)
            angle_list = {'phi1':phi1, 'Phi':Phi}
        else:
            raise ValueError("Exactly two Euler angles are required.")
        axis = (~orient1 * orient2).axis
        a = cls(orient2, axis)
        (k1, v1), (k2, v2) = angle_list.items()
        if not degrees:
            v1 = v1 * 180 / np.pi
            v2 = v2 * 180 / np.pi
        a._repr += f"\n{k1}= {v1}°, {k2}= {v2}°"
        return a

    @classmethod
    def from_Miller_axis(cls, miller, axis):
        """
        Create a perfect fibre crystallographic texture

        Parameters
        ----------
        miller : orix.vector.miller.Miller
            Crystal plane or direction to align with the axis
        axis : tuple or list
            Axis (in sample CS) to align with
        """
        ref_orient = Orientation.from_align_vectors(miller, Vector3d(axis))
        a = cls(ref_orient, axis)
        if miller.coordinate_format == 'uvw' or miller.coordinate_format == 'UVTW':
            miller_str = str(miller.uvw[0])
            miller_str = miller_str.replace('[', '<').replace(']', '>')
        else:
            miller_str = str(miller.hkl[0])
        row_0 =  "\n{miller} || {axis}".format(miller=miller_str, axis=axis)
        point_group = miller.phase.point_group.name
        row_1 = 'Point group: ' + str(point_group)
        a._repr += row_0 + '\n' + row_1
        return a

    def __repr__(self):
        return self._repr

    def mean_tensor(self, tensor):
        tensor_ref_orient = tensor * ~self.orientation
        def fun(theta):
            rotation = ~Orientation.from_axes_angles(self.axis, theta)
            tensor_rotated = tensor_ref_orient * rotation
            return tensor_rotated.to_Kelvin()
        circle = 2 * np.pi
        res, *_ = quad_vec(fun, 0, circle)
        return tensor.__class__.from_Kelvin(res / circle)

    def plot_as_pole_figure(self, miller, n_orientations=100, fig=None, projection='lambert', **kwargs):
        theta = np.linspace(0, 2 * np.pi, n_orientations)
        orientations = self.orientation * Orientation.from_axes_angles(self.axis, theta)
        return _plot_as_pf(orientations, miller, fig, projection)


class CrystalTextureMix:
    def __init__(self, texture_list):
        self.texture_list = [texture_list]


