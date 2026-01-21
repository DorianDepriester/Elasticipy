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
        """
        Create a single-orientation crystallographic texture

        Parameters
        ----------
        orientation : orix.quaternion.orientation.Orientation
            Orientation of the crystallographic texture
        """
        self.orientation = orientation

    def __repr__(self):
        return str(self.orientation.to_euler(degrees=True))

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
        return tensor * self.orientation

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

    def plot_as_pole_figure(self, miller, **kwargs):
        """
        Plot the the pole figure of the crystallographic texture

        Parameters
        ----------
        miller : orix.vector.miller.Miller
            Miller indices of directions/plane to plot
        kwargs
            Keyword arguments passed to orix.vector.Vector3d.scatter

        Returns
        -------
        matplotlib.figure.Figure
            Handle to figure
        """
        v = Vector3d(~self.orientation * miller)
        return v.scatter(return_figure=True, **kwargs)

class FibreTexture(CrystalTexture):
    def __init__(self, miller, axis):
        """
        Create a perfect fibre crystallographic texture

        Parameters
        ----------
        miller : orix.vector.miller.Miller
            Plane or direction to align with the axis
        axis :

        """
        ref_orient = Orientation.from_align_vectors(miller, axis)
        super().__init__(ref_orient)
        self.miller = miller
        self.axis = Vector3d(axis)

    def __repr__(self):
        if self.miller.coordinate_format == 'uvw' or self.miller.coordinate_format == 'UVTW':
            miller = str(self.miller.uvw[0])
            miller = miller.replace('[', '<').replace(']', '>')
        else:
            miller = str(self.miller.hkl[0])
        row_0 =  "Fibre texture with {miller} || {axis}".format(miller=miller, axis=self.axis.data[0])
        point_group = self.miller.phase.point_group.name
        row_1 = 'Point group: ' + str(point_group)
        return row_0 + '\n' + row_1

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
        if fig is None:
            fig = plt.figure()
        ax = add_polefigure(fig, projection=projection)
        for m in miller:
            t = Vector3d(~orientations * m)
            xyz = t.data
            r = np.linalg.norm(xyz, axis=1)
            phi = np.arctan2(xyz[:,1], xyz[:,0])
            phi[phi < 0] += 2 * np.pi
            theta = np.arccos(xyz[:,2] / r)
            ax.plot(phi, theta)
        ax.set_ylim([0, np.pi/2])
        return fig, ax

