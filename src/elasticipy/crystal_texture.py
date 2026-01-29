from copy import deepcopy

from matplotlib import pyplot as plt
from orix.quaternion import Orientation
from orix.vector import Vector3d
from scipy.integrate import quad_vec
import numpy as np
from elasticipy.polefigure import add_polefigure
from elasticipy.tensors.fourth_order import FourthOrderTensor
from abc import ABC

ANGLE_35 = 35.26438968
ANGLE_37 = 36.6992252
ANGLE_54 = 54.73561032
ANGLE_59 = 58.97991646
ANGLE_63 = 63.43494882
ANGLE_74 = 74.20683095

def _plot_as_pf(orientations, miller, fig, projection, plot_type='plot', ax=None, **kwargs):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = add_polefigure(fig, projection=projection)
    m = orientations.shape[0]
    n = miller.shape[0]
    phi = np.zeros((m, n))
    theta = np.zeros((m, n))
    for i in range(0, n):
        mi = miller[i]
        t = Vector3d(~orientations * mi)
        phi[:,i] = t.azimuth
        theta[:,i] = t.polar
    if plot_type == 'scatter':
        ax.scatter(phi, theta, **kwargs)
    else:
        line, = ax.plot(phi[:, 0], theta[:, 0], **kwargs)
        color = line.get_color()
        ax.plot(phi[:, 1:], theta[:, 1:], color=color)
    ax.set_ylim([0, np.pi / 2])
    return fig, ax

class CrystalTexture(ABC):
    _title = 'Abstract class for crystallographic texture'

    def __init__(self):
        self.weight = 1.
        self._details = None

    def mean_tensor(self, tensor):
        """
        Perform the texture-weighted mean of a 4th-order tensor.

        Parameters
        ----------
        tensor : SymmetricFourthOrderTensor
            Reference tensor (unrotated)

        Returns
        -------
        SymmetricFourthOrderTensor
            mean value of the rotated tensor
        """
        pass

    def __mul__(self, other):
        if isinstance(other, FourthOrderTensor):
            return self.mean_tensor(other)
        else:
            t = deepcopy(self)
            t.weight = other
            return t

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        # self + other
        if isinstance(other, CrystalTexture):
            return CompositeTexture([self, other])
        elif isinstance(other, CompositeTexture):
            t = deepcopy(other)
            t.texture_list.insert(0, self)
            return t

    def __repr__(self):
        if self._details is None:
            return self._title
        else:
            return self._title + '\n' + self._details

    def plot_as_pole_figure(self, miller, projection='lambert', fig=None, ax=None, **kwargs):
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
        ax : matplotlib.projections.polar.PolarAxes, optional
            Axes to plot on
        kwargs
            Keyword arguments to pass to matplotlib's scatter/plot functions

        Returns
        -------
        matplotlib.figure.Figure
            Handle to figure
        matplotlib.projections.polar.PolarAxes
            Axes where the pole figure is plotted
        """
        pass


class UniformTexture(CrystalTexture):
    """
    Simple class to define the uniform texture over SO(3)
    """
    _title = 'Uniform texture'

    def __init__(self):
        """
        Create a uniform texture over SO(3)
        """
        super().__init__()
        self._details = 'Uniform distribution over SO(3)'

    def mean_tensor(self, tensor):
        return tensor.infinite_random_average()

    def plot_as_pole_figure(self, miller, projection='lambert', fig=None, ax=None, **kwargs):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = add_polefigure(fig, projection=projection)
        phi = np.linspace(0,2*np.pi)
        theta = np.linspace(0,np.pi)
        phi, theta = np.meshgrid(phi, theta)
        r = np.ones_like(phi)
        ax.contourf(phi, theta, r, **kwargs)
        ax.set_ylim([0, np.pi / 2])
        return fig, ax

class DiscreteTexture(CrystalTexture):
    """
    Class to handle classical crystallographic texture.

    Notes
    -----
    This class implements the crystallographic textures listed by [Lohmuller]_

    References
    ----------
    .. [Lohmuller] Lohmuller, P.; Peltier, L.; Hazotte, A.; Zollinger, J.; Laheurte, P.; Fleury, E. Variations of
       the Elastic Properties of the CoCrFeMnNi High Entropy Alloy Deformed by Groove Cold Rolling.
       Materials 2018, 11, 1337. https://doi.org/10.3390/ma11081337
    """
    _title = "Crystallographic texture"

    def __init__(self, orientation):
        """
        Create a single-orientation crystallographic texture.

        Parameters
        ----------
        orientation : orix.quaternion.orientation.Orientation
            Orientation of the crystals
        """
        super().__init__()
        self.orientation = orientation
        if orientation is None:
            self._details = 'Uniform over SO(3)'
        else:
            self._details = 'φ1={:.2f}°, ϕ={:.2f}°, φ2={:.2f}°'.format(*self.orientation.to_euler(degrees=True)[0])

    def mean_tensor(self, tensor):
        return tensor * self.orientation

    @classmethod
    def cube(cls):
        """
        Create a Cube crystallographic texture: {100}<100>

        Returns
        -------
        DiscreteTexture

        See Also
        --------
        A : Create a A single-orientation crystallographic texture
        brass : Create a Brass single-orientation crystallographic texture
        copper : Create a Copper single-orientation crystallographic texture
        CuT : Create a CuT single-orientation crystallographic texture
        Goss : Create a Goss single-orientation crystallographic texture
        GossBrass : Create a Goss-Brass single-orientation crystallographic texture
        P : Create a P single-orientation crystallographic texture
        S : Create an S single-orientation crystallographic texture
        """
        o = Orientation.from_euler([0, 0, 0], degrees=True)
        return DiscreteTexture(o)

    @classmethod
    def Goss(cls):
        """
        Create a Goss crystallographic texture: {110}<100>

        Returns
        -------
        DiscreteTexture

        See Also
        --------
        A : Create an A single-orientation crystallographic texture
        brass : Create a Brass single-orientation crystallographic texture
        copper : Create a Copper single-orientation crystallographic texture
        cube : Create a Cube single-orientation crystallographic texture
        CuT : Create a CuT single-orientation crystallographic texture
        GossBrass : Create a Goss-Brass single-orientation crystallographic texture
        P : Create a P single-orientation crystallographic texture
        S : Create an S single-orientation crystallographic texture
        """
        o = Orientation.from_euler([0, 45, 0], degrees=True)
        return DiscreteTexture(o)

    @classmethod
    def brass(cls):
        """
        Create a Brass crystallographic texture: {110}<112>

        Returns
        -------
        DiscreteTexture

        See Also
        --------
        A : Create a A single-orientation crystallographic texture
        copper : Create a Copper single-orientation crystallographic texture
        cube : Create a Cube single-orientation crystallographic texture
        CuT : Create a CuT single-orientation crystallographic texture
        Goss : Create a Goss single-orientation crystallographic texture
        GossBrass : Create a Goss-Brass single-orientation crystallographic texture
        P : Create a P single-orientation crystallographic texture
        S : Create an S single-orientation crystallographic texture
        """
        o = Orientation.from_euler([ANGLE_35, 45, 0], degrees=True)
        return DiscreteTexture(o)

    @classmethod
    def GossBrass(cls):
        """
        Create a Goss/Brass crystallographic texture: {110}<115>

        Returns
        -------
        DiscreteTexture

        See Also
        --------
        A : Create an A single-orientation crystallographic texture
        copper : Create a Copper single-orientation crystallographic texture
        cube : Create a Cube single-orientation crystallographic texture
        CuT : Create a CuT single-orientation crystallographic texture
        Goss : Create a Goss single-orientation crystallographic texture
        P : Create a P single-orientation crystallographic texture
        S : Create an S single-orientation crystallographic texture
        """
        o = Orientation.from_euler([ANGLE_74, 90, 45], degrees=True)
        return DiscreteTexture(o)

    @classmethod
    def copper(cls):
        """
        Create a copper crystallographic texture: {112}<111>

        Returns
        -------
        DiscreteTexture

        See Also
        --------
        A : Create an A single-orientation crystallographic texture
        cube : Create a Cube single-orientation crystallographic texture
        CuT : Create a CuT single-orientation crystallographic texture
        Goss : Create a Goss single-orientation crystallographic texture
        GossBrass : Create a Goss-Brass single-orientation crystallographic texture
        P : Create a P single-orientation crystallographic texture
        S : Create an S single-orientation crystallographic texture
        """
        o = Orientation.from_euler([90, ANGLE_35, 45], degrees=True)
        return DiscreteTexture(o)

    @classmethod
    def A(cls):
        """
        Create an "A" crystallographic texture: {110}<111>

        Returns
        -------
        DiscreteTexture

        See Also
        --------
        copper : Create a Copper single-orientation crystallographic texture
        cube : Create a Cube single-orientation crystallographic texture
        CuT : Create a CuT single-orientation crystallographic texture
        Goss : Create a Goss single-orientation crystallographic texture
        GossBrass : Create a Goss-Brass single-orientation crystallographic texture
        P : Create a P single-orientation crystallographic texture
        S : Create an S single-orientation crystallographic texture
        """
        o = Orientation.from_euler([ANGLE_35, 90, 45], degrees=True)
        return DiscreteTexture(o)

    @classmethod
    def P(cls):
        """
        Create a "P"" crystallographic texture: {011}<211>

        Returns
        -------
        DiscreteTexture

        See Also
        --------
        A : Create an A single-orientation crystallographic texture
        copper : Create a Copper single-orientation crystallographic texture
        cube : Create a Cube single-orientation crystallographic texture
        CuT : Create a CuT single-orientation crystallographic texture
        Goss : Create a Goss single-orientation crystallographic texture
        GossBrass : Create a Goss-Brass single-orientation crystallographic texture
        S : Create an S single-orientation crystallographic texture
        """
        o = Orientation.from_euler([ANGLE_54, 90, 45], degrees=True)
        return DiscreteTexture(o)

    @classmethod
    def CuT(cls):
        """
        Create a CuT crystallographic texture: {552}<115>

        Returns
        -------
        DiscreteTexture

        See Also
        --------
        A : Create an A single-orientation crystallographic texture
        copper : Create a Copper single-orientation crystallographic texture
        cube : Create a Cube single-orientation crystallographic texture
        Goss : Create a Goss single-orientation crystallographic texture
        GossBrass : Create a Goss-Brass single-orientation crystallographic texture
        P : Create a P single-orientation crystallographic texture
        S : Create an S single-orientation crystallographic texture
        """
        o = Orientation.from_euler([90, ANGLE_74, 45], degrees=True)
        return DiscreteTexture(o)

    @classmethod
    def S(cls):
        """
        Create an "S" crystallographic texture: {123}<634>

        Returns
        -------
        DiscreteTexture

        See Also
        --------
        A : Create an A single-orientation crystallographic texture
        copper : Create a Copper single-orientation crystallographic texture
        cube : Create a Cube single-orientation crystallographic texture
        CuT : Create a CuT single-orientation crystallographic texture
        Goss : Create a Goss single-orientation crystallographic texture
        GossBrass : Create a Goss-Brass single-orientation crystallographic texture
        P : Create a P single-orientation crystallographic texture
        """
        o = Orientation.from_euler([ANGLE_59, ANGLE_37, ANGLE_63], degrees=True)
        return DiscreteTexture(o)

    def plot_as_pole_figure(self, miller, projection='lambert', fig=None, ax=None, **kwargs):
        return _plot_as_pf(self.orientation, miller, fig, projection, ax=ax, plot_type='scatter', **kwargs)

class FibreTexture(CrystalTexture):
    _title = 'Fibre texture'

    def __init__(self, orientation, axis, point_group=None):
        """
        Create a fibre-type crystallographic texture

        Parameters
        ----------
        orientation : orix.quaternion.orientation.Orientation
            Reference orientation
        axis : list or tuple or numpy.ndarray or orix.vector.Vector3D
            Axis of rotation (in sample CS)
        point_group : orix.phase.point_group.PointGroup, optional
            Point group to use
        """
        super().__init__()
        self.orientation = orientation
        self.axis = Vector3d(axis)
        self.point_group = point_group

    @classmethod
    def from_Euler(cls, phi1=None, Phi=None, phi2=None, degrees=True):
        """
        Create a fibre texture by providing two fixed Bunge-Euler values

        Parameters
        ----------
        phi1 : float
            First Euler angle
        Phi : float
            Second Euler angle
        phi2 : float
            Third Euler angle
        degrees : boolean, optional
            If true (default), the angles must be passed in degrees (in radians otherwise)

        Returns
        -------
        FibreTexture

        See Also
        --------
        from_Miller_axis : Define a fibre texture by aligning a miller direction with a given axis

        Examples
        --------
        A fibre texture corresponding to constant (e.g. zero) values for phi1 and phi2, and uniform distribution of Phi
        on [0,2π[, can be defined as follows:

        >>> from elasticipy.crystal_texture import FibreTexture
        >>> t1 = FibreTexture.from_Euler(phi1=0., phi2=0.)
        >>> t1
        Fibre texture
        φ1= 0.0°, φ2= 0.0°

        Similarly, the following returns a fibre texture for phi1=0 and Phi=0, and uniform distribution of phi2 on
        [0,2π[:

        >>> t2 = FibreTexture.from_Euler(phi1=0., Phi=0.)
        >>> t2
        Fibre texture
        φ1= 0.0°, ϕ= 0.0°
        """
        if phi1 is None:
            orient1 = Orientation.from_euler([0., Phi, phi2] , degrees=degrees)
            orient2 = Orientation.from_euler([1., Phi, phi2] , degrees=degrees)
            angle_list = {'ϕ':Phi, 'φ2':phi2}
        elif Phi is None:
            orient1 = Orientation.from_euler([phi1, 0., phi2], degrees=degrees)
            orient2 = Orientation.from_euler([phi1, 1., phi2], degrees=degrees)
            angle_list = {'φ1':phi1, 'φ2':phi2}
        elif phi2 is None:
            orient1 = Orientation.from_euler([phi1, Phi, 0.] , degrees=degrees)
            orient2 = Orientation.from_euler([phi1, Phi, 1.] , degrees=degrees)
            angle_list = {'φ1':phi1, 'ϕ':Phi}
        else:
            raise ValueError("Exactly two Euler angles are required.")
        axis = (~orient1 * orient2).axis
        a = cls(orient2, axis)
        (k1, v1), (k2, v2) = angle_list.items()
        if not degrees:
            v1 = v1 * 180 / np.pi
            v2 = v2 * 180 / np.pi
        a._details = f"{k1}= {v1}°, {k2}= {v2}°"
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

        Returns
        -------
        FibreTexture

        See Also
        --------
        from_Euler : define a fibre texture from two Euler angles

        Examples
        --------
        Let's consider a cubic poly-crystal (point group: m-3m), whose orientations are defined by a perfect alignment
        of direction <100> with the Z axis of a sample (therefore a uniform distribution around the Z axis). This
        texture can be defined as:

        >>> from orix.crystal_map import Phase
        >>> from orix.vector.miller import Miller
        >>> from elasticipy.crystal_texture import FibreTexture
        >>> phase = Phase(point_group='m-3m')
        >>> m = Miller(uvw=[1,0,0], phase=phase)
        >>> t = FibreTexture.from_Miller_axis(m, [0,0,1])
        >>> t
        Fibre texture
        <1. 0. 0.> || [0, 0, 1]
        """
        ref_orient = Orientation.from_align_vectors(miller, Vector3d(axis))
        a = cls(ref_orient, axis, point_group=miller.phase.point_group.name)
        if miller.coordinate_format == 'uvw' or miller.coordinate_format == 'UVTW':
            miller_str = str(miller.uvw[0])
            miller_str = miller_str.replace('[', '<').replace(']', '>')
        else:
            miller_str = str(miller.hkl[0])
        a.point_group = miller.phase.point_group.name
        row_0 = "{miller} || {axis}".format(miller=miller_str, axis=axis)
        a._details = row_0
        return a

    def mean_tensor(self, tensor):
        tensor_ref_orient = tensor * ~self.orientation
        def fun(theta):
            rotation = ~Orientation.from_axes_angles(self.axis, theta)
            tensor_rotated = tensor_ref_orient * rotation
            return tensor_rotated.to_Kelvin()
        circle = 2 * np.pi
        res, *_ = quad_vec(fun, 0, circle)
        return tensor.__class__.from_Kelvin(res / circle)

    def plot_as_pole_figure(self, miller, n_orientations=100, fig=None, ax=None, projection='lambert', **kwargs):
        theta = np.linspace(0, 2 * np.pi, n_orientations)
        orientations = self.orientation * Orientation.from_axes_angles(self.axis, theta)
        return _plot_as_pf(orientations, miller, fig, projection, ax=ax, **kwargs)


class CompositeTexture:
    def __init__(self, texture_list):
        """
        Create a mix of crystal textures

        Parameters
        ----------
        texture_list : list of CrystalTexture
            List of crystal textures to mix
        """
        self.texture_list = list(texture_list)

    def __mul__(self, other):
        # self * other
        if isinstance(other, (float, int)):
            tm = deepcopy(self)
            for t in tm.texture_list:
                t.weight *= other
            return tm
        elif isinstance(other, FourthOrderTensor):
            return self.mean_tensor(other)

    def __rmul__(self, other):
        # other * self
        return self * other

    def __add__(self, other):
        # self + other
        if isinstance(other, CrystalTexture):
            t = deepcopy(self)
            t.texture_list.append(other)
            return t
        elif isinstance(other, CompositeTexture):
            return CompositeTexture(self.texture_list + other.texture_list)

    def __len__(self):
        return len(self.texture_list)

    def __repr__(self):
        title = 'Mixture of crystallographic textures'
        heading = ' Wgt.  Type      Component'
        sep =     ' ------------------------------------------------------------'
        table = []
        for t in self.texture_list:
            if isinstance(t, DiscreteTexture):
                kind = 'discrete'
            elif isinstance(t, UniformTexture):
                kind = 'uniform '
            else:
                kind = 'fibre   '
            table.append(' {:.2f}  {}  {}'.format(t.weight, kind, t._details))
        return '\n'.join([title, heading, sep] + table)

    def mean_tensor(self, tensor):
        """
        Compute the weighted average of a tensor, considering each texture component separately.

        Parameters
        ----------
        tensor : FourthOrderTensor
            Reference tensor (unrotated)

        Returns
        -------
        FourthOrderTensor

        Examples
        --------
        Let consider a mixture of Goss and fibre tensor (with phi1=0 and phi2=0):

        >>> from elasticipy.crystal_texture import DiscreteTexture, FibreTexture
        >>> from elasticipy.tensors.elasticity import StiffnessTensor
        >>> t = DiscreteTexture.Goss() + FibreTexture.from_Euler(phi1=0.0, phi2=0.0)
        >>> t
        Mixture of crystallographic textures
         Wgt.  Type            Component
         -----------------------------------------
         1.00  single-orient.  φ1=0.00°, ϕ=45.00°, φ2=0.00°
         1.00  fibre           φ1= 0.0°, φ2= 0.0°

        Then, assume that the stiffness tensor is defined as follows:

        >>> C = StiffnessTensor.cubic(C11=186, C12=134, C44=77) # mp-30

        The ODF-weighted Voigt average can be computed as follows:

        >>> Cvoigt = t.mean_tensor(C)
        >>> Cvoigt
        Stiffness tensor (in Voigt mapping):
        [[ 1.86000000e+02  1.34000000e+02  1.34000000e+02  0.00000000e+00
           0.00000000e+00  0.00000000e+00]
         [ 1.34000000e+02  2.24250000e+02  9.57500000e+01  6.96664948e-15
           0.00000000e+00  0.00000000e+00]
         [ 1.34000000e+02  9.57500000e+01  2.24250000e+02 -2.83236976e-15
           0.00000000e+00  0.00000000e+00]
         [ 0.00000000e+00  2.85362012e-16  8.15320034e-17  2.58750000e+01
           0.00000000e+00  0.00000000e+00]
         [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
           5.77500000e+01 -5.48414542e-17]
         [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
           5.48414542e-17  5.77500000e+01]]

        Alternatively, on can directly use the following syntax:

        >>> Cvoigt = C * t
        """
        n = len(self)
        t = tensor.__class__.eye(shape=(n,))
        wgt = []
        for i in range(0, n):
            ti = self.texture_list[i]
            wgt.append(ti.weight)
            t[i] = ti.mean_tensor(tensor)
        return t.tensor_average(weights=wgt)

    def plot_as_pole_figure(self, miller, fig=None, projection='lambert'):
        if fig is None:
            fig = plt.figure(tight_layout=True)
        ax = add_polefigure(fig, projection=projection)
        for t in self.texture_list:
            t.plot_as_pole_figure(miller, fig=fig, projection=projection, ax=ax)
        return fig, ax