from abc import ABC

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from plotly import graph_objects as go

from elasticipy.tensors.stress_strain import StressTensor, StrainTensor


class YieldCriterion(ABC):
    """
    Abstract class for plasticity criteria
    """
    name = 'generic'

    def yield_function(self, stress):
        """
        Return the yield function, with respect to the plasticity criterion.

        Parameters
        ----------
        stress : StressTensor
            Stress tensor to compute the yield function from.

        Returns
        -------
        float or numpy.ndarray
            Negative if, and only if, the elasticity criterion is met
        """
        pass

    def normal(self, stress, **kwargs):
        """
        Apply the normality rule

        Parameters
        ----------
        stress : StressTensor
            Stress tensor to apply the normality rule
        kwargs : dict
            Keyword arguments passed to the function

        Returns
        -------
        StrainTensor
            Normalized direction of plastic flow
        """
        pass

    @property
    def _plot_bounds(self):
        return (0.0, 1.0), (0.0, 1.0)

    def plot_2D(self, color='red', fig=None, ax=None, alpha=0.3,
                xrange=None, yrange=None, npt=400, label=None, linewidth=1., linestyle='solid'):
        """
        Plot the elastic domain in the biaxial tensile space.

        Parameters
        ----------
        color : str, optional
            Color to use for the plot.
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. If not provided, a new figure will be created.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        alpha : float, optional
            Transparency of the inside of the yield surface.
        xrange : tuple, optional
            Set the x-range of the plot. If not provided, it will be set automatically.
        yrange : tuple, optional
            Set the y-range of the plot. If not provided, it will be set automatically.
        npt : int, optional
            Number of points along each direction to use for the plot.
        label : str, optional
            Label for the plot. If not provided, the name of the yield criterion constructor will be used.
        linewidth : float, optional
            Width of the lines in the plot. Default is 1.
        linestyle : str, optional
            Style of the lines in the plot. Default is 'solid'.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure where the plot is drawn.
        ax : matplotlib.axes.Axes
            Axes where the plot is drawn.

        Examples
        --------
        Plot the von Mises yield surface:

        .. plot::

            from elasticipy.plasticity import VonMisesPlasticity
            fig, ax = VonMisesPlasticity().plot_2D()
            fig.show()
        """
        margin = 1.05
        if xrange is None:
            xrange = self._plot_bounds[0]
        if yrange is None:
            yrange = self._plot_bounds[1]
        sigma1 = np.linspace(xrange[0] * margin, xrange[1] * margin , npt)
        sigma2 = np.linspace(yrange[0] * margin, yrange[1] * margin , npt)
        Sigma1, Sigma2 = np.meshgrid(sigma1, sigma2)
        sigma = StressTensor.tensile([1, 0, 0], Sigma1) + StressTensor.tensile([0, 1, 0], Sigma2)
        f = self.yield_function(sigma)

        if ax is None or fig is None:
            fig, ax = plt.subplots()

        # Tracer la zone remplie et le contour
        if alpha != 0.:
            ax.contourf(Sigma1, Sigma2, f, levels=[-np.inf, 0], colors=[color], alpha=alpha)
        ax.contour(Sigma1, Sigma2, f, levels=[0], colors=color, linewidths=linewidth, linestyles=linestyle)

        if label is None:
            label = self.name
        proxy = Line2D([0], [0], color=color, label=label, linewidth=linewidth, linestyle=linestyle)
        ax.add_line(proxy)

        # Configuration de l'axe
        ax.set_xlabel(r'$\sigma_1$')
        ax.set_ylabel(r'$\sigma_2$')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True)
        ax.axis('equal')

        return fig, ax

    def plot_3D(self, xrange=None, yrange=None, zrange=None, color='red', opacity=0.3, npt=100, fig=None):
        """
        Plot the yield surface in the principal stress space

        Parameters
        ----------
        xrange, yrange, zrange : tuple, optional
            range for the first principal stress. If not provided, (-1,1) will be used, except if the figure already
            contains data; in this case, the same ranges will be reused.
        color : str, optional
            colro to use for plotting the surface
        opacity : float, optional
            opacity of the surface
        npt : int, optional
            number of points along each direction to use for the plot.
        fig : plotly.graph_objs._figure.Figure
            handle to existing plotly figure. This is used when one want to plot multiple surfaces on the same graph.

        Returns
        -------
        fig : plotly.graph_objs._figure.Figure
            Figure where the plot is drawn
        """
        if fig is None:
            fig = go.Figure()
        elif len(fig.data):
            if xrange is None:
                xrange = (np.min(fig.data[0].x), np.max(fig.data[0].x))
            if yrange is None:
                yrange = (np.min(fig.data[0].y), np.max(fig.data[0].y))
            if zrange is None:
                zrange = (np.min(fig.data[0].z), np.max(fig.data[0].z))
        if xrange is None:
            xrange = (-1, 1)
        if yrange is None:
            yrange = (-1, 1)
        if zrange is None:
            zrange = (-1, 1)
        sigma_1=np.linspace(*xrange, npt)
        sigma_2=np.linspace(*yrange, npt)
        sigma_3=np.linspace(*zrange, npt)
        s1, s2, s3 = np.meshgrid(sigma_1, sigma_2, sigma_3, indexing='ij')
        s = (StressTensor.tensile([1,0,0], s1)
             + StressTensor.tensile([0,1,0], s2)
             + StressTensor.tensile([0,0,1], s3))
        f = self.yield_function(s)
        fig.add_trace(go.Isosurface(
            x=s1.flatten(),
            y=s2.flatten(),
            z=s3.flatten(),
            value=f.flatten(),
            isomin=0,
            isomax=0,
            caps=dict(x_show=False, y_show=False),
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=opacity,
            name=self.name,
            ))
        fig.update_layout(scene=dict(
            xaxis=dict(
                title=dict(
                    text=r'σ₁'
                )
            ),
            yaxis=dict(
                title=dict(
                    text=r'σ₂'
                )
            ),
            zaxis=dict(
                title=dict(
                    text=r'σ₃'
                )
            ),
        ),)
        return fig


class VonMisesCriterion(YieldCriterion):
    """
    von Mises plasticity criterion, with associated normality rule
    """
    name = 'von Mises'

    def __init__(self, yield_stress=1.0):
        """
        Create a plasticity criterion

        Parameters
        ----------
        yield_stress : float
            Tensile yield stress
        """
        self.yield_stress = yield_stress

    @property
    def _plot_bounds(self):
        sigma_max = self.yield_stress * 2 / np.sqrt(3)
        return (-sigma_max, sigma_max), (-sigma_max, sigma_max)

    @staticmethod
    def eq_stress(stress, **kwargs):
        """
        Return the equivalent stress, with respect to the plasticity criterion.

        Parameters
        ----------
        stress : StressTensor
            Stress to compute the equivalent stress from
        kwargs : dict
            keyword arguments passed to the function

        Returns
        -------
        float or numpy.ndarray
        """
        return stress.vonMises()

    def yield_function(self, stress):
        return self.eq_stress(stress) - self.yield_stress

    @staticmethod
    def normal(stress, **kwargs):
        eq_stress = stress.vonMises()
        dev_stress = stress.deviatoric_part()
        gradient_tensor = dev_stress / eq_stress
        return StrainTensor(3 / 2 * gradient_tensor.matrix)


class TrescaCriterion(VonMisesCriterion):
    """
    Tresca plasticity criterion, with associated normality rule
    """
    name = 'Tresca'

    @property
    def _plot_bounds(self):
        sigma_max = self.yield_stress
        return (-sigma_max, sigma_max), (-sigma_max, sigma_max)

    @staticmethod
    def eq_stress(stress, **kwargs):
        return stress.Tresca()

    @staticmethod
    def normal(stress, **kwargs):
        vals, dirs = stress.eig()
        u1 = dirs[..., 0]
        u3 = dirs[..., 2]
        s1 = vals[..., 0]
        s2 = vals[..., 1]
        s3 = vals[..., 2]
        a = np.einsum('...i,...j->...ij', u1, u1)
        b = np.einsum('...i,...j->...ij', u3, u3)
        normal = a - b
        singular_points = np.logical_or(s2 == s1, s2 == s3)
        normal[singular_points] = VonMisesCriterion().normal(stress[singular_points]).matrix
        normal[np.logical_and(s2 == s1, s2 == s3)] = 0.0
        strain = StrainTensor(normal)
        return strain / strain.eq_strain()


class DruckerPrager(YieldCriterion):
    """
    Drucker-Prager pressure-dependent plasticity criterion, with associated normality rule
    """
    name = 'Drucker-Prager'

    def __init__(self, alpha, k):
        """
        Create a Drucker-Prager (DP) plasticity criterion.

        The DG criterion can be defined by the pai values alpha and c, or c and phi (see notes).

        Parameters
        ----------
        alpha : float
            Pressure dependence parameters (see notes for details)
        k : float
            Constant component of the yield surface (see below)

        Notes
        -----
        The pressure-dependent DP plasticity criterion assumes that the yield surface is defined as:

        .. math::

            \\alpha I_1 + \\sqrt{J_2} - k

        where :math:`I_1` is the first invariant of the stress tensor, and :math:`J_2` is the second invariant of the
        deviatoric stress tensor.
        """
        self.alpha = alpha
        self.k = k

    @classmethod
    def from_cohesion_friction_angle(cls, c, phi, fit='middle', degrees=True):
        """
        Define a Drucker-Prager yield criterion from the cohesion factor and friction angle

        Parameters
        ----------
        c : float
            cohesion factor
        phi : float
            cone angle, in degrees
        fit : str ['inside', 'middle', 'outside'], optional
            How to treat c and phi, as they usually refer to the Mohr-Coulomb (MC) yield criterion. It relates the shape
            of the DP cone in the principal stress space with respect to that of the MC criterion:
              - inside : the DP cone is inside that of MC (tangent to the faces of the MC cone)
              - outside : the DP cone is outside that of MC
              - middle : the DP cone is of intermediate shape between inside and outside.
        degrees : bool, optional
            Whether the friction angle is in given degrees. Default is True.

        Notes
        -----
        if fit=='outside':

        .. math::

            k = \\frac{ 6c\\cos\\phi }{ \\sqrt{3}\\left(3+\\sin\phi\\right) }

            \\alpha = \\frac{-2\\sin\phi}{ \\sqrt{3}\\left(3+\\sin\phi\\right) }

        if fit=='inside':

        .. math::

            k = \\frac{ 3c\\cos\\phi }{ \\sqrt{\\left(9+3\\sin^2\phi\\right)} }

            \\alpha = \\frac{-\\sin\phi}{ \\sqrt{\\left(9+3\\sin^2\phi\\right)} }

        if fit=='middle':

        .. math::

            k = \\frac{ 6c\\cos\\phi }{ \\sqrt{3}\\left(3-\\sin\phi\\right) }

            \\alpha = \\frac{-2\\sin\phi}{ \\sqrt{3}\\left(3-\\sin\phi\\right) }

        Returns
        -------
        DruckerPrager
            DP yield criterion
        """
        if degrees:
            phi = np.radians(phi)
        if fit == 'outside':
            d = 3 ** 0.5 * (3 + np.sin(phi))
            k = 6 * c * np.cos(phi) / d
            alpha = -2 * np.sin(phi) / d
        elif fit == 'inside':
            d = (9 + 3 * np.sin(phi) ** 2) ** 0.5
            k = 3 * c * np.cos(phi) / d
            alpha = -np.sin(phi) / d
        else:
            d = 3 ** 0.5 * (3 - np.sin(phi))
            k = 6 * c * np.cos(phi) / d
            alpha = -2 * np.sin(phi) / d
        return cls(alpha, k)

    @property
    def _plot_bounds(self):
        bounds = (self.k / (2 * self.alpha - 3**(-0.5)), self.k / (2 * self.alpha + 3**(-0.5)))
        s_max = np.max(bounds)
        s_min = np.min(bounds)
        return (1.5*s_min, 1.5*s_max), (1.5*s_min, 1.5*s_max)

    def eq_stress(self, stress, **kwargs):
        return (stress.J2**0.5 + self.alpha * stress.I1) / (1/3**0.5 + self.alpha)

    def normal(self, stress, **kwargs):
        J2 = stress.J2
        gradient = stress.deviatoric_part() / (2 * J2**0.5) + self.alpha * StressTensor.eye(stress.shape)
        strain = StrainTensor(gradient.matrix)
        return strain / strain.eq_strain()

    def yield_function(self, stress):
        return stress.J2**0.5 + self.alpha * stress.I1 - self.k


class MohrCoulomb(YieldCriterion):
    """
    Mohr-Coulomb pressure-dependent plasticity criterion, with associated normality rule
    """
    name = 'Mohr-Coulomb'

    def __init__(self, c=1.0, phi=0.):
        """
        Create a Mohr-Coulomb (MC) yield criterion.

        Parameters
        ----------
        c : float
            Cohesion factor
        phi : float
            Friction angle (in degrees)

        Notes
        -----
        Given the principal stresses :math:`\\sigma_{I}\\geq \\sigma_{II}\\geq \\sigma_{III}`, the MC yield function is
        defined as:

        .. math::

            \\sigma_{I} - \\sigma_{III} + \\left(\\sigma_{I} + \\sigma_{III}\\right)\\sin\\phi - 2c\\cos\\phi

        """
        self.c = c
        self.phi = np.radians(phi)

    def yield_function(self, stress):
        sigma_p = stress.principal_stresses()
        c1 = sigma_p[...,0]
        c3 = sigma_p[...,2]
        return c1 - c3 - (c1 + c3) * np.sin(self.phi) - 2 * self.c * np.cos(self.phi)

    @property
    def _plot_bounds(self):
        s_min = -2 * self.c * np.cos(self.phi) / (1 + np.sin(self.phi))
        s_max = 2 * self.c * np.cos(self.phi) / (1 - np.sin(self.phi))
        return (s_min, s_max), (s_min, s_max)

    def normal(self, stress, **kwargs):
        s, dirs = stress.eig()
        u1 = dirs[..., 0]
        u2 = dirs[..., 1]
        u3 = dirs[..., 2]
        s1 = s[..., 0]
        s2 = s[..., 1]
        s3 = s[..., 2]
        t1 = np.einsum('...i,...j->...ij', u1, u1)
        t2 = np.einsum('...i,...j->...ij', u2, u2)
        t3 = np.einsum('...i,...j->...ij', u3, u3)
        singular_point_12 = s1 == s2
        singular_point_23 = s2 == s3
        a = t1
        b = t3
        a[singular_point_12] = 0.5 * t1 + 0.5 * t2
        b[singular_point_23] = 0.5 * t2 + 0.5 * t3
        normal = ( 1 - np.sin(self.phi)) * a - (1 + np.sin(self.phi)) * b
        strain = StrainTensor(normal)
        return strain / strain.eq_strain()

