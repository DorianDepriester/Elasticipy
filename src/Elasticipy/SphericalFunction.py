import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from numpy import cos, sin
from scipy import integrate as integrate
from scipy import optimize
from Elasticipy.PoleFigure import add_polefigure


def sph2cart(*args):
    """
    Converts spherical/hyperspherical coordinates to cartesian coordinates.

    Parameters
    ----------
    args : tuple
        (phi, theta) angles for spherical coordinates of direction u, where phi denotes the azimuth from X and theta is
        the colatitude angle from Z.
        If a third argument is passed, it defines the third angle in hyperspherical coordinate system (psi), that is
        the orientation of the second vector v, orthogonal to u.

    Returns
    -------
    numpy.ndarray
        directions u expressed in cartesian coordinates system.
    tuple of numpy.ndarray, numpy.ndarray
        If a third angle is passed, returns a tuple:
        - The first element is `u`, the directions expressed in the cartesian coordinate system.
        - The second element is `v`, the direction of the second vector orthogonal to `u`, also expressed in the
        cartesian coordinate system.
    """
    phi, theta, *psi = args
    phi_vec = np.array(phi).flatten()
    theta_vec = np.array(theta).flatten()
    u = np.array([cos(phi_vec) * sin(theta_vec), sin(phi_vec) * sin(theta_vec), cos(theta_vec)]).T
    if not psi:
        return u
    else:
        psi_vec = np.array(psi).flatten()
        e_phi = np.array([-sin(phi_vec), cos(phi_vec), np.zeros(phi_vec.shape)])
        e_theta = np.array([cos(theta_vec) * cos(phi_vec), cos(theta_vec) * sin(phi_vec), -sin(theta_vec)])
        v = cos(psi_vec) * e_phi + sin(psi_vec) * e_theta
        return u, v.T


def _plot3D(fig, u, r, **kwargs):
    norm = Normalize(vmin=r.min(), vmax=r.max())
    colors = cm.viridis(norm(r))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    xyz = (u.T * r.T).T
    ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], facecolors=colors, rstride=1, cstride=1,
                    antialiased=False, **kwargs)
    mappable = cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax


def _create_xyz_section(ax, section_name, polar_angle):
    ax.title.set_text('{}-{} plane'.format(*section_name))
    if section_name == 'XY':
        phi = polar_angle
        theta = np.pi / 2 * np.ones(len(polar_angle))
    elif section_name == 'XZ':
        phi = np.zeros(len(polar_angle))
        theta = np.pi / 2 - polar_angle
    else:
        phi = (np.pi / 2) * np.ones(len(polar_angle))
        theta = np.pi / 2 - polar_angle
    ax.set_xticks(np.linspace(0, 3 * np.pi / 2, 4))
    h_direction, v_direction = section_name
    ax.set_xticklabels((h_direction, v_direction, '-' + h_direction, '-' + v_direction))
    return phi, theta, ax


def uniform_spherical_distribution(n_evals, seed=None, return_orthogonal=False):
    """
    Create a set of vectors whose projections over the unit sphere are uniformly distributed.

    Parameters
    ----------
    n_evals : int
        Number of vectors to generate
    seed : int, default None
        Sets the seed for the random values. Useful if one wants to ensure reproducibility.
    return_orthogonal : bool, default False
        If true, also return a second set of vectors which are orthogonal to the first one.

    Returns
    -------
    u : np.ndarray
        Random set of vectors whose projections over the unit sphere are uniform.
    v : np.ndarray
        Set of vectors with the same properties as u, but orthogonal to u.
        Returned only if return_orthogonal is True

    Notes
    -----
    The returned vector(s) are not unit. If needed, one can use:
        u = (u.T / np.linalg.norm(u, axis=1)).T
    """
    if seed is None:
        rng = np.random
    else:
        rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_evals, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    if return_orthogonal:
        if seed is not None:
            # Ensure that the seed used for generated v is not the same as that for u
            # Otherwise, u and v would be equal.
            rng = np.random.default_rng(seed+1)
        u2 = rng.normal(size=(n_evals, 3))
        return u, np.cross(u, u2)
    else:
        return u

def _integrate_over_unit_sphere(phi, theta, values=None, psi=None):
    sine = np.sin(theta)
    if values is None:
        values = np.ones(phi.shape)
    if psi is None:
        return integrate.trapezoid(
                    integrate.trapezoid(
                        values * sine, axis=0, x=phi[:, 0]
                    ), axis=0, x=theta[0, :])
    else:
        return integrate.trapezoid(
                    integrate.trapezoid(
                        integrate.trapezoid(
                            values * sine, axis=0, x=phi[:, 0, 0]
                        ), axis=0, x=theta[0, :, 0])
                , x=psi[0,0,:])



class SphericalFunction:
    """
    Class for spherical functions, that is, functions that depend on directions in 3D space.

    Attribute
    ---------
    fun : function to use
    """
    domain = np.array([[0, 2 * np.pi],
                       [0, np.pi / 2]])
    name = 'Spherical function'
    """Bounds to consider in spherical coordinates"""

    def __init__(self, fun, symmetry=True):
        """
        Create a spherical function, that is, a function that depends on one direction only.

        Parameters
        ----------
        fun : callable
            Function to return
        symmetry : bool, optional
            Set to true if fun(u)==fun(-u)
        """
        self.fun = fun
        self.symmetry = symmetry

    def __repr__(self):
        val_min, _ = self.min()
        val_max, _ = self.max()
        s = '{}\n'.format(self.name)
        s += 'Min={}, Max={}'.format(val_min, val_max)
        return s

    def __add__(self, other):
        if isinstance(other, self.__class__):
            def fun(*x):
                return self.fun(*x) + other.fun(*x)
            return self.__class__(fun)
        elif isinstance(other, (float, int, np.number)):
            def fun(*x):
                return self.fun(*x) + other
            return self.__class__(fun)
        else:
            msg_error = 'A {} can only be added to another {} or a scalar value.'.format(self.name, self.name)
            raise NotImplementedError(msg_error)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            def fun(*x):
                return self.fun(*x) - other.fun(*x)
            return self.__class__(fun)
        else:
            return self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            def fun(*x):
                return self.fun(*x) * other.fun(*x)
        elif isinstance(other, (float, int, np.number)):
            def fun(*x):
                return self.fun(*x) * other
        else:
            raise NotImplemented
        return self.__class__(fun)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            def fun(*x):
                return self.fun(*x) / other.fun(*x)
        elif isinstance(other, (float, int, np.number)):
            def fun(*x):
                return self.fun(*x) / other
        else:
            raise NotImplemented
        return self.__class__(fun)

    def eval(self, u):
        """
        Evaluate value along a given (set of) direction(s).

        Parameters
        ----------
        u : np.ndarray or list
            Direction(s) to estimate the value along with. It can be of a unique direction [nx, ny, nz],
            or a set of directions (e.g. [[n1x, n1y n1z],[n2x, n2y, n2z],...]).

        Returns
        -------
        float or np.ndarray
            If only one direction is given as a tuple of floats [nx, ny, nz], the result is a float; otherwise, the
            result is a nd.array.

        See Also
        --------
        eval_spherical : evaluate the function along a given direction given using the spherical coordinates
        """
        values = self.fun(u)
        if isinstance(u, list) and np.array(u).shape == (3,):
            return values[0]
        else:
            return values

    def eval_spherical(self, *args, degrees=False):
        """
        Evaluate value along a given (set of) direction(s) defined by its (their) spherical coordinates.

        Parameters
        ----------
        args : list or np.ndarray
            [phi, theta] where phi denotes the azimuth angle from X axis,
            and theta is the latitude angle from Z axis (theta==0 -> Z axis).
        degrees : bool, default False
            If True, the angles are given in degrees instead of radians.

        Returns
        -------
        float or np.ndarray
            If only one direction is given as a tuple of floats [phi, theta], the result is a float; otherwise, the
            result is a nd.array.

        See Also
        --------
        eval : evaluate the function along a direction given by its cartesian coordinates
        """
        angles = np.atleast_2d(args)
        if degrees:
            angles = np.radians(angles)
        phi, theta = angles.T
        u = sph2cart(phi, theta)
        values = self.eval(u)
        if (np.array(args).shape == (2,) or np.array(args).shape == (1, 2)) and not isinstance(args, np.ndarray):
            return values[0]
        else:
            return values

    def _global_minimizer(self, fun):
        n_eval = 50
        phi = np.linspace(*self.domain[0], n_eval)
        theta = np.linspace(*self.domain[1], n_eval)
        if len(self.domain) == 2:
            phi, theta = np.meshgrid(phi, theta)
            angles0 = np.array([phi.flatten(), theta.flatten()]).T
        else:
            psi = np.linspace(*self.domain[2], n_eval)
            phi, theta, psi = np.meshgrid(phi, theta, psi)
            angles0 = np.array([phi.flatten(), theta.flatten(), psi.flatten()]).T
        values = fun(angles0)
        loc_x0 = np.argmin(values)
        angles1 = angles0[loc_x0]
        results = optimize.minimize(fun, angles1, method='L-BFGS-B', bounds=self.domain)
        return results

    def min(self):
        """
        Find minimum value of the function.

        Returns
        -------
        fmin : float
            Minimum value
        dir : np.ndarray
            Direction along which the minimum value is reached

        See Also
        --------
        max : return the maximum value and the location where it is reached
        """
        results = self._global_minimizer(self.eval_spherical)
        return results.fun, sph2cart(*results.x)

    def max(self):
        """
        Find maximum value of the function.

        Returns
        -------
        min : float
            Maximum value
        direction : np.ndarray
            direction along which the maximum value is reached

        See Also
        --------
        min : return the minimum value and the location where it is reached
        """
        def fun(x):
            return -self.eval_spherical(*x)

        results = self._global_minimizer(fun)
        return -results.fun, sph2cart(*results.x)

    def mean(self, method='trapezoid', n_evals=10000, seed=None):
        """
        Estimate the mean value along all directions in the 3D space.

        Parameters
        ----------
        method : str {'exact', 'Monte Carlo', 'trapezoid'}, optional
            If 'exact', the full integration is performed over the unit sphere (see Notes). If 'trapezoid', the trapeze
            method is used to approximate the integral. If 'Monte Carlo', the function is evaluated along a finite set
            of random directions.
        n_evals : int, optional
            If method=='Monte Carlo' or 'trapezoid', sets the number of unit directions to use.
        seed : int, default None
            Sets the seed for random sampling when using the Monte Carlo method. Useful when one wants to reproduce
            results.

        Returns
        -------
        float
            Mean value

        See Also
        --------
        std : Standard deviation of the spherical function over the unit sphere.

        Notes
        -----
        The full integration over the unit sphere, used if method=='exact', takes advantage of numpy.integrate.dblquad.
        This algorithm is robust but usually slow. The Monte Carlo method can be 1000 times faster.
        """
        if self.symmetry:
            dom_size = 2 * np.pi
        else:
            dom_size = 4 * np.pi
        if method == 'exact':
            def fun(theta, phi):
                return self.eval_spherical(phi, theta) * sin(theta)

            domain = self.domain.flatten()
            q = integrate.dblquad(fun, *domain)
            return q[0] / dom_size
        elif method == 'trapezoid':
            (phi, theta), evals = self.evaluate_on_spherical_grid(n_evals)
            dom_size = _integrate_over_unit_sphere(phi, theta)
            return _integrate_over_unit_sphere(phi, theta, values=evals) / dom_size
        else:
            u = uniform_spherical_distribution(n_evals, seed=seed)
            return np.mean(self.eval(u))

    def var(self, method='trapezoid', n_evals=10000, mean=None, seed=None):
        """
        Estimate the variance along all directions in the 3D space

        Parameters
        ----------
        method : str {'exact', 'Monte Carlo', 'trapezoid}, optional
            If 'exact', the full integration is performed over the unit sphere (see Notes). If 'trapezoid', the trapeze
            method is used to approximate the integral. If 'Monte Carlo', the function is evaluated along a finite set
            of random directions.
        n_evals : int, optional
            If method=='Monte Carlo' or 'trapezoid', sets the number of unit directions to use.
        mean : float, optional
            If provided, skip estimation of mean value and use that provided instead (only used for exact and trapezoid
            methods)
        seed : int, optional
            Sets the seed for random sampling when using the Monte Carlo method. Useful when one wants to reproduce
            results.

        Returns
        -------
        float
            Variance of the function

        See Also
        --------
        mean : mean value of the function over the unit sphere.

        Notes
        -----
        The full integration over the unit sphere, used if method=='exact', takes advantage of numpy.integrate.dblquad.
        This algorithm is robust but usually slow. The Monte Carlo method can be 1000 times faster.
        """
        if method == 'exact':
            if mean is None:
                mean = self.mean(method='exact')

            def fun(theta, phi):
                return (self.eval_spherical(phi, theta) - mean) ** 2 * sin(theta)

            domain = self.domain.flatten()
            q = integrate.dblquad(fun, *domain)
            return q[0] / (2 * np.pi)
        elif method == 'trapezoid':
            if mean is None:
                mean = self.mean(method="trapezoid", n_evals=n_evals)
            (phi, theta), evals = self.evaluate_on_spherical_grid(n_evals)
            dom_size = _integrate_over_unit_sphere(phi, theta)
            return _integrate_over_unit_sphere(phi, theta, values=(evals - mean)**2) / dom_size
        else:
            u = uniform_spherical_distribution(n_evals, seed=seed)
            return np.var(self.eval(u))

    def std(self, **kwargs):
        """
        Standard deviation of the function along all directions in the 3D space.

        Parameters
        ----------
        **kwargs
            These parameters will be passed to var() function

        Returns
        -------
        float
            Standard deviation

        See Also
        --------
        var : variance of the function
        """
        return np.sqrt(self.var(**kwargs))

    def evaluate_on_spherical_grid(self, n, return_in_spherical=True, use_symmetry=True):
        """
        Create a set of vectors corresponding to a spherical grid (phi,theta), then flatten it.

        Parameters
        ----------
        n : int or tuple of int
            If int, it give the overall number of evaluations over the quarter unit sphere. If tuple of int, they
            correspond to the number of spherical angles (n_phi, n_theta).
        return_in_spherical : bool, optional
            If true, the first output argument will be the spherical coordinates (phi, theta). Otherwise, the cartesian
            coordinates are returned
        use_symmetry : whether to take consider the upper half-domain only, or the full sphere.

        Returns
        -------
        numpy.ndarray or tuple
            Coordinates of evaluation. If return_in_spherical==True, they will be returned as a tuple of angle (phi,
            theta). Otherwise, they will be returned as a numpy array of shape (n_phi, n_theta, 3).
        numpy.ndarray
            Grid of evaluated values of shape (n_phi, n_theta, 3).
        """
        symmetry = self.symmetry and use_symmetry
        if isinstance(n, int):
            if symmetry:
                n_theta = int(np.sqrt(n)/ 2) + 1
                n_phi = 4 * n_theta
            else:
                n_theta = int(np.sqrt(n / 2)) + 1
                n_phi = 2 * n_theta
        else:
            n_phi, n_theta = n
        if symmetry:
            theta_max = np.pi / 2
        else:
            theta_max = np.pi
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta = np.linspace(0, theta_max, n_theta)
        phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
        u = sph2cart(phi_grid.flatten(), theta_grid.flatten())
        evals = self.eval(u)
        evals_grid = evals.reshape((n_phi, n_theta))
        if return_in_spherical:
            return (phi_grid, theta_grid), evals_grid
        else:
            return u.reshape((n_phi, n_theta, 3)), evals_grid

    def plot3D(self, n_phi=50, n_theta=50, fig=None, **kwargs):
        """
        3D plotting of a spherical function

        Parameters
        ----------
        n_phi : int, default 50
            Number of azimuth angles (phi) to use for plotting. Default is 50.
        n_theta : int, default 50
            Number of latitude angles (theta) to use for plotting. Default is 50.
        fig : matplotlib.figure.Figure, default None
            handle to existing figure object. If None, a new figure will be created. If passed, the figure is not shown,
            (one should use plt.show() afterward).
        **kwargs
            These parameters will be passed to matplotlib plot_surface() function.

        Returns
        -------
        matplotlib.figure.Figure
            Handle to the figure
        matplotlib.Axes3D
            Handle to axes

        See Also
        --------
        plot_xyz_sections : plot values of the function in X-Y, X-Z an Y-Z planes.
        """
        if fig is None:
            new_fig = plt.figure()
        else:
            new_fig = fig
        u, evals = self.evaluate_on_spherical_grid((n_phi, n_theta), return_in_spherical=False, use_symmetry=False)
        ax = _plot3D(new_fig, u, evals, **kwargs)
        ax.axis('equal')
        if fig is None:
            plt.show()
        return new_fig, ax

    def plot_xyz_sections(self, n_theta=500, fig=None, axs=None, **kwargs):
        """
        Plot XYZ sections of spherical data.

        This method generates a figure with three polar plots showing the XY, XZ, and YZ sections of the
        spherical data contained within the instance. Each section is plotted using n_theta evenly spaced
        points over the interval [0, 2π].

        Parameters
        ----------
        n_theta : int, optional
            Number of points to use for the polar plot angles. Default is 500.
        fig : matplotlib.figure.Figure, default None
            Handle to existing figure object. If None, a new figure will be created. If passed, the figure is not shown
        axs : tuple of matplotlib.projections.polar.PolarAxes, optional
            If provided, use these axes to plot the sections, instead of creating new ones.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plot function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the polar plots.
        axs : list of matplotlib.axes._subplots.PolarAxesSubplot
            List of axes objects for each plot.
        """
        if fig is None:
            new_fig = plt.figure()
        else:
            new_fig = fig
        theta_polar = np.linspace(0, 2*np.pi, n_theta)
        titles = ('XY', 'XZ', 'YZ')
        axs_new = []
        for i in range(0, 3):
            if axs is None:
                ax = new_fig.add_subplot(1, 3, i+1, projection='polar')
            else:
                ax = axs[i]
            angles = np.zeros((n_theta, 2))
            phi, theta, ax = _create_xyz_section(ax, titles[i], theta_polar)
            angles[:, 0] = phi
            angles[:, 1] = theta
            r = self.eval_spherical(angles)
            ax.plot(theta_polar, r, **kwargs)
            axs_new.append(ax)
        if fig is None:
            new_fig.show()
        return new_fig, axs_new

    def plot_as_pole_figure(self, n_theta=50, n_phi=200, projection='lambert',
                            fig=None, plot_type='imshow', show=True, title=None,
                            subplot_args=(), subplot_kwargs=None, **kwargs):
        """
        Plots a pole figure visualization of spherical data using specified parameters and plot types.

        The function creates a pole figure based on spherical coordinates and can use different
        types of plots including 'imshow', 'contourf', and 'contour'. It also supports a variety of
        projections and renders the figure using Matplotlib.

        Parameters
        ----------
        n_theta : int, optional
            Number of divisions in the theta dimension, by default 50.
        n_phi : int, optional
            Number of divisions in the phi dimension, by default 200.
        projection : str {'Lambert','equal area'}, optional
            The type of projection to use for the plot, by default 'Lambert'.
        fig : matplotlib.figure.Figure, optional
            A Matplotlib figure object. If None, a new figure will be created, by default None.
        plot_type : str, optional
            The type of plot to generate: 'imshow', 'contourf', or 'contour', by default 'imshow'.
        title : str, optional
            Title to add to the current axis. Defaut is None.
        subplot_args : tuple
            List of arguments to pass to the subplot function, by default ()
        subplot_kwargs : dict
            Dictionary of keyword-arguments to pass to the subplot function, by default {}
        **kwargs : dict
            Additional keyword arguments to pass to the plotting functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Matplotlib figure containing the plot.
        ax : matplotlib.axes._axes.Axes
            The Matplotlib axes of the plot.
        """
        if subplot_kwargs is None:
            subplot_kwargs = {}
        if fig is None:
            new_fig = plt.figure()
        else:
            new_fig = fig
        ax = add_polefigure(new_fig, *subplot_args, projection=projection, **subplot_kwargs)
        phi = np.linspace(*self.domain[0], n_phi)
        theta = np.linspace(*self.domain[1], n_theta)
        phi, theta = np.meshgrid(phi, theta)
        phi_flat = phi.flatten()
        theta_flat = theta.flatten()
        values = self.eval_spherical(np.array([phi_flat, theta_flat]).T)
        if plot_type == 'imshow':
            sc = ax.pcolormesh(phi, theta, values.reshape(phi.shape), **kwargs)
        elif plot_type == 'contourf':
            sc = ax.contourf(phi, theta, values.reshape(phi.shape), **kwargs)
        elif plot_type == 'contour':
            sc = ax.contour(phi, theta, values.reshape(phi.shape), **kwargs)
        else:
            raise ValueError(f'Unknown plot type: {plot_type}')
        ax.set_rlim(*self.domain[1])
        ax.set_title(title)
        new_fig.colorbar(sc)
        if show:
            plt.show()
        return new_fig, ax

    @classmethod
    def ellipsoid(cls, a, b, c):
        """Generate a Spherical function corresponding the the 'radii' of an ellipsoid. The ellipsoid has its
        eigenvectors aligned with X, Y and Z.

        This function is mainly meant for testing purposes.

        Parameters
        ----------
        a, b, c : float
            half-axes of the ellipsoid, along the X direction, Y and Z directions, respectively.
        """
        def fun(u):
            u = np.atleast_2d(u)
            u_x, u_y, u_z = (u.T / np.linalg.norm(u, axis=1))
            return (u_x**2/a**2 + u_y**2/b**2 + u_z**2/c**2)**(-0.5)
        return cls(fun)

class HyperSphericalFunction(SphericalFunction):
    """
    Class for defining functions that depend on two orthogonal directions u and v.
    """
    domain = np.array([[0, 2 * np.pi],
                       [0, np.pi / 2],
                       [0, np.pi]])
    name = 'Hyperspherical function'

    def __init__(self, fun):
        """
        Create a hyperspherical function, that is, a function that depends on two orthogonal directions only.
        """
        super().__init__(fun)

    def eval(self, u, *args):
        """
        Evaluate the Hyperspherical function with respect to two orthogonal directions.

        Parameters
        ----------
        u : list or np.ndarray
            First axis
        args : list or np.ndarray
            Second axis

        Returns
        -------
        float or np.ndarray
            Function value

        See Also
        --------
        eval_spherical : evaluate the function along a direction defined by its spherical coordinates.
        """
        values = self.fun(u, *args)
        if np.array(u).shape == (3,) and not isinstance(u, np.ndarray):
            return values[0]
        else:
            return values

    def mean(self, method='trapezoid', n_evals=int(1e6), seed=None):
        if method == 'exact':
            def fun(psi, theta, phi):
                return self.eval_spherical(phi, theta, psi) * sin(theta)

            domain = self.domain.flatten()
            q = integrate.tplquad(fun, *domain)
            return q[0] / (2 * np.pi ** 2)
        elif method == 'trapezoid':
            (phi, theta, psi), evals = self.evaluate_on_spherical_grid(n_evals)
            dom_size = _integrate_over_unit_sphere(phi, theta, psi=psi)
            integral = _integrate_over_unit_sphere(phi, theta, psi=psi, values=evals)
            return integral / dom_size
        else:
            u, v = uniform_spherical_distribution(n_evals, seed=seed, return_orthogonal=True)
            return np.mean(self.eval(u, v))

    def eval_spherical(self, *args, degrees=False):
        """
        Evaluate value along a given (set of) direction(s) defined by its (their) spherical coordinates.

        Parameters
        ----------
        args : list or np.ndarray
            [phi, theta, psi] where phi denotes the azimuth angle from X axis to the first direction (u), theta is
            the latitude angle from Z axis (theta==0 -> u = Z axis), and psi is the angle defining the orientation of
            the second direction (v) in the plane orthogonal to u, as illustrated below:

            .. image:: ../../docs/_static/images/HyperSphericalCoordinates.png


        degrees : bool, default False
            If True, the angles are given in degrees instead of radians.

        Returns
        -------
        float or np.ndarray
            If only one direction is given as a tuple of floats [nx, ny, nz], the result is a float;
        otherwise, the result is a nd.array.

        See Also
        --------
        eval : evaluate the function along two orthogonal directions (u,v))
        """
        angles = np.atleast_2d(args)
        if degrees:
            angles = np.radians(angles)
        phi, theta, psi = angles.T
        u, v = sph2cart(phi, theta, psi)
        values = self.eval(u, v)
        if np.array(args).shape == (3,) and not isinstance(args, np.ndarray):
            return values[0]
        else:
            return values

    def var(self, method='trapezoid', n_evals=int(1e6), mean=None, seed=None):
        if method == 'exact':
            if mean is None:
                mean = self.mean(method='exact')

            def fun(psi, theta, phi):
                return (mean - self.eval_spherical(phi, theta, psi)) ** 2 * sin(theta)

            domain = self.domain.flatten()
            q = integrate.tplquad(fun, *domain)
            return q[0] / (2 * np.pi ** 2)
        if method == 'trapezoid':
            (phi, theta, psi), evals = self.evaluate_on_spherical_grid(n_evals)
            dom_size = _integrate_over_unit_sphere(phi, theta, psi=psi)
            if mean is None:
                mean = self.mean(method='trapezoid', n_evals=n_evals)
            return _integrate_over_unit_sphere(phi, theta, psi=psi, values=(evals - mean)**2) / dom_size
        else:
            u, v = uniform_spherical_distribution(n_evals, seed=seed, return_orthogonal=True)
            return np.var(self.eval(u, v))

    def evaluate_on_spherical_grid(self, n, return_in_spherical=True, use_symmetry=True):
        """
        Create a set of vectors corresponding to a spherical grid (phi,theta), then flatten it.

        Parameters
        ----------
        n : int or tuple of int
            If int, gives the overall number of evaluations over the unit hypersphere. If a tuple is passed, they gieve
            the number of angles to consider for (hyper)spherical coordinates (n_phi, n_theta, n_psi).
        return_in_spherical : bool, optional
            If true, the first output argument will be the spherical coordinates (phi, theta). Otherwise, the cartersian
            coordinates are returned
        use_symmetry : bool, optional
            Whether to use take advantage ot symmetry

        Returns
        -------
        tuple
            Coordinates of evaluation, either in spherical of cartesian coordinates
        numpy.ndarray
            Grid of evaluated values
        """
        symmetry = self.symmetry and use_symmetry
        if isinstance(n, int):
            if symmetry:
                n_phi = int(2 * n ** (1 / 3)) + 1
                n_theta = int(n_phi / 4) + 1
                n_psi = int(n_phi / 2) + 1
            else:
                n_phi = int(4 * n ** (1 / 3)) + 1
                n_theta = int(n_phi / 2) + 1
                n_psi = int(n_phi / 2) + 1
        else:
            n_phi, n_theta, n_psi = n
        if symmetry:
            theta_max = np.pi / 2
        else:
            theta_max = np.pi
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta = np.linspace(0, theta_max, n_theta)
        psi = np.linspace(0, np.pi, n_psi)
        phi_grid, theta_grid, psi_grid = np.meshgrid(phi, theta, psi, indexing='ij')
        u, v = sph2cart(phi_grid.flatten(), theta_grid.flatten(), psi_grid)
        evals = self.eval(u, v)
        evals_grid = evals.reshape((n_phi, n_theta, n_psi))
        if return_in_spherical:
            return (phi_grid, theta_grid, psi_grid), evals_grid
        else:
            u_r = u.reshape((n_phi, n_theta, n_psi, 3))
            v_r = v.reshape((n_phi, n_theta, n_psi, 3))
            return (u_r, v_r), evals_grid

    def plot3D(self, n_phi=50, n_theta=50, n_psi=50, which='mean', fig=None, **kwargs):
        """
        Generate a 3D plot representing the evaluation of spherical harmonics.

        This function evaluates a function over a grid defined by spherical coordinates
        (phi, theta, psi) and produces a 3D plot. It provides options to display the mean,
        standard deviation, minimum, or maximum of the evaluated values along the third angles (psi).
        The plot can be customized with additional keyword arguments.

        Parameters
        ----------
        n_phi : int, optional
            Number of divisions along the phi axis, default is 50.
        n_theta : int, optional
            Number of divisions along the theta axis, default is 50.
        n_psi : int, optional
            Number of divisions along the psi axis, default is 50.
        which : str, optional
            Determines which statistical measure to plot ('mean', 'std', 'min', 'max'),
            default is 'mean'.
        fig : matplotlib.figure.Figure, optional
            Handle to existing figure object. Default is None. If provided, it disables showing the figure.
        kwargs : dict, optional
            Additional keyword arguments to customize the plot.

        Returns
        -------
        tuple
            A tuple containing the matplotlib figure and axes objects.
        """
        if fig is None:
            new_fig = plt.figure()
        else:
            new_fig = fig
        uv, values = self.evaluate_on_spherical_grid((n_phi, n_theta, n_psi), return_in_spherical=False, use_symmetry=False)
        u, _ = uv
        if which == 'std':
            r_grid = np.std(values, axis=2)
        elif which == 'min':
            r_grid = np.min(values, axis=2)
        elif which == 'max':
            r_grid = np.max(values, axis=2)
        else:
            r_grid = np.mean(values, axis=2)
        ax = _plot3D(new_fig, u[:, :, 0, :], r_grid, **kwargs)
        if fig is None:
            plt.show()
        return new_fig, ax

    def plot_xyz_sections(self, n_theta=500, n_psi=100, color_minmax='blue', alpha_minmax=0.2, color_mean='red',
                          fig=None):
        """
        Plots the XYZ sections using polar projections.

        This function creates a figure with three subplots representing the XY, XZ,
        and YZ sections. It utilizes polar projections to plot the min, max, and mean
        values of the evaluated function over given theta and phi ranges.

        Parameters
        ----------
        n_theta : int, optional
            Number of theta points to use in the grid (default is 500).
        n_psi : int, optional
            Number of psi points to use in the grid (default is 100).
        color_minmax : str, optional
            Color to use for plotting min and max values (default is 'blue').
        alpha_minmax : float, optional
            Alpha transparency level to use for the min/max fill (default is 0.2).
        color_mean : str, optional
            Color to use for plotting mean values (default is 'red').
        fig : matplotlib.figure.Figure, optional
            Handle to existing figure object. Default is None. If provided, it disables showing the figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axs : list of matplotlib.axes._subplots.PolarAxesSubplot
            List of polar axis subplots.
        """
        if fig is None:
            new_fig = plt.figure()
        else:
            new_fig = fig
        theta_polar = np.linspace(0, 2 * np.pi, n_theta)
        titles = ('XY', 'XZ', 'YZ')
        handles, labels = [], []
        axs = []
        for i in range(0, 3):
            ax = new_fig.add_subplot(1, 3, i+1, projection='polar')
            phi, theta, ax = _create_xyz_section(ax, titles[i], theta_polar)
            psi = np.linspace(0, np.pi, n_psi)
            phi_grid, psi_grid = np.meshgrid(phi, psi, indexing='ij')
            theta_grid, _ = np.meshgrid(theta, psi, indexing='ij')
            phi = phi_grid.flatten()
            theta = theta_grid.flatten()
            psi = psi_grid.flatten()
            u, v = sph2cart(phi, theta, psi)
            values = self.eval(u, v).reshape((n_theta, n_psi))
            min_val = np.min(values, axis=1)
            max_val = np.max(values, axis=1)
            ax.plot(theta_polar, min_val, color=color_minmax)
            ax.plot(theta_polar, max_val, color=color_minmax)
            area = ax.fill_between(theta_polar, min_val, max_val, alpha=alpha_minmax, label='Min/Max')
            line, = ax.plot(theta_polar, np.mean(values, axis=1), color=color_mean, label='Mean')
            axs.append(ax)
        handles.extend([line, area])
        labels.extend([line.get_label(), area.get_label()])
        new_fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.95))
        if fig is None:
            new_fig.show()
        return new_fig, axs

    def plot_as_pole_figure(self, n_theta=50, n_phi=200, n_psi=50, which='mean', projection='lambert', fig=None,
                            plot_type='imshow', show=True, title=None, subplot_args=(), subplot_kwargs=None, **kwargs):
        """
        Generate a pole figure plot from spherical function evaluation.

        This function evaluates a spherical function over specified ranges of angles
        (phi, theta, psi) and then generates a 2D pole figure plot using various
        statistical summaries of the data (mean, std, min, max). It also supports
        several types of plot visualizations such as 'imshow', 'contourf', and 'contour'.

        Parameters
        ----------
        n_theta : int, optional
            Number of sampling points for theta angle. Default is 50.
        n_phi : int, optional
            Number of sampling points for phi angle. Default is 200.
        n_psi : int, optional
            Number of sampling points for psi angle. Default is 50.
        which : str, optional
            Specifies the type of statistical summary to use for plotting.
            Options include 'mean', 'std', 'min', 'max'. Default is 'mean'.
        projection : str, optional
            Type of projection for the pole figure plot. Default is 'lambert'.
        fig : matplotlib.figure.Figure, optional
            Pre-existing figure to use for plotting. If None, a new figure is created.
            Default is None.
        plot_type : str, optional
            Type of plot to generate. Can be 'imshow', 'contourf', or 'contour'.
            Default is 'imshow'.
        show : bool, optional
            Set whether to show the plot or not. Default is True. This must be turned off when using multiple subplots.
        subplot_args : tuple, optional
            Arguments to pass to the add_subplot() function. Default is None.
        subplot_kwargs : dict, optional
            Keyword arguments to pass to the add_subplot() function. Default is None.
        **kwargs : dict, optional
            Additional keyword arguments passed to the plotting functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
        if subplot_kwargs is None:
            subplot_kwargs = {}
        if fig is None:
            fig = plt.figure()
        ax = add_polefigure(fig, *subplot_kwargs, projection=projection, **subplot_kwargs)
        phi = np.linspace(*self.domain[0], n_phi)
        theta = np.linspace(*self.domain[1], n_theta)
        psi = np.linspace(*self.domain[2], n_psi)
        phi_grid, theta_grid, psi_grid = np.meshgrid(phi, theta, psi, indexing='ij')
        phi_flat = phi_grid.flatten()
        theta_flat = theta_grid.flatten()
        psi_flat = psi_grid.flatten()
        values = self.eval_spherical(np.array([phi_flat, theta_flat, psi_flat]).T)
        reshaped_values = values.reshape((n_phi, n_theta, n_psi))
        if which == 'std':
            to_plot = np.std(reshaped_values, axis=2)
        elif which == 'min':
            to_plot = np.min(reshaped_values, axis=2)
        elif which == 'max':
            to_plot = np.max(reshaped_values, axis=2)
        else:
            to_plot = np.mean(reshaped_values, axis=2)
        phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
        if plot_type == 'imshow':
            sc = ax.pcolormesh(phi_grid, theta_grid, to_plot, **kwargs)
        elif plot_type == 'contourf':
            sc = ax.contourf(phi_grid, theta_grid, to_plot, **kwargs)
        elif plot_type == 'contour':
            sc = ax.contour(phi_grid, theta_grid, to_plot, **kwargs)
        else:
            raise ValueError(f'Unknown plot type: {plot_type}')
        ax.set_rlim(*self.domain[1])
        ax.set_title(title)
        fig.colorbar(sc)
        if show:
            plt.show()
        return fig, ax