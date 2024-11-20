import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from numpy import cos, sin
from scipy import integrate as integrate
from scipy import optimize


def sph2cart(*args):
    """
    Converts spherical/hyperspherical coordinates to cartesian coordinates.

    Parameters
    ----------
    args : tuple
        (phi, theta) angles for spherical coordinates of direction u, where phi denotes the azimuth from X and theta is
        the colatitude angle from Z.
        If a third argument is passed, it defines the third angle in hyperspherical coordinate system, that is
        the orientation of the second vector v, orthogonal to u.

    Returns
    -------
    np.ndarray
        directions u expressed in cartesian coordinates system.
    tuple of (np.ndarray, np.ndarray)
        direction of the second vector, orthogonal to u, expressed in cartesian coordinates system.
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
    if return_orthogonal:
        if seed is not None:
            # Ensure that the seed used for generated v is not the same as that for u
            # Otherwise, u and v would be equal.
            rng = np.random.default_rng(seed+1)
        u2 = rng.normal(size=(n_evals, 3))
        return u, np.cross(u, u2)
    else:
        return u


class SphericalFunction:
    """
    Class for spherical functions, that is, functions that depend on directions in 3D space.

    Attribute
    ---------
    fun : function to use
    """
    domain = np.array([[0, 2 * np.pi],
                       [0, np.pi / 2]])
    """Bounds to consider in spherical coordinates"""

    def __init__(self, fun):
        """
        Create a spherical function, that is, a function that depends on one direction only.

        Parameters
        ----------
        fun : function
            Function to return
        """
        self.fun = fun

    def __repr__(self):
        val_min, _ = self.min()
        val_max, _ = self.max()
        s = 'Spherical function\n'
        s += 'Min={}, Max={}'.format(val_min, val_max)
        return s

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

    def mean(self, method='exact', n_evals=10000, seed=None):
        """
        Estimate the mean value along all directions in the 3D space.

        Parameters
        ----------
        method : str {'exact', 'Monte Carlo'}
            If 'exact', the full integration is performed over the unit sphere (see Notes). If method=='Monte Carlo', the function is evaluated along a finite set of random directions. The
            Monte Carlo method is usually very fast, compared to the exact method.
        n_evals : int, default 10000
            If method=='Monte Carlo', sets the number of random directions to use.
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
        if method == 'exact':
            def fun(theta, phi):
                return self.eval_spherical(phi, theta) * sin(theta)

            domain = self.domain.flatten()
            q = integrate.dblquad(fun, *domain)
            return q[0] / (2 * np.pi)
        else:
            u = uniform_spherical_distribution(n_evals, seed=seed)
            return np.mean(self.eval(u))

    def var(self, method='exact', n_evals=10000, mean=None, seed=None):
        """
        Estimate the variance along all directions in the 3D space

        Parameters
        ----------
        method : str {'exact', 'Monte Carlo'}
            If 'exact', the full integration is performed over the unit sphere (see Notes). If method=='Monte Carlo', the function is evaluated along a finite set of random directions. The
            Monte Carlo method is usually very fast, compared to the exact method.
        n_evals : int, default 10000
            If method=='Monte Carlo', sets the number of random directions to use.
        mean : float, default None
            If provided, and if method=='exact', skip estimation of mean value and use that provided instead.
        seed : int, default None
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
                mean = self.mean()

            def fun(theta, phi):
                return (self.eval_spherical(phi, theta) - mean) ** 2 * sin(theta)

            domain = self.domain.flatten()
            q = integrate.dblquad(fun, *domain)
            return q[0] / (2 * np.pi)
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

    def plot(self, n_phi=50, n_theta=50, **kwargs):
        """
        3D plotting of a spherical function

        Parameters
        ----------
        n_phi : int, default 50
            Number of azimuth angles (phi) to use for plotting. Default is 50.
        n_theta : int, default 50
            Number of latitude angles (theta) to use for plotting. Default is 50.
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
        fig = plt.figure()
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)
        phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
        phi = phi_grid.flatten()
        theta = theta_grid.flatten()
        u = sph2cart(phi, theta)
        values = self.eval(u)
        u_grid = u.reshape([*phi_grid.shape, 3])
        r_grid = values.reshape(phi_grid.shape)
        ax = _plot3D(fig, u_grid, r_grid, **kwargs)
        plt.show()
        return fig, ax

    def plot_xyz_sections(self, n_theta=500):
        """
        Plot values in X-Y, X-Z and Y-Z planes

        Parameters
        ----------
        n_theta : int, default 500
            Number of values of polar angle to use for plotting

        Returns
        -------
        matplotlib.figure.Figure
            Handle to the figure
        matplotlib.Axes3D
            Handle to axes

        See Also
        --------
        plot : plot the values of the function over the whole unit sphere as a 3D surface
        """
        fig = plt.figure()
        theta_polar = np.linspace(0, 2*np.pi, n_theta)
        titles = ('XY', 'XZ', 'YZ')
        for i in range(0, 3):
            ax = fig.add_subplot(1, 3, i+1, projection='polar')
            angles = np.zeros((n_theta, 2))
            phi, theta, ax = _create_xyz_section(ax, titles[i], theta_polar)
            angles[:, 0] = phi
            angles[:, 1] = theta
            r = self.eval_spherical(angles)
            ax.plot(theta_polar, r)
        fig.show()
        return fig, ax


class HyperSphericalFunction(SphericalFunction):
    """
    Class for defining functions that depend on two orthogonal directions u and v.
    """
    domain = np.array([[0, 2 * np.pi],
                       [0, np.pi / 2],
                       [0, np.pi]])

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

    def mean(self, method='Monte Carlo', n_evals=10000, seed=None):
        if method == 'exact':
            def fun(psi, theta, phi):
                return self.eval_spherical(phi, theta, psi) * sin(theta)

            domain = self.domain.flatten()
            q = integrate.tplquad(fun, *domain)
            return q[0] / (2 * np.pi ** 2)
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

            .. image:: _static/images/HyperSphericalCoordinates.png


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

    def var(self, method='Monte Carlo', n_evals=10000, mean=None, seed=None):
        if method == 'exact':
            if mean is None:
                mean = self.mean()

            def fun(psi, theta, phi):
                return (mean - self.eval_spherical(phi, theta, psi)) ** 2 * sin(theta)

            domain = self.domain.flatten()
            q = integrate.tplquad(fun, *domain)
            return q[0] / (2 * np.pi ** 2)
        else:
            u, v = uniform_spherical_distribution(n_evals, seed=seed, return_orthogonal=True)
            return np.var(self.eval(u, v))

    def plot(self, n_phi=50, n_theta=50, n_psi=50, which='mean', **kwargs):
        """
        3D plotting of a spherical function

        Parameters
        ----------
        n_phi : int, default 50
            Number of azimuth angles (phi) to use for plotting. Default is 50.
        n_theta : int, default 50
            Number of latitude angles (theta) to use for plotting. Default is 50.
        n_psi : int, default 50
            Number of psi value to look for min/max/mean value (see below). Default is 50.
        which : str {'mean', 'std', 'min', 'max'}, default 'mean'
            How to handle the 3rd coordinate. For instance, if which=='mean' (default), for a given value of angles
            (phi, theta), the mean function value over all psi angles is plotted.
        **kwargs
            These arguments will be passed to ax.plot_surface() function.

        Returns
        -------
        matplotlib.figure.Figure
            Handle to the figure
        matplotlib.Axes3D
            Handle to axes

        See Also
        --------
        plot_xyz_sections : plot statistics in X-Y, X-Z and Y-Z planes
        """
        fig = plt.figure()
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)
        psi = np.linspace(0, np.pi, n_psi)
        phi_grid, theta_grid, psi_grid = np.meshgrid(phi, theta, psi, indexing='ij')
        phi = phi_grid.flatten()
        theta = theta_grid.flatten()
        psi = psi_grid.flatten()
        u, v = sph2cart(phi, theta, psi)
        values = self.eval(u, v).reshape((n_phi, n_theta, n_psi))
        if which == 'std':
            r_grid = np.std(values, axis=2)
        elif which == 'min':
            r_grid = np.min(values, axis=2)
        elif which == 'max':
            r_grid = np.max(values, axis=2)
        else:
            r_grid = np.mean(values, axis=2)
        u_grid = u.reshape((n_phi, n_theta, n_psi, 3))
        ax = _plot3D(fig, u_grid[:, :, 0, :], r_grid, **kwargs)
        plt.show()
        return fig, ax

    def plot_xyz_sections(self, n_theta=500, n_psi=100):
        """
        Plot statistics in X-Y, X-Z and Y-Z planes.

        The function value will be plotted as functions of the first direction (u), whereas the second direction (v)
        will be used to evaluate the statistics (min/max and mean).

        Parameters
        ----------
        n_theta : int, default 500
            Number of values of polar angle to use for plotting
        n_psi : int, default 100
            Number of psi value to use for evaluating the statistics (mean, min and max)

        Returns
        -------
        matplotlib.figure.Figure
            Handle to the figure
        matplotlib.Axes3D
            Handle to axes

        See Also
        --------
        plot : plot a given statistic as a 3D surface
        """
        fig = plt.figure()
        theta_polar = np.linspace(0, 2 * np.pi, n_theta)
        titles = ('XY', 'XZ', 'YZ')
        handles, labels = [], []
        for i in range(0, 3):
            ax = fig.add_subplot(1, 3, i+1, projection='polar')
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
            ax.plot(theta_polar, min_val, color='blue')
            ax.plot(theta_polar, max_val, color='blue')
            ax.plot(theta_polar, np.mean(values, axis=1), color='red', label='Mean')
            area = ax.fill_between(theta_polar, min_val, max_val, alpha=0.2, label='Min/Max')
            line, = ax.plot(theta_polar, np.mean(values, axis=1), color='red', label='Mean')
        handles.extend([line, area])
        labels.extend([line.get_label(), area.get_label()])
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.95))
        fig.show()
        return fig, ax
