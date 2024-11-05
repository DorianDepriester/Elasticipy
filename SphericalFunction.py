from inspect import signature

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from numpy import cos, sin
from scipy import integrate as integrate
from scipy.optimize import minimize


def _sph2cart(phi, theta, psi=None):
    phi_vec = np.array(phi).flatten()
    theta_vec = np.array(theta).flatten()
    u = np.array([cos(phi_vec) * sin(theta_vec), sin(phi_vec) * sin(theta_vec), cos(theta_vec)]).T
    if psi is None:
        return u
    else:
        e_phi = np.array([-sin(phi_vec), cos(phi_vec), np.zeros(phi_vec.shape)])
        e_theta = np.array([cos(theta_vec) * cos(phi_vec), cos(theta_vec) * sin(phi_vec), -sin(theta_vec)])
        v = cos(psi) * e_phi + sin(psi) * e_theta
        return u, v.T


def _cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return phi, theta


def _multistart_minimization(fun, bounds):
    # Ensure that the initial guesses are uniformly
    # distributed over the half unit sphere
    xyz_0 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [1, 1, 1],
                      [-1, 1, 1],
                      [1, -1, 1],
                      [1, 1, -1],
                      [-1, -1, 1],
                      [-1, 1, -1],
                      [1, -1, -1]])
    phi_theta_0 = _cart2sph(*xyz_0.T)
    angles_0 = np.transpose(phi_theta_0)
    if len(bounds) == 3:
        psi_0 = np.array([[0, np.pi / 2, np.pi]]).T
        phi_theta_0 = np.tile(angles_0, (len(psi_0), 1))
        psi_0 = np.repeat(psi_0, len(angles_0), axis=0)
        angles_0 = np.hstack((phi_theta_0, psi_0))
    best_result = None
    for x0 in angles_0:
        result = minimize(fun, x0, method='L-BFGS-B', bounds=bounds)
        if best_result is None or (result.fun < best_result.fun):
            best_result = result
    return best_result


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


class SphericalFunction:
    def __init__(self, fun, domain=None):
        if domain is None:
            domain = [[0, 2 * np.pi],
                      [0, np.pi / 2]]
        self.domain = np.array(domain)
        self.fun = fun

    def __repr__(self):
        val_min, _ = self.min
        val_max, _ = self.max
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
            If only one direction is given as a tuple of floats [nx, ny, nz], the result is a float;
            otherwise, the result is a nd.array.
        """
        values = self.fun(u)
        if isinstance(u, list) and np.array(u).shape == (3,):
            return values[0]
        else:
            return values

    def eval_spherical(self, *args):
        """
        Evaluate value along a given (set of) direction(s) defined by its (their) spherical coordinates.

        Parameters
        ----------
        args : list or np.ndarray
            [phi, theta] where phi denotes the azimuth angle from X axis,
            and theta is the latitude angle from Z axis (theta==0 -> Z axis).

        Returns
        -------
        float or np.ndarray
            If only one direction is given as a tuple of floats [nx, ny, nz], the result is a float;
        otherwise, the result is a nd.array.
        """
        angles = np.atleast_2d(args)
        phi, theta = angles.T
        u = _sph2cart(phi, theta)
        values = self.eval(u)
        if (np.array(args).shape == (2,) or np.array(args).shape == (1, 2)) and not isinstance(args, np.ndarray):
            return values[0]
        else:
            return values

    @property
    def min(self):
        """
        Find minimum value of the function.

        Returns
        -------
        tuple
            Minimum value and location where it is reached (direction)
            min[0] (float): minimum value
            min[1] (np.ndarray): direction along which this value is reached
        """
        def fun(x):
            return self.eval_spherical(*x)

        q = _multistart_minimization(fun, bounds=self.domain)
        val = q.fun
        angles = q.x
        return val, _sph2cart(*angles)

    @property
    def max(self):
        """
        Find maximum value of the function.

        Returns
        -------
        tuple
            Minimum value and location where it is reached (direction)
            max[0] (float): maximum value
            max[1] (np.ndarray): direction along which this value is reached
        """
        def fun(x):
            return -self.eval_spherical(*x)

        q = _multistart_minimization(fun, bounds=self.domain)
        val = -q.fun
        angles = q.x
        return val, _sph2cart(*angles)

    def mean(self):
        """
        Estimate the mean value along all directions in the 3D space

        Returns
        -------
        float
            Mean value
        """
        def fun(theta, phi):
            return self.eval_spherical(phi, theta) * sin(theta)

        domain = self.domain.flatten()
        q = integrate.dblquad(fun, *domain)
        return q[0] / (2 * np.pi)

    def var(self, mean=None):
        """
        Estimate the variance along all directions in the 3D space

        Parameters
        ----------
        mean : float, optional
            If provided, skip estimation of mean value and use that provided instead.

        Returns
        -------
        float
            Variance of the functio
        """
        if mean is None:
            mean = self.mean()

        def fun(theta, phi):
            return (self.eval_spherical(phi, theta) - mean) ** 2 * sin(theta)

        domain = self.domain.flatten()
        q = integrate.dblquad(fun, *domain)
        return q[0] / (2 * np.pi)

    def std(self, **kwargs):
        """
        Standard deviation of the function along all directions in the 3D space.

        Parameters
        ----------
        kwargs : optional
            Keyword arguments passed-by to var()

        Returns
        -------
        float
            Standard deviation
        """
        return np.sqrt(self.std(**kwargs))

    def plot(self, n_phi=50, n_theta=50, **kwargs):
        """
        3D plotting of a spherical function

        Parameters
        ----------
        n_phi : int, optional
            Number of azimuth angles (phi) to use for plotting. Default is 50.
        n_theta : int, optional
            Number of latitude angles (theta) to use for plotting. Default is 50.
        **kwargs : keyword arguments passed-by to ax.plot_surface()

        Returns
        -------
        matplotlib.figure.Figure
            Handle to the figure
        """
        fig = plt.figure()
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)
        phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
        phi = phi_grid.flatten()
        theta = theta_grid.flatten()
        u = _sph2cart(phi, theta)
        values = self.eval(u)
        u_grid = u.reshape([*phi_grid.shape, 3])
        r_grid = values.reshape(phi_grid.shape)
        ax = _plot3D(fig, u_grid, r_grid, **kwargs)
        plt.show()
        return fig, ax


class HyperSphericalFunction(SphericalFunction):
    def __init__(self, fun, domain=None):
        if domain is None:
            domain = [[0, 2 * np.pi],
                      [0, np.pi / 2],
                      [0, np.pi]]
        super().__init__(fun, domain=domain)

    def eval(self, u, *args):
        values = self.fun(u, *args)
        if np.array(u).shape == (3,) and not isinstance(u, np.ndarray):
            return values[0]
        else:
            return values

    def mean(self):
        def fun(psi, theta, phi):
            return self.eval_spherical(phi, theta, psi) * sin(theta)

        domain = self.domain.flatten()
        q = integrate.tplquad(fun, *domain)
        return q[0] / (2 * np.pi ** 2)

    def _psi_min(self, phi, theta):
        def fun(psi):
            return self.eval_spherical(phi, theta, psi[0])

        result = minimize(fun, np.pi, bounds=[self.domain[2]])
        return result.fun

    def _psi_max(self, phi, theta):
        def fun(psi):
            return -self.eval_spherical(phi, theta, psi)

        result = minimize(fun, np.pi, bounds=[self.domain[2]])
        return -result.fun

    def eval_spherical(self, *args):
        angles = np.atleast_2d(args)
        phi, theta, psi = angles.T
        u, v = _sph2cart(phi, theta, psi)
        values = self.eval(u, v)
        if np.array(args).shape == (3,) and not isinstance(args, np.ndarray):
            return values[0]
        else:
            return values

    def var(self, mean=None):
        if mean is None:
            mean = self.mean()

        def fun(psi, theta, phi):
            return (mean - self.eval_spherical(phi, theta, psi)) ** 2 * sin(theta)

        domain = self.domain.flatten()
        q = integrate.tplquad(fun, *domain)
        return q[0] / (2 * np.pi ** 2)

    def plot(self, n_phi=50, n_theta=50, n_psi=50, which='mean', **kwargs):
        """
        3D plotting of a spherical function

        Parameters
        ----------
        n_phi : int, optional
            Number of azimuth angles (phi) to use for plotting. Default is 50.
        n_theta : int, optional
            Number of latitude angles (theta) to use for plotting. Default is 50.
        **kwargs : keyword arguments passed-by to ax.plot_surface()

        Returns
        -------
        matplotlib.figure.Figure
            Handle to the figure
        """
        fig = plt.figure()
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)
        psi = np.linspace(0, np.pi, n_psi)
        phi_grid, theta_grid, psi_grid = np.meshgrid(phi, theta, psi, indexing='ij')
        phi = phi_grid.flatten()
        theta = theta_grid.flatten()
        psi = psi_grid.flatten()
        u, v = _sph2cart(phi, theta, psi=psi)
        values = self.eval(u, v).reshape((n_phi, n_theta, n_psi))
        if which == 'mean':
            r_grid = np.mean(values, axis=2)
        elif which == 'min':
            r_grid = np.min(values, axis=2)
        else:
            r_grid = np.max(values, axis=2)
        u_grid = u.reshape((n_phi, n_theta, n_psi, 3))
        ax = _plot3D(fig, u_grid[:, :, 0, :], r_grid, **kwargs)
        plt.show()
        return fig, ax
