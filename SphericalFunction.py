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
        e_phi = np.array([-sin(phi_vec), cos(phi_vec), 0])
        e_theta = np.array([cos(theta_vec) * cos(phi_vec), cos(theta_vec) * sin(phi_vec), -sin(theta_vec)])
        v = cos(psi) * e_phi + sin(psi) * e_theta
    return u, v


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

    def eval(self, *args):
        return self.fun(*args)

    def eval_spherical(self, *args):
        phi, theta = args
        u = _sph2cart(phi, theta)
        return self.eval(u)

    @property
    def min(self):
        def fun(x):
            return self.eval_spherical(*x)

        q = _multistart_minimization(fun, bounds=self.domain)
        val = q.fun
        angles = q.x
        return val, _sph2cart(*angles)

    @property
    def max(self):
        def fun(x):
            return -self.eval_spherical(*x)

        q = _multistart_minimization(fun, bounds=self.domain)
        val = -q.fun
        angles = q.x
        return val, _sph2cart(*angles)

    def mean(self):
        def fun(theta, phi):
            return self.eval_spherical(phi, theta) * sin(theta)

        domain = self.domain.flatten()
        q = integrate.dblquad(fun, *domain)
        return q[0] / (2 * np.pi)

    def var(self, mean=None):
        if mean is None:
            mean = self.mean()

        def fun(theta, phi):
            return (self.eval_spherical(phi, theta) - mean) ** 2 * sin(theta)

        domain = self.domain.flatten()
        q = integrate.dblquad(fun, *domain)
        return q[0] / (2 * np.pi)

    def std(self, **kwargs):
        return np.sqrt(self.std(**kwargs))

    def _plot(self, funs, fig, n_phi=50, n_theta=50, opacities=None):
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi, n_theta)
        val = np.zeros((n_phi, n_theta))
        phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
        for k, fun in enumerate(funs):
            x, y, z = _sph2cart(phi_grid, theta_grid)
            for i in range(n_phi):
                for j in range(n_theta):
                    val[i, j] = fun(phi_grid[i,j], theta_grid[i,j])
            x *= val
            y *= val
            z *= val
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            val_min, _ = self.min
            val_max, _ = self.max
            norm = Normalize(vmin=val_min, vmax=val_max)
            colors = cm.viridis(norm(val))
            if opacities is None:
                alpha = 1.0
            else:
                alpha = opacities[k]
            ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, antialiased=False, alpha=alpha)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax, norm

    def plot(self, **kwargs):
        fig = plt.figure()
        ax, norm = self._plot([self.eval_spherical], fig, **kwargs)
        mappable = cm.ScalarMappable(cmap='viridis', norm=norm)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax)
        plt.show()
        return fig


class HyperSphericalFunction(SphericalFunction):
    def __init__(self, fun, domain=None):
        if domain is None:
            domain = [[0, 2 * np.pi],
                      [0, np.pi / 2],
                      [0, np.pi]]
        super().__init__(fun, domain=domain)

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
        phi, theta, psi = args
        u, v = _sph2cart(phi, theta, psi)
        return self.eval(u, v)

    def var(self, mean=None):
        if mean is None:
            mean = self.mean()

        def fun(psi, theta, phi):
            return (mean - self.eval_spherical(phi, theta, psi)) ** 2 * sin(theta)

        domain = self.domain.flatten()
        q = integrate.tplquad(fun, *domain)
        return q[0] / (2 * np.pi ** 2)

    def plot(self, n_phi=50, n_theta=50, **kwargs):
        fig = plt.figure()
        ax, norm = self._plot([self._psi_min], fig,
                              n_phi=n_phi, n_theta=n_theta, opacities=[1.0, 0.2])
        plt.show()
        return fig