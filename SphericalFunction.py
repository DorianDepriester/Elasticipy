from inspect import signature

import numpy as np
from numpy import cos, sin
from scipy import integrate as integrate
from scipy.optimize import minimize


def _sph2cart(phi, theta, psi=None):
    u = np.array([cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)])
    if psi is None:
        return u
    else:
        e_phi = np.array([-sin(phi), cos(phi), 0])
        e_theta = np.array([cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)])
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
                      [1, -1, 1]])
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
        sig = signature(fun)
        n_args = len(sig.parameters)
        if domain is None:
            if n_args == 1:
                domain = [[0, 2 * np.pi],
                          [0, np.pi / 2]]
            else:
                domain = [[0, 2 * np.pi],
                          [0, np.pi / 2],
                          [0, np.pi]]
        self.n_args = n_args
        self.domain = domain
        self.fun = fun

    def __repr__(self):
        val_min, _ = self.min
        val_max, _ = self.max
        s = 'Spherical function\n'
        s += 'Min={}, Max={}'.format(val_min, val_max)
        return s

    def eval(self, *args):
        return self.fun(*args)

    def evalsph(self, *args):
        if self.n_args == 1:
            phi, theta = args
            u = _sph2cart(phi, theta)
            return self.eval(u)
        else:
            phi, theta, psi = args
            u, v = _sph2cart(phi, theta, psi)
            return self.eval(u, v)

    @property
    def min(self):
        def fun(x):
            return self.evalsph(*x)

        q = _multistart_minimization(fun, bounds=self.domain)
        val = q.fun
        angles = q.x
        return val, _sph2cart(*angles)

    @property
    def max(self):
        def fun(x):
            return -self.evalsph(*x)

        q = _multistart_minimization(fun, bounds=self.domain)
        val = -q.fun
        angles = q.x
        return val, _sph2cart(*angles)

    def mean(self):
        if self.n_args == 1:
            def fun(theta, phi):
                return self.evalsph(phi, theta) * sin(theta)

            q = integrate.dblquad(fun, 0, 2 * np.pi, 0, np.pi / 2)
            return q[0] / (2 * np.pi)
        else:
            def fun(psi, theta, phi):
                return self.evalsph(phi, theta, psi) * sin(theta)

            q = integrate.tplquad(fun, 0, 2 * np.pi, 0, np.pi / 2, 0, np.pi)
            return q[0] / (2 * np.pi ** 2)

    def std(self, mean=None):
        if mean is None:
            mean = self.mean()
        if self.n_args == 1:
            def fun(theta, phi):
                return (self.evalsph(phi, theta) - mean) ** 2 * sin(theta)

            q = integrate.dblquad(fun, 0, 2 * np.pi, 0, np.pi / 2)
            var = q[0] / (2 * np.pi)
        else:
            def fun(psi, theta, phi):
                return (mean - self.evalsph(phi, theta, psi)) ** 2 * sin(theta)

            q = integrate.tplquad(fun, 0, 2 * np.pi, 0, np.pi / 2, 0, np.pi)
            var = q[0] / (2 * np.pi ** 2)
        return np.sqrt(var)
