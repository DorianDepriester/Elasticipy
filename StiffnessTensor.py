import numpy as np
from numpy import cos, sin
import scipy.integrate as integrate
from scipy.optimize import minimize
import re


def _sph2cart(phi, theta, psi=None):
    u = np.array([cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)])
    if psi is None:
        return u
    else:
        e_phi = np.array([-sin(phi), cos(phi), 0])
        e_theta = np.array([cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)])
        v = cos(psi) * e_phi + sin(psi) * e_theta
    return np.vstack((u, v))


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
        psi_0 = np.array([[0, np.pi/2, np.pi]]).T
        phi_theta_0 = np.tile(angles_0, (len(psi_0), 1))
        psi_0 = np.repeat(psi_0, len(angles_0), axis=0)
        angles_0 = np.hstack((phi_theta_0, psi_0))
    best_result = None
    for x0 in angles_0:
        result = minimize(fun, x0, method='L-BFGS-B', bounds=bounds)
        if best_result is None or (result.fun < best_result.fun):
            best_result = result
    return best_result


def _parse_tensor_components(prefix, **kwargs):
    pattern = r'^{}(\d{{2}})$'.format(prefix)
    value = dict()
    for k, v in kwargs.items():
        match = re.match(pattern, k)  # Extract 'C11' to '11' and so
        if match:
            value[match.group(1)] = v
    return value


def voigt(i, j):
    voigt_mat = np.array([[0, 5, 4],
                          [5, 1, 3],
                          [4, 3, 2]])
    return voigt_mat[i, j]


def unvoigt(i):
    inverse_voigt_mat = np.array([[0, 0],
                                  [1, 1],
                                  [2, 2],
                                  [1, 2],
                                  [0, 2],
                                  [0, 1]])
    return inverse_voigt_mat[i]


def _compute_unit_strain_along_direction(S, m, n):
    if not isinstance(S, ComplianceTensor):
        S = S.inv()
    if (np.linalg.norm(m) < 1e-9) or (np.linalg.norm(n) < 1e-9):
        raise ValueError('The input vector cannot be zeros')
    m = m / np.linalg.norm(m)
    n = n / np.linalg.norm(n)

    indices = np.indices((3, 3, 3, 3))
    i, j, k, ell = indices[0], indices[1], indices[2], indices[3]
    cosine = m[i] * n[j] * m[k] * n[ell]
    return np.sum(cosine * S.full_tensor())


def tensorFromCrystalSymmetry(symmetry='Triclinic', point_group=None, tensor='Stiffness', unit='GPa', **kwargs):
    tensor = tensor.lower()
    if tensor == 'stiffness':
        prefix = 'C'
        k = 1/2
    else:
        prefix = 'S'
        k = 2
    values = _parse_tensor_components(prefix, **kwargs)
    C = np.zeros((6, 6))
    symmetry = symmetry.lower()
    if ((symmetry == 'tetragonal') or (symmetry == 'trigonal')) and (point_group is None):
        raise ValueError('For tetragonal and trigonal symmetries, the point group is mandatory.')
    tetra_1 = ['-42m', '422', '4mm', '4/mm']
    tetra_2 = ['4', '-4', '4m']
    trigo_1 = ['32', '3m', '-3m']
    trigo_2 = ['3', '-3']
    if point_group is not None:
        if (point_group in tetra_1) or (point_group in tetra_2):
            symmetry = 'tetragonal'
        elif (point_group in trigo_1) or (point_group in trigo_1):
            symmetry = 'trigonal'
    try:
        if symmetry == 'isotropic':
            C[0, 0] = C[1, 1] = C[2, 2] = values['11']
            C[0, 1] = C[0, 2] = C[1, 2] = values['12']
            C[3, 3] = C[4, 4] = C[5, 5] = (C[0, 0] - C[0, 1]) * k
        elif symmetry == 'cubic':
            C[0, 0] = C[1, 1] = C[2, 2] = values['11']
            C[0, 1] = C[0, 2] = C[1, 2] = values['12']
            C[3, 3] = C[4, 4] = C[5, 5] = values['44']
        elif symmetry == 'hexagonal':
            C[0, 0] = C[1, 1] = values['11']
            C[0, 1] = values['12']
            C[0, 2] = C[1, 2] = values['13']
            C[2, 2] = values['33']
            C[3, 3] = C[4, 4] = values['44']
            C[5, 5] = (C[0, 0] - C[0, 1]) * k
        elif symmetry == 'tetragonal':
            C[0, 0] = C[1, 1] = values['11']
            C[0, 1] = values['12']
            C[0, 2] = C[1, 2] = values['13']
            C[2, 2] = values['33']
            C[3, 3] = C[4, 4] = values['44']
            C[5, 5] = values['66']
            if point_group in tetra_2:
                C[0, 5] = values['16']
                C[1, 5] = -C[0, 5]
        elif symmetry == 'trigonal':
            C[0, 0] = C[1, 1] = values['11']
            C[0, 1] = values['12']
            C[0, 2] = C[1, 2] = values['13']
            C[0, 3] = C[4, 5] = values['14']
            C[1, 3] = -C[0, 3]
            C[2, 2] = values['33']
            C[3, 3] = C[4, 4] = values['44']
            C[5, 5] = (C[0, 0] - C[0, 1]) * k
            if point_group in trigo_2:
                C[1, 4] = values['25']
                C[3, 5] = C[1, 4]
                C[0, 4] = -C[1, 4]
        else:  # Orthorombic, monoclinic or triclinic
            C[0, 0], C[0, 1], C[0, 2] = values['11'], values['12'], values['13']
            C[1, 1], C[1, 2] = values['22'], values['23']
            C[2, 2] = values['33']
            C[3, 3], C[4, 4], C[5, 5] = values['44'], values['55'], values['66']
            if (symmetry == 'monoclinic') or (symmetry == 'triclinic'):
                C[0, 5], C[1, 5], C[2, 5] = values['C16'], values['C26'], values['C36']
                C[3, 4] = values['C45']
                if symmetry == 'triclinic':
                    C[0, 3], C[1, 3], C[2, 3] = values['14'], values['24'], values['34']
                    C[0, 4], C[1, 4], C[2, 4] = values['15'], values['25'], values['35']
                    C[3, 5], C[4, 5] = values['46'], values['56']
        if tensor == 'stiffness':
            return StiffnessTensor(C + np.tril(C.T, -1), stress_unit=unit)
        else:
            return ComplianceTensor(C + np.tril(C.T, -1), stress_unit=unit)
    except KeyError as key:
        entry_error = prefix + key.args[0]
        if (symmetry == 'tetragonal') or (symmetry == 'trigonal'):
            err_msg = "For point group {}, keyword argument {} is required".format(point_group, entry_error)
        else:
            err_msg = "For {} symmetry, keyword argument {} is required".format(symmetry, entry_error)
        raise ValueError(err_msg)


class SymmetricTensor:
    tensor_name = 'Symmetric'
    voigt_map = np.ones((6, 6))

    def __init__(self, M):
        self.matrix = M

    def __repr__(self):
        heading = '{} tensor (in Voigt notation):\n'.format(self.tensor_name)
        return heading + self.matrix.__str__()

    def full_tensor(self):
        i, j, k, ell = np.indices((3, 3, 3, 3))
        ij = voigt(i, j)
        kl = voigt(k, ell)
        m = self.matrix[ij, kl] / self.voigt_map[ij, kl]
        return m

    def rotate(self, m):
        rotated_tensor = np.einsum('im,jn,ko,lp,mnop->ijkl', m, m, m, m, self.full_tensor())
        ij, kl = np.indices((6, 6))
        i, j = unvoigt(ij).T
        k, ell = unvoigt(kl).T
        rotated_matrix = rotated_tensor[i, j, k, ell] * self.voigt_map[ij, kl]
        return self.__class__(rotated_matrix)


class StiffnessTensor(SymmetricTensor):
    tensor_name = 'Stiffness'

    def __init__(self, S, stress_unit='GPa'):
        super().__init__(S)
        self.unit = stress_unit

    def __repr__(self):
        print_tensor = super().__repr__()
        return print_tensor + ' {}'.format(self.unit)

    def inv(self):
        C = np.linalg.inv(self.matrix)
        return ComplianceTensor(C, stress_unit=self.unit)

    @property
    def Young_modulus(self):
        return YoungModulus(self, unit=self.unit)

    @property
    def shear_modulus(self):
        return ShearModulus(self, unit=self.unit)


class ComplianceTensor(StiffnessTensor):
    tensor_name = 'Compliance'
    voigt_map = np.vstack((
        np.hstack((np.ones((3, 3)), 2*np.ones((3, 3)))),
        np.hstack((2*np.ones((3, 3)), 4*np.ones((3, 3))))
    ))

    def __init__(self, C, stress_unit='TPa'):
        super().__init__(C, stress_unit=stress_unit)

    def __repr__(self):
        print_tensor = super(type(self).__bases__[0], self).__repr__()
        return print_tensor + ' /{}'.format(self.unit)

    def inv(self):
        S = np.linalg.inv(self.matrix)
        return StiffnessTensor(S, stress_unit=self.unit)


class SphericalFunction:
    min = 0, (0., 0., 0.)
    max = np.inf, (0., 0., 0.)

    def __repr__(self):
        val_min, _ = self.min
        val_max, _ = self.max
        s = 'Spherical function\n'
        s += 'Min={}, Max={}'.format(val_min, val_max)
        return s


class YoungModulus(SphericalFunction):
    def __init__(self, tensor, unit='GPa'):
        def compute_young_modulus(n):
            eps = _compute_unit_strain_along_direction(tensor, n, n)
            return 1 / eps

        self.E = compute_young_modulus
        self.unit = unit

    def eval(self, n):
        return self.E(n)

    def evalsph(self, phi, theta):
        return self.eval(_sph2cart(phi, theta))

    @property
    def min(self):
        def fun(x):
            return self.evalsph(*x)
        q = _multistart_minimization(fun, bounds=[[0, 2*np.pi], [0, np.pi/2]])
        Emin = q.fun
        phi, theta = q.x
        return Emin, _sph2cart(phi, theta)

    @property
    def max(self):
        def fun(x):
            return -self.evalsph(*x)
        q = _multistart_minimization(fun, bounds=[[0, 2*np.pi], [0, np.pi/2]])
        Emax = -q.fun
        phi, theta = q.x
        return Emax, _sph2cart(phi, theta)

    def mean(self):
        def fun(theta, phi):
            return self.evalsph(phi, theta) * sin(theta)
        q = integrate.dblquad(fun, 0, 2 * np.pi, 0, np.pi / 2)
        return q[0] / (2 * np.pi)

    def std(self, Emean=None):
        if Emean is None:
            Emean = self.mean()

        def fun(theta, phi):
            return (self.evalsph(phi, theta) - Emean) ** 2 * sin(theta)
        q = integrate.dblquad(fun, 0, 2 * np.pi, 0, np.pi / 2)
        var = q[0] / (2 * np.pi)
        return np.sqrt(var)


class ShearModulus(SphericalFunction):
    def __init__(self, S, unit='GPa'):
        def compute_shear_modulus(m, n):
            eps = _compute_unit_strain_along_direction(S, m, n)
            return 1 / (4 * eps)
        self.G = compute_shear_modulus
        self.unit = unit

    def eval(self, u, v):
        if np.abs(np.dot(u, v)) > 1e-9:
            raise ValueError('The input vectors must be orthogonal.')
        return self.G(u, v)

    def evalsph(self, phi, theta, psi):
        uv = _sph2cart(phi, theta, psi=psi)
        return self.eval(*uv)

    @property
    def min(self):
        def fun(x):
            return self.evalsph(*x)
        q = _multistart_minimization(fun, bounds=[[0, 2*np.pi], [0, np.pi/2], [0, np.pi]])
        Gmin = q.fun
        phi, theta, psi = q.x
        return Gmin, _sph2cart(phi, theta, psi=psi)

    @property
    def max(self):
        def fun(x):
            return -self.evalsph(*x)
        q = _multistart_minimization(fun, bounds=[[0, 2*np.pi], [0, np.pi/2], [0, np.pi]])
        Gmax = -q.fun
        phi, theta, psi = q.x
        return Gmax, _sph2cart(phi, theta, psi=psi)

    def mean(self):
        def fun(psi, theta, phi):
            return self.evalsph(phi, theta, psi) * sin(theta)

        q = integrate.tplquad(fun, 0, 2 * np.pi, 0, np.pi / 2, 0, np.pi)
        return q[0] / (2 * np.pi**2)

    def std(self, Gmean=None):
        if Gmean is None:
            Gmean = self.mean()

        def fun(psi, theta, phi):
            return (Gmean - self.evalsph(phi, theta, psi))**2 * sin(theta)

        q = integrate.tplquad(fun, 0, 2 * np.pi, 0, np.pi / 2, 0, np.pi)
        var = q[0] / (2 * np.pi**2)
        return np.sqrt(var)
