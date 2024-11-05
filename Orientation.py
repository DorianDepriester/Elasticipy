import numpy as np
from collections.abc import Iterable
from scipy.stats import special_ortho_group


def _rotation_matrix_z(phi):
    M = np.zeros((len(phi), 3, 3))
    M[:, 0, 0] = np.cos(phi)
    M[:, 0, 1] = np.sin(phi)
    M[:, 1, 0] = -np.sin(phi)
    M[:, 1, 1] = np.cos(phi)
    M[:, 2, 2] = 1
    return M


def _rotation_matrix_x(Phi):
    M = np.zeros((len(Phi), 3, 3))
    M[:, 0, 0] = 1
    M[:, 1, 1] = np.cos(Phi)
    M[:, 1, 2] = np.sin(Phi)
    M[:, 2, 1] = -np.sin(Phi)
    M[:, 2, 2] = np.cos(Phi)
    return M


def EulerAngles(phi1, Phi, phi2, degrees=False):
    phi1_vec = np.atleast_1d(phi1)
    Phi_vec = np.atleast_1d(Phi)
    phi2_vec = np.atleast_1d(phi2)
    if degrees:
        phi1_vec, Phi_vec, phi2_vec = np.radians([phi1_vec, Phi_vec, phi2_vec])
    M1 = _rotation_matrix_z(phi1_vec)
    M2 = _rotation_matrix_x(Phi_vec)
    M3 = _rotation_matrix_z(phi2_vec)
    M = np.matmul(np.matmul(M3, M2), M1)
    if isinstance(phi1, Iterable):
        return M
    else:
        return M[0]


def random_orientation(n=1):
    return special_ortho_group.rvs(3, size=n)
