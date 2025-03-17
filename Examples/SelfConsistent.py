import numpy as np
from Elasticipy.tensors.fourth_order import FourthOrderTensor
from Elasticipy.tensors.elasticity import StiffnessTensor, ComplianceTensor
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation
from scipy.integrate import dblquad
from scipy.optimize import minimize, Bounds

I = FourthOrderTensor.identity(mapping='Voigt')
global phi, theta


def extract_upper_triangular_stiffness(C):
    upper_triangular = C.matrix[np.triu_indices(6)]
    return upper_triangular

def reconstruct_symmetric_from_1d(upper_triangular_1d):
    reconstructed_matrix = np.zeros((6, 6))
    indices = np.triu_indices(6)
    reconstructed_matrix[indices] = upper_triangular_1d
    reconstructed_matrix = reconstructed_matrix + reconstructed_matrix.T - np.diag(reconstructed_matrix.diagonal())

    return reconstructed_matrix

def gamma(C_macro_local, a1=1, a2=1, a3=1):
    s1 = np.sin(theta)*np.cos(phi) / a1
    s2 = np.sin(theta)*np.sin(phi) / a2
    s3 = np.cos(theta) / a3
    s = [s1, s2, s3]
    D = np.einsum('lmnp,pqr,lqr->qrmn', C_macro_local.full_tensor(), s, s)
    return np.einsum('qrik,jqr,lqr->qrijkl', np.linalg.inv(D), s, s)

def Morris_tensor(C_macro_local):
    g = gamma(C_macro_local)
    gsin = (g.T*np.sin(theta.T)).T
    a = trapezoid(gsin, phi[:,0], axis=0)
    b= trapezoid(a, theta[0], axis=0)/(4*np.pi)
    return b

def gamma_int(C_macro_local, theta1, phi1):
    s1 = np.sin(theta1)*np.cos(phi1)
    s2 = np.sin(theta1)*np.sin(phi1)
    s3 = np.cos(theta1)
    s = [s1, s2, s3]
    D = np.einsum('lmnp,p,l->mn', C_macro_local.full_tensor(), s, s)
    return np.einsum('nr,w,s->nwrs', np.linalg.inv(D), s, s)

def Morris_tensor_int(C_macro_local):
    E = np.zeros((3,3,3,3))
    for n in range(3):
        for w in range(3):
            for r in range(3):
                for s in range(3):
                    def fun(phi_y, theta_x):
                        return np.sin(theta_x) * gamma_int(C_macro_local, theta_x, phi_y)[n,w,r,s]
                    E[n,w,r,s] = 1/(4*np.pi) * dblquad(fun, 0, np.pi, 0, 2*np.pi)[0]
    return E

def localization_tensor(C_macro_local, C_incl):
    E = Morris_tensor(C_macro_local)
    delta = FourthOrderTensor(C_incl.full_tensor() - C_macro_local.full_tensor())
    Ainv = FourthOrderTensor(np.einsum('ijmn,mnkl->ijkl', E, delta.full_tensor())) + I
    Sesh= np.einsum('ijmn,mnkl->ijkl', E, C_macro_local.full_tensor())
    nu = C_macro_local.Poisson_ratio.mean()
    beta = 2*(4-5*nu) / (15 * (1 - nu))
    Sesh_th_full = (1-2*beta) * np.einsum('ij,kl->ijkl', np.eye(3), np.eye(3)) + beta * I.full_tensor()
    Sesh_th = FourthOrderTensor(Sesh_th_full, mapping='Voigt')
    return Ainv.inv().full_tensor()

def global_spherical_grid(n_theta=100, n_phi=100):
    global phi, theta
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    phi, theta = np.meshgrid(phi, theta, indexing='ij')

def Kroner_Eshelby(Ci, g, max_iter=5, atol=1e-3, rtol=1e-4, display=False):
    Ci_rotated = (Ci * g)
    C_macro = Ci.Hill_average()
    eigen_stiff = C_macro.eig_stiffnesses
    global_spherical_grid()
    keep_on = True
    k = 0
    message = 'Maximum number of iterations is reached'
    m = len(g)
    A_local = FourthOrderTensor.zeros(m)
    while keep_on:
        eigen_stiff_old = eigen_stiff
        C_macro_local = C_macro * (g.inv())
        for i in range(m):
            A_local[i] = localization_tensor(C_macro_local[i], Ci)
        A = A_local * g
        CiAi = Ci_rotated.ddot(A)
        CiAi_mean = CiAi.mean()
        C_macro = StiffnessTensor(CiAi_mean.full_tensor(), force_symmetry=True)

        # Stopping criteria
        eigen_stiff = C_macro.eig_stiffnesses
        abs_change = np.abs(eigen_stiff - eigen_stiff_old)
        rel_change = np.max(abs_change / eigen_stiff_old)
        max_abs_change = np.max(abs_change)
        k += 1
        if  max_abs_change < atol:
            keep_on = False
            message = 'Absolute change is below threshold value'
        if rel_change < rtol:
            keep_on = False
            message = 'Relative change is below threshold value'
        if k == max_iter:
            keep_on = False
        if display:
            err = A.mean() - FourthOrderTensor.identity()
            err = np.max(np.abs(err.matrix))
            print('Iter #{}: abs. change={:0.5f}; rel. change={:0.5f}; error={:0.5f}'.format(k, max_abs_change, rel_change,err))
    return C_macro, message


def KE_iteration(Cmacro_flat, Ci, g):
    C_matrix = reconstruct_symmetric_from_1d(Cmacro_flat)
    C_macro = StiffnessTensor(C_matrix)
    global_spherical_grid()
    m = len(g)
    A_local = FourthOrderTensor.zeros(m)
    C_macro_local = C_macro * (g.inv())
    for i in range(m):
        A_local[i] = localization_tensor(C_macro_local[i], Ci)
    CiAi_local = Ci.ddot(A_local)
    CiAi = CiAi_local * g
    CiAi_mean = CiAi.mean()
    C_macro_new = StiffnessTensor(CiAi_mean.full_tensor(), force_symmetry=True)
    return C_macro_new
 #   return np.sum((C_macro_new.matrix - C_matrix)**2)

Cstrip = StiffnessTensor.transverse_isotropic(Ex= 10.2, Ez=146.8, nu_zx=0.274, nu_yx=0.355, Gxz=7)
Cstrip = Cstrip * Rotation.from_euler('Y', 90, degrees=True)
orientations = Rotation.random(100)

Ccub = StiffnessTensor.cubic(C11=110, C12=10, C44=44)
C_stress, reason = Kroner_Eshelby(Cstrip, orientations, max_iter=50, rtol=1e-6, atol=1e-5, display=True)

# C_rotated = (Cstrip * orientations)
# C0 = C_rotated.Hill_average()
# C0_triu = extract_upper_triangular_stiffness(C0)
# Cmin_flat = C_rotated.Reuss_average().matrix.flatten()
# Cmax_flat = C_rotated.Voigt_average().matrix.flatten()
# unknown_bound = Cmin_flat > Cmax_flat
# Cmin_flat[unknown_bound] = -np.inf
# Cmax_flat[unknown_bound] = np.inf
# bounds = Bounds(Cmin_flat, Cmax_flat)
#
# def fun(C):
#     return KE_iteration(C, Cstrip, orientations)
#
# def print_inter(x):
#     print(reconstruct_symmetric_from_1d(x)[0,0])
#
# m = minimize(fun, C0_triu, options={'disp':True}, callback=print_inter, tol=1e-3)