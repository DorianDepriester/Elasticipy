import numpy as np
from Elasticipy.tensors.fourth_order import FourthOrderTensor
from Elasticipy.tensors.elasticity import StiffnessTensor
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation

I = FourthOrderTensor.identity()

def gamma(C_macro_local, phi, theta, a1, a2, a3):
    s1 = np.sin(theta)*np.cos(phi) / a1
    s2 = np.sin(theta)*np.sin(phi) / a2
    s3 = np.cos(theta) / a3
    s = np.array([s1, s2, s3])
    D = np.einsum('lmnp,pqr,lqr->qrmn', C_macro_local.full_tensor(), s, s)
    return np.einsum('qrik,jqr,lqr->qrijkl', np.linalg.inv(D), s, s)

def polarization_tensor(C_macro_local, phi, theta, a1, a2, a3):
    g = gamma(C_macro_local, phi, theta, a1, a2, a3)
    gsin = (g.T*np.sin(theta.T)).T
    a = trapezoid(gsin, phi[:,0], axis=0)
    b= trapezoid(a, theta[0], axis=0)/(4*np.pi)
    return b

def localization_tensor(C_macro_local, C_incl, phi, theta):
    E = polarization_tensor(C_macro_local, phi, theta, a1=1, a2=1, a3=1)
    delta = C_incl.full_tensor() - C_macro_local.full_tensor()
    Ainv = FourthOrderTensor(np.einsum('ijmn,mnkl->ijkl', E, delta)) + I
    return Ainv.inv().full_tensor()

def spherical_grid(n_theta=50, n_phi=100):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    return np.meshgrid(phi, theta, indexing='ij')

def Kroner_Eshelby(Ci, g, max_iter=5, atol=1e-3, rtol=1e-3, display=False):
    phi, theta = spherical_grid()
    Ci_rotated = (Ci * g)
    C_macro = Ci_rotated.Hill_average()
    eigen_stiff = C_macro.eig_stiffnesses
    keep_on = True
    k = 0
    message = 'Maximum number of iterations is reached'
    m = len(g)
    A_local = FourthOrderTensor.zeros(m)
    while keep_on:
        eigen_stiff_old = eigen_stiff
        C_macro_local = C_macro * (g.inv())
        for i in range(m):
            A_local[i] = localization_tensor(C_macro_local[i], Ci, phi, theta)
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

Cstrip = StiffnessTensor.transverse_isotropic(Ex= 10.2, Ez=146.8, nu_zx=0.274, nu_yx=0.355, Gxz=7)
Cstrip = Cstrip * Rotation.from_euler('Y', 90, degrees=True)
#orientations = Rotation.from_euler('Z', np.linspace(0, 180, 10, endpoint=False), degrees=True)
orientations = Rotation.random(10, random_state=1234)

C_stress, reason = Kroner_Eshelby(Cstrip, orientations, display=True)
