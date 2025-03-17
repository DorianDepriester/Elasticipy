import numpy as np
from Elasticipy.tensors.fourth_order import StiffnessTensor, ComplianceTensor, FourthOrderTensor
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation

I = FourthOrderTensor.identity()
#I = np.einsum('ik,jl->ijkl', np.eye(3), np.eye(3))
global phi, theta

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
    return FourthOrderTensor(b)

def localization_tensor(C_macro_local, C_incl):
    E = Morris_tensor(C_macro_local)
    delta = FourthOrderTensor(C_incl.full_tensor() - C_macro_local)
    Ainv = E.ddot(delta) + I
    return Ainv.inv()

def global_spherical_grid(n_theta=50, n_phi=100):
    global phi, theta
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    phi, theta = np.meshgrid(phi, theta, indexing='ij')

def Kroner_Eshelby(C, g, method='stress', max_iter=50, atol=1e-3, rtol=1e-4, display=False):
    C_rotated = C * g
    C_macro = StiffnessTensor.isotropic(E=100, nu=0.3)
    eigen_stiff = C_macro.eig_stiffnesses
    global_spherical_grid()
    keep_on = True
    k = 0
    message = 'Maximum number of iterations is reached'
    m = len(g)
    A_local = FourthOrderTensor.zeros(m)
    while keep_on:
        eigen_stiff_old = eigen_stiff
        C_macro_local = C_macro * g.inv()
        for i in range(m):
            A_local[i] = localization_tensor(C_macro_local[i].full_tensor(), C)
        A = A_local * g
        CiAi = C_rotated.ddot(A)
        if method == 'stress':
            LiAi_mean = CiAi.mean().full_tensor()
            C_macro = StiffnessTensor(LiAi_mean, force_symmetry=True)
            AB = A
        elif method == 'strain':
            B = CiAi.ddot(C_macro.inv())
            LiinvBi = C_rotated.inv().ddot(B)
            mean = LiinvBi.mean()
            S_macro = ComplianceTensor(mean.matrix, force_symmetry=True)
            C_macro = S_macro.inv()
            AB = B
        else:
            raise ValueError('Only "strain" and "stress" are valid method names')

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
            err = AB.matrix - I.matrix
            err = np.max(np.abs(err))
            print('Iter #{}: abs. change={:0.5f}; rel. change={:0.5f}; error={:0.5f}'.format(k, max_abs_change, rel_change,err))
    return C_macro, message


Cstrip = StiffnessTensor.transverse_isotropic(Ex= 10.2, Ez=146.8, nu_zx=0.274, nu_yx=0.355, Gxz=7)
Cstrip = Cstrip * Rotation.from_euler('Y', 90, degrees=True)
orientations = Rotation.from_euler('Z', np.linspace(0,180,10, endpoint=False), degrees=True)

Ciso = StiffnessTensor.isotropic(E=200, nu=0.3)
C_stress, reason = Kroner_Eshelby(Cstrip, orientations, method='stress', max_iter=50, rtol=1e-6, atol=1e-5, display=True)

