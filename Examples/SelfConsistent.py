import numpy as np
from Elasticipy.tensors.fourth_order import FourthOrderTensor
from Elasticipy.tensors.elasticity import StiffnessTensor
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation
from scipy.integrate import dblquad

I = FourthOrderTensor.identity()

def gamma_int(C_macro_local, s):

    return np.einsum('nr,w,s->nwrs', np.linalg.inv(D), s, s)

def Morris_tensor_int(C_macro_local, a1=1, a2=1, a3=1):
    E = np.zeros((3,3,3,3))
    for n in range(3):
        for w in range(3):
            for r in range(3):
                for s in range(3):
                    def fun(phi, theta):
                        s1 = np.sin(theta) * np.cos(phi) / a1
                        s2 = np.sin(theta) * np.sin(phi) / a2
                        s3 = np.cos(theta) / a3
                        k = [s1, s2, s3]
                        D = np.einsum('lmnp,p,l->mn', C_macro_local.full_tensor(), k, k)
                        # np.einsum('nr,w,s->nwrs', np.linalg.inv(D), s, s)
                        return np.sin(theta) * np.linalg.inv(D)[n,r] * k[w] * k[s]
                    E[n,w,r,s] = 1/(4*np.pi) * dblquad(fun, 0, np.pi, 0, 2*np.pi)[0]
    return E

def gamma(C_macro_local, phi, theta, a1, a2, a3):
    s1 = np.sin(theta)*np.cos(phi) / a1
    s2 = np.sin(theta)*np.sin(phi) / a2
    s3 = np.cos(theta) / a3
    s = np.array([s1, s2, s3])
    D = np.einsum('lmnp,pqr,lqr->qrmn', C_macro_local.full_tensor(), s, s)
    Dinv = np.linalg.inv(D)
    M1 = np.einsum('qrjk,iqr,lqr->qrijkl', Dinv, s, s)
    M2 = np.einsum('qrik,jqr,lqr->qrijkl', Dinv, s, s)
    M3 = np.einsum('qrjl,iqr,kqr->qrijkl', Dinv, s, s)
    M4 = np.einsum('qril,jqr,kqr->qrijkl', Dinv, s, s)
    return (M1+M2+M3+M4)/4

def polarization_tensor(C_macro_local, a1, a2, a3, n_phi, n_theta):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
    g = gamma(C_macro_local, phi_grid, theta_grid, a1, a2, a3)
    gsin = (g.T*np.sin(theta_grid.T)).T
    a = trapezoid(gsin, phi, axis=0)
    b= trapezoid(a, theta, axis=0)/(4*np.pi)
    return b

def localization_tensor(C_macro_local, C_incl, n_phi, n_theta):
    E = polarization_tensor(C_macro_local, 0.1, 1, 10, n_phi, n_theta)
    delta = FourthOrderTensor(C_incl.matrix - C_macro_local.matrix)
    Ainv = E.ddot(delta) + I
    return Ainv.inv().full_tensor()

def Kroner_Eshelby(Ci, g, max_iter=5, atol=1e-3, rtol=1e-3, display=False, n_phi=100, n_theta=100):
    Ci_rotated = (Ci * g)
    C_macro = Ci.Hill_average()
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
            A_local[i] = localization_tensor(C_macro_local[i], Ci, n_phi, n_theta)
        A = A_local * g
        CiAi = Ci_rotated.ddot(A)
        CiAi_mean = CiAi.mean()
        C_macro = StiffnessTensor.from_Kelvin(CiAi_mean.matrix, force_symmetry=True)

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
orientations = Rotation.random(100, random_state=1234)

C_stress, reason = Kroner_Eshelby(Cstrip, orientations, display=True)
