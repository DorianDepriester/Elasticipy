import numpy as np
from Elasticipy.FourthOrderTensor import StiffnessTensor, SymmetricTensor, ComplianceTensor
from scipy.integrate import trapezoid
from Elasticipy.FourthOrderTensor import rotate_tensor
from scipy.spatial.transform import Rotation

I = StiffnessTensor.identity(return_full_tensor=True)


def ddot(a, b):
    if isinstance(a, SymmetricTensor):
        a_full = a.full_tensor()
    else:
        a_full = a
    if isinstance(b, SymmetricTensor):
        b_full = b.full_tensor()
    else:
        b_full = b
    return np.einsum('...ijmn,...mnkl->...ijkl', a_full, b_full)

def invert_4th_order_tensor(T):
    shape = T.shape
    *a,_,_,_,_ = shape
    T_mat = T.reshape(tuple(a) + (9, 9))
    T_inv_mat = np.linalg.pinv(T_mat)
    return T_inv_mat.reshape(shape)

def Kroner_tensor(L, theta, phi, a1=1., a2=1., a3=1.):
    theta = np.atleast_2d(theta)
    phi = np.atleast_2d(phi)
    s1 = np.sin(theta)*np.cos(phi) / a1
    s2 = np.sin(theta)*np.sin(phi) / a2
    s3 = np.cos(theta) / a3
    s = [s1, s2, s3]
    D = np.einsum('kijl,kpq,lpq->ijpq', L, s, s)
    return np.einsum('ikmn,jmn,lmn->ijklmn', np.linalg.inv(D.T).T, s, s)

def Morris_tensor(L):
    gamma = Kroner_tensor(L, theta, phi)
    a = trapezoid(gamma*np.sin(theta), theta[0], axis=-1)
    return trapezoid(a, phi[:,0], axis=-1)/(4*np.pi)

def localization_tensor(C_macro_local, C_incl):
    E = Morris_tensor(C_macro_local)
    Ainv = ddot(E, C_incl.full_tensor() - C_macro_local) + I
    return invert_4th_order_tensor(Ainv)

def global_spherical_grid(n_theta=50, n_phi=100):
    global phi, theta
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    phi, theta = np.meshgrid(phi, theta, indexing='ij')

def Kroner_Eshelby(C, orientations, method='stress',  max_iter=50, atol=1e-3, rtol=1e-4, display=False):
    C_rotated = C * orientations
    C_macro = C_rotated.Hill_average()
    eigen_stiff = C_macro.eig_stiffnesses
    global_spherical_grid()
    keep_on = True
    iter = 0
    reason = 'Maximum number of iterations is reached'
    m = len(orientations)
    while keep_on:
        eigen_stiff_old = eigen_stiff
        A_local = np.zeros((m,3,3,3,3))
        C_macro_local = C_macro * orientations.inv()
        for i in range(m):
            A_local[i] = localization_tensor(C_macro_local[i].full_tensor(), C)
        LiAi_local = ddot(C, A_local)
        if method == 'stress':
            LiAi = rotate_tensor(LiAi_local, orientations)
            C_matrix = np.mean(LiAi, axis=0)
            C_macro = StiffnessTensor(C_matrix, force_symmetry=True)
        elif method == 'strain':
            B_local = ddot(LiAi_local, C_macro_local.inv().full_tensor())
            Bi = rotate_tensor(B_local, orientations)
            LiBi_mean = np.mean(ddot(C_rotated.inv(), Bi),axis=0)
            S_macro = ComplianceTensor(LiBi_mean, force_symmetry=True)
            C_macro = S_macro.inv()
        else:
            raise ValueError('Only "strain" and "stress" are valid method names')

        eigen_stiff = C_macro.eig_stiffnesses
        abs_change = np.abs(eigen_stiff - eigen_stiff_old)
        rel_change = np.max(abs_change / eigen_stiff_old)
        max_abs_change = np.max(abs_change)
        iter += 1
        if  max_abs_change < atol:
            keep_on = False
            reason = 'Absolute change is below threshold value'
        if rel_change < rtol:
            keep_on = False
            reason = 'Relative change is below threshold value'
        if iter == max_iter:
            keep_on = False
        if display:
            err = np.max(np.mean(A_local, axis=0) - StiffnessTensor.identity(return_full_tensor=True))
            print('Iter #{}: abs. change={:0.5f}; rel. change={:0.5f}; error={:0.5f}'.format(iter, max_abs_change, rel_change,err))
    return C_macro, reason


Cstrip = StiffnessTensor.transverse_isotropic(Ex= 10.2, Ez=146.8, nu_zx=0.274, nu_yx=0.355, Gxz=7)
orientations = Rotation.from_euler('X', np.linspace(0,180,10), degrees=True)
C_stress, reason = Kroner_Eshelby(Cstrip, orientations, method='strain', max_iter=20, display=True)

