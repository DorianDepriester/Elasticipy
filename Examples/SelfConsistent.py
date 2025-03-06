import numpy as np
from Elasticipy.FourthOrderTensor import StiffnessTensor, SymmetricTensor
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

def Kroner_matrix(L, theta, phi, a1=1., a2=1., a3=1.):
    theta = np.atleast_2d(theta)
    phi = np.atleast_2d(phi)
    s1 = np.sin(theta)*np.cos(phi) / a1
    s2 = np.sin(theta)*np.sin(phi) / a2
    s3 = np.cos(theta) / a3
    s = [s1, s2, s3]
    D = np.einsum('kijl,kpq,lpq->ijpq', L.full_tensor(), s, s)
    return np.einsum('ikmn,jmn,lmn->ijklmn', np.linalg.inv(D.T).T, s, s)

def Morris_tensor(L):
    gamma = Kroner_matrix(L, theta, phi)
    a = trapezoid(gamma*np.sin(theta), theta[0], axis=-1)
    return trapezoid(a, phi[:,0], axis=-1)/(4*np.pi)

def localization_tensor(C_macro, C_incl, orientation):
    E = Morris_tensor(C_macro)
    E_local = rotate_tensor(E, orientation.inv())
    C_macro_local = rotate_tensor(C_macro.full_tensor(), orientation.inv())
    Ainv = ddot(E_local, C_incl.full_tensor() - C_macro_local) + I
    A = invert_4th_order_tensor(Ainv)
    return rotate_tensor(A, orientation)

def global_spherical_grid(n_theta=50, n_phi=100):
    global phi, theta
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    phi, theta = np.meshgrid(phi, theta, indexing='ij')

def Kroner_Eshelby(C, orientations, method='stress',  max_iter=50, atol=1e-3, rtol=1e-4, display=False):
    C_rotated = C * orientations
    C_macro = C.Hill_average()
    global_spherical_grid()
    keep_on = True
    iter = 0
    while keep_on:
        A = np.zeros((m,3,3,3,3))
        for i in range(m):
            A[i] = localization_tensor(C_macro, C, orientations[i])
        LiAi = ddot(C_rotated, A)
        if method == 'stress':
            C_macro_new = np.mean(LiAi, axis=0)
        elif method == 'strain':
            B = ddot(LiAi, C_macro.inv().full_tensor())
            LiBi_mean = np.mean(ddot(C_rotated.inv(), B),axis=0)
            C_macro_new = invert_4th_order_tensor(LiBi_mean)
        C_macro_old = C_macro.full_tensor()
        abs_change = np.abs(C_macro_new - C_macro_old)
        rel_change = np.max(abs_change[C_macro_old!=0] / np.abs(C_macro_old[C_macro_old!=0]))
        max_abs_change = np.max(abs_change)
        iter += 1
        reason = 'Maximum number of iterations is reached'
        if  max_abs_change < atol:
            keep_on = False
            reason = 'Absolute change is below threshold value'
        if rel_change < rtol:
            keep_on = False
            reason = 'Relative change is below threshold value'
        if iter > max_iter:
            keep_on = False
        if display:
            err = np.linalg.norm(np.mean(A, axis=0) - StiffnessTensor.identity(return_full_tensor=True))
            print('Iter #{}: abs. change={:0.5f}; rel. change={:0.5f}; error={:0.5f}'.format(iter, max_abs_change, rel_change,err))
        C_macro = StiffnessTensor(C_macro_new, force_symmetry=True)
    return C_macro, reason

m = 100
orientations = Rotation.random(m)
C = StiffnessTensor.cubic(C11=110, C12=12, C44=44)

C_stress, reason = Kroner_Eshelby(C, orientations, method='stress', max_iter=10, display=True)
print(C_stress)

C_strain, reason = Kroner_Eshelby(C, orientations, method='strain', max_iter=10, display=True)
print(C_strain)
