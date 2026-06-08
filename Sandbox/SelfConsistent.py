import numpy as np
from elasticipy.tensors.fourth_order import FourthOrderTensor, SymmetricFourthOrderTensor
from elasticipy.tensors.elasticity import StiffnessTensor
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation

I = FourthOrderTensor.identity()

def gamma(C_macro_local, phi, theta, a1, a2, a3):
    s1 = np.sin(theta)*np.cos(phi) / a1
    s2 = np.sin(theta)*np.sin(phi) / a2
    s3 = np.cos(theta) / a3
    s = np.array([s1, s2, s3])
    C = C_macro_local.full_tensor
    D = np.einsum('ijkl,j...,l...->ik...', C, s, s)
    Dinv = np.linalg.inv(D.T).T
    a1 = np.einsum('ik...,j...,l...->ijkl...', Dinv, s, s)
    a2 = np.einsum('jk...,i...,l...->ijkl...', Dinv, s, s)
    a3 = np.einsum('il...,j...,k...->ijkl...', Dinv, s, s)
    a4 = np.einsum('jl...,i...,k...->ijkl...', Dinv, s, s)
    return (a1 + a2 + a3 + a4) /4

def polarization_tensor(C_macro_local, a1, a2, a3, n_phi, n_theta):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
    g = gamma(C_macro_local, phi_grid, theta_grid, a1, a2, a3)
    gsin = g * np.sin(theta_grid)
    a = trapezoid(gsin, theta, axis=-1)
    b= trapezoid(a, phi, axis=-1)/(4*np.pi)
    return FourthOrderTensor(b, force_minor_symmetry=True)

def localization_tensor(C_macro_local, C_incl, n_phi, n_theta, a1, a2, a3):
    E = polarization_tensor(C_macro_local, a1, a2, a3, n_phi, n_theta)
    Ainv = E.ddot(C_incl - C_macro_local) + I
    return Ainv.inv()

def Kroner_Eshelby(Ci, g=None, max_iter=100, atol=1e-3, rtol=1e-3, display=False, n_phi=50, n_theta=100, particle_size=None):
    if isinstance(Ci, (tuple, list)):
        Ci = StiffnessTensor.stack(Ci)
    if g is not None:
        Ci_rotated = (Ci * g)
    else:
        Ci_rotated = Ci
    C_macro = Ci.Hill_average()
    C_macro = SymmetricFourthOrderTensor(C_macro)
    eigen_stiff = C_macro.eigvals()
    keep_on = True
    k = 0
    message = 'Maximum number of iterations is reached'
    m = Ci.shape[0]
    A_local = FourthOrderTensor.zeros(m)
    if particle_size is None:
        a1 = a2 = a3 = 1
    else:
        a1, a2, a3 = particle_size
    while keep_on:
        eigen_stiff_old = eigen_stiff
        if g is not None:
            C_macro_local = C_macro * (g.inv())
        else:
            C_macro_local = C_macro
        for i in range(m):
            if C_macro_local.shape:
                A_local[i] = localization_tensor(C_macro_local[i], Ci[i], n_phi, n_theta, a1, a2, a3)
            else:
                A_local[i] = localization_tensor(C_macro_local, Ci[i], n_phi, n_theta, a1, a2, a3)
        if g is None:
            A = A_local
        else:
            A = A_local * g
        Q = Ci_rotated.ddot(A)
        CiAi_mean = Q.mean()
        C_macro = SymmetricFourthOrderTensor(CiAi_mean, force_symmetries=True)
        err = A.mean() - FourthOrderTensor.identity()

        # Stopping criteria
        eigen_stiff = C_macro.eigvals()
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
            err = np.max(np.abs(err._matrix))
            print('Iter #{}: abs. change={:0.5f}; rel. change={:0.5f}; error={:0.5f}'.format(k, max_abs_change, rel_change,err))
    return C_macro, message

C1 = StiffnessTensor.isotropic(E=210, nu=0.25)
C2 = StiffnessTensor.isotropic(E=70, nu=0.3)
Cmacro, msg = Kroner_Eshelby((C1, C2), display=True, particle_size=(1000,1000,1))
