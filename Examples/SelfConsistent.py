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
    Dinv = np.linalg.inv(D)
    return np.einsum('qrjk,iqr,lqr->qrijkl', Dinv, s, s) # The symmetrization is made afterward (see below)

def polarization_tensor(C_macro_local, a1, a2, a3, n_phi, n_theta):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
    g = gamma(C_macro_local, phi_grid, theta_grid, a1, a2, a3)
    gsin = (g.T*np.sin(theta_grid.T)).T
    a = trapezoid(gsin, phi, axis=0)
    b= trapezoid(a, theta, axis=0)/(4*np.pi)
    return FourthOrderTensor(b, force_minor_symmetry=True) # Symmetrization here (see above)

def localization_tensor(C_macro_local, C_incl, n_phi, n_theta, a1, a2, a3):
    E = polarization_tensor(C_macro_local, a1, a2, a3, n_phi, n_theta)
    delta = FourthOrderTensor(C_incl.matrix - C_macro_local.matrix)
    Ainv = E.ddot(delta) + I
    return Ainv.inv()

def Kroner_Eshelby(Ci, g, method='strain', max_iter=5, atol=1e-3, rtol=1e-3, display=False, n_phi=100, n_theta=100, particle_size=None):
    Ci_rotated = (Ci * g)
    C_macro = Ci_rotated.Hill_average()
    eigen_stiff = C_macro.eig_stiffnesses
    keep_on = True
    k = 0
    message = 'Maximum number of iterations is reached'
    m = len(g)
    A_local = FourthOrderTensor.zeros(m)
    if particle_size is None:
        a1 = a2 = a3 = 1
    else:
        a1, a2, a3 = particle_size
    while keep_on:
        eigen_stiff_old = eigen_stiff
        C_macro_local = C_macro * (g.inv())
        for i in range(m):
            A_local[i] = localization_tensor(C_macro_local[i], Ci, n_phi, n_theta, a1, a2, a3)
        A = A_local * g
        Q = Ci_rotated.ddot(A)
        if method=='stress':
            CiAi_mean = Q.mean()
            C_macro = StiffnessTensor.from_Kelvin(CiAi_mean.matrix, force_symmetries=True)
            err = A.mean() - FourthOrderTensor.identity()
        else:
            B = Q.ddot(C_macro.inv())
            R = Ci_rotated.inv().ddot(B)
            R_mean = R.mean()
            C_macro = StiffnessTensor.from_Kelvin(R_mean.inv().matrix, force_symmetries=True)
            err = B.mean() - FourthOrderTensor.identity()

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
            err = np.max(np.abs(err.matrix))
            print('Iter #{}: abs. change={:0.5f}; rel. change={:0.5f}; error={:0.5f}'.format(k, max_abs_change, rel_change,err))
    return C_macro, message

Cstrip = StiffnessTensor.transverse_isotropic(Ex= 10.2, Ez=146.8, nu_zx=0.274, nu_yx=0.355, Gxz=7)
Cstrip = Cstrip * Rotation.from_euler('Y', 90, degrees=True)
#orientations = Rotation.from_euler('Z', np.linspace(0, 180, 10, endpoint=False), degrees=True)
orientations = Rotation.random(100, random_state=1234)
Cstrip = StiffnessTensor.cubic(C11=186, C12=134, C44=77)
C_stress, reason = Kroner_Eshelby(Cstrip, orientations, display=True, method='strain')
