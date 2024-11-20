import numpy as np
import re
from Elasticipy.StressStrainTensors import SecondOrderTensor, StrainTensor, StressTensor
from Elasticipy.SphericalFunction import SphericalFunction, HyperSphericalFunction
from scipy.spatial.transform import Rotation


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


def _compute_unit_strain_along_direction(S, m, n, transverse=False):
    m_vec = np.atleast_2d(m)
    n_vec = np.atleast_2d(n)
    if not isinstance(S, ComplianceTensor):
        S = S.inv()
    if np.any(np.linalg.norm(m_vec) < 1e-9) or np.any(np.linalg.norm(n_vec) < 1e-9):
        raise ValueError('The input vector cannot be zeros')
    m_vec = (m_vec.T / np.linalg.norm(m_vec, axis=1)).T
    n_vec = (n_vec.T / np.linalg.norm(n_vec, axis=1)).T

    indices = np.indices((3, 3, 3, 3))
    i, j, k, ell = indices[0], indices[1], indices[2], indices[3]
    dot = np.abs(np.einsum('ij,ij->i', m_vec, n_vec))
    if np.any(np.logical_and(dot > 1e-9, dot < (1 - 1e-9))):
        raise ValueError('The two directions must be either equal or orthogonal.')
    if transverse:
        cosine = m_vec[:, i] * m_vec[:, j] * n_vec[:, k] * n_vec[:, ell]
    else:
        cosine = m_vec[:, i] * n_vec[:, j] * m_vec[:, k] * n_vec[:, ell]
    return np.einsum('pijkl,ijkl->p', cosine, S.full_tensor())


def tensorFromCrystalSymmetry(symmetry='Triclinic', point_group=None, diad='x',
                              tensor='Stiffness', phase_name='', **kwargs):
    """
    Create a fourth-order tensor from limited number of components, taking advantage of crystallographic symmetries

    Parameters
    ----------
    symmetry : str, default Triclinic
        Name of the crystallographic symmetry
    point_group : str
        Point group of the considered crystal. Only used (and mandatory) for tetragonal and trigonal symmetries.
    diad : str {'x', 'y'}, default 'x'
        Alignment convention. Sets whether x||a or y||b. Only used for monoclinic symmetry.
    tensor : str {'Stiffness', 'Compliance'}, default 'Stiffness'
        Type of tensor to create
    phase_name : str, default None
        Name to use when printing the tensor
    kwargs
        Keywords describing all the necessary components, depending on the crystal's symmetry and the type of tensor.
        For Stiffness, they should be named as 'Cij' (e.g. C11=..., C12=...).
        For Comliance, they should be named as 'Sij' (e.g. S11=..., S12=...).
        See examples below.

    Returns
    -------
    StiffnessTensor or ComplianceTensor

    Examples
    --------
    >>> tensorFromCrystalSymmetry(symmetry='monoclinic', diad='y', phase_name='TiNi',
    ...                          C11=231, C12=127, C13=104,
    ...                          C22=240, C23=131, C33=175,
    ...                          C44=81, C55=11, C66=85,
    ...                          C15=-18, C25=1, C35=-3, C46=3)
    Stiffness tensor (in Voigt notation) for TiNi:
    [[231. 127. 104.   0. -18.   0.]
     [127. 240. 131.   0.   1.   0.]
     [104. 131. 175.   0.  -3.   0.]
     [  0.   0.   0.  81.   0.   3.]
     [-18.   1.  -3.   0.  11.   0.]
     [  0.   0.   0.   3.   0.  85.]]
    Symmetry: monoclinic

    >>> tensorFromCrystalSymmetry(symmetry='monoclinic', diad='y', phase_name='TiNi',
    ...                          C11=8, C12=-3, C13=-2,
    ...                          C22=8, C23=-5, C33=10,
    ...                          C44=12, C55=116, C66=12,
    ...                          C15=14, C25=-8, C35=0, C46=0)
    Stiffness tensor (in Voigt notation) for TiNi:
    [[  8.  -3.  -2.   0.  14.   0.]
     [ -3.   8.  -5.   0.  -8.   0.]
     [ -2.  -5.  10.   0.   0.   0.]
     [  0.   0.   0.  12.   0.   0.]
     [ 14.  -8.   0.   0. 116.   0.]
     [  0.   0.   0.   0.   0.  12.]]
    Symmetry: monoclinic
    """
    tensor = tensor.lower()
    if tensor == 'stiffness':
        prefix = 'C'
        k = 1 / 2
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
                if (symmetry == 'monoclinic') and (diad == 'x'):
                    C[0, 5], C[1, 5], C[2, 5] = values['16'], values['26'], values['36']
                    C[3, 4] = values['45']
                else:
                    C[0, 4], C[1, 4], C[2, 4] = values['15'], values['25'], values['35']
                    C[3, 5] = values['46']
                if symmetry == 'triclinic':
                    C[0, 3], C[1, 3], C[2, 3] = values['14'], values['24'], values['34']
                    C[0, 4], C[1, 4], C[2, 4] = values['15'], values['25'], values['35']
                    C[3, 5], C[4, 5] = values['46'], values['56']
        if tensor == 'stiffness':
            constructor = StiffnessTensor
        else:
            constructor = ComplianceTensor
        return constructor(C + np.tril(C.T, -1), symmetry=symmetry, phase_name=phase_name)
    except KeyError as key:
        entry_error = prefix + key.args[0]
        if (symmetry == 'tetragonal') or (symmetry == 'trigonal'):
            err_msg = "For point group {}, keyword argument {} is required".format(point_group, entry_error)
        elif symmetry == 'monoclinic':
            err_msg = "For {} symmetry with diag='{}', keyword argument {} is required".format(symmetry, diad, entry_error)
        else:
            err_msg = "For {} symmetry, keyword argument {} is required".format(symmetry, entry_error)
        raise ValueError(err_msg)


class SymmetricTensor:
    """
    Template class for manipulating symmetric fourth-order tensors.

    Attributes
    ----------
    matrix : np.ndarray
        (6,6) matrix gathering all the components of the tensor, using the Voigt notation.
    symmetry : str
        Symmetry of the tensor

    """
    tensor_name = 'Symmetric'
    voigt_map = np.ones((6, 6))

    def __init__(self, M, phase_name='', symmetry='Triclinic', orientations=None):
        """
        Construct of stiffness tensor from a (6,6) matrix

        Parameters
        ----------
        M : np.ndarray
            (6,6) matrix corresponding to the stiffness tensor, written using the Voigt notation
        phase_name : str, default None
            Name to display
        symmetry : str, default Triclinic
        """
        self.matrix = np.array(M)
        self.phase_name = phase_name
        self.symmetry = symmetry
        self.orientations = orientations

    def __repr__(self):
        if self.phase_name == '':
            heading = '{} tensor (in Voigt notation):\n'.format(self.tensor_name)
        else:
            heading = '{} tensor (in Voigt notation) for {}:\n'.format(self.tensor_name, self.phase_name)
        print_symmetry = '\nSymmetry: {}'.format(self.symmetry)
        msg = heading + self.matrix.__str__() + print_symmetry
        if self.orientations is not None:
            msg = msg + '\n{} orientations'.format(len(self))
        return msg

    def __len__(self):
        if self.orientations is None:
            return ()
        elif self.orientations.single:
            return 1
        else:
            return len(self.orientations)

    def full_tensor(self):
        """
        Returns the full (unvoigted) tensor, as a [3, 3, 3, 3] array

        Returns
        -------
        np.ndarray
            Full tensor (4-index notation)
        """
        i, j, k, ell = np.indices((3, 3, 3, 3))
        ij = voigt(i, j)
        kl = voigt(k, ell)
        m = self.matrix[ij, kl] / self.voigt_map[ij, kl]
        if self.orientations is None:
            return m
        else:
            ori = self.orientations.as_matrix()
            rotated_tensors = np.einsum('qim,qjn,qko,qlp,mnop->qijkl', ori, ori, ori, ori, m)
            return rotated_tensors

    def rotate(self, rotation):
        """
        Apply a single rotation to a tensor, and return its component into the rotated frame.

        Parameters
        ----------
        rotation : Rotation
            Rotation to apply

        Returns
        -------
        SymmetricTensor
            Rotated tensor
        """
        rot_mat = rotation.as_matrix()
        rotated_tensor = np.einsum('im,jn,ko,lp,mnop->ijkl', rot_mat, rot_mat, rot_mat, rot_mat, self.full_tensor())
        ij, kl = np.indices((6, 6))
        i, j = unvoigt(ij).T
        k, ell = unvoigt(kl).T
        rotated_matrix = rotated_tensor[i, j, k, ell] * self.voigt_map[ij, kl]
        return self.__class__(rotated_matrix)

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            if other.shape == (6, 6):
                mat = self.matrix + other
            elif other.shape == (3, 3, 3, 3):
                ten = self.full_tensor() + other
                ij, kl = np.indices((6, 6))
                i, j = unvoigt(ij).T
                k, ell = unvoigt(kl).T
                mat = ten[i, j, k, ell] * self.voigt_map[ij, kl]
            else:
                raise ValueError('The input argument must be either a 6x6 matrix or a (3,3,3,3) array.')
        elif isinstance(other, SymmetricTensor):
            if type(other) == type(self):
                mat = self.matrix + other.matrix
            else:
                raise ValueError('The two tensors to add must be of the same class.')
        else:
            raise ValueError('I don''t know how to add {} with {}.'.format(type(self), type(other)))
        return self.__class__(mat)

    def __mul__(self, other):
        if isinstance(other, SecondOrderTensor):
            return SecondOrderTensor(self * other.matrix)
        elif isinstance(other, np.ndarray):
            if other.shape[-2:] == (3, 3):
                if self.orientations is None:
                    return np.einsum('ijkl,...kl->...ij', self.full_tensor(), other)
                else:
                    return np.einsum('qijkl,...kl->q...ij', self.full_tensor(), other)
        elif isinstance(other, Rotation):
            if other.single:
                return self.rotate(other)
            else:
                return self.__class__(self.matrix, symmetry=self.symmetry, orientations=other)
        else:
            return self.__class__(self.matrix * other, symmetry=self.symmetry)

    def __rmul__(self, other):
        if isinstance(other, (Rotation, float, int)):
            return self * other
        else:
            raise ValueError('A fourth order tensor can be left-multiplied by rotations or scalar only.')

    def _orientation_average(self, orientations):
        """
        Rotate the tensor by a series of rotations, then evaluate its mean value.

        Parameters
        ----------
        orientations : np.ndarray or Rotation
            If an array is provided, it must be of shape [m, 3, 3], where orientations[i,:,:] gives i-th the orientation
            matrix

        Returns
        -------
        SymmetricTensor
            Mean tensor

        """
        if isinstance(orientations, Rotation):
            orientations = orientations.as_matrix()
        if len(orientations.shape) == 2:
            raise ValueError('The orientation must be a 3x3 or a Nx3x3 matrix')
        elif len(orientations.shape) == 2:
            return self * orientations
        else:
            m = orientations
            rotated_tensor = np.einsum('qim,qjn,qko,qlp,mnop->ijkl', m, m, m, m, self.full_tensor())
            ij, kl = np.indices((6, 6))
            i, j = unvoigt(ij).T
            k, ell = unvoigt(kl).T
            rotated_matrix = rotated_tensor[i, j, k, ell] * self.voigt_map[ij, kl] / orientations.shape[0]
            return self.__class__(rotated_matrix)


class StiffnessTensor(SymmetricTensor):
    """
    Class for manipulating fourth-order stiffness tensors.
    """
    tensor_name = 'Stiffness'

    def __init__(self, S, **kwargs):
        super().__init__(S, **kwargs)

    def __mul__(self, other):
        if isinstance(other, StrainTensor):
            new_tensor = super().__mul__(other)
            return StressTensor(new_tensor.matrix)
        elif isinstance(other, StressTensor):
            raise ValueError('You cannot multiply a stiffness tensor with a Stress tensor.')
        else:
            return super().__mul__(other)

    def inv(self):
        """
        Compute the reciprocal compliance tensor

        Returns
        -------
        ComplianceTensor
            Reciprocal tensor
        """
        C = np.linalg.inv(self.matrix)
        return ComplianceTensor(C, symmetry=self.symmetry, phase_name=self.phase_name, orientations=self.orientations)

    @property
    def Young_modulus(self):
        """
        Directional Young's modulus

        Returns
        -------
        SphericalFunction
            Young's modulus
        """
        def compute_young_modulus(n):
            eps = _compute_unit_strain_along_direction(self, n, n)
            return 1 / eps
        return SphericalFunction(compute_young_modulus)

    @property
    def shear_modulus(self):
        """
        Directional shear modulus

        Returns
        -------
        HyperSphericalFunction
            Shear modulus
        """
        def compute_shear_modulus(m, n):
            eps = _compute_unit_strain_along_direction(self, m, n)
            return 1 / (4 * eps)

        return HyperSphericalFunction(compute_shear_modulus)

    @property
    def Poisson_ratio(self):
        """
        Directional Poisson's ratio

        Returns
        -------
        HyperSphericalFunction
            Poisson's ratio
        """
        def compute_PoissonRatio(m, n):
            eps1 = _compute_unit_strain_along_direction(self, m, m)
            eps2 = _compute_unit_strain_along_direction(self, m, n, transverse=True)
            return -eps2 / eps1
        return HyperSphericalFunction(compute_PoissonRatio)

    def Voigt_average(self, orientations=None):
        """
        Compute the Voigt average of stiffness tensor

        Parameters
        ----------
        orientations : np.ndarray or None
            Set of m orientation matrices, defined as a [m, 3, 3] array.
            If None, uniform distribution is assumed, resulting in isotropic tensor

        Returns
        -------
        StiffnessTensor
            Voigt average of stiffness tensor

        See Also
        --------
        Reuss_average : compute the Reuss average
        Hill_average : compute the Voigt-Reuss-Hill average
        """
        if orientations is None:
            c = self.matrix
            C11 = (c[0, 0] + c[1, 1] + c[2, 2]) / 5 \
                + (c[0, 1] + c[0, 2] + c[1, 2]) * 2 / 15 \
                + (c[3, 3] + c[4, 4] + c[5, 5]) * 4 / 15
            C12 = (c[0, 0] + c[1, 1] + c[2, 2]) / 15 \
                + (c[0, 1] + c[0, 2] + c[1, 2]) * 4 / 15 \
                - (c[3, 3] + c[4, 4] + c[5, 5]) * 2 / 15
            C44 = (c[0, 0] + c[1, 1] + c[2, 2] - c[0, 1] - c[0, 2] - c[1, 2]) / 15 + (c[3, 3] + c[4, 4] + c[5, 5]) / 5
            mat = np.array([[C11, C12, C12, 0,   0,   0],
                            [C12, C11, C12, 0,   0,   0],
                            [C12, C12, C11, 0,   0,   0],
                            [0,   0,   0,   C44, 0,   0],
                            [0,   0,   0,   0,   C44, 0],
                            [0,   0,   0,   0,   0,   C44]])
            return StiffnessTensor(mat, symmetry='isotropic', phase_name=self.phase_name)
        else:
            return self._orientation_average(orientations)

    def Reuss_average(self, **kwargs):
        """
        Compute the Reuss average of tensor

        Parameters
        ----------
        orientations : np.ndarray or None
            Set of m orientation matrices, defined as a [m, 3, 3] array.
            If None, uniform distribution is assumed, resulting in isotropic tensor

        Returns
        -------
        StiffnessTensor
            Reuss average of stiffness tensor

        See Also
        --------
        Voigt_average : compute the Voigt average
        Hill_average : compute the Voigt-Reuss-Hill average
        """
        return self.inv().Reuss_average(**kwargs).inv()

    def Hill_average(self, orientations=None):
        """
        Compute the (Voigt-Reuss-)Hill average of tensor

        Parameters
        ----------
        orientations : np.ndarray or None
            Set of m orientation matrices, defined as a [m, 3, 3] array.
            If None, uniform distribution is assumed, resulting in isotropic tensor

        Returns
        -------
        StiffnessTensor
            Voigt-Reuss-Hill average of tensor

        See Also
        --------
        Voigt_average : compute the Voigt average
        Reuss_average : compute the Reuss average
        """
        Reuss = self.Reuss_average(orientations=orientations)
        Voigt = self.Voigt_average(orientations=orientations)
        return (Reuss + Voigt) * 0.5


class ComplianceTensor(StiffnessTensor):
    """
    Class for manipulating compliance tensors

    """
    tensor_name = 'Compliance'
    voigt_map = np.vstack((
        np.hstack((np.ones((3, 3)), 2 * np.ones((3, 3)))),
        np.hstack((2 * np.ones((3, 3)), 4 * np.ones((3, 3))))
    ))

    def __init__(self, C, **kwargs):
        super().__init__(C, **kwargs)

    def __mul__(self, other):
        if isinstance(other, StressTensor):
            return StrainTensor(self * other.matrix)
        elif isinstance(other, StrainTensor):
            raise ValueError('You cannot multiply a compliance tensor with Strain tensor.')
        else:
            return super().__mul__(other)

    def inv(self):
        """
        Compute the reciprocal stiffness tensor

        Returns
        -------
        StiffnessTensor
            Reciprocal tensor
        """
        S = np.linalg.inv(self.matrix)
        return StiffnessTensor(S, symmetry=self.symmetry, phase_name=self.phase_name, orientations=self.orientations)

    def Reuss_average(self, orientations=None):
        if orientations is None:
            s = self.matrix
            C11 = (s[0, 0] + s[1, 1] + s[2, 2]) / 5 \
                + (s[0, 1] + s[0, 2] + s[1, 2]) * 2 / 15 \
                + (s[3, 3] + s[4, 4] + s[5, 5]) * 1 / 15
            C12 = (s[0, 0] + s[1, 1] + s[2, 2]) / 15 \
                + (s[0, 1] + s[0, 2] + s[1, 2]) * 4 / 15 \
                - (s[3, 3] + s[4, 4] + s[5, 5]) * 1 / 30
            C44 = (s[0, 0] + s[1, 1] + s[2, 2] - s[0, 1] - s[0, 2] - s[1, 2]) * 4 / 15 + (s[3, 3] + s[4, 4] + s[5, 5]) / 5
            mat = np.array([[C11, C12, C12, 0,   0,   0],
                            [C12, C11, C12, 0,   0,   0],
                            [C12, C12, C11, 0,   0,   0],
                            [0,   0,   0,   C44, 0,   0],
                            [0,   0,   0,   0,   C44, 0],
                            [0,   0,   0,   0,   0,   C44]])
            return ComplianceTensor(mat, symmetry='isotropic', phase_name=self.phase_name)
        else:
            return self._orientation_average(orientations)

    def Voigt_average(self, **kwargs):
        return self.inv().Voigt_average(**kwargs).inv()

    def Hill_average(self, **kwargs):
        return self.inv().Hill_average(**kwargs)

