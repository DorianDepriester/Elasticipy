from Elasticipy.tensors.fourth_order import SymmetricFourthOrderTensor
from Elasticipy.spherical_function import SphericalFunction, HyperSphericalFunction
from Elasticipy.crystal_symmetries import SYMMETRIES
from Elasticipy.tensors.stress_strain import StrainTensor, StressTensor
from Elasticipy.tensors.mapping import VoigtMapping, KelvinMapping
import numpy as np
import re

def _parse_tensor_components(prefix, **kwargs):
    pattern = r'^{}(\d{{2}})$'.format(prefix)
    value = dict()
    for k, v in kwargs.items():
        match = re.match(pattern, k)  # Extract 'C11' to '11' and so
        if match:
            value[match.group(1)] = v
    return value

def _indices2str(ij):
    return f'{ij[0] + 1}{ij[1] + 1}'

def _isotropic_matrix(C11, C12, C44):
    return np.array([[C11, C12, C12, 0, 0, 0],
                     [C12, C11, C12, 0, 0, 0],
                     [C12, C12, C11, 0, 0, 0],
                     [0, 0, 0, C44, 0, 0],
                     [0, 0, 0, 0, C44, 0],
                     [0, 0, 0, 0, 0, C44]])


def _check_definite_positive(mat):
    try:
        np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        eigen_val = np.linalg.eigvals(mat)
        raise ValueError('The input matrix is not definite positive (eigenvalues: {})'.format(eigen_val))

def _switch_poisson_ratios(nu_xy, nu_yx, Ex, Ey, indices):
    if nu_yx is None and nu_xy is not None:
        return nu_xy * Ey / Ex
    elif nu_yx is not None and nu_xy is None:
        return nu_yx
    else:
        raise ValueError('Either nu_{0}{1} or nu_{1}{0} must be provided'.format(*indices))

class StiffnessTensor(SymmetricFourthOrderTensor):
    """
    Class for manipulating fourth-order stiffness tensors.
    """
    tensor_name = 'Stiffness'
    C11_C12_factor = 0.5
    C46_C56_factor = 1.0
    component_prefix = 'C'

    def __init__(self, M, symmetry='Triclinic', check_positive_definite=True, phase_name= None, mapping=VoigtMapping(), **kwargs):
        """
        Construct of stiffness tensor from a (6,6) matrix.

        The input matrix must be symmetric, otherwise an error is thrown (except if check_symmetry==False, see below)

        Parameters
        ----------
        M : np.ndarray
            (6,6) matrix corresponding to the stiffness tensor, written using the Voigt notation, or array of shape
            (3,3,3,3).
        phase_name : str, default None
            Name to display
        symmetry : str, default Triclinic
            Name of the crystal's symmetry
        check_symmetry : bool, optional
            Whether to check or not that the input matrix is symmetric.
        force_symmetry : bool, optional
            If true, the major symmetry of the tensor is forces
        mapping : str or MappingConvention
            mapping convention to use. Default is VoigtMapping.

        Notes
        -----
        The units used when building the stiffness tensor are up to the user (GPa, MPa, psi etc.). Therefor, the
        results you will get when performing operations (Young's modulus, "product" with strain tensor etc.) will be
        consistent with these units. For instance, if the stiffness tensor is defined in GPa, the computed stress will
        be given in GPa as well.
        """
        super().__init__(M, mapping=mapping, **kwargs)
        if check_positive_definite:
            _check_definite_positive(self._matrix)
        self.symmetry = symmetry
        self.phase_name = phase_name

    def __mul__(self, other):
        if isinstance(other, StrainTensor):
            new_tensor = self.ddot(other)
            return StressTensor(new_tensor.matrix)
        elif isinstance(other, StressTensor):
            raise ValueError('You cannot multiply a stiffness tensor with a Stress tensor.')
        else:
            return super().__mul__(other)

    def __repr__(self):
        string = super().__repr__()
        if self.phase_name is not None:
            string += '\nPhase: {}'.format(self.phase_name)
        string += '\nSymmetry: {}'.format(self.symmetry)
        return string

    def inv(self):
        """
        Compute the reciprocal compliance tensor

        Returns
        -------
        ComplianceTensor
            Reciprocal tensor
        """
        C = np.linalg.inv(self._matrix)
        return ComplianceTensor(C, symmetry=self.symmetry, phase_name=self.phase_name)

    @classmethod
    def from_txt_file(cls, filename):
        """
        Load the tensor from a text file.

        The two first lines can have data about phase name and symmetry, but this is not mandatory.

        Parameters
        ----------
        filename : str
            Filename to load the tensor from.

        Returns
        -------
        SymmetricFourthOrderTensor
            The reconstructed tensor read from the file.

        See Also
        --------
        save_to_txt : create a tensor from text file

        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Initialize defaults
        phase_name = None
        symmetry = 'Triclinic'
        matrix_start_index = 0

        # Parse phase name if available
        if lines and lines[0].startswith("Phase Name:"):
            phase_name = lines[0].split(": ", 1)[1].strip()
            matrix_start_index += 1

        # Parse symmetry if available
        if len(lines) > matrix_start_index and lines[matrix_start_index].startswith("Symmetry:"):
            symmetry = lines[matrix_start_index].split(": ", 1)[1].strip()
            matrix_start_index += 1

        # Parse matrix
        matrix = np.loadtxt(lines[matrix_start_index:])

        # Return the reconstructed object
        return cls(matrix, phase_name=phase_name, symmetry=symmetry)

    def save_to_txt(self, filename, matrix_only=False):
        """
        Save the tensor to a text file.

        Parameters
        ----------
        filename : str
            Filename to save the tensor to.
        matrix_only : bool, False
            If true, only the components of tje stiffness tensor is saved (no data about phase nor symmetry)

        See Also
        --------
        from_txt_file : create a tensor from text file

        """
        with open(filename, 'w') as f:
            if not matrix_only:
                if self.phase_name is not None:
                    f.write(f"Phase Name: {self.phase_name}\n")
                f.write(f"Symmetry: {self.symmetry}\n")
            for row in self._matrix:
                f.write("  " + "  ".join(f"{value:8.2f}" for value in row) + "\n")

    @classmethod
    def _matrixFromCrystalSymmetry(cls, symmetry='Triclinic', point_group=None, diad='y', prefix=None, **kwargs):
        if prefix is None:
            prefix = cls.component_prefix
        values = _parse_tensor_components(prefix, **kwargs)
        C = np.zeros((6, 6))
        symmetry = symmetry.capitalize()
        if ((symmetry == 'tetragonal') or (symmetry == 'trigonal')) and (point_group is None):
            raise ValueError('For tetragonal and trigonal symmetries, the point group is mandatory.')
        tetra_1 = ['4', '-4', '4/m']
        tetra_2 = ['4mm', '-42m', '422', '4/mmm']
        trigo_1 = ['3', '-3']
        trigo_2 = ['32', '-3m', '3m']
        if point_group is not None:
            if (point_group in tetra_1) or (point_group in tetra_2):
                symmetry = 'Tetragonal'
            elif (point_group in trigo_1) or (point_group in trigo_2):
                symmetry = 'Trigonal'
        symmetry_description = SYMMETRIES[symmetry]
        if symmetry == 'Tetragonal':
            if point_group in tetra_1:
                symmetry_description = symmetry_description[', '.join(tetra_1)]
            else:
                symmetry_description = symmetry_description[', '.join(tetra_2)]
        elif symmetry == 'Trigonal':
            if point_group in trigo_1:
                symmetry_description = symmetry_description[', '.join(trigo_1)]
            else:
                symmetry_description = symmetry_description[', '.join(trigo_2)]
        elif symmetry == 'Monoclinic':
            symmetry_description = symmetry_description["Diad || " + diad]
        for required_field in symmetry_description.required:
            C[required_field] = values[_indices2str(required_field)]

        # Now apply relationships between components
        for equality in symmetry_description.equal:
            for index in equality[1]:
                C[index] = C[equality[0]]
        for opposite in symmetry_description.opposite:
            for index in opposite[1]:
                C[index] = -C[opposite[0]]
        C11_C12 = symmetry_description.C11_C12
        if C11_C12:
            for index in C11_C12:
                C[index] = (C[0, 0] - C[0, 1]) * cls.C11_C12_factor

        if symmetry == 'Trigonal':
            C[3, 5] = cls.C46_C56_factor * C[3, 5]
            C[4, 5] = cls.C46_C56_factor * C[4, 5]

        return C + np.tril(C.T, -1)

    @classmethod
    def fromCrystalSymmetry(cls, symmetry='Triclinic', point_group=None, diad='y', phase_name=None, prefix=None,
                            **kwargs):
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
        phase_name : str, default None
            Name to use when printing the tensor
        prefix : str, default None
            Define the prefix to use when providing the components. By default, it is 'C' for stiffness tensors, 'S' for
            compliance.
        kwargs
            Keywords describing all the necessary components, depending on the crystal's symmetry and the type of tensor.
            For Stiffness, they should be named as 'Cij' (e.g. C11=..., C12=...).
            For Comliance, they should be named as 'Sij' (e.g. S11=..., S12=...).
            See examples below. The behaviour can be overriten with the prefix option (see above)

        Returns
        -------
        FourthOrderTensor

        See Also
        --------
        isotropic : creates an isotropic stiffness tensor from two paremeters (e.g. E and v).

        Notes
        -----
        The relationships between the tensor's components depend on the crystallogrpahic symmetry [1]_.

        References
        ----------
        .. [1] Nye, J. F. Physical Properties of Crystals. London: Oxford University Press, 1959.

        Examples
        --------
        >>> from Elasticipy.tensors.elasticity import StiffnessTensor\n
        >>> StiffnessTensor.fromCrystalSymmetry(symmetry='monoclinic', diad='y', phase_name='TiNi',
        ...                                     C11=231, C12=127, C13=104,
        ...                                     C22=240, C23=131, C33=175,
        ...                                     C44=81, C55=11, C66=85,
        ...                                     C15=-18, C25=1, C35=-3, C46=3)
        Stiffness tensor (in Voigt mapping):
        [[231. 127. 104.   0. -18.   0.]
         [127. 240. 131.   0.   1.   0.]
         [104. 131. 175.   0.  -3.   0.]
         [  0.   0.   0.  81.   0.   3.]
         [-18.   1.  -3.   0.  11.   0.]
         [  0.   0.   0.   3.   0.  85.]]
        Phase: TiNi
        Symmetry: monoclinic

        >>> from Elasticipy.tensors.elasticity import ComplianceTensor\n
        >>> ComplianceTensor.fromCrystalSymmetry(symmetry='monoclinic', diad='y', phase_name='TiNi',
        ...                                      S11=8, S12=-3, S13=-2,
        ...                                      S22=8, S23=-5, S33=10,
        ...                                      S44=12, S55=116, S66=12,
        ...                                      S15=14, S25=-8, S35=0, S46=0)
        Compliance tensor (in Voigt mapping):
        [[  8.  -3.  -2.   0.  14.   0.]
         [ -3.   8.  -5.   0.  -8.   0.]
         [ -2.  -5.  10.   0.   0.   0.]
         [  0.   0.   0.  12.   0.   0.]
         [ 14.  -8.   0.   0. 116.   0.]
         [  0.   0.   0.   0.   0.  12.]]
        Phase: TiNi
        Symmetry: monoclinic
        """
        matrix = cls._matrixFromCrystalSymmetry(point_group=point_group, diad=diad, symmetry=symmetry, prefix=prefix,
                                                **kwargs)
        return cls(matrix, phase_name=phase_name, symmetry=symmetry)


    @classmethod
    def hexagonal(cls, *, C11=0., C12=0., C13=0., C33=0., C44=0., phase_name=None):
        """
        Create a fourth-order tensor from hexagonal symmetry.

        Parameters
        ----------
        C11, C12 , C13, C33, C44 : float
            Components of the tensor, using the Voigt notation
        phase_name : str, optional
            Phase name to display
        Returns
        -------
        FourthOrderTensor

        See Also
        --------
        transverse_isotropic : creates a transverse-isotropic tensor from engineering parameters
        cubic : create a tensor from cubic symmetry
        tetragonal : create a tensor from tetragonal symmetry
        """
        return cls.fromCrystalSymmetry(symmetry='hexagonal', C11=C11, C12=C12, C13=C13, C33=C33, C44=C44,
                                       phase_name=phase_name, prefix='C')

    @classmethod
    def trigonal(cls, *, C11=0., C12=0., C13=0., C14=0., C33=0., C44=0., C15=0., phase_name=None):
        """
        Create a fourth-order tensor from trigonal symmetry.

        Parameters
        ----------
        C11, C12, C13, C14, C33, C44 : float
            Components of the tensor, using the Voigt notation
        C15 : float, optional
            C15 component of the tensor, only used for point groups 3 and -3.
        phase_name : str, optional
            Phase name to display
        Returns
        -------
        FourthOrderTensor

        See Also
        --------
        tetragonal : create a tensor from tetragonal symmetry
        orthorhombic : create a tensor from orthorhombic symmetry
        """
        return cls.fromCrystalSymmetry(point_group='3', C11=C11, C12=C12, C13=C13, C14=C14, C15=C15,
                                       C33=C33, C44=C44, phase_name=phase_name, prefix='C')

    @classmethod
    def tetragonal(cls, *, C11=0., C12=0., C13=0., C33=0., C44=0., C16=0., C66=0., phase_name=None):
        """
        Create a fourth-order tensor from tetragonal symmetry.

        Parameters
        ----------
        C11,  C12, C13, C33, C44, C66 : float
            Components of the tensor, using the Voigt notation
        C16 : float, optional
            C16 component in Voigt notation (for point groups 4, -4 and 4/m only)
        phase_name : str, optional
            Phase name to display

        Returns
        -------
        FourthOrderTensor

        See Also
        --------
        trigonal : create a tensor from trigonal symmetry
        orthorhombic : create a tensor from orthorhombic symmetry
        """
        return cls.fromCrystalSymmetry(point_group='4', C11=C11, C12=C12, C13=C13, C16=C16,
                                       C33=C33, C44=C44, C66=C66, phase_name=phase_name, prefix='C')

    @classmethod
    def cubic(cls, *, C11=0., C12=0., C44=0., phase_name=None):
        """
        Create a fourth-order tensor from cubic symmetry.

        Parameters
        ----------
        C11 , C12, C44 : float
        phase_name : str, optional
            Phase name to display

        Returns
        -------
        StiffnessTensor

        See Also
        --------
        hexagonal : create a tensor from hexagonal symmetry
        orthorhombic : create a tensor from orthorhombic symmetry
        """
        return cls.fromCrystalSymmetry(symmetry='cubic', C11=C11, C12=C12, C44=C44, phase_name=phase_name, prefix='C')

    @classmethod
    def orthorhombic(cls, *, C11=0., C12=0., C13=0., C22=0., C23=0., C33=0., C44=0., C55=0., C66=0., phase_name=None):
        """
        Create a fourth-order tensor from orthorhombic symmetry.

        Parameters
        ----------
        C11, C12, C13, C22, C23, C33, C44, C55, C66 : float
            Components of the tensor, using the Voigt notation
        phase_name : str, optional
            Phase name to display

        Returns
        -------
        FourthOrderTensor

        See Also
        --------
        monoclinic : create a tensor from monoclinic symmetry
        orthorhombic : create a tensor from orthorhombic symmetry
        """
        return cls.fromCrystalSymmetry(symmetry='orthorhombic',
                                       C11=C11, C12=C12, C13=C13, C22=C22, C23=C23, C33=C33, C44=C44, C55=C55, C66=C66,
                                       phase_name=phase_name, prefix='C')

    @classmethod
    def monoclinic(cls, *, C11=0., C12=0., C13=0., C22=0., C23=0., C33=0., C44=0., C55=0., C66=0.,
                   C15=None, C25=None, C35=None, C46=None,
                   C16=None, C26=None, C36=None, C45=None,
                   phase_name=None):
        """
        Create a fourth-order tensor from monoclinic symmetry. It automatically detects whether the components are given
        according to the Y or Z diad, depending on the input arguments.

        For Diad || y, C15, C25, C35 and C46 must be provided.
        For Diad || z, C16, C26, C36 and C45 must be provided.

        Parameters
        ----------
        C11, C12 , C13, C22, C23, C33, C44, C55, C66 : float
            Components of the tensor, using the Voigt notation
        C15 : float, optional
            C15 component of the tensor (if Diad || y)
        C25 : float, optional
            C25 component of the tensor (if Diad || y)
        C35 : float, optional
            C35 component of the tensor (if Diad || y)
        C46 : float, optional
            C46 component of the tensor (if Diad || y)
        C16 : float, optional
            C16 component of the tensor (if Diad || z)
        C26 : float, optional
            C26 component of the tensor (if Diad || z)
        C36 : float, optional
            C36 component of the tensor (if Diad || z)
        C45 : float, optional
            C45 component of the tensor (if Diad || z)
        phase_name : str, optional
            Name to display

        Returns
        -------
        FourthOrderTensor

        See Also
        --------
        triclinic : create a tensor from triclinic symmetry
        orthorhombic : create a tensor from orthorhombic symmetry
        """
        diad_y = not (None in (C15, C25, C35, C46))
        diad_z = not (None in (C16, C26, C36, C45))
        if diad_y and diad_z:
            raise KeyError('Ambiguous diad. Provide either C15, C25, C35 and C46; or C16, C26, C36 and C45')
        elif diad_y:
            return cls.fromCrystalSymmetry(symmetry='monoclinic', diad='y',
                                           C11=C11, C12=C12, C13=C13, C22=C22, C23=C23, C33=C33, C44=C44, C55=C55,
                                           C66=C66,
                                           C15=C15, C25=C25, C35=C35, C46=C46, phase_name=phase_name, prefix='C')
        elif diad_z:
            return cls.fromCrystalSymmetry(symmetry='monoclinic', diad='z',
                                           C11=C11, C12=C12, C13=C13, C22=C22, C23=C23, C33=C33, C44=C44, C55=C55,
                                           C66=C66,
                                           C16=C16, C26=C26, C36=C36, C45=C45, phase_name=phase_name, prefix='C')
        else:
            raise KeyError('For monoclinic symmetry, one should provide either C15, C25, C35 and C46, '
                           'or C16, C26, C36 and C45.')

    @classmethod
    def triclinic(cls, C11=0., C12=0., C13=0., C14=0., C15=0., C16=0.,
                  C22=0., C23=0., C24=0., C25=0., C26=0.,
                  C33=0., C34=0., C35=0., C36=0.,
                  C44=0., C45=0., C46=0.,
                  C55=0., C56=0.,
                  C66=0., phase_name=None):
        """

        Parameters
        ----------
        C11 , C12 , C13 , C14 , C15 , C16 , C22 , C23 , C24 , C25 , C26 , C33 , C34 , C35 , C36 , C44 , C45 , C46 , C55 , C56 , C66 : float
            Components of the tensor
        phase_name : str, optional
            Name to display

        Returns
        -------
        FourthOrderTensor

        See Also
        --------
        monoclinic : create a tensor from monoclinic symmetry
        orthorhombic : create a tensor from orthorhombic symmetry
        """
        matrix = np.array([[C11, C12, C13, C14, C15, C16],
                           [C12, C22, C23, C24, C25, C26],
                           [C13, C23, C33, C34, C35, C36],
                           [C14, C24, C34, C44, C45, C46],
                           [C15, C25, C35, C45, C55, C56],
                           [C16, C26, C36, C46, C56, C66]])
        return cls(matrix, phase_name=phase_name)

    def _single_tensor_only(self, fun_name=''):
        if self.ndim:
            err_msg = fun_name + ' is not suitable for tensor array. Consider subscripting (e.g. C[0].{}).'.format(fun_name)
            raise ValueError(err_msg)

    @property
    def Young_modulus(self):
        """
        Directional Young's modulus

        Returns
        -------
        SphericalFunction
            Young's modulus
        """
        self._single_tensor_only('Young_modulus')
        if isinstance(self, ComplianceTensor):
            S = self
        else:
            S = self.inv()
        def compute_young_modulus(u):
            a = np.einsum('ijkl,...i,...j,...k,...l->...', S.full_tensor(), u, u, u, u)
            return 1 / a

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
        self._single_tensor_only('shear_modulus')
        if isinstance(self, ComplianceTensor):
            S = self
        else:
            S = self.inv()
        def compute_shear_modulus(u, v):
            G =  0.25/np.einsum('ijkl,...i,...j,...k,...l->...',S.full_tensor(),u,v,u,v)
            return G

        return HyperSphericalFunction(compute_shear_modulus)

    @property
    def Poisson_ratio(self):
        """
        Directional Poisson's ratio

        Returns
        -------
        HyperSphericalFunction
            Poisson's ratio

        Notes
        -----
        If the material undergoes tensile strain :math:`\\varepsilon_{ii}` along the i-th direction, the Poisson ratios
        are defined as:

        .. math::

            \\nu_{ij}=-\\frac{\\partial \\varepsilon_{jj}}{\\partial \\varepsilon_{ii}}

        where :math:`\\varepsilon_{jj}` denotes the (compressive) longitudinal strain along the j-th direction.
        """
        self._single_tensor_only('Poisson_ratio')
        if isinstance(self, ComplianceTensor):
            Sfull = self.full_tensor()
        else:
            Sfull = self.inv().full_tensor()
        def compute_PoissonRatio(u, v):
            numer = np.einsum('ijkl,...i,...j,...k,...l->...',Sfull,v,v,u,u)
            denom = np.einsum('ijkl,...i,...j,...k,...l->...',Sfull,u,u,u,u)
            return -numer / denom

        return HyperSphericalFunction(compute_PoissonRatio)

    @property
    def linear_compressibility(self):
        """
        Compute the directional linear compressibility.

        Returns
        -------
        SphericalFunction
            Directional linear compressibility

        See Also
        --------
        bulk_modulus : bulk modulus of the material
        """
        self._single_tensor_only('linear_compressibility')
        if isinstance(self, ComplianceTensor):
            S = self
        else:
            S = self.inv()
        def compute_linear_compressibility(u):
            return np.einsum('ijkk,...i,...j->...',S.full_tensor(),u,u)

        return SphericalFunction(compute_linear_compressibility)

    @property
    def bulk_modulus(self):
        """
        Compute the bulk modulus of the material

        Returns
        -------
        float or numpy.ndarray
            Bulk modulus

        See Also
        --------
        linear_compressibility : directional linear compressibility
        """
        return self.inv().bulk_modulus

    def Voigt_average(self, axis=None):
        """
        Compute the Voigt average of the stiffness tensor.

        If the tensor is a tensor array, all its values are considered. Otherwise (i.e. if single), the corresponding
        isotropic tensor is returned.

        Parameters
        ----------
        axis : int, optional
            If provided, the average is computed along this axis. Otherwise, the mean is computed on the flattened array.

        Returns
        -------
        StiffnessTensor
            Voigt average of stiffness tensor

        See Also
        --------
        Reuss_average : compute the Reuss average
        Hill_average : compute the Voigt-Reuss-Hill average
        average : generic function for calling either the Voigt, Reuss or Hill average
        """
        if self.ndim:
            return self.mean(axis=axis)
        else:
            c = self._matrix
            A = c[0, 0] + c[1, 1] + c[2, 2]
            B = c[0, 1] + c[0, 2] + c[1, 2]
            C = c[3, 3] + c[4, 4] + c[5, 5]
            C11 = 1 / 5 *  A  + 2 / 15 * B + 4 / 15 * C
            C12 = 1 / 15 * A  + 4 / 15 * B - 2 / 15 * C
            C44 = (A - B) / 15 + C / 5
            mat = _isotropic_matrix(C11, C12, C44)
            return StiffnessTensor(mat, symmetry='isotropic', phase_name=self.phase_name)

    def Reuss_average(self, axis=None):
        """
        Compute the Reuss average of the stiffness tensor. If the tensor contains no orientation, we assume isotropic
        behaviour. Otherwise, the mean is computed over all orientations.

        Parameters
        ----------
        axis : int, optional
            If provided, axis to compute the average along with. If none, the average is computed on the flattened array

        Returns
        -------
        StiffnessTensor
            Reuss average of stiffness tensor

        See Also
        --------
        Voigt_average : compute the Voigt average
        Hill_average : compute the Voigt-Reuss-Hill average
        average : generic function for calling either the Voigt, Reuss or Hill average
        """
        return self.inv().Reuss_average(axis=axis).inv()

    def Hill_average(self, axis=None):
        """
        Compute the (Voigt-Reuss-)Hill average of the stiffness tensor. If the tensor contains no orientation, we assume
        isotropic behaviour. Otherwise, the mean is computed over all orientations.

        Parameters
        ----------
        axis : int, optional
            If provided, axis to compute the average along with. If none, the average is computed on the flattened array

        Returns
        -------
        StiffnessTensor
            Voigt-Reuss-Hill average of tensor

        See Also
        --------
        Voigt_average : compute the Voigt average
        Reuss_average : compute the Reuss average
        average : generic function for calling either the Voigt, Reuss or Hill average
        """
        Reuss = self.Reuss_average(axis=axis)
        Voigt = self.Voigt_average(axis=axis)
        return (Reuss + Voigt) * 0.5

    def average(self, method, axis=None):
        """
        Compute either the Voigt, Reuss, or Hill average of the stiffness tensor.

        This function is just a shortcut for Voigt_average(), Reuss_average(), or Hill_average() and Hill_average().

        Parameters
        ----------
        axis : int, optional
            If provided, axis to compute the average along with. If none, the average is computed on the flattened array
        method : str {'Voigt', 'Reuss', 'Hill'}
            Method to use to compute the average.

        Returns
        -------
        StiffnessTensor

        See Also
        --------
        Voigt_average : compute the Voigt average
        Reuss_average : compute the Reuss average
        Hill_average : compute the Voigt-Reuss-Hill average
        """
        method = method.capitalize()
        if method in ('Voigt', 'Reuss', 'Hill'):
            fun = getattr(self, method + '_average')
            return fun(axis=axis)
        else:
            raise NotImplementedError('Only Voigt, Reus, and Hill are implemented.')

    @classmethod
    def isotropic(cls, E=None, nu=None, lame1=None, lame2=None, phase_name=None):
        """
        Create an isotropic stiffness tensor from two elasticity coefficients, namely: E, nu, lame1, or lame2. Exactly
        two of these coefficients must be provided.

        Parameters
        ----------
        E : float, None
            Young modulus
        nu : float, None
            Poisson ratio
        lame1 : float, None
            First Lamé coefficient
        lame2 : float, None
            Second Lamé coefficient
        phase_name : str, None
            Name to print

        Returns
        -------
            Corresponding isotropic stiffness tensor

        See Also
        --------
        transverse_isotropic : create a transverse-isotropic tensor

        Notes
        -----
        The units you use when passing the elastic moduli must be consistent with that of the stress tensor. For
        instance, if you expect to work in MPa, the Young's modulus and the Lamé's coefficient must be given in MPa as
        well.

        Examples
        --------
        On can check that the shear modulus for steel is around 82 GPa:

        >>> from Elasticipy.tensors.elasticity import StiffnessTensor
        >>> C=StiffnessTensor.isotropic(E=210e3, nu=0.28)
        >>> C.shear_modulus
        Hyperspherical function
        Min=82031.24999999997, Max=82031.25000000006
        """
        return ComplianceTensor.isotropic(E=E, nu=nu, lame1=lame1, lame2=lame2, phase_name=phase_name).inv()

    @classmethod
    def orthotropic(cls, *, Ex, Ey, Ez, Gxy, Gxz, Gyz,
                    nu_yx=None, nu_zx=None, nu_zy=None,
                    nu_xy=None, nu_xz=None, nu_yz=None, **kwargs):
        """
        Create a stiffness tensor corresponding to orthotropic symmetry, given the engineering constants.

        Exactly three Poisson ratios must be provided. See Notes for details.

        Parameters
        ----------
        Ex : float
            Young modulus along the x axis
        Ey : float
            Young modulus along the y axis
        Ez : float
            Young modulus along the z axis
        Gxy : float
            Shear modulus in the x-y plane
        Gxz : float
            Shear modulus in the x-z plane
        Gyz : float
            Shear modulus in the y-z plane
        nu_xy, nu_yx : float, optional
            Poisson ratio along x and y axes. Either nu_xy or nu_yx must be provided, not both.
        nu_xz, nu_zx : float, optional
            Poisson ratio along x and z axes. Either nu_xz or nu_zx must be provided, not both.
        nu_yz, nu_zy : float, optional
            Poisson ratio along y and z axes. Either nu_yz or nu_zy must be provided, not both.
        kwargs : dict, optional
            Keyword arguments to pass to the StiffnessTensor constructor

        Returns
        -------
        StiffnessTensor

        See Also
        --------
        transverse_isotropic : create a stiffness tensor for transverse-isotropic symmetry

        Notes
        -----
        If the material undergoes tensile strain :math:`\\varepsilon_{ii}` along the i-th direction, the Poisson ratios
        are defined as:

        .. math::

            \\nu_{ij}=-\\frac{\\partial \\varepsilon_{jj}}{\\partial \\varepsilon_{ii}}

        where :math:`\\varepsilon_{jj}` denotes the (compressive) longitudinal strain along the j-th direction. If
        :math:`E_x` and :math:`E_y` are the Young moduli along `x` and `y`, we have:

        .. math::

            \\frac{\\nu_{xy}}{E_x} = \\frac{\\nu_{yx}}{E_y}

        """
        nu_yx = _switch_poisson_ratios(nu_xy, nu_yx, Ex, Ey,'xy')
        nu_zx = _switch_poisson_ratios(nu_xz, nu_zx, Ex, Ez,'xz')
        nu_zy = _switch_poisson_ratios(nu_yz, nu_zy, Ey, Ez,'yz')
        tri_sup = np.array([[1 / Ex, -nu_yx / Ey, -nu_zx / Ez, 0, 0, 0],
                            [0, 1 / Ey, -nu_zy / Ez, 0, 0, 0],
                            [0, 0, 1 / Ez, 0, 0, 0],
                            [0, 0, 0, 1 / Gyz, 0, 0],
                            [0, 0, 0, 0, 1 / Gxz, 0],
                            [0, 0, 0, 0, 0, 1 / Gxy]])
        S = tri_sup + np.tril(tri_sup.T, -1)
        return StiffnessTensor(np.linalg.inv(S), symmetry='orthotropic', **kwargs)

    @classmethod
    def transverse_isotropic(cls, *, Ex, Ez, Gxz, nu_yx=None, nu_xy=None, nu_zx=None, nu_xz=None, **kwargs):
        """
        Create a stiffness tensor corresponding to the transversely isotropic symmetry with respect to Z axis, given the
        engineering constants.

        Exactly two Poisson ratios must be provided (nu_xy or nu_yx, and nu_xz or nu_zx). See Notes for details.

        Parameters
        ----------
        Ex : float
            Young modulus along the x axis
        Ez : float
            Young modulus along the y axis
        Gxz : float
            Shear modulus in the x-z plane
        nu_xy, nu_yx : float, optional
            Poisson ratio along x and y. Either nu_xy or nu_yx must be provided, not both.
        nu_xz, nu_zx : float, optional
            Poisson ratio along x and z. Either nu_xz or nu_zx must be provided, not both.
        kwargs : dict
            Keyword arguments to pass to the StiffnessTensor constructor

        Returns
        -------
        StiffnessTensor

        See Also
        --------
        orthotropic : create a stiffness tensor for orthotropic symmetry

        Notes
        -----
        If the material undergoes tensile strain :math:`\\varepsilon_{ii}` along the i-th direction, the Poisson ratios
        are defined as:

        .. math::

            \\nu_{ij}=-\\frac{\\partial \\varepsilon_{jj}}{\\partial \\varepsilon_{ii}}

        where :math:`\\varepsilon_{jj}` denotes the (compressive) longitudinal strain along the j-th direction. If
        :math:`E_x` and :math:`E_y` are the Young moduli along `x` and `y`, we have:

        .. math::

            \\frac{\\nu_{xy}}{E_x} = \\frac{\\nu_{yx}}{E_y}
        """
        nu_yx = _switch_poisson_ratios(nu_xy, nu_yx, Ex, Ex,'xy')
        Gxy = Ex / (2 * (1 + nu_yx))
        C = StiffnessTensor.orthotropic(Ex=Ex, Ey=Ex, Ez=Ez,
                                        nu_yx=nu_yx, nu_zx=nu_zx, nu_zy=nu_zx,
                                        nu_xz=nu_xz, nu_yz=nu_xz,
                                        Gxy=Gxy, Gxz=Gxz, Gyz=Gxz, **kwargs)
        C.symmetry = 'transverse-isotropic'
        return C

    def Christoffel_tensor(self, u):
        """
        Create the Christoffel tensor along a given direction, or set or directions.

        Parameters
        ----------
        u : list or np.ndarray
            3D direction(s) to compute the Christoffel tensor along with

        Returns
        -------
        Gamma : np.ndarray
            Array of Christoffel tensor(s). if u is a list of directions, Gamma[i] is the Christoffel tensor for
            direction  u[i].

        See Also
        --------
        wave_velocity : computes the p- and s-wave velocities.

        Notes
        -----
        For a given stiffness tensor **C** and a given unit vector **u**, the Christoffel tensor is defined as [2]_ :

            .. math:: M_{ij} = C_{iklj}.u_k.u_l

        """
        u_vec = np.atleast_2d(u)
        u_vec = (u_vec.T / np.linalg.norm(u_vec, axis=1)).T
        return np.einsum('inmj,pn,pm->pij', self.full_tensor(), u_vec, u_vec)

    def wave_velocity(self, rho):
        """
        Compute the wave velocities, given the mass density.

        Parameters
        ----------
        rho : float
            mass density. Its unit must be consistent with that of the stiffness tensor. See notes for hints.

        See Also
        --------
        Christoffel_tensor : Computes the Christoffel tensor along a given direction

        Returns
        -------
        c_p : SphericalFunction
            Velocity of the primary (compressive) wave
        c_s1 : SphericalFunction
            Velocity of the fast secondary (shear) wave
        c_s2 : SphericalFunction
            Velocity of the slow secondary (shear) wave

        Notes
        -----
        The estimation of the wave velocities is made by finding the eigenvalues of the Christoffel tensor [2]_.

        One should double-check the units. The table below provides hints about the unit you get, depending on the units
        you use for stiffness and the mass density:

        +-----------------+--------------+------------+-----------------------+
        | Stiffness       | Mass density | Velocities | Notes                 |
        +=================+==============+============+=======================+
        | Pa (N/m²)       | kg/m³        | m/s        | SI units              |
        +-----------------+--------------+------------+-----------------------+
        | GPa (10⁹ Pa)    | kg/dm³       | km/s       | Conversion factor     |
        +-----------------+--------------+------------+-----------------------+
        | GPa (10³ N/mm²) | kg/mm³       | m/s        | Consistent units      |
        +-----------------+--------------+------------+-----------------------+
        | MPa (10⁶ Pa)    | kg/m³        | km/s       | Conversion factor     |
        +-----------------+--------------+------------+-----------------------+
        | MPa (10³ N/mm²) | g/mm³        | m/s        | Consistent units      |
        +-----------------+--------------+------------+-----------------------+

        References
        ----------
        .. [2] J. W. Jaeken, S. Cottenier, Solving the Christoffel equation: Phase and group velocities, Computer Physics
               Communications (207), 2016, https://doi.org/10.1016/j.cpc.2016.06.014.

        """
        self._single_tensor_only('wave_velocity')
        def make_fun(index):
            def fun(n):
                Gamma = self.Christoffel_tensor(n)
                eig, _ = np.linalg.eig(Gamma)
                if index == 0:
                    eig_of_interest = np.max(eig, axis=-1)
                elif index == 1:
                    eig_of_interest = np.median(eig, axis=-1)
                else:
                    eig_of_interest = np.min(eig, axis=-1)
                return np.sqrt(eig_of_interest / rho)

            return fun

        return [SphericalFunction(make_fun(i)) for i in range(3)]

    @classmethod
    def from_MP(cls, ids, api_key=None):
        """
        Import stiffness tensor(s) from the Materials Project API, given their material ids.

        You need to register to `<https://materialsproject.org>`_ first to get an API key. This key can be explicitly
        passed as an argument (see below), or provided as an environment variable named MP_API_KEY.

        Parameters
        ----------
        ids : str or list of str
            ID(s) of the material to import (e.g. "mp-1048")
        api_key : str, optional
            API key to the Materials Project API. If not provided, it should be available as the API_KEY environment
            variable.

        Returns
        -------
        list of StiffnessTensor
            If one of the requested material ids was not found, the corresponding value in the list will be None.
        """
        try:
            from mp_api.client import MPRester
        except ImportError:
            raise ModuleNotFoundError('mp_api module is required for this function.')
        if type(ids) is str:
            Cdict = dict.fromkeys([ids])
        else:
            Cdict = dict.fromkeys(ids)
        with MPRester(api_key=api_key) as mpr:
            elasticity_doc = mpr.materials.elasticity.search(material_ids=ids)
            for material in elasticity_doc:
                key = str(material.material_id)
                if material.elastic_tensor is not None:
                    matrix = material.elastic_tensor.ieee_format
                    symmetry = material.symmetry.crystal_system.value
                    phase_name = material.formula_pretty
                    C = StiffnessTensor(np.asarray(matrix), symmetry=str(symmetry), phase_name=phase_name)
                else:
                    C = None
                Cdict[key] = C
            if elasticity_doc:
                if isinstance(ids, str):
                    return C
                else:
                    return [Cdict[id] for id in ids]
            else:
                return None

    @classmethod
    def weighted_average(cls, Cs, volume_fractions, method):
        """
        Compute the weighted average of a list of stiffness tensors, with respect to a given method (Voigt, Reuss or
        Hill).

        Parameters
        ----------
        Cs : list of StiffnessTensor or list of ComplianceTensor or tuple of StiffnessTensor or tuple of ComplianceTensor
            Series of tensors to compute the average from
        volume_fractions : iterable of floats
            Volume fractions of each phase
        method : str, {'Voigt', 'Reuss', 'Hill'}
            Method to use. It can be 'Voigt', 'Reuss', or 'Hill'.

        Returns
        -------
        StiffnessTensor
            Average tensor
        """
        if np.all([isinstance(a, ComplianceTensor) for a in Cs]):
            Cs = [C.inv() for C in Cs]
        if np.all([isinstance(a, StiffnessTensor) for a in Cs]):
            C_stack = np.array([C._matrix for C in Cs])
            method = method.capitalize()
            if method == 'Voigt':
                C_avg = np.average(C_stack, weights=volume_fractions, axis=0)
                return StiffnessTensor(C_avg)
            elif method == 'Reuss':
                S_stack = np.linalg.inv(C_stack)
                S_avg = np.average(S_stack, weights=volume_fractions, axis=0)
                return StiffnessTensor(np.linalg.inv(S_avg))
            elif method == 'Hill':
                C_voigt = cls.weighted_average(Cs, volume_fractions, 'Voigt')
                C_reuss = cls.weighted_average(Cs, volume_fractions, 'Reuss')
                return (C_voigt + C_reuss) * 0.5
            else:
                raise ValueError('Method must be either Voigt, Reuss or Hill.')
        else:
            raise ValueError('The first argument must be either a list of ComplianceTensors or '
                             'a list of StiffnessTensor.')

    @property
    def universal_anisotropy(self):
        """
        Compute the universal anisotropy factor.

        The larger the value, the more likely the material will behave in an anisotropic way.

        Returns
        -------
        float
            The universal anisotropy factor.

        Notes
        -----
        The universal anisotropy factor is defined as [3]_:

        .. math::

            5\\frac{G_v}{G_r} + \\frac{K_v}{K_r} - 6

        References
        ----------
        .. [3] S. I. Ranganathan and M. Ostoja-Starzewski, Universal Elastic Anisotropy Index,
           *Phys. Rev. Lett.*, 101(5), 055504, 2008. https://doi.org/10.1103/PhysRevLett.101.055504
        """
        self._single_tensor_only('universal_anisotropy')
        Cvoigt = self.Voigt_average()
        Creuss = self.Reuss_average()
        Gv = Cvoigt._matrix[3, 3]
        Gr = Creuss._matrix[3, 3]
        Kv = Cvoigt.bulk_modulus
        Kr = Creuss.bulk_modulus
        return 5 * Gv / Gr + Kv / Kr - 6

    def Zener_ratio(self, tol=1e-4):
        """
        Compute the Zener ratio (Z). Only valid for cubic symmetry.

        This function first checks that the tensor has cubic symmetry within a given tolerance. If not, an error is
        raised.

        Parameters
        ----------
        tol : float, optional
            Tolerance to consider that the material has cubic symmetry

        Returns
        -------
        float
            Zener ratio (NaN is the symmetry is not cubic)

        Notes
        -----
        If the tensor is written in canonical base with Voigt mapping, the Zener ratio is defined as:

        .. math::

                Z=\\frac{ 2C_{44} }{C_{11} - C_{12}}

        The present implementation takes advantage of eigenstiffness to compute the Zener ratio in *any* base, i.e. even
        if the tensor is not given in canonical base (e.g. if rotated).

        See Also
        --------
        universal_anisotropy : compute the universal anisotropy factor
        eig_stiffnesses : eigenstiffnesses of the tensor
        is_cubic : check whether the tensor has cubic symmetry or not

        Examples
        --------
        >>> from Elasticipy.tensors.elasticity import StiffnessTensor
        >>> C = StiffnessTensor.cubic(C11=200, C12=40, C44=20)
        >>> C.Zener_ratio()
        0.25

        which obvisouly corresponds to 2.C44/(C11-C12).

        Now, rotate the tensor and see how it looks like:

        >>> from scipy.spatial.transform import Rotation
        >>> g = Rotation.from_euler('Z', 30, degrees=True)
        >>> C_rot = C*g
        >>> C_rot
        Stiffness tensor (in Voigt mapping):
        [[155.          85.          40.           0.           0.
           25.98076211]
         [ 85.         155.          40.           0.           0.
          -25.98076211]
         [ 40.          40.         200.           0.           0.
            0.        ]
         [  0.           0.           0.          20.           0.
            0.        ]
         [  0.           0.           0.           0.          20.
            0.        ]
         [ 25.98076211 -25.98076211   0.           0.           0.
           65.        ]]
        Symmetry: cubic

        Still, we have
        >>> C_rot.Zener_ratio()
        0.24999999999999983
        """
        if self.is_isotropic():
            return 1.0
        elif self.is_cubic():
            eigs, orders = self.eig_stiffnesses_multiplicity(1e-4)
            numer = eigs[orders==3][0]  # 2*C44
            denom = eigs[orders==2][0]  # C11-C12
            return numer / denom
        else:
            raise ValueError('The tensor does not seem to have cubic symmetry within the given tolerance ({})'.format(tol))

    def to_pymatgen(self):
        """
        Convert the stiffness tensor (from Elasticipy) to Python Materials Genomics (Pymatgen) format.

        Returns
        -------
        pymatgen.analysis.elasticity.elastic.ElasticTensor
            Stiffness tensor for pymatgen
        """
        try:
            from pymatgen.analysis.elasticity import elastic as matgenElast
        except ImportError:
            raise ModuleNotFoundError('pymatgen module is required for this function.')
        return matgenElast.ElasticTensor(self.full_tensor())

    def to_Kelvin(self):
        """
        Returns all the tensor components using the Kelvin(-Mandel) mapping convention.

        Returns
        -------
        numpy.ndarray
            (6,6) matrix, according to the Kelvin mapping

        See Also
        --------
        eig : returns the eigenvalues and the eigenvectors of the Kelvin's matrix
        from_Kelvin : Construct a fourth-order tensor from its (6,6) Kelvin matrix

        Notes
        -----
        This mapping convention is defined as follows [4]_:

        .. math::

            C_K = \\begin{bmatrix}
                C_{11}          & C_{12}            & C_{13}            & \\sqrt{2}C_{14} & \\sqrt{2}C_{15} & \\sqrt{2}C_{16}\\\\
                C_{12}          & C_{22}            & C_{23}            & \\sqrt{2}C_{24} & \\sqrt{2}C_{25} & \\sqrt{2}C_{26}\\\\
                C_{13}          & C_{23}            & C_{33}            & \\sqrt{2}C_{34} & \\sqrt{2}C_{35} & \\sqrt{2}C_{36}\\\\
                \\sqrt{2}C_{14} & \\sqrt{2}C_{24}   & \\sqrt{2}C_{34}   & 2C_{44}         & 2C_{45}         & 2C_{46}\\\\
                \\sqrt{2}C_{15} & \\sqrt{2}C_{25}   & \\sqrt{2}C_{35}   & 2C_{45}         & 2C_{55}         & 2C_{56}\\\\
                \\sqrt{2}C_{16} & \\sqrt{2}C_{26}   & \\sqrt{2}C_{36}   & 2C_{46}         & 2C_{56}         & 2C_{66}\\\\
            \\end{bmatrix}


        References
        ----------
        .. [4] Helbig, K. (2013). What Kelvin might have written about Elasticity. Geophysical Prospecting, 61(1), 1-20.
            doi: 10.1111/j.1365-2478.2011.01049.x
        """
        kelvin_mapping = KelvinMapping()
        return self._matrix /self.mapping.matrix * kelvin_mapping.matrix

    def eig(self):
        """
        Compute the eigenstiffnesses and the eigenstrains.

        Solve the eigenvalue problem from the Kelvin matrix of the stiffness tensor (see Notes).

        Returns
        -------
        numpy.ndarray
            Array of 6 eigenstiffnesses (eigenvalues of the stiffness matrix)
        numpy.ndarray
            (6,6) array of eigenstrains (eigenvectors of the stiffness matrix)

        See Also
        --------
        to_Kelvin : returns the stiffness components as a (6,6) matrix, according to the Kelvin mapping convention.
        eig_stiffnesses : returns the eigenstiffnesses only
        eig_strains : returns the eigenstrains only

        Notes
        -----
        The definition for eigenstiffnesses and the eigenstrains are introduced in [4]_.
        """
        return np.linalg.eigh(self.to_Kelvin())

    @property
    def eig_stiffnesses(self):
        """
        Compute the eigenstiffnesses given by the Kelvin's matrix for stiffness.

        Returns
        -------
        numpy.ndarray
            6 eigenvalues of the Kelvin's stiffness matrix, in ascending order

        See Also
        --------
        eig : returns the eigenstiffnesses and the eigenstrains
        eig_strains : returns the eigenstrains only
        eig_stiffnesses_multiplicity : returns the unique values of eigenstiffnesses with multiplicity
        """
        return np.linalg.eigvalsh(self.to_Kelvin())

    @property
    def eig_strains(self):
        """
        Compute the eigenstrains from the Kelvin's matrix for stiffness

        Returns
        -------
        numpy.ndarray
            (6,6) matrix of eigenstrains, sorted by ascending order of eigenstiffnesses.

        See Also
        --------
        eig : returns both the eigenvalues and the eigenvectors of the Kelvin matrix
        """
        return self.eig()[1]

    @property
    def eig_compliances(self):
        """
        Compute the eigencompliances from the Kelvin's matrix of stiffness

        Returns
        -------
        numpy.ndarray
            Inverses of the 6 eigenvalues of the Kelvin's stiffness matrix, in descending order

        See Also
        --------
        eig_stiffnesses : compute the eigenstiffnesses from the Kelvin's matrix of stiffness
        """
        return 1/self.eig_stiffnesses

    @classmethod
    def from_Kelvin(cls, matrix, **kwargs):
        """
        Create a tensor from the (6,6) matrix following the Kelvin(-Mandel) mapping convention

        Parameters
        ----------
        matrix : list or numpy.ndarray
            (6,6) matrix of components
        kwargs
            keyword arguments passed to the constructor

        Returns
        -------
        StiffnessTensor

        See Also
        --------
        to_Kelvin : return the components as a (6,6) matrix following the Kelvin convention
        """
        kelvin_mapping = KelvinMapping()
        t = cls(matrix / kelvin_mapping.matrix, **kwargs)
        t._matrix *= t.mapping.matrix
        return t

    def eig_stiffnesses_multiplicity(self, tol=1e-4):
        """
        Compute the eigenstiffnesses, then returns the multiplicity of each eigenstiffness.

        Given an absolute tolerance, duplicates in eigenstiffnesses are considered to compute the multiplicity of each
        value.

        Parameters
        ----------
        tol : float, optional
            Absolute tolerance to assume that two distinct eigenstiffnesses are the same

        Returns
        -------
        numpy.ndarray
            Unique values of eigenstiffnesses, sorted by increasing multiplicity
        numpy.ndarray
            Multiplicity of each unique eigenstiffness, sorted by ascending value

        See Also
        --------
        eig_stiffnesses : compute the eigenstiffnesses

        Examples
        --------
        >>> from Elasticipy.tensors.elasticity import StiffnessTensor
        >>> C = StiffnessTensor.cubic(C11=186, C12=134, C44=77)
        >>> C.eig_stiffnesses
        array([ 52.,  52., 154., 154., 154., 454.])
        >>> C.eig_stiffnesses_multiplicity()
        (array([ 52., 154., 454.]), array([2, 3, 1]))
        """
        eig = self.eig_stiffnesses
        counts = []
        uniques = []
        while eig.size:
            duplicates = np.isclose(eig[0], eig, atol=tol)
            counts.append(np.count_nonzero(duplicates))
            uniques.append(eig[0])
            eig = eig[np.logical_not(duplicates)]
        return np.array(uniques), np.array(counts)

    def _check_eig_signature(self, signature, tol):
        if self.shape:
            flat = self.flatten()
            bool_array = np.zeros(flat.shape[0], dtype=bool)
            for i, t in enumerate(flat):
                _, order = t.eig_stiffnesses_multiplicity(tol=tol)
                bool_array[i] = np.array_equal(np.sort(order), np.sort(signature))
            return bool_array.reshape(self.shape)
        else:
            _, order = self.eig_stiffnesses_multiplicity(tol=tol)
            return np.array_equal(np.sort(order), np.sort(signature))

    def is_isotropic(self, tol=0.01):
        """Check that the tensor corresponds to isotropic symmetry, within a given tolerance.

        The method relies on the multiplicity of eigenstiffnesses

        Parameters
        ----------
        tol : float
            Absolute tolerance to consider multiplicity of eigenstiffnesses

        Returns
        -------
        bool or numpy.ndarray
            If the tensor is single, the returned value is boolean.
            If the object is a tensor array, the returned value is an array of bools, the same shape as the tensor array.

        See Also
        --------
        is_cubic : check if the stiffness tensor has cubic symmetry
        is_tetragonal : check if the stiffness tensor has tetragonal symmetry
        eig_stiffnesses : compute eigenstiffnesses
        """
        return self._check_eig_signature([1, 5], tol)

    def is_cubic(self, tol=0.01):
        """Check that the tensor corresponds to cubic symmetry, within a given tolerance.

        The method relies on the multiplicity of eigenstiffnesses.

        Parameters
        ----------
        tol : float, optional
            Absolute tolerance to consider multiplicity of eigenstiffnesses

        Returns
        -------
        bool or numpy.ndarray
            If the tensor is single, the returned value is boolean.
            If the object is a tensor array, the returned value is an array of bools, the same shape as the tensor array.

        See Also
        --------
        is_isotropic : check if the stiffness tensor is isotropic
        is_tetragonal : check if the stiffness tensor has tetragonal symmetry
        eig_stiffnesses : compute eigenstiffnesses

        Examples
        --------
        >>> from Elasticipy.tensors.elasticity import StiffnessTensor
        >>> from scipy.spatial.transform import Rotation
        >>> C = StiffnessTensor.cubic(C11=186, C12=134, C44=77)
        >>> C_rotated = C * Rotation.random(random_state=123)
        >>> C_rotated
        Stiffness tensor (in Voigt mapping):
        [[237.71171578  96.41409344 119.87419078   8.1901353   -3.63846312
          -20.34233446]
         [ 96.41409344 250.74909842 106.83680814   9.33462785  -6.52548033
            0.99714278]
         [119.87419078 106.83680814 227.28900108 -17.52476315  10.16394345
           19.34519167]
         [  8.1901353    9.33462785 -17.52476315  49.83680814  19.34519167
           -6.52548033]
         [ -3.63846312  -6.52548033  10.16394345  19.34519167  62.87419078
            8.1901353 ]
         [-20.34233446   0.99714278  19.34519167  -6.52548033   8.1901353
           39.41409344]]
        Symmetry: cubic

        Once rotated, it is not clear if the stiffness tensors has cubic symmetry. Yet:
        >>> C_rotated.is_cubic()
        True

        """
        return np.logical_or(self._check_eig_signature([1, 2, 3], tol), self.is_isotropic(tol=tol))

    def is_tetragonal(self, tol=0.01):
        """Check that the tensor corresponds to tetragonal symmetry, within a given tolerance.

        The method relies on the multiplicity of eigenstiffnesses.

        Parameters
        ----------
        tol : float
            Absolute tolerance to consider multiplicity of eigenstiffnesses

        Returns
        -------
        bool or numpy.ndarray
            If the tensor is single, the returned value is boolean.
            If the object is a tensor array, the returned value is an array of bools, the same shape as the tensor array.

        See Also
        --------
        is_isotropic : check if the stiffness tensor is isotropic
        is_cubic : check if the stiffness tensor has cubic symmetry
        eig_stiffnesses : compute eigenstiffnesses
        """
        return np.logical_or(self._check_eig_signature([1, 1, 1, 1, 2], tol), self.is_cubic(tol=tol))


class ComplianceTensor(StiffnessTensor):
    """
    Class for manipulating compliance tensors
    """
    tensor_name = 'Compliance'
    C11_C12_factor = 2.0
    component_prefix = 'S'
    C46_C56_factor = 2.0

    def __init__(self, C, check_positive_definite=True, mapping=VoigtMapping(tensor='Compliance'), **kwargs):
        super().__init__(C, check_positive_definite=check_positive_definite, mapping=mapping, **kwargs)
        self.mapping_name = 'Voigt'

    def __mul__(self, other):
        if isinstance(other, StressTensor):
            new_tensor = self.ddot(other)
            return StrainTensor(new_tensor.matrix)
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
        S = np.linalg.inv(self._matrix)
        return StiffnessTensor(S, symmetry=self.symmetry, phase_name=self.phase_name)

    def Reuss_average(self, axis=None):
        if self.ndim:
            return self.mean(axis=axis)
        else:
            s = self._matrix
            A = s[0, 0] + s[1, 1] + s[2, 2]
            B = s[0, 1] + s[0, 2] + s[1, 2]
            C = s[3, 3] + s[4, 4] + s[5, 5]
            S11 = 1 / 5 *  A + 2 / 15 * B + 1 / 15 * C
            S12 = 1 / 15 * A + 4 / 15 * B - 1 / 30 * C
            S44 = 4 / 15 * (A - B) + 1 / 5 * C
            mat = _isotropic_matrix(S11, S12, S44)
            return ComplianceTensor(mat, symmetry='isotropic', phase_name=self.phase_name)

    def Voigt_average(self, axis=None):
        return self.inv().Voigt_average(axis=axis).inv()

    def Hill_average(self, axis=None):
        return self.inv().Hill_average(axis=axis).inv()

    @classmethod
    def isotropic(cls, E=None, nu=None, lame1=None, lame2=None, K=None, phase_name=None):
        n_specified = sum(v is not None for v in [E, nu, lame1, lame2])
        if n_specified != 2:
            raise ValueError("Exactly two values are required among E, nu, lame1 and lame2.")
        if K is not None:
            if E is not None:
                lame2 = 3 * K * E / (9 * K - E)
                nu = (3 * K - E) / 6 / K
            elif lame1 is not None:
                E = 9 * K * (K - lame1) / (3 * K -lame1)
                lame2= 3 * (K - lame1) / 2
                nu = lame1 / (3*K-lame1)
            elif lame2 is not None:
                E = 9 * K * lame2 / (3 * K + lame2)
                nu = (3 * K - 2 * lame2) / 2 / (3 * K + lame2)
            elif nu is not None:
                E = 3 * K * (1 - 2 * nu)
                lame2 = E / 2 / (1 + nu)
        elif E is not None:
            if lame1 is not None:
                R = np.sqrt(E**2 + 9*lame1**2+2*E*lame1)
                lame2 = (E - 3 * lame1 + R) / 4
                nu = 2 * lame1 / (E + lame1 + R)
            elif lame2 is not None:
                nu = E / 2 / lame2 - 1
            elif nu is not None:
                lame2 = E / 2 / (1 + nu)
        elif lame1 is not None:
            if lame2 is not None:
                E = lame2 *  (3 * lame1 + 2 * lame2) / (lame1 + lame2)
            elif nu is not None:
                E = lame1 * ( 1 + nu) * (1 - 2 * nu) / nu
        elif (nu is not None) and (lame2 is not None):
            E = 2 * lame2 * (1 + nu)
        S11 = 1/E
        S12 = -nu/E
        S44 = 1/lame2
        S_mat = _isotropic_matrix(S11, S12, S44)
        return ComplianceTensor(S_mat, symmetry='isotropic', phase_name=phase_name)

    @classmethod
    def orthotropic(cls, *args, **kwargs):
        return super().orthotropic(*args, **kwargs).inv()

    @classmethod
    def transverse_isotropic(cls, *args, **kwargs):
        return super().transverse_isotropic(*args, **kwargs).inv()

    @classmethod
    def weighted_average(cls, *args):
        return super().weighted_average(*args).inv()

    @property
    def bulk_modulus(self):
        matrix_t = self._matrix.T
        sub_matrix = matrix_t[0:3, 0:3]
        return 1 / np.sum(sub_matrix, axis=(0,1))

    @property
    def universal_anisotropy(self):
        """
        Compute the universal anisotropy factor.

        It is actually an alias for inv().universal_anisotropy.

        Returns
        -------
        float
            Universal anisotropy factor
        """
        return self.inv().universal_anisotropy

    def to_pymatgen(self):
        """
        Convert the compliance tensor (from Elasticipy) to Python Materials Genomics (Pymatgen) format.

        Returns
        -------
        ComplianceTensor
            Compliance tensor for pymatgen
        """
        try:
            from pymatgen.analysis.elasticity import elastic as matgenElast
        except ImportError:
            raise ModuleNotFoundError('pymatgen module is required for this function.')
        return matgenElast.ComplianceTensor(self.full_tensor())

    def eig(self):
        """
        Compute the eigencompliances and the eigenstresses.

        Solve the eigenvalue problem from the Kelvin matrix of the compliance tensor (see Notes).

        Returns
        -------
        numpy.ndarray
            Array of 6 eigencompliances (eigenvalues of the stiffness matrix)
        numpy.ndarray
            (6,6) array of eigenstresses (eigenvectors of the stiffness matrix)

        See Also
        --------
        Kelvin : returns the stiffness components as a (6,6) matrix, according to the Kelvin mapping convention.
        eig_compliances : returns the eigencompliances only
        eig_stresses : returns the eigenstresses only

        Notes
        -----
        The definition for eigencompliances and the eigenstresses are introduced in [4]_.
        """
        return np.linalg.eigh(self.to_Kelvin())

    @property
    def eig_compliances(self):
        """
        Compute the eigencompliances given by the Kelvin's matrix for stiffness.

        Returns
        -------
        numpy.ndarray
            6 eigenvalues of the Kelvin's compliance matrix, in ascending order

        See Also
        --------
        eig : returns the eigencompliances and the eigenstresses
        eig_strains : returns the eigenstresses only
        """
        return np.linalg.eigvalsh(self.to_Kelvin())

    @property
    def eig_stresses(self):
        """
        Compute the eigenstresses from the Kelvin's matrix for stiffness

        Returns
        -------
        numpy.ndarray
            (6,6) matrix of eigenstresses, sorted by ascending order of eigencompliances.

        See Also
        --------
        eig : returns both the eigencompliances and the eigenstresses
        """
        return self.eig()[1]

    @property
    def eig_stiffnesses(self):
        """
        Compute the eigenstiffnesses from the Kelvin's matrix of compliance

        Returns
        -------
        numpy.ndarray
            inverses of 6 eigenvalues of the Kelvin's compliance matrix, in descending order

        See Also
        --------
        eig_compliances : compute the eigencompliances from the Kelvin's matrix of compliance
        """
        return 1/self.eig_compliances