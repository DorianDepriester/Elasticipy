import unittest
from elasticipy.tensors.finite_strain import DeformationGradient
import numpy as np

from elasticipy.tensors.second_order import SymmetricSecondOrderTensor
from elasticipy.tensors.stress_strain import StrainTensor, StressTensor


class TestDeformationGradient(unittest.TestCase):
    def test_polar(self):
        # Test with single tensor
        F = DeformationGradient.rand()
        R, U = F.polar()
        assert U == F.U
        np.testing.assert_array_equal(R, F.R)

        R, V = F.polar(side='left')
        assert V == F.V
        np.testing.assert_array_equal(R, F.R)
        
        # Test with tensor array (stacked matrices)
        shape = (3, 4)
        F_array = DeformationGradient.rand(shape=shape)
        R_array, U_array = F_array.polar()
        np.testing.assert_array_equal(U_array.matrix, F_array.U.matrix)
        np.testing.assert_array_equal(R_array, F_array.R)

        R_array, V_array = F_array.polar(side='left')
        np.testing.assert_array_equal(V_array.matrix, F_array.V.matrix)
        np.testing.assert_array_equal(R_array, F_array.R)

    def test_right_CauchyGreen(self):
        F = DeformationGradient.rand()
        C = F.Cauchy_Green()
        U = F.U
        np.testing.assert_array_almost_equal(C.matrix, U.dot(U).matrix)

    def test_left_CauchyGreen(self):
        F = DeformationGradient.rand()
        B = F.Cauchy_Green(side='left')
        V = F.V
        np.testing.assert_array_almost_equal(B.matrix, V.dot(V).matrix)

    def test_green_lagrange(self):
        shape = (5, 3)
        F = DeformationGradient.rand(shape=shape)
        E = F.E
        assert isinstance(E, SymmetricSecondOrderTensor)
        assert E.shape == F.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                Fij = F[i,j].matrix
                Eij = 0.5*(Fij.T @ Fij - np.eye(3))
                np.testing.assert_array_almost_equal(E[i,j].matrix, Eij)

    def test_negative_J(self):
        with self.assertRaises(ValueError) as context:
            _ = DeformationGradient.diag([-1,2,3])
        self.assertEqual(str(context.exception), "The determinant of the deformation tensor must be positive.")

    def test_small_strain(self):
        gamma = 0.1
        F = DeformationGradient.eye() + np.array([[0, 0.1, 0],[0, 0, 0], [0, 0, 0]])
        eps = F.small_strain()
        assert isinstance(eps, StrainTensor)
        assert eps == StrainTensor.shear([1, 0, 0], [0, 1, 0], gamma/2)

    def test_tensile(self):
        u = [1, 0, 0]
        mag = 0.5
        F = DeformationGradient.tensile(u, mag)
        np.testing.assert_array_equal(F.matrix, np.diag([1.5, 1, 1]))

    def test_shear(self):
        u = [1, 0, 0]
        v = [0, 1, 0]
        mag = 0.5
        F = DeformationGradient.shear(u, v, mag)
        F_the = np.eye(3)
        F_the[0,1] = mag
        np.testing.assert_array_equal(F.matrix, F_the)

    def test_isochoric_tensile(self):
        F = DeformationGradient.isochoric_tensile([1,0,0], 0)
        assert F == DeformationGradient.eye()
        assert F.volumetric_strain() == 0.0

        F = DeformationGradient.isochoric_tensile([0,1,0], 99)
        np.testing.assert_array_equal(F.matrix, np.diag([0.1, 100, 0.1]))

        F = DeformationGradient.isochoric_tensile([1,1,0], [1,2,3,4,5])
        np.testing.assert_array_almost_equal(F.J, np.ones(5))
        np.testing.assert_array_almost_equal(F.volumetric_strain(), np.zeros(5))

    def test_elongation(self):
        F = DeformationGradient.diag([2, 1, 1])
        assert F.elongation([1, 0, 0]) == 1.0
        assert F.elongation([0, 1, 0]) == 0.0
        assert F.elongation([0, 0, 1]) == 0.0

    def test_sqrt(self):
        shape = (4, 3)
        F = DeformationGradient.rand(shape=shape)
        C = F.Cauchy_Green()
        C_sqrt = C.sqrt()
        assert C_sqrt.shape == shape
        assert isinstance(C_sqrt, SymmetricSecondOrderTensor)
        C_sqrt_sq = C_sqrt.dot(C_sqrt)
        np.testing.assert_array_almost_equal(C_sqrt_sq.matrix, C.matrix)

    def test_power(self):
        shape = (4, 3)
        F = DeformationGradient.rand(shape=shape)
        F_inv = F.inv()
        np.testing.assert_array_almost_equal(F_inv.matrix, F.power(-1).matrix, decimal=5)

    def test_seth_hill(self):
        shape = (3, 4)
        F = DeformationGradient.rand(shape=shape)
        C = F.Cauchy_Green()
        I = SymmetricSecondOrderTensor.eye()

        # Green-Lagrange
        assert np.all(F.Seth_Hill(1) == F.E)

        # Biot
        Biot = F.Seth_Hill(0.5)
        Biot_th = C.sqrt() - I
        np.testing.assert_array_almost_equal(Biot.matrix, Biot_th.matrix)

        # Hencky
        assert np.all(F.Seth_Hill(0) == 0.5 * C.log())

        # Almansi
        Alma = F.Seth_Hill(-1)
        Alma_th = 0.5 * (I - F.U.power(-2))
        np.testing.assert_allclose(Alma.matrix, Alma_th.matrix, rtol=1e-5)

    def test_biot_strain(self):
        """Test the Biot strain tensor method"""
        # Test with random deformation gradient
        shape = (3, 4)
        F = DeformationGradient.rand(shape=shape)
        C = F.Cauchy_Green()
        I = SymmetricSecondOrderTensor.eye()

        # Biot should equal Seth_Hill(0.5)
        Biot = F.Biot()
        Biot_th = F.Seth_Hill(0.5)
        np.testing.assert_array_almost_equal(Biot.matrix, Biot_th.matrix)

        # Biot should equal C.sqrt() - I
        np.testing.assert_array_almost_equal(Biot.matrix, (C.sqrt() - I).matrix)

        # Test with identity: Biot should be zero
        F_eye = DeformationGradient.eye()
        Biot_eye = F_eye.Biot()
        np.testing.assert_array_almost_equal(Biot_eye.matrix, np.zeros((3, 3)))

        # Test with diagonal stretch
        F_diag = DeformationGradient.diag([2, 1.5, 1])
        Biot_diag = F_diag.Biot()
        C_diag = F_diag.Cauchy_Green()
        expected = C_diag.sqrt().matrix - np.eye(3)
        np.testing.assert_array_almost_equal(Biot_diag.matrix, expected)

    def test_hencky_strain(self):
        """Test the Hencky (logarithmic) strain tensor method"""
        # Test with random deformation gradient
        shape = (3, 4)
        F = DeformationGradient.rand(shape=shape)
        C = F.Cauchy_Green()

        # Hencky should equal Seth_Hill(0)
        Hencky = F.Hencky()
        Hencky_th = F.Seth_Hill(0)
        np.testing.assert_array_almost_equal(Hencky.matrix, Hencky_th.matrix)

        # Hencky should equal 0.5 * log(C)
        np.testing.assert_array_almost_equal(Hencky.matrix, (0.5 * C.log()).matrix)

        # Test with identity: Hencky should be zero
        F_eye = DeformationGradient.eye()
        Hencky_eye = F_eye.Hencky()
        np.testing.assert_array_almost_equal(Hencky_eye.matrix, np.zeros((3, 3)))

        # Test with diagonal stretch
        F_diag = DeformationGradient.diag([2, 1.5, 1])
        Hencky_diag = F_diag.Hencky()
        C_diag = F_diag.Cauchy_Green()
        expected = 0.5 * C_diag.log().matrix
        np.testing.assert_array_almost_equal(Hencky_diag.matrix, expected)

    def test_logarithmic_strain(self):
        """Test the logarithmic_strain alias method"""
        shape = (3, 4)
        F = DeformationGradient.rand(shape=shape)

        # logarithmic_strain should equal Hencky
        log_strain = F.logarithmic_strain()
        Hencky = F.Hencky()
        np.testing.assert_array_almost_equal(log_strain.matrix, Hencky.matrix)

        # Test with identity
        F_eye = DeformationGradient.eye()
        log_strain_eye = F_eye.logarithmic_strain()
        np.testing.assert_array_almost_equal(log_strain_eye.matrix, np.zeros((3, 3)))

    def test_almansi_strain(self):
        """Test the Almansi strain tensor method"""
        # Test with random deformation gradient
        shape = (3, 4)
        F = DeformationGradient.rand(shape=shape, seed=321)
        C = F.Cauchy_Green()
        I = SymmetricSecondOrderTensor.eye()

        # Almansi should equal Seth_Hill(-1)
        Almansi = F.Almansi()
        Almansi_th = F.Seth_Hill(-1)
        np.testing.assert_array_almost_equal(Almansi.matrix, Almansi_th.matrix)

        # Almansi should equal 0.5 * (I - C^-1) using power for consistency
        np.testing.assert_array_almost_equal(Almansi.matrix, (0.5 * (I - C.inv())).matrix, decimal=7)

        # Almansi should equal 0.5 * (I - U^-2)
        # Note: U^-2 and C^-1 may have small numerical differences, so we use rtol
        np.testing.assert_allclose(Almansi.matrix, (0.5 * (I - F.U.power(-2))).matrix, rtol=1e-5)

        # Test with identity: Almansi should be zero
        F_eye = DeformationGradient.eye()
        Almansi_eye = F_eye.Almansi()
        np.testing.assert_array_almost_equal(Almansi_eye.matrix, np.zeros((3, 3)))

        # Test with diagonal stretch
        F_diag = DeformationGradient.diag([2, 1.5, 1])
        Almansi_diag = F_diag.Almansi()
        C_diag = F_diag.Cauchy_Green()
        expected = 0.5 * (np.eye(3) - C_diag.power(-1).matrix)
        np.testing.assert_array_almost_equal(Almansi_diag.matrix, expected, decimal=2)

    def test_seth_hill_family_consistency(self):
        """Test consistency between all Seth-Hill family strain tensors"""
        shape = (5, 3)
        F = DeformationGradient.rand(shape=shape)

        # Green-Lagrangian (n=1) should equal E
        np.testing.assert_array_almost_equal(F.Seth_Hill(1).matrix, F.E.matrix)

        # Biot (n=0.5)
        np.testing.assert_array_almost_equal(F.Biot().matrix, F.Seth_Hill(0.5).matrix)

        # Hencky (n=0)
        np.testing.assert_array_almost_equal(F.Hencky().matrix, F.Seth_Hill(0).matrix)
        np.testing.assert_array_almost_equal(F.logarithmic_strain().matrix, F.Seth_Hill(0).matrix)

        # Almansi (n=-1)
        np.testing.assert_array_almost_equal(F.Almansi().matrix, F.Seth_Hill(-1).matrix)

    def test_seth_hill_special_values(self):
        """Test Seth-Hill with various n values"""
        F = DeformationGradient.diag([2, 1.5, 1])
        C = F.Cauchy_Green()
        I = SymmetricSecondOrderTensor.eye()

        # Test n=1 (Green-Lagrangian)
        E1 = F.Seth_Hill(1)
        expected = 0.5 * (C - I)
        np.testing.assert_array_almost_equal(E1.matrix, expected.matrix)

        # Test n=0.5 (Biot)
        E05 = F.Seth_Hill(0.5)
        expected = C.sqrt() - I
        np.testing.assert_array_almost_equal(E05.matrix, expected.matrix)

        # Test n=-1 (Almansi)
        E_neg1 = F.Seth_Hill(-1)
        # For n=-1: 1/(2*(-1)) * (C^(-1) - I) = -0.5 * (C^(-1) - I) = 0.5 * (I - C^(-1))
        expected = 0.5 * (I - C.inv())
        np.testing.assert_array_almost_equal(E_neg1.matrix, expected.matrix, decimal=2)


class TestPiolaKirchoff(unittest.TestCase):
    def test_Piola_Kirchoff_1_identity(self):
        """Quand F = I, P = σ (pas de transformation)"""
        F = DeformationGradient.eye()
        sigma = StressTensor.diag([1, 2, 3])
        P = F.piola_kirchoff_1(sigma)
        np.testing.assert_array_almost_equal(P.matrix, sigma.matrix)

    def test_Piola_Kirchoff_2_identity(self):
        """Quand F = I, S = σ (pas de transformation)"""
        F = DeformationGradient.eye()
        sigma = StressTensor.diag([1, 2, 3])
        S = F.piola_kirchoff_2(sigma)
        np.testing.assert_array_almost_equal(S.matrix, sigma.matrix)

    def test_Piola_Kirchoff_1_2_relationship(self):
        """Vérifie que S = F⁻¹·P"""
        F = DeformationGradient.rand()
        sigma = StressTensor.rand()

        P = F.piola_kirchoff_1(sigma)
        S_from_P = F.inv().dot(P)

        S = F.piola_kirchoff_2(sigma)
        np.testing.assert_array_almost_equal(S.matrix, S_from_P.matrix)

    def test_Piola_Kirchoff_1_formula(self):
        """Vérifie la formule P = J·σ·F⁻ᵀ"""
        F = DeformationGradient.diag([2, 1, 1])
        sigma = StressTensor.diag([3, 4, 5])

        Finv = F.inv()
        P_expected = F.J * sigma.dot(Finv.T)
        P_actual = F.piola_kirchoff_1(sigma)

        np.testing.assert_array_almost_equal(P_actual.matrix, P_expected.matrix)

    def test_Piola_Kirchoff_2_formula(self):
        """Vérifie la formule S = J·F⁻¹·σ·F⁻ᵀ"""
        F = DeformationGradient.diag([2, 1, 1])
        sigma = StressTensor.diag([3, 4, 5])

        Finv = F.inv()
        J = F.J
        S_expected = J * Finv.dot(sigma).dot(Finv.T)
        S_actual = F.piola_kirchoff_2(sigma)

        np.testing.assert_array_almost_equal(S_actual.matrix, S_expected.matrix)

    def test_Piola_Kirchoff_2_symmetry(self):
        """PK2 doit être symétrique si σ est symétrique"""
        F = DeformationGradient.rand()
        sigma = StressTensor.rand()

        S = F.piola_kirchoff_2(sigma)
        np.testing.assert_array_almost_equal(S.matrix, S.matrix.T)

    def test_Piola_Kirchoff_array_input(self):
        """Test avec des tenseurs de contrainte en tableau"""
        shape = (3, 4)
        F = DeformationGradient.rand(shape=shape)
        sigma = StressTensor.rand(shape=shape)

        P = F.piola_kirchoff_1(sigma)
        S = F.piola_kirchoff_2(sigma)

        assert P.shape == shape
        assert S.shape == shape


if __name__ == '__main__':
    unittest.main()