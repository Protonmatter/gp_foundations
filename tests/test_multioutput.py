import unittest

import numpy as np

from gp_foundations.kernels import MaternKernel
from gp_foundations.multioutput import CoregionalizationMatrix, IntrinsicCoregionalizedGP


class MultiOutputTests(unittest.TestCase):
    def test_coregionalization_from_unconstrained_is_psd(self) -> None:
        raw = np.array([[0.0, 0.0], [0.2, -0.1]])
        matrix = CoregionalizationMatrix.from_unconstrained(raw).matrix
        eigenvalues = np.linalg.eigvalsh(matrix)
        self.assertTrue(np.all(eigenvalues >= -1e-8))

    def test_sparse_observations_borrow_strength(self) -> None:
        kernel = MaternKernel(length_scale=0.3)
        coreg = CoregionalizationMatrix.from_factor(np.array([[1.0, 0.0], [0.8, 0.6]]))
        model = IntrinsicCoregionalizedGP(kernel, coreg, noise=1e-6)
        model.fit_observations(np.array([[0.5]]), np.array([1.0]), np.array([0]))
        posterior = model.posterior(np.array([[0.5]]))
        self.assertGreater(posterior.mean[0, 1], 0.5)

    def test_joint_thompson_sample_shape(self) -> None:
        kernel = MaternKernel(length_scale=0.3)
        coreg = CoregionalizationMatrix.identity(2)
        model = IntrinsicCoregionalizedGP(kernel, coreg, noise=1e-4)
        X = np.array([[0.0], [1.0]])
        Y = np.array([[0.0, 1.0], [1.0, 0.0]])
        model.fit(X, Y)
        sample = model.joint_thompson_sample(np.array([[0.25], [0.75]]), rng=np.random.default_rng(1))
        self.assertEqual(sample.shape, (2, 2))

    def test_posterior_for_output_returns_valid_shapes(self) -> None:
        kernel = MaternKernel(length_scale=0.3)
        coreg = CoregionalizationMatrix.from_factor(np.array([[1.0, 0.0], [0.3, 0.7]]))
        model = IntrinsicCoregionalizedGP(kernel, coreg, noise=1e-4)
        X = np.array([[0.0], [1.0]])
        Y = np.array([[1.0, 0.2], [0.0, -0.1]])
        model.fit(X, Y)
        posterior = model.posterior_for_output(np.array([[0.25], [0.75]]), output_index=1, return_covariance=True)
        self.assertEqual(posterior.mean.shape, (2,))
        self.assertEqual(posterior.variance.shape, (2,))
        self.assertEqual(posterior.covariance.shape, (2, 2))
        self.assertTrue(np.all(posterior.variance >= 0.0))

    def test_evaluate_grid_matches_joint_sample_layout(self) -> None:
        kernel = MaternKernel(length_scale=0.3)
        coreg = CoregionalizationMatrix.identity(2)
        model = IntrinsicCoregionalizedGP(kernel, coreg, noise=1e-4)
        evaluation = model.evaluate_grid(np.array([[0.2], [0.8]]), rng=np.random.default_rng(2))
        self.assertEqual(evaluation.points.shape, (2, 1))
        self.assertEqual(evaluation.posterior.mean.shape, (2, 2))
        self.assertEqual(evaluation.posterior.variance.shape, (2, 2))
        self.assertEqual(evaluation.sample.shape, (2, 2))

    def test_negative_log_marginal_likelihood_is_finite(self) -> None:
        X = np.linspace(0.0, 1.0, 6)[:, None]
        kernel = MaternKernel(length_scale=0.25, variance=1.1)
        coreg = CoregionalizationMatrix.from_factor(np.array([[1.0, 0.0], [0.6, 0.4]]))
        rng = np.random.default_rng(5)
        Lx = np.linalg.cholesky(kernel(X, X) + 1e-8 * np.eye(X.shape[0]))
        Z = rng.standard_normal((X.shape[0], 2))
        Y = Lx @ Z @ coreg.factor.T + 0.03 * rng.standard_normal((X.shape[0], 2))

        model = IntrinsicCoregionalizedGP(
            MaternKernel(length_scale=0.9, variance=0.2),
            CoregionalizationMatrix.identity(2),
            noise=0.2,
        ).fit(X, Y)
        self.assertTrue(np.isfinite(model.negative_log_marginal_likelihood()))

    def test_hyperparameter_optimization_recovers_positive_cross_output_structure(self) -> None:
        X = np.linspace(0.0, 1.0, 8)[:, None]
        true_kernel = MaternKernel(length_scale=0.2, variance=1.3)
        true_coreg = CoregionalizationMatrix.from_factor(np.array([[1.0, 0.0], [0.7, 0.45]]))
        rng = np.random.default_rng(6)
        Lx = np.linalg.cholesky(true_kernel(X, X) + 1e-8 * np.eye(X.shape[0]))
        Y = Lx @ rng.standard_normal((X.shape[0], 2)) @ true_coreg.factor.T
        Y = Y + 0.02 * rng.standard_normal((X.shape[0], 2))

        model = IntrinsicCoregionalizedGP(
            MaternKernel(length_scale=0.85, variance=0.3),
            CoregionalizationMatrix.identity(2),
            noise=0.2,
        ).fit(X, Y)
        initial_objective = model.negative_log_marginal_likelihood()
        result = model.optimize_hyperparameters(maxiter=70)

        self.assertTrue(np.isfinite(result.objective))
        self.assertLessEqual(result.objective, initial_objective + 1e-6)
        eigenvalues = np.linalg.eigvalsh(model.coregionalization.matrix)
        self.assertTrue(np.all(eigenvalues >= -1e-8))
        self.assertGreater(model.coregionalization.matrix[0, 1], 0.05)


if __name__ == '__main__':
    unittest.main()
