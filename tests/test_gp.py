import unittest

import numpy as np

from gp_foundations.gp import GaussianProcessRegressor, OptimizationResult
from gp_foundations.kernels import MaternKernel


class GaussianProcessTests(unittest.TestCase):
    def test_prior_without_data_has_zero_mean(self) -> None:
        gp = GaussianProcessRegressor(MaternKernel(length_scale=0.2), noise=1e-6)
        posterior = gp.posterior(np.array([[0.0], [0.5]]))
        np.testing.assert_allclose(posterior.mean, np.zeros(2))
        self.assertEqual(posterior.covariance.shape, (2, 2))

    def test_posterior_interpolates_training_points(self) -> None:
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp = GaussianProcessRegressor(MaternKernel(length_scale=0.25), noise=1e-8).fit(X, y)
        posterior = gp.posterior(X)
        np.testing.assert_allclose(posterior.mean, y, atol=1e-4)

    def test_log_marginal_likelihood_is_finite(self) -> None:
        X = np.linspace(0.0, 1.0, 6)[:, None]
        y = np.sin(2.0 * np.pi * X[:, 0])
        gp = GaussianProcessRegressor(MaternKernel(length_scale=0.2), noise=1e-4).fit(X, y)
        self.assertTrue(np.isfinite(gp.log_marginal_likelihood()))

    def test_sample_posterior_returns_expected_shape(self) -> None:
        X = np.linspace(0.0, 1.0, 4)[:, None]
        y = np.sin(2.0 * np.pi * X[:, 0])
        gp = GaussianProcessRegressor(MaternKernel(length_scale=0.2), noise=1e-4).fit(X, y)
        samples = gp.sample_posterior(np.linspace(0.0, 1.0, 10)[:, None], n_samples=3, rng=np.random.default_rng(2))
        self.assertEqual(samples.shape, (3, 10))

    def test_negative_log_marginal_likelihood_is_finite(self) -> None:
        X = np.linspace(0.0, 1.0, 8)[:, None]
        covariance = MaternKernel(length_scale=0.18, variance=1.3)(X, X) + 0.03 * np.eye(X.shape[0])
        y = np.random.default_rng(3).multivariate_normal(np.zeros(X.shape[0]), covariance)
        gp = GaussianProcessRegressor(MaternKernel(length_scale=0.7, variance=0.4), noise=0.2).fit(X, y)
        self.assertTrue(np.isfinite(gp.negative_log_marginal_likelihood()))

    def test_hyperparameter_optimization_improves_objective_and_parameters(self) -> None:
        X = np.linspace(0.0, 1.0, 10)[:, None]
        true_length_scale = 0.18
        true_variance = 1.4
        true_noise = 0.03
        covariance = MaternKernel(length_scale=true_length_scale, variance=true_variance)(X, X)
        covariance = covariance + true_noise * np.eye(X.shape[0])
        y = np.random.default_rng(4).multivariate_normal(np.zeros(X.shape[0]), covariance)

        gp = GaussianProcessRegressor(MaternKernel(length_scale=0.8, variance=0.25), noise=0.2).fit(X, y)
        initial_objective = gp.negative_log_marginal_likelihood()
        initial_distance = (
            abs(np.log(float(gp.kernel.length_scale) / true_length_scale))
            + abs(np.log(float(gp.kernel.variance) / true_variance))
            + abs(np.log(float(gp.noise) / true_noise))
        )

        result = gp.optimize_hyperparameters(maxiter=60)

        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(np.isfinite(result.objective))
        self.assertLessEqual(result.objective, initial_objective + 1e-6)

        learned_distance = (
            abs(np.log(float(gp.kernel.length_scale) / true_length_scale))
            + abs(np.log(float(gp.kernel.variance) / true_variance))
            + abs(np.log(float(gp.noise) / true_noise))
        )
        self.assertLess(learned_distance, initial_distance)


if __name__ == '__main__':
    unittest.main()
