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


if __name__ == '__main__':
    unittest.main()
