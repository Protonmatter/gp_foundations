import unittest

import numpy as np

from gp_foundations.kernels import MaternKernel, SpectralMixtureKernel, TimeDecayKernel


class KernelTests(unittest.TestCase):
    def test_matern_kernel_is_symmetric_and_psd(self) -> None:
        x = np.linspace(0.0, 1.0, 6)[:, None]
        kernel = MaternKernel(length_scale=0.3, nu=2.5)
        gram = kernel(x)
        np.testing.assert_allclose(gram, gram.T)
        eigenvalues = np.linalg.eigvalsh(gram)
        self.assertTrue(np.all(eigenvalues >= -1e-8))

    def test_matern_kernel_decays_with_distance(self) -> None:
        kernel = MaternKernel(length_scale=0.2)
        x0 = np.array([[0.5]])
        near = kernel(x0, np.array([[0.55]]))[0, 0]
        far = kernel(x0, np.array([[1.0]]))[0, 0]
        self.assertGreater(near, far)

    def test_spectral_mixture_kernel_is_symmetric(self) -> None:
        x = np.linspace(0.0, 1.0, 5)[:, None]
        kernel = SpectralMixtureKernel(weights=[1.0, 0.5], means=[[0.0], [1.0]], scales=[[1.0], [0.4]])
        gram = kernel(x)
        np.testing.assert_allclose(gram, gram.T)

    def test_time_decay_kernel_downweights_far_inputs(self) -> None:
        base = MaternKernel(length_scale=0.5)
        kernel = TimeDecayKernel(base, decay=1.0, reference_point=0.0)
        close = kernel(np.array([[0.0]]), np.array([[0.0]]))[0, 0]
        far = kernel(np.array([[1.0]]), np.array([[1.0]]))[0, 0]
        self.assertGreater(close, far)


if __name__ == '__main__':
    unittest.main()
